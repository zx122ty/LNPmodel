#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Do ionizable-lipid pKa PROXY features carry signal the existing features miss?

Motivation: the standard advice for this domain is that the protonation state of the
ionizable amine (set by pKa) is the dominant driver of transfection, and that injecting
pKa as a feature should be worth 0.05-0.10 R2. That claim has never been tested here,
and it CANNOT be tested with the priors already in the codebase:
  lip_tert_amine_pka is 0.0 or 10.0 only, with 91.7% of Sheet2 molecules at the mode
  -- it is a near-constant binary flag, not a pKa.
  lip_ionizable_n has 3 distinct values with 76.7% at the mode.
Only lip_tail_logp (488 distinct values) is informative.

So this script builds REAL local-environment proxies for the ionizable nitrogen, in the
spirit of what determines an amine's pKa: how much alkyl character surrounds it (more
alkyl -> more electron-donating -> higher pKa), how far it sits from electron-withdrawing
ester/hydroxyl groups, and its substitution degree.

Decision rule: this is a FALSIFICATION test, run on CPU in the same 15-fold protocol as
the tuned baseline. If adding these proxies to the strongest traditional model does not
move R2, then "inject pKa features" is dead for this dataset and should not be carried
into the (far more expensive) neural fine-tune. If it does move R2, that is the trigger
to wire the same features into the neural head.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traditional_baseline_tuned import (  # noqa: E402
    DS, REPEATS, TARGET_COL, XGB_GRID, XLSX, _cache, _fit_one, folds_for, morgan,
    rdkit_descriptors,
)

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
OUT = ROOT / "5_outputs/pka_proxy_probe.json"

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import rdmolops  # noqa: E402

RDLogger.DisableLog("rdApp.*")

# Aliphatic tertiary amine — the ionizable group of an ionizable lipid. Same SMARTS as
# v6_descriptors so this probe and the pretraining priors agree on what "ionizable" means.
TERT_AMINE = Chem.MolFromSmarts(
    "[NX3;H0;!$(N-C=O);!$(N~[#8,#16]);!$(N-c);!$(N-[#6]=[#6])]")
ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2H0][#6]")
HYDROXYL = Chem.MolFromSmarts("[OX2H]")


def pka_proxies(smi: str) -> dict[str, float]:
    """Local-environment proxies for amine basicity. All pure-RDKit, no external tools."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {}
    dist = rdmolops.GetDistanceMatrix(mol)
    n_idx = [m[0] for m in mol.GetSubstructMatches(TERT_AMINE)]
    ester_idx = [m[1] for m in mol.GetSubstructMatches(ESTER)]
    oh_idx = [m[0] for m in mol.GetSubstructMatches(HYDROXYL)]

    out = {
        "n_ionizable_N": float(len(n_idx)),
        "n_ester": float(len(ester_idx)),
        "n_hydroxyl": float(len(oh_idx)),
    }

    if n_idx:
        # Alkyl character around the amine: carbons reachable within 3 bonds that are
        # not part of a carbonyl/ester. More alkyl character -> more basic amine.
        alkyl_counts = []
        for n in n_idx:
            c = 0
            for a in mol.GetAtoms():
                if a.GetSymbol() != "C" or a.GetIsAromatic():
                    continue
                if dist[n, a.GetIdx()] <= 3:
                    heavy_nbrs = [nb for nb in a.GetNeighbors()
                                  if nb.GetSymbol() != "H"]
                    if not any(nb.GetSymbol() == "O" for nb in heavy_nbrs):
                        c += 1
            alkyl_counts.append(c)
        out["n_alkyl_c_per_N"] = float(np.mean(alkyl_counts))
        out["max_alkyl_c_per_N"] = float(np.max(alkyl_counts))

        # Electron-withdrawing proximity: ester/hydroxyl groups close to the amine pull
        # electron density away and lower pKa.
        d_est = [int(dist[n, e]) for n in n_idx for e in ester_idx] or [99]
        d_oh = [int(dist[n, o]) for n in n_idx for o in oh_idx] or [99]
        out["min_dist_N_ester"] = float(min(d_est))
        out["min_dist_N_hydroxyl"] = float(min(d_oh))
        out["n_ester_within4_N"] = float(sum(1 for d in d_est if d <= 4))
        out["n_oh_within4_N"] = float(sum(1 for d in d_oh if d <= 4))

        # Substitution degree of the amine nitrogens (3 = tertiary by construction here,
        # but this also catches H-bearing variants the SMARTS filters out).
        out["mean_N_degree"] = float(np.mean([mol.GetAtomWithIdx(n).GetDegree()
                                              for n in n_idx]))
    else:
        for k in ("n_alkyl_c_per_N", "max_alkyl_c_per_N", "min_dist_N_ester",
                  "min_dist_N_hydroxyl", "n_ester_within4_N", "n_oh_within4_N",
                  "mean_N_degree"):
            out[k] = 0.0
    return out


NAMES = ["n_ionizable_N", "n_ester", "n_hydroxyl", "n_alkyl_c_per_N",
         "max_alkyl_c_per_N", "min_dist_N_ester", "min_dist_N_hydroxyl",
         "n_ester_within4_N", "n_oh_within4_N", "mean_N_degree"]


def build_proxies(smis) -> np.ndarray:
    rows = [pka_proxies(s) for s in smis]
    return np.asarray([[r.get(k, 0.0) for k in NAMES] for r in rows], dtype=np.float32)


def main():
    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", TARGET_COL])
    smis = d["SMILES"].astype(str).str.strip().tolist()
    y = pd.to_numeric(d[TARGET_COL]).values.astype(float)

    P = _cache("pka_proxies", lambda: build_proxies(smis))
    print(f"proxy features: {P.shape}")
    print(f"{'feature':22s} {'n_uniq':>7} {'std':>9} {'frac_at_mode':>13}")
    for i, k in enumerate(NAMES):
        v = P[:, i]
        mode = pd.Series(v).mode()[0]
        print(f"  {k:22s} {len(np.unique(v)):>7} {v.std():>9.3f} "
              f"{np.mean(v == mode):>13.3f}")

    base = np.hstack([
        _cache("rdkit_desc", lambda: rdkit_descriptors(smis)),
        _cache("morgan_r3_2048", lambda: morgan(smis, 3, 2048)),
    ])
    withP = np.hstack([base, P])
    print(f"\nbase  {base.shape}   base+proxies {withP.shape}")

    # Best XGB config from the tuned sweep, so the comparison isolates the FEATURES.
    params = XGB_GRID[1]
    res = {}
    for tag, X in (("base", base), ("base+pka_proxies", withP)):
        allf = []
        per_seed = {}
        for seed in REPEATS:
            folds = folds_for(seed)
            mean_r2, pf, _ = _fit_one("xgb", params, X, y, folds)
            per_seed[str(seed)] = round(mean_r2, 4)
            allf.extend(pf)
        res[tag] = dict(mean=round(float(np.mean(allf)), 4),
                        std=round(float(np.std(allf, ddof=1)), 4),
                        per_seed=per_seed)
        print(f"  {tag:18s} R2(15 folds) = {res[tag]['mean']:.4f} "
              f"+- {res[tag]['std']:.4f}   per-seed {per_seed}")

    from scipy.stats import ttest_rel
    # Recompute fold vectors for the paired test
    foldv = {}
    for tag, X in (("base", base), ("base+pka_proxies", withP)):
        v = []
        for seed in REPEATS:
            _, pf, _ = _fit_one("xgb", params, X, y, folds_for(seed))
            v.extend(pf)
        foldv[tag] = np.asarray(v)
    delta = foldv["base+pka_proxies"] - foldv["base"]
    t, p = ttest_rel(foldv["base+pka_proxies"], foldv["base"])
    print(f"\npaired (15 folds): delta = {delta.mean():+.4f}  t = {t:.3f}  p = {p:.4f}  "
          f"wins {int((delta > 0).sum())}/{len(delta)}")
    verdict = ("PROXIES HELP -> wire them into the neural head"
               if p < 0.05 and delta.mean() > 0 else
               "NO SIGNIFICANT GAIN -> pKa-proxy injection is not the lever here")
    print(f"VERDICT: {verdict}")

    OUT.write_text(json.dumps({"results": res,
                               "delta": round(float(delta.mean()), 4),
                               "t": round(float(t), 4), "p": round(float(p), 4),
                               "wins": int((delta > 0).sum()), "n_folds": len(delta),
                               "verdict": verdict}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
