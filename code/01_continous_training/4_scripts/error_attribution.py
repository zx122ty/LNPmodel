#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Error attribution: WHERE do the arms disagree, and does the pattern have a cause?

Context. On 15 paired folds the neural fine-tune (0.6807) beats the tuned traditional
baseline (0.6448) by +0.036 (p=0.0035) -- yet their RESIDUALS correlate only 0.76,
lower than the 0.79 between the two neural arms, and averaging them recovers +0.023
(p=0.0003). Complementarity that large is worth explaining BEFORE spending days on
architecture work to "fix" either arm.

Method. Per-molecule squared error (the quantity R2 measures), averaged over the 3
scaffold partitions -- every molecule lands in a test fold exactly once per partition,
so each gets 3 independent test errors. Then

    dE = E_neural - E_traditional        (>0  => TRADITIONAL was closer on that molecule)

Molecules are binned by physicochemical descriptor and by label value.

Two pairings are run for different reasons:
  * neural_2d vs traditional  -- explains the ENSEMBLE gain (where does the tree help?)
  * neural_3d vs neural_2d    -- tests the CONFORMATIONAL-VARIANCE hypothesis directly.
    If 3D's deficit grows with molecular flexibility, that is the mechanism; if the
    deficit is flat, then 3D is simply uniformly weaker and an invariance loss will
    not rescue it.

Statistical care. ~10 bins per descriptor invites over-reading, so:
  * every bin carries a bootstrap 95% CI (1000 resamples);
  * a Spearman trend test across all molecules is reported alongside -- a monotone
    drift is far more credible than a single bin whose CI clears zero;
  * bins smaller than 40 molecules are dropped rather than reported.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
OUT = ROOT / "5_outputs/error_attribution.json"

DS = "Sheet2"
TARGET_COL = "RLU (log2)"
SEEDS = (42, 1337, 2024)
EXP = ("Sheet2_exp_unimol2_lipids_84m_v5_checkpoint_1_30000.pt_scaffold_seed{}"
       "_lr5e-05_ep40")
N_BOOT = 1000
MIN_BIN = 40

PAIRINGS = [
    ("neural_2d", "traditional",
     "explains the ensemble: where does the tree model actually help?"),
    ("neural_3d", "neural_2d",
     "tests conformational variance: does 3D's deficit grow with flexibility?"),
]


def load_labels():
    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", TARGET_COL])
    smis = d["SMILES"].astype(str).str.strip().tolist()
    y = pd.to_numeric(d[TARGET_COL]).values.astype(float)
    return smis, y


def neural_pred(seed: int, is3d: bool):
    p = ROOT / "5_outputs/finetune_moltrain" / (EXP.format(seed) + ("_3dconf" if is3d else ""))
    if not (p / "cv.data").exists():
        return None
    z = np.asarray(joblib.load(p / "cv.data")).ravel()
    sc = joblib.load(p / "target_scaler.ss")
    return sc.inverse_transform(z.reshape(-1, 1)).ravel()


def per_molecule_error(arm: str, y) -> np.ndarray:
    """Mean squared error per molecule, averaged over the partitions where it was tested.

    Averaging across partitions is what makes a per-molecule number stable enough to
    bin: a single fold's error for one molecule is dominated by noise.
    """
    errs = []
    if arm == "traditional":
        z = np.load(ROOT / "5_outputs/traditional_oof.npz")
        for seed in SEEDS:
            k = f"pred_{seed}"
            if k in z.files:
                errs.append((z[k] - y) ** 2)
    else:
        is3d = arm == "neural_3d"
        for seed in SEEDS:
            p = neural_pred(seed, is3d)
            if p is not None:
                errs.append((p - y) ** 2)
    if not errs:
        raise SystemExit(f"no predictions for arm {arm}")
    return np.mean(np.vstack(errs), axis=0)


def build_features(smis):
    """Descriptors chosen to probe distinct hypotheses, not to maximise R2."""
    scaf_counts = Counter()
    rows = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        try:
            scaf = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
        except Exception:
            scaf = "__fail__"
        scaf_counts[scaf] += 1
        rows.append(dict(
            n_rot_bonds=rdMolDescriptors.CalcNumRotatableBonds(m),
            heavy_atoms=m.GetNumHeavyAtoms(),
            mol_logp=Crippen.MolLogP(m),
            tpsa=rdMolDescriptors.CalcTPSA(m),
            frac_csp3=rdMolDescriptors.CalcFractionCSP3(m),
            n_rings=rdMolDescriptors.CalcNumRings(m),
            _scaf=scaf,
        ))
    df = pd.DataFrame(rows)
    # Chemotype frequency: under a scaffold split a test molecule's scaffold is entirely
    # absent from its own training fold, so this measures how well the chemotype is
    # represented ACROSS the dataset -- a proxy for how much transfer is even possible.
    df["scaf_freq"] = df["_scaf"].map(scaf_counts)
    return df.drop(columns=["_scaf"])


def boot_ci(x, n=N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def report_bins(dE, values, name, label, out):
    """Quantile-bin a descriptor and report mean dE per bin with a bootstrap CI."""
    try:
        bins = pd.qcut(values, q=min(6, len(np.unique(values))), duplicates="drop")
    except Exception:
        return
    print(f"\n  by {name}  (dE>0 => traditional/2nd arm closer)")
    print(f"    {'bin':>22} {'n':>5} {'mean dE':>9} {'95% CI':>20} {'winner':>10}")
    rows = []
    for interval, grp in pd.DataFrame({"b": bins, "dE": dE}).groupby("b", observed=True):
        x = grp["dE"].values
        if len(x) < MIN_BIN:
            continue
        lo, hi = boot_ci(x)
        winner = "traditional" if x.mean() > 0 else "neural"
        star = " *" if (lo > 0 or hi < 0) else ""
        print(f"    {str(interval):>22} {len(x):>5} {x.mean():>+9.3f} "
              f"[{lo:>+7.3f},{hi:>+7.3f}] {winner:>10}{star}")
        rows.append(dict(bin=str(interval), n=int(len(x)), mean=round(float(x.mean()), 4),
                         ci=[round(lo, 4), round(hi, 4)], winner=winner))
    rho, p = spearmanr(values, dE)
    print(f"    trend: Spearman rho = {rho:+.4f}  p = {p:.4f}"
          f"{'  <-- MONOTONE' if p < 0.05 else ''}")
    out[label] = dict(bins=rows, spearman_rho=round(float(rho), 4),
                      spearman_p=round(float(p), 4))


def main():
    smis, y = load_labels()
    print(f"{DS}: n={len(y)}")

    arms = {}
    for a in ("neural_2d", "neural_3d", "traditional"):
        arms[a] = per_molecule_error(a, y)
        print(f"  {a:12s} mean squared error per molecule = {arms[a].mean():.4f}")

    feats = build_features(smis)
    print(f"\ndescriptors: {list(feats.columns)}")
    print(feats.describe().T[["min", "50%", "max"]].round(2).to_string())

    store = {}
    for a, b, why in PAIRINGS:
        dE = arms[a] - arms[b]
        print(f"\n{'='*72}")
        print(f"  {a} vs {b}")
        print(f"  dE = E_{a} - E_{b};  dE>0 => {b} closer.  mean dE = {dE.mean():+.4f}")
        print(f"  {b} wins on {int((dE > 0).sum())}/{len(dE)} molecules")
        print(f"  ({why})")
        sub = {}
        for fname in feats.columns:
            report_bins(dE, feats[fname].values, fname, fname, sub)
        # label-dimension: is one arm better at high transfection and the other at low?
        report_bins(dE, y, "target value (y)", "target_y", sub)
        store[f"{a}_vs_{b}"] = sub

    OUT.write_text(json.dumps(store, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
