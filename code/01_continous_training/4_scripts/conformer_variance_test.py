#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the 3D arm's deficit caused by conformational variance? A direct test.

The error attribution found neural_3d's deficit grows monotonically with rotatable
bonds (rho=+0.059, p=0.0095). That is consistent with the conformational-variance
story but does not prove it -- rotatable bonds could proxy for something else.

This script tests the mechanism rather than the proxy. The claim is:

    a molecule with ONE rigid 3D conformer should be fine under a single-conformer
    encoder; a molecule whose conformer ensemble is spread out will not be.

So for each molecule we measure the SPREAD of its own conformer ensemble (embed N
conformers, measure how much the interatomic-distance matrix moves across them) and
ask whether neural_3d's deficit grows with that spread. This is the causal quantity
the hypothesis names, not a correlate of it.

Two spreads are computed because they fail differently:
  * interatomic-distance spread over a real ETKDG ensemble -- the quantity Uni-Mol2's
    distance-based attention bias literally has to absorb.
  * end-to-end distance spread -- captures whether the flexible tails actually
    sample different global shapes.

If the deficit tracks ensemble spread with a steeper, cleaner trend than it tracks
rotatable bonds, the invariance-loss route (penalising disagreement between two
conformers' CLS embeddings) is targeting the right thing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from scipy.stats import spearmanr

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
sys.path.insert(0, str(ROOT / "4_scripts"))
from conformer_3d import EMBED_ATTEMPTS, optimise_conformer  # noqa: E402

XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
OUT = ROOT / "5_outputs/conformer_variance_test.json"
CACHE = ROOT / "5_outputs/feat_cache/Sheet2_ensemble_spread.npy"

DS = "Sheet2"
TARGET_COL = "RLU (log2)"
SEEDS = (42, 1337, 2024)
EXP = ("Sheet2_exp_unimol2_lipids_84m_v5_checkpoint_1_30000.pt_scaffold_seed{}"
       "_lr5e-05_ep40")
N_CONF = 8


def ensemble_spread(smi: str, n_conf: int = N_CONF) -> tuple[float, float, int]:
    """(mean spread of the interatomic-distance matrix, end-to-end distance std, n_conf).

    Spread is measured on the INTERATOMIC DISTANCE MATRIX, not on Cartesian RMSD.
    Two reasons, one practical and one principled:

    * Practical: GetBestRMS solves a graph-isomorphism alignment per pair. With N=8
      that is 28 alignment calls per molecule; measured, it failed to finish 1920
      molecules in 45 minutes across 16 workers.
    * Principled: Uni-Mol2's attention bias is built from interatomic distances
      (se3_invariant_kernel over `dist`). Distance-matrix spread is therefore the
      quantity the encoder literally has to absorb, whereas Cartesian RMSD conflates
      that with global rotation/translation that the model is invariant to anyway.

    Conformers come from the same ETKDGv3 recipe the fine-tune feeds the model, so the
    spread measured here is the spread the encoder actually sees.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.nan, np.nan, 0
    mol = Chem.AddHs(mol)
    conf_ids = []
    for max_iters in EMBED_ATTEMPTS:
        p = AllChem.ETKDGv3()
        p.randomSeed = 20260805
        p.enforceChirality = True
        p.useRandomCoords = True
        p.pruneRmsThresh = -1.0
        p.numThreads = 1
        p.maxIterations = max_iters
        try:
            conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_conf, params=p))
        except Exception:
            conf_ids = []
        if len(conf_ids) >= 2:
            break
    if len(conf_ids) < 2:
        return np.nan, np.nan, len(conf_ids)

    heavy = Chem.RemoveHs(mol)
    nh = heavy.GetNumAtoms()
    iu = np.triu_indices(nh, k=1)

    mats, e2e = [], []
    for cid in conf_ids:
        pos = mol.GetConformer(cid).GetPositions()[:nh]
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        mats.append(d[iu])
        e2e.append(float(d.max()))
    mats = np.asarray(mats)                     # [n_conf, n_pairs]

    # Mean per-pair standard deviation across the ensemble: "how much does a typical
    # interatomic distance move between conformers". This is the direct analogue of
    # the noise the single-conformer encoder is asked to be invariant to.
    spread = float(np.mean(np.std(mats, axis=0)))
    return spread, float(np.std(e2e)), len(conf_ids)


def build_spread(smis):
    from concurrent.futures import ProcessPoolExecutor
    import os
    nw = max(1, min(16, (os.cpu_count() or 8) - 4))
    print(f"  embedding {N_CONF} conformers x {len(smis)} molecules ({nw} workers) …",
          flush=True)
    with ProcessPoolExecutor(max_workers=nw) as ex:
        out = list(ex.map(_spread_task, smis, chunksize=4))
    return np.asarray(out, dtype=np.float64)


def _spread_task(smi):
    r, e, n = ensemble_spread(smi)
    return [r, e, n]


def neural_pred(seed, is3d):
    p = ROOT / "5_outputs/finetune_moltrain" / (EXP.format(seed) + ("_3dconf" if is3d else ""))
    z = np.asarray(joblib.load(p / "cv.data")).ravel()
    sc = joblib.load(p / "target_scaler.ss")
    return sc.inverse_transform(z.reshape(-1, 1)).ravel()


def main():
    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", TARGET_COL])
    smis = d["SMILES"].astype(str).str.strip().tolist()
    y = pd.to_numeric(d[TARGET_COL]).values.astype(float)
    print(f"{DS}: n={len(y)}")

    if CACHE.exists():
        S = np.load(CACHE)
        print(f"  spread cache hit <- {CACHE.name}")
    else:
        S = build_spread(smis)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(CACHE, S)
    rmsd_spread, e2e_spread, n_conf = S[:, 0], S[:, 1], S[:, 2]
    ok = np.isfinite(rmsd_spread) & (n_conf >= 2)
    print(f"  usable molecules: {ok.sum()}/{len(smis)}  "
          f"median distance spread = {np.nanmedian(rmsd_spread):.3f} A")

    e3 = np.mean([(neural_pred(s, True) - y) ** 2 for s in SEEDS], axis=0)
    e2 = np.mean([(neural_pred(s, False) - y) ** 2 for s in SEEDS], axis=0)
    dE = e3 - e2                      # >0 => 2D arm closer
    rotb = np.array([rdMolDescriptors.CalcNumRotatableBonds(Chem.MolFromSmiles(s))
                     for s in smis])

    print(f"\ndE = E_3d - E_2d  (>0 => 2D closer);  mean = {dE.mean():+.4f}")
    print(f"  2D closer on {int((dE > 0).sum())}/{len(dE)} molecules\n")

    results = {}
    for name, v in (("distance_matrix_spread", rmsd_spread),
                    ("end_to_end_spread", e2e_spread),
                    ("n_rot_bonds", rotb.astype(float))):
        m = ok if name != "n_rot_bonds" else np.ones(len(smis), bool)
        rho, p = spearmanr(v[m], dE[m])
        results[name] = dict(rho=round(float(rho), 4), p=round(float(p), 6),
                             n=int(m.sum()))
        print(f"  {name:22s} rho = {rho:+.4f}  p = {p:.2e}  (n={m.sum()})")

    # Head-to-head: does ensemble spread explain MORE than rotatable bonds?
    # Partial correlation controls for rotatable bonds, since more rotatable bonds
    # mechanically produce more ensemble spread.
    m = ok
    def resid(a, b):
        a, b = a[m], b[m]
        A = np.vstack([np.ones_like(b), b]).T
        beta, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ beta
    pr_rho, pr_p = spearmanr(resid(rmsd_spread, rotb.astype(float)), resid(dE, rotb.astype(float)))
    print(f"\n  PARTIAL correlation of ensemble spread with dE, controlling for rotatable bonds:")
    print(f"    rho = {pr_rho:+.4f}  p = {pr_p:.2e}")
    results["partial_rmsd_controlling_rotb"] = dict(rho=round(float(pr_rho), 4),
                                                    p=round(float(pr_p), 6))
    verdict = ("ensemble spread carries signal BEYOND rotatable-bond count"
               if pr_p < 0.05 else
               "ensemble spread adds nothing beyond rotatable-bond count")
    print(f"    -> {verdict}")

    OUT.write_text(json.dumps({
        "n": int(len(y)), "usable": int(ok.sum()), "n_conformers": N_CONF,
        "trends": results, "verdict": verdict,
        "median_distance_spread": round(float(np.nanmedian(rmsd_spread)), 4),
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
