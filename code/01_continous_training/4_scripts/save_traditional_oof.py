#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Save the tuned traditional model's OOF predictions so it can be ensembled.

The tuned sweep (traditional_baseline_tuned.py) recorded per-fold R2 only. Averaging
R2 numbers is not an ensemble, so the predictions themselves are needed to test
whether the traditional model adds anything on top of the neural one.

Only the single best configuration from that sweep is re-run, on every partition, and
its per-fold R2 is asserted against the recorded value -- if they disagree the sweep
and this script are not measuring the same thing and any ensemble built on them would
be meaningless.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from traditional_baseline_tuned import (  # noqa: E402
    DS, REPEATS, TARGET_COL, XLSX,
    _cache, _fit_one, folds_for, morgan, rdkit_descriptors,
)

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
OUT = ROOT / "5_outputs/traditional_oof.npz"


def main():
    sweep = json.loads((ROOT / "5_outputs/traditional_baseline_tuned.json").read_text())
    key = max(sweep["results"], key=lambda k: sweep["results"][k]["r2_mean_15"])
    info = sweep["results"][key]
    fs_name, kind = key.split("|")
    print(f"best config: {key}  (recorded {info['r2_mean_15']:.4f})")

    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", TARGET_COL])
    smis = d["SMILES"].astype(str).str.strip().tolist()
    y = pd.to_numeric(d[TARGET_COL]).values.astype(float)

    if fs_name == "desc+morgan_r3_2048":
        X = np.hstack([_cache("rdkit_desc", lambda: rdkit_descriptors(smis)),
                       _cache("morgan_r3_2048", lambda: morgan(smis, 3, 2048))])
    elif fs_name == "desc+morgan_r2_2048":
        X = np.hstack([_cache("rdkit_desc", lambda: rdkit_descriptors(smis)),
                       _cache("morgan_r2_2048", lambda: morgan(smis, 2, 2048))])
    else:
        raise SystemExit(f"feature set {fs_name} not wired into this script")

    store = {}
    for seed in REPEATS:
        folds = folds_for(seed)
        mean_r2, pf, pred = _fit_one(kind, info["params"], X, y, folds)
        recorded = [info["per_seed"][str(seed)]["folds"]][0]
        assert np.allclose(sorted(pf), sorted(recorded), atol=1e-3), (
            f"seed {seed}: recomputed folds {pf} != recorded {recorded}")
        store[f"pred_{seed}"] = pred
        store[f"r2_{seed}"] = np.asarray(pf)
        print(f"  seed {seed}: {mean_r2:.4f} (matches sweep)")

    np.savez(OUT, **store)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
