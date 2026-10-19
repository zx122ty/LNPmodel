#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What does "single forward" actually buy? Measured, on the real deployed pipeline.

The question as posed
--------------------
"One model outputs both mu and sigma" vs "use the fingerprint sigma_hat". That framing
suggests the alternative to distillation is running the 13-arm ensemble at inference,
which would be expensive. It is not what happens.

What the deployed pipeline actually is
--------------------------------------
  mu   = NNLS blend of neural_2d + traditional + xgb_atompair   (3 models)
  sigma = RF(fingerprints) -> log sigma                          (1 model)

The 13-arm panel was only ever needed to CREATE sigma_hat's training target
(sigma = spread across the arm panel, dumped per molecule). At inference the arm
panel is not touched: sigma_hat reads fingerprints. So the deployed system is FOUR
models, only ONE of which is a neural forward.

That means distillation would save the marginal cost of:
    - 'traditional'    (XGB on desc+morgan)
    - 'xgb_atompair'   (XGB on atom pair)
    - the RF sigma_hat
while the neural forward -- the expensive part -- is paid in BOTH designs.

This script measures that marginal cost so the requirement can be settled with
numbers instead of intuition. Everything is timed on the same 1920 molecules and
extrapolated to the 300k-molecule virtual library (the library the project actually
screens, see build_lipids_pretrain_lmdb / the IL virtual library).

Usage:
    python 4_scripts/deploy_cost_probe.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
FEAT = ROOT / "5_outputs/feat_cache"
DS = "Sheet2"
LIBRARY_N = 300_000          # the IL virtual library the project screens
N_TIME = 1920


def timeit(fn, repeat=3):
    """Best-of-N wall time in seconds (best-of removes scheduler noise)."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> int:
    print(f"timing on n={N_TIME}, extrapolating to the {LIBRARY_N:,}-molecule library\n")
    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", "RLU (log2)"])
    y = pd.to_numeric(d["RLU (log2)"]).values.astype(float)

    F = {k: np.load(FEAT / f"{DS}_{k}.npy") for k in
         ("rdkit_desc", "morgan_r3_2048", "morgan_r2_2048", "maccs", "atompair",
          "torsion")}
    from sklearn.ensemble import RandomForestRegressor
    import xgboost as xgb

    folds = json.loads((ROOT / f"5_outputs/split_cache/{DS}_scaffold_k5_seed42.json"
                        ).read_text())["folds"][0]
    tr = np.asarray(folds["train"], int)

    rows = []

    def add(name, secs):
        per_mol = secs / N_TIME
        rows.append({"component": name, "seconds_1920": round(secs, 4),
                     "ms_per_molecule": round(per_mol * 1e3, 4),
                     "minutes_300k": round(per_mol * LIBRARY_N / 60, 2)})

    # ---- the arms the deployed mu needs ----
    X_trad = np.hstack([F["rdkit_desc"], F["morgan_r3_2048"]])
    trad = xgb.XGBRegressor(tree_method="hist", n_jobs=8, random_state=0,
                            n_estimators=1200, max_depth=8, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.4, reg_lambda=1.0)
    t_fit = timeit(lambda: trad.fit(X_trad[tr], y[tr]), repeat=1)
    add("traditional  (XGB fit)", t_fit)
    add("traditional  (XGB predict)", timeit(lambda: trad.predict(X_trad)))

    atp = xgb.XGBRegressor(tree_method="hist", n_jobs=8, random_state=0,
                           n_estimators=600, max_depth=6, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.4, reg_lambda=1.0)
    t_fit = timeit(lambda: atp.fit(F["atompair"][tr], y[tr]), repeat=1)
    add("xgb_atompair (XGB fit)", t_fit)
    add("xgb_atompair (XGB predict)", timeit(lambda: atp.predict(F["atompair"])))

    # ---- the sigma head: RF on all fingerprints (the sigma_hat estimator) ----
    z = np.load(ROOT / "5_outputs/ensemble_oof.npz")
    sigma = z["scaffold__panel"].mean(1).std(0, ddof=1)
    X_all = np.hstack([F["rdkit_desc"], F["morgan_r2_2048"], F["maccs"],
                       F["atompair"], F["torsion"]])
    rf = RandomForestRegressor(n_estimators=400, max_features=0.3, n_jobs=12,
                               random_state=0)
    t_fit = timeit(lambda: rf.fit(X_all[tr], np.log(sigma)[tr]), repeat=1)
    add("sigma_hat     (RF fit)", t_fit)
    add("sigma_hat     (RF predict)", timeit(lambda: rf.predict(X_all)))

    # ---- the neural arm (the ONLY GPU forward in either design) ----
    try:
        import torch
        import sys
        sys.path.insert(0, str(ROOT / "4_scripts"))
        sys.path.insert(0, str(ROOT / "4_scripts/Uni-Mol/unimol_tools"))
        import finetune_moltrain as fm
        fm.USE_STOCK_84M = False
        fm.CHECKPOINT = ROOT / ("2_models/continued_pretrain/unimol2_lipids_84m_v5/"
                                "checkpoints/checkpoint_1_30000.pt")
        fm._install_lipidfm_loader()
        from unimol_tools.models.unimolv2 import UniMolV2Model
        from unimol_tools.data.datahub import DataHub

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = UniMolV2Model(output_dim=1, model_size="84m").to(dev).eval()
        hub = DataHub(data={"SMILES": d["SMILES"].astype(str).tolist(), "TARGET": y},
                      is_train=False, task="regression", smiles_col="SMILES",
                      target_cols="TARGET", model_name="unimolv2", model_size="84m",
                      data_type="molecule", max_atoms=256, remove_hs=False,
                      multi_process=False, seed=42)
        Xn = hub.data["unimol_input"]
        samples = [(m, 0.0) for m in Xn[:N_TIME]]
        batches = [model.batch_collate_fn(samples[i:i + 32])
                   for i in range(0, len(samples), 32)]

        def fwd():
            with torch.no_grad():
                for b, _ in batches:
                    b = {k: (v.to(dev) if torch.is_tensor(v) else v)
                         for k, v in b.items()}
                    model(**b)
        add(f"neural_2d     (forward, {dev.upper()})", timeit(fwd, repeat=2))
        note = (f"neural forward timed on {dev}; feature/conformer generation is a "
                f"SEPARATE cost paid identically by both designs and is not counted")
    except Exception as e:                                  # noqa: BLE001
        note = f"neural timing skipped: {type(e).__name__}: {e}"

    print(f"  {'component':34s} {'ms/mol':>9} {'min / 300k':>11}")
    for r in rows:
        print(f"  {r['component']:34s} {r['ms_per_molecule']:>9.4f} "
              f"{r['minutes_300k']:>11.2f}")
    print(f"\n  {note}")

    # ---- the comparison that decides the requirement ----
    get = {r["component"]: r["minutes_300k"] for r in rows}
    def pred_only(prefix):
        return sum(v for k, v in get.items() if k.startswith(prefix) and
                   "predict" in k or (k.startswith(prefix) and "forward" in k))
    trad_p = get["traditional  (XGB predict)"]
    atp_p = get["xgb_atompair (XGB predict)"]
    rf_p = get["sigma_hat     (RF predict)"]
    nn_p = next((v for k, v in get.items() if "forward" in k), float("nan"))

    print(f"\n  --- 300k-molecule screening cost, predict-only ---")
    print(f"    neural_2d forward (paid in BOTH designs)  {nn_p:>8.2f} min")
    print(f"    traditional predict                       {trad_p:>8.2f} min")
    print(f"    xgb_atompair predict                      {atp_p:>8.2f} min")
    print(f"    sigma_hat  predict                        {rf_p:>8.2f} min")
    saved = trad_p + atp_p + rf_p
    total = nn_p + saved
    print(f"\n    distillation removes:                     {saved:>8.2f} min "
          f"({saved/total*100:.1f}% of inference)")
    print(f"    inference that remains either way:        {nn_p:>8.2f} min "
          f"({nn_p/total*100:.1f}% of inference)")

    out = {"n_timed": N_TIME, "library_n": LIBRARY_N, "components": rows,
           "note": note,
           "min_per_300k": {"neural_forward": nn_p, "traditional_predict": trad_p,
                            "xgb_atompair_predict": atp_p, "sigma_hat_predict": rf_p},
           "saved_by_distillation_min": saved,
           "remaining_either_way_min": nn_p,
           "pct_of_inference_removed": round(saved / total * 100, 1)}
    p = ROOT / "5_outputs/deploy_cost_probe.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
