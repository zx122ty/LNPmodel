#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict transfection efficiency + calibrated uncertainty for a batch of SMILES.

    mu    = NNLS blend of neural_2d + traditional + xgb_atompair
    sigma = 1.9 * exp(RF(fingerprints))          <- the calibration factor is applied

Both are returned per molecule. See prompt/26-09-15/10-附录 §7.1 for why the
deployed system is four models, and §7.2 for what each one costs.

READ THIS BEFORE TRUSTING ANY NUMBER
------------------------------------
1. **sigma is a REPORTING quantity, not a decision rule.** P1 measured this on both
   protocols: ranking candidates by sigma finds 0.08 recall@25 where ranking by mu
   finds 0.36, and adding sigma to a UCB acquisition function does not improve on
   plain greedy. Use sigma to say how much to trust a mu -- do NOT use it to choose
   what to make. (prompt/26-09-15/09-P1不确定性报告.md §4.5)

2. **The calibration factor 1.9 makes intervals honest, not the ranking better.**
   sigma ranked errors well (Spearman +0.36) but its SCALE was ~2x too small
   (|z| < 1 held for 42% of molecules where a Gaussian would give 68%). Sorting is
   unchanged by the factor. Report sigma*1.9 as an error bar; never report the raw
   value as a std without it.

3. **Weights live here, not in the ensemble code.** The three arms are combined with
   FIXED weights from ensemble_weighted.json, not refitted per batch. Refitting on a
   screening batch would silently produce a different predictor from the one that
   scored 0.7143.

4. **Out-of-domain molecules get plausible-looking numbers.** Sheet2 is a 4CR
   combinatorial library and 1914/1920 of its molecules fall in ONE Butina cluster.
   A molecule far from that cluster is extrapolation, and this script will not tell
   you -- the fingerprint features are all bounded, so nothing saturates. If you
   screen something structurally new, check its Tanimoto to the training set first
   (--report-ood does this) before believing mu.

Usage
-----
    # single SMILES
    python 4_scripts/predict_lipids.py --smiles "CCCCCCCC(=O)N(CCN(C)C)C(CCCCC)C(=O)NC"

    # a file, top-100 by predicted mu, with domain check
    python 4_scripts/predict_lipids.py --input cands.csv --smiles-col smiles \\
        --top 100 --report-ood --out ranked.csv

    # the project's screening library
    python 4_scripts/predict_lipids.py \\
        --input ../other_model/generated_virtual_library_processed_candidates_for_screen.txt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parents[1]
MODELS = Path(os.environ.get("LIPIDFM_DEPLOY_MODELS_DIR", ROOT / "5_outputs/deploy_models"))
FEAT_CACHE = ROOT / "5_outputs/feat_cache"
FOLD_DIR = Path(os.environ.get(
    "LIPIDFM_FINETUNED_MODEL_DIR",
    ROOT / (
        "5_outputs/finetune_moltrain/"
        "Sheet2_exp_unimol2_lipids_84m_v5_checkpoint_1_30000.pt_"
        "scaffold_seed42_lr5e-05_ep40"
    ),
))
TRAIN_XLSX = Path(os.environ.get(
    "LNP_BENCHMARK_XLSX",
    PROJECT_ROOT / "benchmark/input_data/dataset.xlsx",
))

DS = "Sheet2"
BLEND_ORDER = ["neural_2d", "traditional", "xgb_atompair"]
SIGMA_FEATS = ("rdkit_desc", "morgan_r2_2048", "maccs", "atompair", "torsion")


# ============================================================ feature builders
def _desc_spec() -> tuple[list[str], np.ndarray]:
    """The descriptor column spec, pinned to the training-time definition.

    ⚠️ This is load-bearing and was a real bug. `arm_zoo.rdkit_descriptors` computes
    all 217 descriptors and then drops columns that are constant OR exceed 1e6 --
    PER CALL, on the data it was given. On Sheet2's 1920 molecules that leaves 132
    columns; on a 200-molecule screening batch it leaves 113. Feeding the second
    matrix to a model fitted on the first does not raise -- XGBoost just uses the
    wrong columns -- it silently returns numbers that look fine and mean nothing.

    So the surviving column names and the imputation medians are persisted at fit
    time and replayed here. New molecules never get to change the column set.
    """
    spec = MODELS / "desc_spec.npz"
    if not spec.exists():
        raise SystemExit(
            f"missing {spec}\n"
            f"Run: python 4_scripts/fit_deploy_models.py   (it writes the descriptor "
            f"spec alongside the models)")
    z = np.load(spec, allow_pickle=True)
    return list(z["keys"]), z["med"]


def build_desc(smis: list[str], keys: list[str], med: np.ndarray) -> np.ndarray:
    rows = []
    for s in smis:
        v = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s))
        rows.append([v[k] if v.get(k) is not None else np.nan for k in keys])
    D = np.asarray(rows, dtype=np.float64)
    D[~np.isfinite(D)] = np.nan
    idx = np.where(~np.isfinite(D))
    D[idx] = np.take(med, idx[1])
    return D.astype(np.float32)


def morgan(smis, radius, nbits):
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
    return np.asarray([np.asarray(gen.GetFingerprint(Chem.MolFromSmiles(s)),
                                  dtype=np.float32) for s in smis], dtype=np.float32)


def maccs(smis):
    return np.asarray([np.asarray(AllChem.GetMACCSKeysFingerprint(Chem.MolFromSmiles(s)),
                                  dtype=np.float32) for s in smis], dtype=np.float32)


def atompair(smis, nbits=1024):
    gen = AllChem.GetAtomPairGenerator(fpSize=nbits)
    return np.asarray([np.asarray(gen.GetFingerprint(Chem.MolFromSmiles(s)),
                                  dtype=np.float32) for s in smis], dtype=np.float32)


def torsion(smis, nbits=1024):
    gen = AllChem.GetTopologicalTorsionGenerator(fpSize=nbits)
    return np.asarray([np.asarray(gen.GetFingerprint(Chem.MolFromSmiles(s)),
                                  dtype=np.float32) for s in smis], dtype=np.float32)


def build_all(smis: list[str]) -> dict[str, np.ndarray]:
    keys, med = _desc_spec()
    return {"desc": build_desc(smis, keys, med),
            "morgan_r2_2048": morgan(smis, 2, 2048),
            "morgan_r3_2048": morgan(smis, 3, 2048),
            "maccs": maccs(smis),
            "atompair": atompair(smis),
            "torsion": torsion(smis)}


# ================================================================ neural arm
def neural_predict(smis: list[str], device: str = "cuda") -> np.ndarray:
    """The fine-tuned LipidFM forward. GPU if available.

    Features are built through the model's own pipeline (conformers + the full bond
    graph). ⚠️ Passing RDKit mols is mandatory -- the atoms/coordinates route loses
    the bond graph entirely (CLAUDE.md 坑 #2, measured -0.108).

    The checkpoint here is the 15-fold fold-0 model of scaffold/seed42: the same
    fold whose 384 test molecules the CPU arms were refit without, so all four
    components of this predictor share one hold-out.
    """
    import torch
    for tools_dir in (
        ROOT / "4_scripts/Uni-Mol/unimol_tools",
        PROJECT_ROOT / "code/unimol_tools-main",
    ):
        if tools_dir.is_dir():
            sys.path.insert(0, str(tools_dir))
            break
    from unimol_tools.models.unimolv2 import UniMolV2Model
    from unimol_tools.data.conformer import UniMolV2Feature

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model = UniMolV2Model(output_dim=1, model_size="84m", load_pretrained=False)
    sd = torch.load(FOLD_DIR / "model_0.pth", map_location="cpu")
    model.load_state_dict(sd["model_state_dict"])
    model = model.to(dev).eval()
    scaler = __import__("joblib").load(FOLD_DIR / "target_scaler.ss")

    feats = UniMolV2Feature(data_type="molecule", max_atoms=256, remove_hs=False,
                            multi_process=True, seed=42).transform(smis)
    payload = feats[0] if isinstance(feats, tuple) else feats
    samples = [(m, 0.0) for m in payload]
    preds = []
    with torch.no_grad():
        for i in range(0, len(samples), 128):
            b, _ = model.batch_collate_fn(samples[i:i + 128])
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            preds.append(model(**b).cpu().numpy().ravel())
    z = np.concatenate(preds)
    return scaler.inverse_transform(z.reshape(-1, 1)).ravel()


# ============================================================== domain check
def tanimoto_to_train(smis: list[str], sample: int = 2000) -> np.ndarray:
    """Max Tanimoto (ECFP4) of each molecule to the training set.

    There is no OOD detector in this pipeline and the features cannot saturate, so
    a molecule nothing like the training data still gets a confident-looking mu.
    This is the cheapest available sanity check.
    """
    d = pd.read_excel(TRAIN_XLSX, sheet_name=DS).dropna(subset=["SMILES", "RLU (log2)"])
    tr = d["SMILES"].astype(str).str.strip().tolist()
    gen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    A = np.asarray([np.asarray(gen.GetFingerprint(Chem.MolFromSmiles(s)), np.float32)
                    for s in smis], np.float32)
    B = np.asarray([np.asarray(gen.GetFingerprint(Chem.MolFromSmiles(s)), np.float32)
                    for s in tr[:sample]], np.float32)
    inter = A @ B.T
    a = A.sum(1, keepdims=True)
    b = B.sum(1, keepdims=True)
    return (inter / np.maximum(a + b.T - inter, 1e-9)).max(1)


# ==================================================================== main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smiles", default=None, help="one SMILES")
    ap.add_argument("--input", type=Path, default=None,
                    help="CSV/TXT of candidates (one SMILES per line if .txt)")
    ap.add_argument("--smiles-col", default=None,
                    help="column name; omit for a .txt or a single-column file")
    ap.add_argument("--out", type=Path, default=None, help="write ranked CSV here")
    ap.add_argument("--top", type=int, default=None, help="keep only the top-N by mu")
    ap.add_argument("--report-ood", action="store_true",
                    help="also compute max Tanimoto to the training set")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--no-neural", action="store_true",
                    help="skip the neural arm (CPU-only, for a quick smoke test; "
                         "mu is then NOT the validated predictor)")
    args = ap.parse_args()

    if args.smiles:
        smis = [args.smiles]
    elif args.input:
        if args.input.suffix.lower() == ".txt":
            smis = [l.strip() for l in args.input.read_text().splitlines()
                    if l.strip() and not l.lower().startswith("smiles")]
        else:
            df = pd.read_csv(args.input)
            col = args.smiles_col or df.columns[0]
            smis = df[col].astype(str).str.strip().tolist()
    else:
        raise SystemExit("need --smiles or --input")

    # drop anything RDKit cannot parse, and say so -- silently skipping molecules
    # in a screening run is how you end up reporting a ranking over a subset nobody
    # knows about.
    ok, bad = [], []
    for s in smis:
        (ok if Chem.MolFromSmiles(s) is not None else bad).append(s)
    if bad:
        print(f"  !! {len(bad)}/{len(smis)} SMILES failed to parse and were DROPPED")
        for b in bad[:5]:
            print(f"       {b[:70]}")
    smis = ok
    print(f"predicting {len(smis)} molecules")

    manifest = json.loads((MODELS / "manifest.json").read_text())
    w = np.asarray(manifest["blend"]["weights_used"], float)
    cal = float(manifest["calibration"]["sigma_scale_factor"])
    F = build_all(smis)
    print(f"  features built (desc {F['desc'].shape[1]} cols)")

    import xgboost as xgb
    import joblib
    t0 = time.time()
    trad = xgb.XGBRegressor()
    trad.load_model(str(MODELS / "traditional.json"))
    atom = xgb.XGBRegressor()
    atom.load_model(str(MODELS / "xgb_atompair.json"))
    # Feature sets must match the fitted models exactly: traditional = desc +
    # morgan_r3_2048, xgb_atompair = atompair alone.
    mu_trad = trad.predict(np.hstack([F["desc"], F["morgan_r3_2048"]]))
    mu_atom = atom.predict(F["atompair"])
    print(f"  CPU arms done in {time.time()-t0:.1f}s")

    rf = joblib.load(MODELS / "sigma_hat_rf.joblib")
    # ⚠️ Column ORDER must match how the RF was fitted, not just the set of blocks.
    # fit_deploy_models.py stacks (desc, morgan_r2_2048, maccs, atompair, torsion);
    # numpy re-stacks by position, so a different order silently produces a
    # differently-meaning predictor rather than an error.
    X_sig = np.hstack([F["desc"], F["morgan_r2_2048"], F["maccs"],
                       F["atompair"], F["torsion"]])
    sigma = cal * np.exp(rf.predict(X_sig))

    if args.no_neural:
        print("  !! --no-neural: mu is the 2-CPU-arm blend, NOT the validated "
              "predictor. Use for plumbing checks only.")
        mu_neural = np.zeros(len(smis))
        w = np.asarray([0.0, w[1] / (w[1] + w[2]), w[2] / (w[1] + w[2])])
    else:
        t0 = time.time()
        mu_neural = neural_predict(smis, args.device)
        print(f"  neural arm done in {time.time()-t0:.1f}s")

    mu = w[0] * mu_neural + w[1] * mu_trad + w[2] * mu_atom

    out = pd.DataFrame({
        "smiles": smis,
        "mu_log2_RLU": mu,
        "sigma_calibrated_log2": sigma,
        "mu_minus_1sigma_log2": mu - sigma,
        "mu_plus_1sigma_log2": mu + sigma,
        "predicted_RLU": np.power(2.0, mu),
        "lower_1sigma_RLU": np.power(2.0, mu - sigma),
        "upper_1sigma_RLU": np.power(2.0, mu + sigma),
    })
    if args.report_ood:
        out["max_tanimoto_to_train"] = tanimoto_to_train(smis)

    if args.top:
        out = out.sort_values("mu_log2_RLU", ascending=False).head(args.top).reset_index(drop=True)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n" + out.head(20).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.out:
        out.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}  ({len(out)} rows)")

    print(f"\n⚠️  mu = {w[0]:.3f}*neural + {w[1]:.3f}*traditional + {w[2]:.3f}*atompair"
          f"   (weights from ensemble_weighted.json, NOT refit)")
    print(f"⚠️  sigma is CALIBRATED (x{cal}). Use it to judge trust in mu; "
          f"do NOT use it to rank candidates (P1 §4.5).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
