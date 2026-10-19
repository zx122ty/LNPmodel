#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tta_moltrain.py — conformer TTA (test-time augmentation) for MolTrain fine-tuned
LipidFM models.

Motivation: our pretraining-side ablations are all within ±0.02 noise, and the
single largest untouched lever is the conformer side. MolTrain generates ONE
conformer per SMILES at both train and predict time (`inner_smi2coords`, seed=42).
TTA evaluates each test molecule over N independently-seeded conformers and
averages the fold-model predictions — a pure inference-side change (no retraining).

Paired design (do not change): identical cached folds, identical fold models
(`model_{fold}.pth`) produced by finetune_moltrain.py, identical scaler. The ONLY
difference from the baseline run is how many conformers each test molecule gets.

Usage:
  conda run -n unimol2 python 4_scripts/tta_moltrain.py \
      --exp-dir 5_outputs/finetune_moltrain/<exp_tag> \
      --dataset Sheet2 --split scaffold --seed 1337 --n-confs 11
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "4_scripts/Uni-Mol/unimol_tools"))

from unimol_tools.data.conformer import inner_smi2coords, mol2unimolv2  # noqa: E402
from unimol_tools.models.unimolv2 import UniMolV2Model  # noqa: E402

BENCHMARK_XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
SPLIT_CACHE_DIR = REPO_ROOT / "5_outputs/split_cache"
CKPT_DIR = REPO_ROOT / "2_models/continued_pretrain"

DATASETS = {
    "Sheet2": {"target_col": "RLU (log2)", "log10": False},
    # Sheet3's target gained an IGROV1_ prefix in the 2026-09-13 data fix.
    "Sheet3": {"target_col": ("IGROV1_transfection_efficiency", "transfection_efficiency"),
               "log10": True},
}

# v6d backbone = the model this TTA experiment is run on (per user decision)
LIPIDFM_CKPT = CKPT_DIR / "unimol2_lipids_84m_v6d/checkpoints/checkpoint_1_30000.pt"


def build_conformer_feats(smiles: str, max_atoms: int, seed: int):
    """One conformer for one SMILES — via MolTrain's own inner_smi2coords.

    Reusing the library's generator (rather than a stricter hand-rolled ETKDGv3)
    matters: inner_smi2coords falls back to 2D coordinates when 3D embedding
    fails, so coverage is ~100% instead of the ~19% a strict ETKDGv3 gives on
    these large flexible lipids.
    """
    try:
        mol = inner_smi2coords(smiles, seed=seed, mode="fast",
                               remove_hs=False, return_mol=True)
        if mol is None:
            return None
        return mol2unimolv2(mol, max_atoms, remove_hs=False)
    except Exception:
        return None


def _conformer_task(task: tuple) -> tuple:
    """Worker wrapper (module-level so ProcessPool can pickle it)."""
    smiles, max_atoms, seed = task
    return build_conformer_feats(smiles, max_atoms, seed)


def pregenerate_conformers(smiles_list: list[str], n_confs: int, max_atoms: int,
                           n_workers: int, cache_path: Path | None = None) -> list[list]:
    """All (mol, conf) featurizations, computed once and reused across folds.

    Conformer embedding is CPU-bound and dominates runtime; the 5 folds partition
    the same molecule set, so generating once and caching to disk makes reruns
    (e.g. after a downstream bug fix) nearly free.

    Conformer r uses seed 42 for r=0 (byte-identical to MolTrain's own default
    path) and 42 + 1000*r for the extra conformers.
    """
    import pickle
    from concurrent.futures import ProcessPoolExecutor

    if cache_path is not None and cache_path.exists():
        print(f"[conf] loading cached conformers <- {cache_path}", flush=True)
        with cache_path.open("rb") as fh:
            return pickle.load(fh)

    tasks = []
    for smi in smiles_list:
        for r in range(n_confs):
            tasks.append((smi, max_atoms, 42 if r == 0 else 42 + 1000 * r))
    print(f"[conf] generating {len(tasks):,} conformers "
          f"({len(smiles_list)} mols x {n_confs} confs, {n_workers} workers) …",
          flush=True)
    feats = [None] * len(tasks)
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, res in enumerate(ex.map(_conformer_task, tasks, chunksize=8)):
            feats[i] = res
            if (i + 1) % 2000 == 0:
                print(f"[conf]   {i + 1}/{len(tasks)}", flush=True)
    fail = sum(1 for f in feats if f is None)
    print(f"[conf] done; failed {fail:,}/{len(tasks):,}", flush=True)

    out: list[list] = [
        [feats[j * n_confs + r] for r in range(n_confs)]
        for j in range(len(smiles_list))
    ]
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as fh:
            pickle.dump(out, fh)
        print(f"[conf] cached -> {cache_path}", flush=True)
    return out


def install_lipidfm_loader(ckpt: Path) -> None:
    """Make UniMolV2Model load OUR pretrained backbone at construction time."""
    orig = UniMolV2Model.load_pretrained_weights

    def _load(self, path=None, strict=False):
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        backbone_missing = [k for k in missing if not k.startswith("classification_head")]
        if backbone_missing:
            raise RuntimeError(f"Backbone incomplete: {len(backbone_missing)} keys missing")
        print(f"    [load] backbone {ckpt.name}: {len(sd)} keys")

    UniMolV2Model.load_pretrained_weights = _load


def build_model(cfg: dict, device: str) -> UniMolV2Model:
    install_lipidfm_loader(LIPIDFM_CKPT)
    return UniMolV2Model(
        output_dim=1,
        model_size=cfg.get("model_size", "84m"),
        remove_hs=cfg.get("remove_hs", False),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Conformer TTA evaluation for MolTrain fold models.")
    ap.add_argument("--exp-dir", type=Path, required=True,
                    help="MolTrain run dir containing model_{fold}.pth + target_scaler.ss")
    ap.add_argument("--dataset", choices=list(DATASETS), default="Sheet2")
    ap.add_argument("--split", default="scaffold")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n-confs", type=int, default=11,
                    help="conformers per test molecule (seed 42 + i)")
    ap.add_argument("--max-atoms", type=int, default=256)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    exp_dir = args.exp_dir.resolve()
    cfg_path = exp_dir / "config.yaml"
    cfg = {}
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    print(f"exp_dir={exp_dir}")
    print(f"n_confs={args.n_confs}  split={args.split}  seed={args.seed}")

    # ── data + folds (identical to finetune_moltrain.py) ───────────────────
    ds = DATASETS[args.dataset]
    df = pd.read_excel(BENCHMARK_XLSX, sheet_name=args.dataset)
    smiles = df["SMILES"].astype(str).tolist()
    col = ds["target_col"]
    if isinstance(col, tuple):
        col = next((c for c in col if c in df.columns), None)
        if col is None:
            raise SystemExit(f"{args.dataset} target column not found in {list(df.columns)}")
    y = df[col].astype(float).to_numpy()
    if ds["log10"]:
        y = np.log10(y)

    if args.split == "sequential":
        sys.path.insert(0, str(REPO_ROOT / "4_scripts"))
        import lumi_protocol
        folds = lumi_protocol.sequential_folds(len(smiles), args.kfold)
    else:
        import hashlib
        cache = SPLIT_CACHE_DIR / f"{args.dataset}_{args.split}_k{args.kfold}_seed{args.seed}.json"
        cached = json.loads(cache.read_text())
        want = hashlib.sha256("\n".join(smiles).encode()).hexdigest()[:16]
        if cached.get("smiles_sha256") != want:
            raise SystemExit(f"fold cache mismatch: {cache}")
        folds = [(np.asarray(f["train"], int), np.asarray(f["test"], int))
                 for f in cached["folds"]]

    import joblib
    scaler = joblib.load(exp_dir / "target_scaler.ss")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── pre-generate all conformers ONCE (shared by all folds) ─────────────
    import os
    n_workers = max(1, (os.cpu_count() or 2) - 2)
    conf_cache = exp_dir / f"tta_confs_{args.dataset}_{args.n_confs}.pkl"
    conf_feats = pregenerate_conformers(smiles, args.n_confs, args.max_atoms,
                                        n_workers, cache_path=conf_cache)

    # ── per-fold: load fold model, predict each test mol over n_confs ──────
    tta_per_fold, base_per_fold = [], []
    for fold, (tr_idx, te_idx) in enumerate(folds):
        fold_model_path = exp_dir / f"model_{fold}.pth"
        if not fold_model_path.exists():
            raise SystemExit(f"missing fold model: {fold_model_path}")
        model = build_model(cfg, device)
        sd = torch.load(fold_model_path, map_location="cpu", weights_only=False)
        state = sd.get("model_state_dict", sd)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  fold {fold}: loaded model_{fold}.pth "
              f"(missing {len(missing)}, unexpected {len(unexpected)})")
        model = model.to(device).eval()

        test_smiles = [smiles[i] for i in te_idx]
        n_test = len(test_smiles)

        # predictions[r][j] = fold-model prediction for molecule j, conformer r
        # NOTE: conf_feats is indexed by the GLOBAL molecule index (te_idx[j]),
        # not by the within-test position j — using j was a silent misalignment
        # bug that produced Spearman ~ 0.
        preds = np.zeros((args.n_confs, n_test), dtype=np.float64)
        ok = np.zeros((args.n_confs, n_test), dtype=bool)
        for r in range(args.n_confs):
            flat = [(j, conf_feats[int(te_idx[j])][r]) for j in range(n_test)
                    if conf_feats[int(te_idx[j])][r] is not None]
            for s in range(0, len(flat), 16):
                chunk = flat[s:s + 16]
                samples = [(f, None) for _, f in chunk]
                batch, _ = model.batch_collate_fn(samples)
                batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in batch.items()}
                with torch.no_grad():
                    out = model(**batch)
                logits = out[0] if isinstance(out, tuple) else out
                logits = logits.detach().float().cpu().numpy().reshape(-1)
                for vi, (j, _) in enumerate(chunk):
                    preds[r, j] = logits[vi]
                    ok[r, j] = True
            print(f"    fold {fold} conf {r + 1}/{args.n_confs}: "
                  f"covered {int(ok[r].sum())}/{n_test}", flush=True)

        # TTA = mean over conformers that succeeded; also keep per-conformer
        # "baseline" (r=0, seed=42 — the same conformer MolTrain's own predict
        # path would draw with its default seed) for a paired comparison.
        tta = np.where(ok.any(0), (preds * ok).sum(0) / np.maximum(ok.sum(0), 1), np.nan)
        base = preds[0].copy()

        # fold models emit predictions in the STANDARDISED target space
        # (target_normalize: standard); the metric must be computed on the
        # original scale, so un-standardise before scoring.
        tta_raw = tta.copy()
        base_raw = base.copy()
        m_any = ~np.isnan(tta_raw)
        if m_any.any():
            tta_raw[m_any] = scaler.inverse_transform(
                tta_raw[m_any].reshape(-1, 1)).ravel()
        m_any_b = ~np.isnan(base_raw)
        if m_any_b.any():
            base_raw[m_any_b] = scaler.inverse_transform(
                base_raw[m_any_b].reshape(-1, 1)).ravel()

        y_true_fold = y[te_idx]
        mask = ~np.isnan(tta_raw)
        tta_per_fold.append({
            "r2": float(r2_score(y_true_fold[mask], tta_raw[mask])),
            "rmse": float(np.sqrt(mean_squared_error(y_true_fold[mask], tta_raw[mask]))),
            "spearman": float(spearmanr(y_true_fold[mask], tta_raw[mask]).correlation),
            "n": int(mask.sum()),
            "coverage": float(mask.mean()),
        })
        mk = ~np.isnan(base_raw)
        base_per_fold.append({
            "r2": float(r2_score(y_true_fold[mk], base_raw[mk])),
            "rmse": float(np.sqrt(mean_squared_error(y_true_fold[mk], base_raw[mk]))),
            "spearman": float(spearmanr(y_true_fold[mk], base_raw[mk]).correlation),
            "n": int(mk.sum()),
            "coverage": float(mk.mean()),
        })
        del model

    def agg(per_fold):
        r2s = np.array([f["r2"] for f in per_fold])
        sps = np.array([f["spearman"] for f in per_fold])
        return {
            "r2": float(r2s.mean()), "r2_std": float(r2s.std()),
            "rmse": float(np.mean([f["rmse"] for f in per_fold])),
            "spearman": float(sps.mean()), "spearman_std": float(sps.std()),
            "per_fold": per_fold,
        }

    result = {
        "exp_dir": str(exp_dir), "dataset": args.dataset, "split": args.split,
        "seed": args.seed, "n_confs": args.n_confs,
        "tta": agg(tta_per_fold),
        "single_conf_r0": agg(base_per_fold),
    }
    print("\n" + "=" * 64)
    print(f"  {args.dataset} | {args.split} | seed{args.seed} | TTA n={args.n_confs}")
    print("=" * 64)
    t, b = result["tta"], result["single_conf_r0"]
    print(f"  TTA (mean over {args.n_confs} confs) R2={t['r2']:+.4f}±{t['r2_std']:.4f}  "
          f"Sp={t['spearman']:.4f}")
    print(f"  single-conformer (r0)      R2={b['r2']:+.4f}±{b['r2_std']:.4f}  "
          f"Sp={b['spearman']:.4f}")
    print(f"  ΔR2 (TTA - single) = {t['r2'] - b['r2']:+.4f}")

    out = args.out or (exp_dir / f"tta_result_{args.dataset}_{args.split}_s{args.seed}_n{args.n_confs}.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
