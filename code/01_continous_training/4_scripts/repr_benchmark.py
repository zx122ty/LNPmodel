"""Frozen-representation benchmark: stock-84M vs LipidFM v2 vs v3.

Extracts [CLS] molecular representations with ``UniMolRepr`` (backbone FROZEN, no
fine-tuning) and fits classical regressors on top. The question this answers:

    Did continued pretraining damage the backbone's representations themselves,
    or is the end-to-end fine-tuning result a fine-tuning-stage artifact?

If v3 is also worse here, the regression lives in the pretraining stage. This costs
minutes instead of the ~12 h an end-to-end MolTrain sweep needs.

Comparability rules (do not relax these):
  * Same ``unimol_tools`` copy as ``finetune_moltrain.py`` -- the vendored
    ``unimol_tools-main`` has no modelzoo/84M/checkpoint.pt, so the stock baseline
    would try to hit the network instead of loading the local weights.
  * Same CACHED scaffold folds as the end-to-end runs (5_outputs/split_cache/), so
    frozen-feature and fine-tuned numbers are read on one ruler.
  * Ridge alpha is chosen by an inner CV on the TRAINING fold only, and features are
    standardised with a scaler fit on train only. Different checkpoints produce
    representations of different scale/variance; a hardcoded alpha=1.0 would quietly
    reward whichever base happens to match that scale.

Usage:
  conda run -n unimol2 python 4_scripts/repr_benchmark.py --dataset all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
# Same copy finetune_moltrain.py uses: it ships weights/modelzoo/84M/checkpoint.pt,
# so USE_STOCK loads offline through the official path.
sys.path.insert(0, str(REPO_ROOT / "4_scripts/Uni-Mol/unimol_tools"))

import unimol_tools.models.unimolv2 as v2mod  # noqa: E402
from unimol_tools import UniMolRepr  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "4_scripts"))
import lumi_protocol  # noqa: E402

BENCHMARK_XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
SPLIT_CACHE_DIR = REPO_ROOT / "5_outputs/split_cache"
OUTPUT_DIR = REPO_ROOT / "5_outputs/repr_benchmark"
FEATURE_CACHE = OUTPUT_DIR / "features"
CKPT_DIR = REPO_ROOT / "2_models/continued_pretrain"

DATASETS = {
    "Sheet2": {"target_col": "RLU (log2)", "log10": False, "scale": "log2(RLU)"},
    "Sheet3": {"target_col": ("IGROV1_transfection_efficiency", "transfection_efficiency"),
               "log10": True, "scale": "log10(RLU)"},
}


def resolve_col(df, names) -> str:
    """Pick the first present column name.

    Sheet3's target was renamed with an ``IGROV1_`` prefix when the structures were
    corrected (2026-09-13), so accept either spelling instead of hard-coding one.
    """
    if isinstance(names, str):
        names = (names,)
    for n in names:
        if n in df.columns:
            return n
    raise SystemExit(f"none of {list(names)} found in {list(df.columns)}")

BASES = {
    "stock-84M": None,  # None -> official load path (local modelzoo weights)
    "v2": CKPT_DIR / "unimol2_lipids_84m_v2/checkpoints/checkpoint_best.pt",
    "v3": CKPT_DIR / "unimol2_lipids_84m_v3/checkpoints/checkpoint_best.pt",
    "v4": CKPT_DIR / "unimol2_lipids_84m_v4/checkpoints/checkpoint_best.pt",
    "v5": CKPT_DIR / "unimol2_lipids_84m_v5/checkpoints/checkpoint_1_30000.pt",
    "v5e": CKPT_DIR / "unimol2_lipids_84m_v5e/checkpoints/checkpoint_1_30000.pt",
    "v5_60k": CKPT_DIR / "unimol2_lipids_84m_v5_60k/checkpoints/checkpoint_2_60000.pt",
    "v6": CKPT_DIR / "unimol2_lipids_84m_v6/checkpoints/checkpoint1.pt",
    "v6b": CKPT_DIR / "unimol2_lipids_84m_v6b/checkpoints/checkpoint_2_30000.pt",
    "v6c": CKPT_DIR / "unimol2_lipids_84m_v6c/checkpoints/checkpoint_1_30000.pt",
    "v6d": CKPT_DIR / "unimol2_lipids_84m_v6d/checkpoints/checkpoint_1_30000.pt",
}


def load_data(ds_key: str) -> tuple[list[str], np.ndarray]:
    cfg = DATASETS[ds_key]
    df = pd.read_excel(BENCHMARK_XLSX, sheet_name="Sheet2" if ds_key == "Sheet2" else "Sheet3")
    smiles = df["SMILES"].astype(str).tolist()
    y = df[resolve_col(df, cfg["target_col"])].astype(float).to_numpy()
    if cfg["log10"]:
        y = np.log10(y)
    if len(smiles) != len(y):
        raise SystemExit(f"{ds_key}: {len(smiles)} SMILES vs {len(y)} targets")
    return smiles, y


def load_cached_folds(ds_key: str, smiles: list[str], seed: int, kfold: int = 5,
                      split: str = "scaffold") -> list:
    """Load the SAME folds the end-to-end runs were scored on.

    ``split='sequential'`` needs no cache: LUMI's protocol is a deterministic
    function of n (contiguous blocks, no shuffle), so it is generated on the fly
    and is byte-identical to what finetune_moltrain.py builds.
    """
    if split == "sequential":
        return lumi_protocol.sequential_folds(len(smiles), kfold)
    path = SPLIT_CACHE_DIR / f"{ds_key}_{split}_k{kfold}_seed{seed}.json"
    if not path.exists():
        raise SystemExit(f"missing fold cache {path}; run finetune_moltrain.py --split-only first")
    cached = json.loads(path.read_text(encoding="utf-8"))
    want = hashlib.sha256("\n".join(smiles).encode()).hexdigest()[:16]
    if cached.get("smiles_sha256") != want:
        raise SystemExit(f"{path} was built for a different dataset ({cached.get('smiles_sha256')} != {want})")
    return [(np.asarray(f["train"], int), np.asarray(f["test"], int)) for f in cached["folds"]]


_STOCK_LOADER = v2mod.UniMolV2Model.load_pretrained_weights  # captured before any patch


def _install_loader(ckpt: Path | None) -> None:
    """Point the model at OUR checkpoint; ``None`` restores the stock loader.

    Restoring explicitly (rather than importlib.reload) matters: extracting v3 and
    then stock in one process would otherwise leave the v3 patch installed and
    silently benchmark v3 twice under two different names.
    """
    if ckpt is None:
        v2mod.UniMolV2Model.load_pretrained_weights = _STOCK_LOADER
        return

    def _load(self, path=None, strict=False):
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        missing, unexpected = self.load_state_dict(sd, strict=False)
        backbone_missing = [k for k in missing if not k.startswith("classification_head")]
        if backbone_missing:
            raise RuntimeError(f"Backbone incomplete: {len(backbone_missing)} keys missing")
        print(f"    [load] {ckpt.name}: {len(sd)} keys "
              f"(head skipped {len(missing) - len(backbone_missing)}, unexpected {len(unexpected)})")
    v2mod.UniMolV2Model.load_pretrained_weights = _load


def extract_features(base: str, ds_key: str, smiles: list[str]) -> np.ndarray:
    """Frozen [CLS] representations, cached to .npy (extraction is the slow part)."""
    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    tag = hashlib.sha256("\n".join(smiles).encode()).hexdigest()[:8]
    cache = FEATURE_CACHE / f"{base}_{ds_key}_{tag}.npy"
    if cache.exists():
        X = np.load(cache)
        print(f"    [feat] cached {cache.name} {X.shape}")
        return X

    ckpt = BASES[base]
    if ckpt is not None and not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    _install_loader(ckpt)  # None -> restores the stock loader

    model = UniMolRepr(data_type="molecule", model_name="unimolv2", model_size="84m",
                       remove_hs=False, use_cuda=torch.cuda.is_available())
    model.model.eval()
    with torch.no_grad():
        out = model.get_repr(smiles, return_atomic_reprs=False)
    X = np.asarray(out["cls_repr"], dtype=np.float32)
    if X.shape[0] != len(smiles):
        raise SystemExit(f"{base}/{ds_key}: got {X.shape[0]} reprs for {len(smiles)} SMILES "
                         f"— silent molecule dropping would misalign y")
    np.save(cache, X)
    print(f"    [feat] extracted {X.shape} -> {cache.name}")
    return X


def evaluate(X: np.ndarray, y: np.ndarray, folds: list) -> dict:
    """Fold-mean R2 / RMSE / Spearman, same headline convention as the e2e runs."""
    from sklearn.linear_model import RidgeCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error
    from scipy.stats import spearmanr

    alphas = np.logspace(-2, 4, 13)
    out = {}
    for name in ("ridge", "rf"):
        r2s, rmses, sps, rsps = [], [], [], []
        for tr, te in folds:
            if name == "ridge":
                sc = StandardScaler().fit(X[tr])          # fit on TRAIN only
                m = RidgeCV(alphas=alphas).fit(sc.transform(X[tr]), y[tr])
                pred = m.predict(sc.transform(X[te]))
            else:
                m = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=8)
                m.fit(X[tr], y[tr])
                pred = m.predict(X[te])
            r2s.append(float(r2_score(y[te], pred)))
            rmses.append(float(np.sqrt(mean_squared_error(y[te], pred))))
            sps.append(float(spearmanr(y[te], pred).statistic))
            rsps.append(lumi_protocol.relaxed_spearman(pred, y[te]))
        out[name] = {
            "r2": round(float(np.mean(r2s)), 4),
            "r2_std": round(float(np.std(r2s, ddof=1)), 4),
            "per_fold_r2": [round(v, 4) for v in r2s],
            "rmse": round(float(np.mean(rmses)), 4),
            "spearman": round(float(np.mean(sps)), 4),
            "spearman_std": round(float(np.std(sps, ddof=1)), 4),
            # LUMI's metric_for_best_model (hg_mterics.py:10-39)
            "relaxed_spearman": round(float(np.mean(rsps)), 4),
            "relaxed_spearman_std": round(float(np.std(rsps, ddof=1)), 4),
            "per_fold_relaxed_spearman": [round(v, 4) for v in rsps],
        }
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Frozen-representation benchmark across bases.")
    p.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    p.add_argument("--seed", type=int, default=42, help="which cached fold set to score on")
    p.add_argument("--split", choices=["scaffold", "sequential"], default="scaffold",
                   help="'scaffold' = our extrapolation protocol (cached folds); "
                        "'sequential' = LUMI-lab's own default (contiguous, no shuffle) "
                        "= combinatorial interpolation. --seed is ignored for sequential.")
    p.add_argument("--bases", nargs="*", default=list(BASES))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds_keys = list(DATASETS) if args.dataset == "all" else [args.dataset]
    results: dict = {}

    for ds_key in ds_keys:
        tag = (f"{args.split} folds" if args.split == "sequential"
               else f"{args.split} folds seed={args.seed}")
        print(f"\n{'='*66}\n  {ds_key}  (frozen repr, {tag})\n{'='*66}")
        smiles, y = load_data(ds_key)
        folds = load_cached_folds(ds_key, smiles, args.seed, split=args.split)
        print(f"  n={len(smiles)}  target sd={y.std():.4f}  fold test sizes={[len(te) for _t, te in folds]}")
        results[ds_key] = {}
        for base in args.bases:
            print(f"  -- {base}")
            X = extract_features(base, ds_key, smiles)
            res = evaluate(X, y, folds)
            results[ds_key][base] = res
            for mname in ("ridge", "rf"):
                r = res[mname]
                print(f"     {mname:<6} R²={r['r2']:+.4f} ± {r['r2_std']:.4f}   "
                      f"Spearman={r['spearman']:.4f}   "
                      f"relaxSp={r['relaxed_spearman']:.4f}   folds={r['per_fold_r2']}")

    out = OUTPUT_DIR / f"repr_benchmark_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({
        "split": (f"{args.split} (LUMI protocol, deterministic)" if args.split == "sequential"
                  else f"{args.split} (cached, seed={args.seed})"),
        "protocol": "frozen [CLS] repr + RidgeCV/RF, fold-mean R² + LUMI relaxed-Spearman",
        "results": results,
    }, indent=2), encoding="utf-8")
    print(f"\nResults -> {out}")

    print(f"\n{'='*66}\n  SUMMARY (fold-mean R², frozen representations)\n{'='*66}")
    for ds_key in results:
        print(f"\n{ds_key}:")
        print(f"  {'base':<12} {'ridge':<22} {'rf':<22}")
        for base in args.bases:
            r = results[ds_key].get(base)
            if not r:
                continue
            print(f"  {base:<12} "
                  f"{r['ridge']['r2']:+.4f} ± {r['ridge']['r2_std']:.4f}      "
                  f"{r['rf']['r2']:+.4f} ± {r['rf']['r2_std']:.4f}")


if __name__ == "__main__":
    main()
