"""MolTrain-based end-to-end fine-tuning of LipidFM on ionizable lipid benchmarks.

This uses the real ``MolTrain`` pipeline (from unimol_tools): SMILES → on-the-fly
conformer generation → backbone + prediction head **jointly trained end-to-end**.
Unlike the frozen linear probe, this lets the backbone adapt to the downstream task.

The key plumbing: ``MolTrain`` loads its weights via ``UniMolV2Model.load_pretrained_weights``
(or ``NNModel.load_model_dir``). Our LipidFM checkpoint is Uni-Core format, so we
monkey-patch ``UniMolV2Model.load_pretrained_weights`` to load our checkpoint instead.

The patch only swaps in our checkpoint at model construction time; every reload of a
fine-tuned fold model (model_{fold}.pth) is passed to the stock loader. (The earlier
version ignored ``path`` and always reloaded the pretrain checkpoint, silently throwing
away each fold's fine-tuned weights -> R² of -120..-2e100.)

Run modes:
  * LipidFM v2 (default): patch applied, loads CHECKPOINT (env LIPIDFM_CHECKPOINT).
  * Stock 84M baseline: set USE_STOCK_84M=1 (no patch -> official MolTrain load path).

Usage:
  conda run -n unimol2 python 4_scripts/finetune_moltrain.py \
      --dataset all --epochs 20 --lr 1e-4 --kfold 5                 # LipidFM v2
  USE_STOCK_84M=1 conda run -n unimol2 python 4_scripts/finetune_moltrain.py \
      --dataset all --epochs 20 --lr 1e-4 --kfold 5                 # stock 84M base
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for _unimol_tools_dir in (
    REPO_ROOT / "4_scripts/Uni-Mol/unimol_tools",
    REPO_ROOT.parent / "unimol_tools-main",
):
    if _unimol_tools_dir.is_dir():
        sys.path.insert(0, str(_unimol_tools_dir))
        break

import unimol_tools.models.unimolv2 as v2mod

# Chemically-aware splits (scaffold / butina) — pure RDKit, no deepchem.
sys.path.insert(0, str(REPO_ROOT / "4_scripts"))
import chem_splits
import conformer_3d
import lumi_protocol

BENCHMARK_XLSX = Path(os.environ.get(
    "LNP_BENCHMARK_XLSX",
    str(REPO_ROOT.parents[1] / "benchmark/input_data/dataset.xlsx"),
))
CHECKPOINT_VAR = "LIPIDFM_CHECKPOINT"
CHECKPOINT = Path(
    os.environ.get(
        CHECKPOINT_VAR,
        str(REPO_ROOT / "2_models/continued_pretrain/unimol2_lipids_84m_v2/checkpoints/checkpoint_best.pt"),
    )
)
# Default behaviour is to fine-tune LipidFM. Set USE_STOCK_84M=1 to fine-tune the
# stock Uni-Mol2 84M base instead (no monkeypatch -> official MolTrain load path),
# for a clean baseline-vs-LipidFM comparison in a separate process.
USE_STOCK_84M = os.environ.get("USE_STOCK_84M", "0") == "1"
OUTPUT_DIR = REPO_ROOT / "5_outputs/finetune_moltrain"

DATASETS = {
    # --- evaluation 1 (TE) ----------------------------------------------------
    # `frozen_split` was turned on for Sheet2/Sheet3 on 2026-09-18, together with
    # the Butina correction (坑 #14). Before that these two read
    # `5_outputs/split_cache/`, whose butina entries had been built by the broken
    # splitter -- and the corrected folds were being written to
    # `downstream_splits/`, i.e. the neural arm and the baseline arm would have
    # been scored on DIFFERENT partitions with nothing to say so.
    #
    # The switch is verified not to move a single molecule for scaffold or
    # sequential: every frozen file reproduces its `split_cache` twin fold-for-fold
    # (checked 2026-09-18). butina DOES change -- that is the fix, not a
    # regression. `downstream/sync_legacy_split_cache.py` mirrors the frozen files
    # back into `split_cache/` so the legacy analysis scripts (arm_zoo,
    # ensemble_weighted, uncertainty_eval, ...) keep reading one consistent store.
    "Sheet2": {"sheet_name": "Sheet2", "smiles_col": "SMILES", "target_col": "RLU (log2)",
               "log10": False, "frozen_split": True, "butina_cutoff": 0.7},
    "Sheet3": {"sheet_name": "Sheet3", "smiles_col": "SMILES",
               # renamed with an IGROV1_ prefix in the 2026-09-13 structure fix
               "target_col": ("IGROV1_transfection_efficiency", "transfection_efficiency"),
               "log10": True, "frozen_split": True, "butina_cutoff": 0.7},
    # --- downstream evaluation 2 (plan 26-09-16 §5.3) -------------------------
    # Sourced from the downstream registry rather than a workbook sheet, because
    # the pKa cleaning pipeline (parse -> drop implausible -> collapse replicates)
    # lives there and must be the single implementation. `frozen_split` switches
    # fold loading from split_cache/ to downstream_splits/, which is the file
    # third parties are given; the cache and the deliverable must not be allowed
    # to drift apart.
    #
    # `butina_cutoff` is recorded but NOT used to build folds on this path (the
    # frozen file already encodes them). It is 0.8 here, not the Sheet2 value of
    # 0.7 -- butina cutoffs do not transfer across datasets (坑 #10), and 0.7 on
    # this library produced a 54.7%-of-data fold.
    "pKa": {"source": "registry", "key": "pKa", "target_col": "TARGET",
            "log10": False, "frozen_split": True, "butina_cutoff": 0.8},
    # --- downstream evaluation 3 (MTR, plan 26-09-16 §5.4) --------------------
    # 83 molecules x 16 targets. This is the ONLY dataset here that is not
    # single-target: `target_cols` is a LIST, the frame carries all 16 columns,
    # and the whole training path switches to MolTrain's `multilabel_regression`
    # task (one linear head per column, z-scored per column with that column's
    # own train-fold statistics -- see datascaler.py `multilabel_regression`).
    #
    # Protocol (user decision 2026-09-18): `random` is the MAIN split. 83 molecules
    # x 16 targets has almost no statistical power, so the splitter choice cannot
    # rescue it, and that reason is independent of any splitter. `scaffold` is an
    # explicitly-warned appendix: its family-concentration ratio is 1.93 (largest
    # scaffold 32 vs mean fold 16.6), well past the 1.0 threshold where a fold is
    # filled by a single family and the constant predictor does as well -- measured
    # R2 -3.93. butina is NOT offered at any cutoff (split_files.SPLITS_BY_DATASET).
    #
    # `butina_cutoff` is recorded for symmetry but is unreachable on this dataset.
    "MTR": {"source": "registry", "key": "MTR",
            "target_col": "TARGET",  # placeholder; the real list is MTR_TARGETS
            "log10": False, "frozen_split": True, "butina_cutoff": 0.8},
}

# Per-dataset Butina cutoff. Mirrors downstream/split_files.BUTINA_CUTOFF_BY_DATASET;
# kept in sync by hand because importing that module at finetune time would pull the
# downstream package into every historical Sheet2 run. Sheet2/Sheet3 = 0.7 (the
# project's canonical value); pKa = 0.8. See effective_butina_cutoff.
BUTINA_CUTOFF_BY_DATASET = {"Sheet2": 0.7, "Sheet3": 0.7, "pKa": 0.8}
BUTINA_CUTOFF_DEFAULT = 0.6  # argparse default; see effective_butina_cutoff


# ---------------------------------------------------------------------------
# [LIPIDFM ONLY] Monkey-patch UniMolV2Model to load OUR checkpoint at init.
#
# In the stock (USE_STOCK_84M=1) baseline the patch is NOT applied, so MolTrain
# loads the stock 84M weights through the official download/load path — keeping
# baseline and LipidFM on an identical evaluation pipeline.
# ---------------------------------------------------------------------------
def _install_lipidfm_loader() -> None:
    """Install the fix described in the module docstring; see #files-touched."""
    # The stock 84M modelzoo file path carries this token (see models/weights.py).
    _stock_84m_token = "84M/checkpoint.pt"

    def _load_lipidfm_weights(self, path=None, strict=False):
        # Paths pointing at a trained fold model (model_{fold}.pth) or a real
        # trained checkpoint must be honoured by the stock loader — do NOT
        # overwrite them (this was the -120..-2e100 R² bug).
        is_stock_init = path is None or _stock_84m_token in path
        if not is_stock_init:
            state_dict = torch.load(path, map_location="cpu", weights_only=False)
            if "model" in state_dict:
                state_dict = state_dict["model"]
            elif "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            missing, unexpected = self.load_state_dict(state_dict, strict=strict)
            if unexpected or missing:
                print(f"  [Reload] loaded {path} (missing={len(missing)}, "
                      f"unexpected={len(unexpected)})")
            return
        # Construction-time: swap in our Uni-Core-format LipidFM checkpoint.
        if not CHECKPOINT.exists():
            raise FileNotFoundError(f"LipidFM checkpoint not found: {CHECKPOINT}")
        sd = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        # strict=False so the fresh regression head (classification_head) is
        # NOT overwritten — our checkpoint has no head keys (pretrain-only). Adapters are
        # the same story: they are new with PEFT, so no released checkpoint carries them
        # and they must keep their identity initialisation.
        missing, unexpected = self.load_state_dict(sd, strict=False)
        fresh = ("classification_head", "adapters.")
        head_missing = [k for k in missing if not k.startswith(fresh)]
        if head_missing:
            raise RuntimeError(f"Backbone incomplete: {len(head_missing)} keys missing")
        print(f"  [LipidFM] loaded {len(sd)} backbone keys into UniMolV2Model "
              f"(head/adapter skipped: {len(missing) - len(head_missing)}, "
              f"unexpected: {len(unexpected)})")

    v2mod.UniMolV2Model.load_pretrained_weights = _load_lipidfm_weights


SPLIT_CACHE_DIR = REPO_ROOT / "5_outputs/split_cache"

# Everything in the pretrained checkpoint, i.e. what "frozen backbone" means. Expressed
# as prefix matches for nnmodel.py's freeze_layers (which freezes on `name.startswith`).
# `adapters.` and `classification_head.` are deliberately absent, so under
# --freeze-backbone they are the only trainable tensors.
BACKBONE_PREFIXES = [
    "encoder.", "embed_tokens.", "atom_feature.", "edge_feature.",
    "se3_invariant_kernel.", "movement_pred_head.",
]


def trainable_prefix_summary(model) -> dict:
    """Trainable parameter count per top-level module name (diagnostic for PEFT arms)."""
    out: dict[str, int] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        out[name.split(".")[0]] = out.get(name.split(".")[0], 0) + p.numel()
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def build_run_label(args) -> str:
    """Name this run by its checkpoint directory plus every adaptation variable.

    The label feeds BOTH the output directory and the aggregate-results key, and 坑 #6
    is that anything omitted from those keys lets a new run silently overwrite an old
    one's evidence. Freeze/adapter settings change what is trained, so they belong here
    -- a `--freeze-layers 6` run previously reused the unfrozen run's aggregate key and
    clobbered it (the exp_tag had the suffix; this key did not). Kept as one function
    because the label is needed in two places and they must not drift apart.
    """
    if USE_STOCK_84M:
        label = "stock-84M"
    else:
        variant = CHECKPOINT.parent.parent.name  # unimol2_lipids_84m_vN
        label = f"{variant}/{CHECKPOINT.name}"
    if args.freeze_layers > 0:
        label = f"{label}_freeze{args.freeze_layers}"
    if args.freeze_backbone:
        label = f"{label}_frozen"
    if args.adapter_dim > 0:
        label = f"{label}_adapter{args.adapter_dim}"
    if args.es_val == "internal":
        # The early-stopping validation set is a PROTOCOL dimension (坑 #12), not a
        # hyperparameter: this run answers a different question from the default arm
        # on the same folds. Omitting it here would let the control arm overwrite the
        # main arm's directory, aggregate entry and OOF dump -- three separate
        # clobberings of the exact kind 坑 #6 is about.
        label = f"{label}_esval"
    return label


def effective_butina_cutoff(ds_key: str, args: argparse.Namespace) -> float:
    """The cutoff actually used for (dataset, split), and the value recorded.

    Butina cutoffs do not transfer across datasets (坑 #10): 0.7 was calibrated on
    Sheet2's 1920-molecule 4CR library, and on pKa's 574 it yields a fold holding
    54.7% of the data. Every queue script hardcodes ``--butina-cutoff 0.7``, so
    reading the CLI value alone would silently re-break pKa.

    Resolution order: an explicit CLI value wins ONLY if it differs from the
    argparse default -- i.e. the caller actually asked for it. Otherwise a
    dataset with a registered cutoff uses that. Sheet2/Sheet3 are unaffected
    either way (their registered value is also 0.7).
    """
    explicit = args.butina_cutoff != BUTINA_CUTOFF_DEFAULT
    if explicit or ds_key not in BUTINA_CUTOFF_BY_DATASET:
        return args.butina_cutoff
    return BUTINA_CUTOFF_BY_DATASET[ds_key]


def load_or_build_folds(ds_key: str, smiles: list[str], split: str, kfold: int,
                        seed: int, butina_cutoff: float) -> list:
    """Return the canonical fold list for (dataset, split, k, seed), building it once.

    Every base (stock / v2 / v3 / v4 …) must be scored on byte-identical folds, or the
    comparison is meaningless. RDKit/sklearn version drift can silently perturb scaffold
    perception, so the fold assignment is computed ONCE, cached to JSON keyed by the
    full protocol, and reused verbatim afterwards. The cache stores a hash of the SMILES
    column so a changed dataset can never be scored against stale folds.

    Datasets marked ``frozen_split`` read from ``downstream_splits/`` instead. Those
    files are the published deliverable (they carry per-fold SMILES so a third party
    can rebuild the partition without RDKit), and the training path must consume the
    same bytes a reviewer would -- two fold sources for one protocol is how a
    published number stops matching its own reproducibility bundle.
    """
    if DATASETS.get(ds_key, {}).get("frozen_split"):
        sys.path.insert(0, str(REPO_ROOT / "4_scripts/downstream"))
        import split_files  # noqa: PLC0415
        folds = split_files.load_folds(ds_key, split, seed, kfold, smiles=smiles)
        print(f"  [split] frozen {split} folds -> "
              f"{ds_key}_{split}_k{kfold}_seed{seed}.json (downstream_splits/) "
              f"(test sizes {[len(te) for _tr, te in folds]})")
        return folds

    SPLIT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = SPLIT_CACHE_DIR / f"{ds_key}_{split}_k{kfold}_seed{seed}.json"
    smiles_hash = hashlib.sha256("\n".join(smiles).encode()).hexdigest()[:16]

    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("smiles_sha256") != smiles_hash:
            raise SystemExit(
                f"Split cache {cache_path} was built for a DIFFERENT dataset "
                f"(hash {cached.get('smiles_sha256')} != {smiles_hash}). Refusing to "
                f"score on stale folds — delete the cache to rebuild."
            )
        folds = [(np.asarray(f["train"], dtype=int), np.asarray(f["test"], dtype=int))
                 for f in cached["folds"]]
        print(f"  [split] reusing cached {split} folds -> {cache_path.name}")
        return folds

    if split == "scaffold":
        fold_splits, _fold_ids, labels = chem_splits.scaffold_folds(smiles, kfold, seed=seed)
    elif split == "butina":
        fold_splits, _fold_ids, labels = chem_splits.butina_folds(
            smiles, kfold, cutoff=butina_cutoff, seed=seed)
    elif split == "sequential":
        # LUMI-lab's own default (--kfold-mode sequential): contiguous blocks in
        # file order, NO shuffling, so `seed` is deliberately ignored here. On the
        # 4CR-1920 combinatorial library this measures interpolation over R-group
        # assemblies rather than scaffold extrapolation. See lumi_protocol.py.
        fold_splits = lumi_protocol.sequential_folds(len(smiles), kfold)
        labels = np.arange(len(smiles))
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=kfold, shuffle=True, random_state=seed)
        fold_splits = list(kf.split(np.arange(len(smiles))))
        labels = np.arange(len(smiles))

    # Leak check: a scaffold/cluster must never straddle train and test of one fold.
    if split in ("scaffold", "butina"):
        for k, (tr, te) in enumerate(fold_splits):
            if set(np.asarray(labels)[tr]) & set(np.asarray(labels)[te]):
                raise SystemExit(f"{split} fold {k} leaks a group across train/test")

    cache_path.write_text(json.dumps({
        "dataset": ds_key, "split": split, "kfold": kfold, "seed": seed,
        "butina_cutoff": butina_cutoff if split == "butina" else None,
        "n": len(smiles), "smiles_sha256": smiles_hash,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "fold_test_sizes": [int(len(te)) for _tr, te in fold_splits],
        "folds": [{"train": np.asarray(tr).tolist(), "test": np.asarray(te).tolist()}
                  for tr, te in fold_splits],
    }, indent=2), encoding="utf-8")
    print(f"  [split] built and cached {split} folds -> {cache_path.name} "
          f"(test sizes {[len(te) for _t, te in fold_splits]})")
    return fold_splits


# ---------------------------------------------------------------------------
# [ALL BASES] Force MolTrain to use OUR cached folds, for stock and LipidFM alike.
#
# This patch was previously installed only on the LipidFM path, so the stock baseline
# silently used MolTrain's own splitter. That is not the same algorithm: stock
# 'scaffold' is GroupKFold over Murcko scaffolds, and every acyclic lipid shares the
# empty scaffold "" -> one giant group. On Sheet3 that yields fold sizes
# [339,175,126,108,20] (one fold = 44% of the data) vs our balanced [148,149,148,148,175],
# agreeing with our assignment only ~5% of the time. Comparing a base evaluated under
# one to a base evaluated under the other is meaningless, so both now go through the
# identical cached fold list.
# ---------------------------------------------------------------------------
def result_target_scale(ds_key: str) -> str:
    """Label for the scale the metrics are reported on, for (dataset).

    Defined once and used by both the result JSON and the OOF dump: two
    hand-written copies of this expression is how the numbers and the file
    describing them drift apart, and the scale is not recoverable from the
    predictions alone.
    """
    cfg = DATASETS[ds_key]
    if cfg.get("log10"):
        return "log10(RLU)"
    return "raw" if cfg.get("source") == "registry" else "log2(RLU)"


def _fold_of_row(folds: list, n_rows: int) -> np.ndarray:
    """Map each dataframe row to the fold it was a TEST member of.

    Every row must be in exactly one test fold. A row in none means the fold list
    does not cover the dataset (and its prediction would be a silent zero from
    ``y_pred = np.zeros(n)``); a row in two means a fold overlap. Both are
    detectable here and neither is detectable from the fold R² values alone, so
    this raises rather than returning a best-effort array.
    """
    out = np.full(n_rows, -1, dtype=int)
    for k, (_tr, te) in enumerate(folds):
        te = np.asarray(te, dtype=int)
        clash = out[te] != -1
        if clash.any():
            raise ValueError(
                f"row(s) {te[clash][:5].tolist()} appear in the test set of fold "
                f"{k} and of fold {out[te[clash]][0]}; folds overlap")
        out[te] = k
    missing = np.flatnonzero(out < 0)
    if missing.size:
        raise ValueError(
            f"{missing.size} row(s) (e.g. {missing[:5].tolist()}) are in no test "
            f"fold; the fold list does not cover the dataset")
    return out


def _install_fold_injector(folds: list) -> None:
    from unimol_tools.data import split as _split_mod

    def _fixed_split(self, smiles=None, target=None, group=None, scaffolds=None, **params):
        self.split_folds = folds
        return folds

    _split_mod.Splitter.split = _fixed_split


# ---------------------------------------------------------------------------
# [--es-val internal] The 坑 #12 control arm.
#
# MolTrain validates on the TEST fold (nnmodel.py: `X_valid, y_valid = X[te_idx]`),
# so every fold's early-stopping epoch is chosen by looking at the very molecules the
# fold is scored on. The classical arms do not do this: `downstream/baselines.py`
# carves `val_frac=0.15` out of the TRAIN fold for its MLP, and the tree arms select
# on a train-internal split too. The two protocols are therefore not on the same
# scale -- which is why Sheet3/scaffold's raw +0.3431 cannot be quoted beside the
# other three cells (see prompt/26-09-18/08-* §3).
#
# This flag trains the identical model on the identical folds with the identical
# budget and changes exactly one thing: early stopping watches a train-internal
# holdout instead of the test fold. Each test fold is then scored with the checkpoint
# early stopping saved, exactly as the default path does. The resulting number is on
# the SAME protocol as the classical arms, so it can go in the same column.
#
# What it is NOT: a different model. It is the same model under a harder protocol, so
# its score is expected to be LOWER than the default arm's. That gap measures the
# 坑 #12 advantage directly on the raw R² scale, instead of inferring it from a
# mean-shift decomposition.
#
# The split rule is copied from `baselines.py` (`val_frac=0.15`, floor 8) on purpose,
# including its refusal to stratify: matching the classical arms' protocol is the
# whole point of the control, and stratifying would create a third protocol that
# nothing else uses, so the comparison would lose its referent.
#
# Two deliberate deviations, both recorded per fold in `es_val_meta`:
#   * baselines.py draws every fold's permutation from ONE stream (MODEL_SEED=0), so
#     folds with equal train size get the same permutation PATTERN. Here each fold
#     gets `default_rng([0, k])`. The holdout SIZES are identical either way; only
#     which rows land in them differs, and matching a permutation pattern across
#     folds is a defect rather than a protocol.
#   * below the floor, baselines.py disables early stopping entirely (trains the full
#     budget on all of train); here the holdout is raised to 8 rows instead, because
#     MolTrain has no per-fold "no early stopping" switch. `floor_applied` marks every
#     fold where the two rules diverge, and on evaluation 1 no fold triggers it (the
#     smallest train fold is 620 rows -> 93 validation rows).
# ---------------------------------------------------------------------------
ES_VAL_SEED = 0      # fixed, like baselines.MODEL_SEED -- deliberately NOT the fold seed
ES_VAL_FRAC = 0.15
ES_VAL_MIN = 8


def make_internal_es_folds(folds: list) -> tuple[list, list]:
    """Rewrite each fold's ``(train, test)`` as ``(train_inner, val_inner)``.

    The result is what MolTrain trains on: ``train_inner`` is the training set and
    ``val_inner`` is the early-stopping set, BOTH carved out of the original train
    fold. The original test fold is dropped from the list entirely, so MolTrain
    cannot see it -- which is the entire point.

    Indices stay in the ORIGINAL dataframe's numbering (MolTrain indexes ``X[tr_idx]``
    on the full feature array), so nothing needs remapping.

    Per-fold streams (``default_rng([seed, k])``) rather than one shared stream:
    several folds share a train size (Sheet3/scaffold has three folds at 620 rows), and
    a single stream would hand all three the same permutation pattern.

    ``floor_applied`` is recorded rather than silently absorbed: the floor changes the
    effective validation fraction, and a reader comparing two datasets must be able to
    see where that happened. On evaluation 1 no fold triggers it.
    """
    out, meta = [], []
    for k, (tr, _te) in enumerate(folds):
        tr = np.asarray(tr, dtype=int)
        rng = np.random.default_rng([ES_VAL_SEED, k])
        perm = rng.permutation(len(tr))
        n_val = int(len(tr) * ES_VAL_FRAC)
        floored = n_val < ES_VAL_MIN
        if floored:
            n_val = min(ES_VAL_MIN, max(len(tr) - 1, 1))
        if n_val < 1 or len(tr) - n_val < 1:
            raise ValueError(
                f"fold {k}: a train fold of {len(tr)} row(s) cannot be split into a "
                f"non-empty training and validation part")
        vi, ti = perm[:n_val], perm[n_val:]
        out.append((tr[ti], tr[vi]))
        meta.append({"fold": k, "n_train_inner": int(ti.size), "n_val": int(vi.size),
                     "val_frac": round(float(vi.size) / len(tr), 4),
                     "floor_applied": bool(floored)})
    return out, meta


def repredict_real_folds(clf, folds: list, n_t: int = 1) -> np.ndarray:
    """Score the REAL test folds with the per-fold checkpoints early stopping saved.

    With ``--es-val internal``, ``nnmodel.run()`` writes its predictions for the
    early-stopping holdout into ``y_pred[te_idx]`` -- i.e. into the TRAIN rows -- and
    leaves every real test row at the zeros it was allocated with. Those zeros are not
    obviously zeros downstream: ``MolTrain.fit()`` inverse-transforms the array and
    hands it back as ``cv_pred``, so a fold's stored R² would be computed against a
    constant and would read as a spectacular failure rather than as a wiring mistake.
    The OOF vector is therefore rebuilt here, by loading ``model_{k}.pth`` for each
    fold k and predicting that fold's test set.

    This is exactly what ``fit_predict_wo_ddp`` already does at the end of every fold
    on the default path (``load_model=True``), so the scoring path is unchanged --
    only the dataset it is pointed at changes.

    Returns z-scored predictions in original row order, matching what
    ``nnmodel.run()`` puts in ``cv['pred']`` before MolTrain un-standardises it.
    Shape is ``(n_rows, n_t)`` -- ``n_t`` is 1 for the single-target datasets and
    16 for MTR, and it is passed in rather than inferred because ``Trainer.predict``
    returns whatever the head produces and a silent (n, 1) here would be scored
    as 16 columns' worth of one column.
    """
    from unimol_tools.models.nnmodel import NNDataset

    trainer, nn = clf.trainer, clf.model
    X = np.asarray(nn.features)
    y = np.asarray(clf.data["target"])
    out = np.zeros((y.shape[0], n_t), dtype=float)
    filled = np.zeros(y.shape[0], dtype=bool)
    for k, (_tr, te) in enumerate(folds):
        te = np.asarray(te, dtype=int)
        yp, _loss, _score = trainer.predict(
            nn.model, NNDataset(X[te], y[te]), nn.loss_func, nn.activation_fn,
            clf.save_path, k, clf.data["target_scaler"], epoch=1, load_model=True)
        yp = np.asarray(yp, dtype=float).reshape(len(te), n_t)
        if yp.shape[0] != te.size:
            raise ValueError(
                f"fold {k}: predicted {yp.shape[0]} row(s) for {te.size} test row(s)")
        if filled[te].any():
            raise ValueError(f"fold {k}: test rows overlap an earlier fold's")
        out[te] = yp
        filled[te] = True
    if not filled.all():
        raise ValueError(f"{int((~filled).sum())} row(s) are in no test fold")
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MolTrain end-to-end finetuning of LipidFM.")
    p.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--kfold", type=int, default=5)
    p.add_argument("--early-stopping", type=int, default=5)
    p.add_argument("--split", choices=["random", "scaffold", "butina", "sequential"],
                   default="scaffold",
                   help="Primary CV split. 'random' is for baseline comparison only; "
                        "'scaffold' (Bemis-Murcko, size-balanced) is the default final "
                        "evaluation; 'butina' uses ECFP4/Tanimoto clusters; "
                        "'sequential' reproduces LUMI-lab's own --kfold-mode sequential "
                        "(contiguous blocks, no shuffle -> combinatorial interpolation, "
                        "NOT scaffold extrapolation; `--seed` does not affect the folds).")
    p.add_argument("--butina-cutoff", type=float, default=0.6,
                   help="Tanimoto similarity cutoff for the Butina split.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for BOTH the fold assignment and MolTrain's training RNG. "
                        "Sheet3 folds are high-variance (n=768), so average over "
                        "several seeds before drawing conclusions.")
    p.add_argument("--freeze-layers", type=int, default=0,
                   help="Freeze the first N encoder layers during fine-tuning "
                        "(UniMol2 84M has 12 layers; freezing early layers is a "
                        "regulariser for the small ~1900-sample benchmark).")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Freeze the ENTIRE pretrained backbone (encoder, embedding, "
                        "atom/edge/SE3 features, movement head), leaving only the "
                        "regression head -- and the adapters, if --adapter-dim is set "
                        "-- trainable. This is the PEFT control arm: it isolates how "
                        "much of the end-to-end score comes from adapting the backbone "
                        "at all.")
    p.add_argument("--adapter-dim", type=int, default=0,
                   help="PEFT: insert a Houlsby bottleneck adapter "
                        "(h <- h + W_up.GELU(W_down.LN(h))) after every encoder layer's "
                        "FFN, with bottleneck width r. 0 (default) adds nothing -- no "
                        "params, no state_dict keys, no forward work -- so every "
                        "pre-PEFT result still reproduces bit-for-bit. Intended use is "
                        "with --freeze-backbone, giving ~1%% trainable params.")
    p.add_argument("--split-only", action="store_true",
                   help="Build and save the folds then exit before the expensive "
                        "MolTrain fit() (useful to inspect/verify splits cheaply).")
    p.add_argument("--use-3d-conformers", action="store_true",
                   help="Feed MolTrain pre-generated ETKDGv3+MMFF 3D conformers through "
                        "its atoms/coordinates input path, instead of its built-in "
                        "generator. The built-in path silently substitutes a FLAT 2D "
                        "depiction whenever RDKit's default 10 embedding attempts fail, "
                        "which is ~82%% of these flexible lipids -- so the default "
                        "fine-tune feeds a 3D model planar input. See conformer_3d.py.")
    p.add_argument("--conf-cache", type=Path, default=None,
                   help="3D conformer cache (default: 5_outputs/conf3d_cache/). Shared "
                        "across bases so every model sees identical geometries.")
    p.add_argument("--es-val", choices=["test", "internal"], default="test",
                   help="Which fold the early-stopping validation set comes from. "
                        "'test' (default) is MolTrain's own behaviour and every number "
                        "published before 2026-09-19: it validates on the TEST fold "
                        "(nnmodel.py: X_valid, y_valid = X[te_idx]) -- 坑 #12. "
                        # NB: the literal percent must be %%%% -- the string is
                        # %-formatted HERE and then %-formatted AGAIN by argparse's
                        # help renderer. A single %% collapses to a bare % in the
                        # first pass and argparse dies with
                        # "TypeError: %o format: an integer is required, not dict".
                        "'internal' carves a %.0f%%%% holdout out of the TRAIN fold "
                        "instead (val_frac=%.2f, floor %d, seed %d -- the same rule "
                        "downstream/baselines.py uses for its MLP), so the arm is scored "
                        "on the protocol the classical baselines use and its raw R² is "
                        "comparable to theirs. The real test folds are re-predicted from "
                        "the saved per-fold checkpoints. Expect this arm to score LOWER; "
                        "the gap IS the 坑 #12 advantage. See prompt/26-09-18/08-* §3."
                        % (ES_VAL_FRAC * 100, ES_VAL_FRAC, ES_VAL_MIN, ES_VAL_SEED))
    p.add_argument("--output", type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def target_columns(ds_key: str) -> list[str]:
    """The target column names to train on, in the dataset's canonical order.

    Single-target datasets (Sheet2/Sheet3/pKa) return ``["TARGET"]``: the loaders
    rename their one label to that name so every downstream index and every
    historical result key keeps working unchanged.

    MTR returns its 16 real names. That is deliberate, not cosmetic: MolTrain's
    ``multilabel_regression`` path reads the targets from these columns and its
    per-column scaler is built in this order, so the order here IS the output-head
    order. It also has to match ``ds_registry.load("MTR")``'s list exactly, because
    the collector writes ``per_target`` keyed by those names.
    """
    if ds_key == "MTR":
        sys.path.insert(0, str(REPO_ROOT / "4_scripts/downstream"))
        import ds_registry  # noqa: PLC0415
        return list(ds_registry.MTR_TARGETS)
    return ["TARGET"]


def _is_low_variance(ds_key: str, target: str) -> bool:
    """Is this a target whose spread is too small to support a regression headline?

    Read from the registry so the threshold has exactly one home. Near-constant
    columns are kept in the matrix (dropping them would change the shared head and
    make the run incomparable to any other MTR arm) but flagged, because a high R²
    on a near-constant column means nothing.
    """
    if ds_key != "MTR":
        return False
    sys.path.insert(0, str(REPO_ROOT / "4_scripts/downstream"))
    import ds_registry  # noqa: PLC0415
    return target in ds_registry.MTR_LOW_VARIANCE


def load_data(ds_key: str) -> pd.DataFrame:
    """Load a benchmark sheet and expose it as ``SMILES`` + target column(s).

    Sheet2's ``RLU (log2)`` is already log-scaled; Sheet3's raw RLU spans ~4 orders
    of magnitude (2e3..1e8), so we take ``log10`` first. This puts both targets on
    a comparable log scale so the reported metrics are not dominated by the
    different units. MolTrain then applies ``target_normalize="standard"`` on top
    for training.

    A dataset with ``source: registry`` is loaded through
    ``downstream/ds_registry`` instead of a workbook sheet. That is not
    indirection for its own sake: the pKa pipeline's drops (17 unparseable, 4
    implausible, 72 replicate groups collapsed) and its canonicalisation are the
    reason the frozen split file's sha256 means anything. Re-implementing them
    here would let the trained model and the frozen split disagree about row
    order while both still "worked".

    MTR comes back with all 16 of its columns under their own names -- see
    ``target_columns``. Nothing is renamed to ``TARGET``, because a 16-way head
    with 16 identical column names is not addressable.
    """
    cfg = DATASETS[ds_key]
    if cfg.get("source") == "registry":
        sys.path.insert(0, str(REPO_ROOT / "4_scripts/downstream"))
        import ds_registry  # noqa: PLC0415
        df, _targets = ds_registry.load(cfg["key"], verbose=True)
        cols = target_columns(ds_key)
        return pd.DataFrame({"SMILES": df["SMILES"].astype(str),
                             **{c: df[c].astype(float) for c in cols}})

    df = pd.read_excel(BENCHMARK_XLSX, sheet_name=cfg["sheet_name"])
    col = cfg["target_col"]
    if isinstance(col, tuple):  # accept either spelling of Sheet3's target
        col = next((c for c in col if c in df.columns), None)
        if col is None:
            raise SystemExit(f"{ds_key}: none of {cfg['target_col']} in {list(df.columns)}")
    target = df[col].astype(float)
    if cfg["log10"]:
        target = np.log10(target)
    return pd.DataFrame({"SMILES": df["SMILES"].astype(str), "TARGET": target})


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from unimol_tools import MolTrain

    # `all` keeps its historical meaning: the benchmark SHEETS only. It used to be
    # every key in DATASETS, and every existing queue script (run_v5_eval_queue.sh,
    # the PEFT queues, ...) relies on that. Letting a newly registered downstream
    # dataset join `all` would silently add hours of unrequested GPU work to those
    # scripts and change what their logs mean. Registry datasets are named explicitly.
    ds_keys = ([k for k, v in DATASETS.items() if v.get("source") != "registry"]
               if args.dataset == "all" else [args.dataset])
    all_results = {}

    # Identify the run by the checkpoint's own directory (…/unimol2_lipids_84m_v3/
    # checkpoints/checkpoint_best.pt -> "unimol2_lipids_84m_v3/checkpoint_best.pt"),
    # NOT a hardcoded "LipidFM-v2". The old label was fixed at v2 for every LipidFM
    # run, so a v3 run silently overwrote v2's entry in the aggregate file.
    run_label = build_run_label(args)
    print(f"  run_label={run_label}  split={args.split}  seed={args.seed}")

    for ds_key in ds_keys:
        print(f"\n{'='*60}\n  {ds_key}\n{'='*60}")
        df = load_data(ds_key)
        tcols = target_columns(ds_key)
        print(f"  {len(df)} samples, {len(tcols)} target column(s)"
              + (f": {tcols}" if len(tcols) > 1 else ""))

        # Save to CSV for MolTrain
        csv_path = args.output / f"{ds_key}_data.csv"
        df.to_csv(csv_path, index=False)

        # MolTrain: real end-to-end finetuning with k-fold CV.
        # Compute and save the chemically-diverse folds FIRST so --split-only can
        # inspect them without a (multi-hour) training run.
        # Canonical folds: built once, cached, and reused by EVERY base so stock/v2/v3
        # are scored on byte-identical splits. Installed for the stock path too.
        folds = load_or_build_folds(ds_key, df["SMILES"].astype(str).tolist(),
                                    args.split, args.kfold, args.seed,
                                    effective_butina_cutoff(ds_key, args))
        split_info = build_and_save_splits(ds_key, df, args, folds)
        _install_fold_injector(folds)

        # Build the MolTrain payload. Order is load-bearing: rows must stay in dataset
        # order so the cached folds (integer indices into THIS dataframe) still select
        # the right molecules, and so atoms[i] pairs with TARGET[i] and with the SMILES
        # whose sha256 guards the split cache.
        if args.use_3d_conformers:
            conf_cache = args.conf_cache or conformer_3d.default_cache_path(
                ds_key, len(df))
            mols, kept, stats = conformer_3d.generate_3d_conformers(
                df["SMILES"].astype(str).tolist(), seed=args.seed, cache_path=conf_cache)
            # Drop the rows that could not be embedded, then re-index the folds onto
            # the surviving rows. Skipping the remap would shift every index past the
            # first drop and quietly mix train with test.
            if len(kept) != len(df):
                keep_mask = np.zeros(len(df), dtype=bool)
                keep_mask[kept] = True
                df = df.iloc[kept].reset_index(drop=True)
                folds = conformer_3d.remap_folds(folds, kept)
                _install_fold_injector(folds)
                print(f"  [conf3d] kept {len(kept)}/{len(keep_mask)} rows; folds remapped")
            # Pass RDKit mols, NOT atoms/coordinates. A 'mols' entry makes DataHub
            # call UniMolV2Feature.transform_mols (datahub.py:112-114), which reads
            # each mol's own conformer and keeps its bond orders. The
            # atoms/coordinates route instead goes through
            # create_mol_from_atoms_and_coords, which builds a mol with NO BONDS:
            # degree collapses, edge_attr empties and shortest_path saturates at its
            # 511 "unreachable" sentinel. Measured, that variant scored 0.589 vs 0.698
            # for the built-in generator -- the bond graph, not the coordinates,
            # dominated. A mol from SMILES keeps real connectivity, so the conformer
            # is the only thing that differs from the baseline.
            # A dict (not a DataFrame) is the payload the vendored unimol_tools 0.1.6
            # accepts; its read_data has no DataFrame branch. The column must be named
            # ROMol: read_data surfaces mols only when it sees that exact column
            # (datareader.py: `if 'ROMol' in data.columns: dd['mols'] = ...`), and a
            # column literally named 'mols' would be ignored.
            fit_payload = {
                "SMILES": df["SMILES"].astype(str).tolist(),
                "TARGET": df["TARGET"].values,
                "ROMol": mols,
            }
            print(f"  [conf3d] feeding MolTrain {len(mols):,} RDKit mols carrying 3D "
                  f"conformers + full bond graph ({stats.rate_3d*100:.1f}% verified 3D)")
        else:
            fit_payload = str(csv_path)

        # 坑 #12 control arm: hand MolTrain a fold list whose "test" half is a
        # train-internal holdout, so early stopping never sees the scored molecules.
        # `folds` keeps the REAL partition -- the OOF dump, the split CSVs and the
        # re-prediction below all need it, and re-deriving it from the rewritten list
        # is impossible. `es_folds` is what the injector gets.
        es_meta = None
        if args.es_val == "internal":
            es_folds, es_meta = make_internal_es_folds(folds)
            _install_fold_injector(es_folds)
            print(f"  [es-val internal] early stopping watches a "
                  f"{ES_VAL_FRAC:.0%} train-internal holdout (seed {ES_VAL_SEED}); "
                  f"the {args.kfold} real test folds are NOT shown to MolTrain and "
                  f"are re-predicted from model_<k>.pth afterwards")
            for m in es_meta:
                print(f"      fold {m['fold']}: train {m['n_train_inner']} / "
                      f"val {m['n_val']} ({m['val_frac']:.1%})"
                      + ("  [floor applied]" if m["floor_applied"] else ""))

        if args.split_only:
            print(f"  [--split-only] splits written; skipping MolTrain fit().")
            continue

        # Per-run save_path: every run used to share "<ds>_exp/", so each new run
        # silently overwrote the previous one's cv.data / metric.result / scaler —
        # which is how v3's artifacts ended up standing in for v2's. Keying the
        # directory by base+split+seed keeps every run's evidence intact.
        # The conformer SOURCE belongs in that key for the same reason: a 3D-conformer
        # run and a built-in-generator run differ in input, not just hyperparameters,
        # so sharing a directory would let one silently destroy the other's evidence
        # (which is exactly what happened on the first 3D run, before this suffix).
        # Freeze/adapter suffixes ride on run_label (build_run_label), because they too
        # change what is trained: a frozen run must not overwrite the fully-tuned one.
        conf_suffix = "_3dconf" if args.use_3d_conformers else ""
        exp_tag = (f"{ds_key}_exp_{run_label.replace('/', '_')}_{args.split}"
                   f"_seed{args.seed}_lr{args.lr}_ep{args.epochs}{conf_suffix}")
        print(f"  [tag] {exp_tag}")
        # MolTrain's task. `multilabel_regression` is the 16-head MTR path: one
        # linear head per target column, its own per-column scaler, and the
        # NaN-tolerant MAE loss (loss.py MAEwithNan). Everything else in this file
        # -- folds, early stopping, the OOF dump, the result schema -- is shared.
        #
        # Which task is chosen is NOT a flag: it follows from the dataset's target
        # list, because a 16-column frame trained as `regression` would silently
        # reshape to (n*16, 1) and score the columns as if they were samples.
        task = "multilabel_regression" if len(tcols) > 1 else "regression"
        clf = MolTrain(
            task=task,
            data_type="molecule",
            epochs=args.epochs,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            early_stopping=args.early_stopping,
            metrics="r2",
            # MolTrain's Splitter validates the method name in __init__ (split.py:54)
            # BEFORE _install_fold_injector replaces Splitter.split, and it has no
            # 'sequential'/'butina' branch. Since the injector makes the built-in
            # splitter unreachable, the name only has to survive that validation —
            # 'random' is the neutral alias. The folds actually used are always our
            # cached ones; args.split remains the label recorded in the results.
            split=(args.split if args.split in ("random", "scaffold", "group",
                                                "stratified", "select") else "random"),
            kfold=args.kfold,
            save_path=str(args.output / exp_tag),
            model_name="unimolv2",
            model_size="84m",
            smiles_col="SMILES",
            # A comma-joined string, not a list: MolTrain's config round-trips this
            # through JSON, and datareader splits it back on ','. The target NAMES
            # are the real column names for MTR and the literal "TARGET" otherwise.
            target_cols=",".join(tcols),
            target_normalize="standard",
            use_cuda=True,
            use_amp=True,
            conf_cache_level=1,
            # UniMolV2Feature defaults max_atoms to 128, and mol2unimolv2 crops with a
            # RANDOMLY SHUFFLED mask once a molecule exceeds it. Sheet2 tops out at 69
            # heavy atoms, but 8 Sheet3 molecules run past 128 and would silently lose
            # a random subset of atoms -- different on every access. The pretraining
            # corpus is capped at 256, so 256 keeps fine-tune and pretrain consistent
            # and makes the crop branch unreachable.
            max_atoms=conformer_3d.MAX_ATOMS,
        )
        # MolTrain's __init__ accepts **params but never merges it into self.config,
        # so MolTrain(seed=...) is silently ignored. Set it on the config directly.
        clf.config["seed"] = args.seed
        clf.config["split_seed"] = args.seed
        # PEFT: the adapter width reaches the model as self.args.adapter_dim because
        # MolTrain.__init__ keeps **params in self.config and NNModel forwards them to
        # UniMolV2Model. Setting it on the config (not as a MolTrain kwarg) is what makes
        # that hop, exactly as freeze_layers does below. Left at 0 -> empty ModuleList.
        clf.config["adapter_dim"] = args.adapter_dim
        if args.adapter_dim > 0:
            print(f"  [peft] bottleneck adapters: r={args.adapter_dim} after every "
                  f"encoder layer, applied after the FFN residual")
            # The adapters start as an exact identity (W_up is zero-initialised), so a
            # frozen-backbone run with adapters begins at the frozen part's own
            # predictions and improves from there.
            assert args.adapter_dim < 768, "adapter wider than the 768-d stream"
        if args.freeze_layers > 0:
            # Trailing dot keeps "encoder.layers.1" from matching ".10"/".11".
            clf.config["freeze_layers"] = [
                f"encoder.layers.{i}." for i in range(args.freeze_layers)
            ]
            print(f"  [freeze] freezing first {args.freeze_layers} encoder layers "
                  f"({clf.config['freeze_layers'][0]} .. "
                  f"{clf.config['freeze_layers'][-1]})")
        if args.freeze_backbone:
            # Freeze everything the pretrained checkpoint supplied. `freeze_layers` is a
            # FREEZE list here (nnmodel.py: requires_grad = not should_freeze), so the
            # adapters must NOT be listed — staying off it is exactly what leaves them
            # trainable. They are top-level "adapters." keys, deliberately outside every
            # BACKBONE_PREFIX, which is what makes the split clean.
            clf.config["freeze_layers"] = list(BACKBONE_PREFIXES)
            if args.adapter_dim > 0:
                print(f"  [peft] frozen backbone; trainable = adapters + head")
            else:
                print(f"  [peft] frozen backbone; trainable = linear head only (769 params)")
        # fit_payload is either the CSV path (MolTrain's own conformer generation, with
        # its 2D fallback) or the atoms/coordinates dict built above.
        clf.fit(fit_payload)

        # Per-fold predictions and targets — SCALE CONTRACT (do not "fix" again):
        #   * `clf.cv_pred` has ALREADY been un-standardised inside MolTrain.fit()
        #     (train.py:191-203 does `y_pred = scalar.inverse_transform(...)` before
        #     assigning `self.cv_pred`). It is on the MODELLING scale.
        #   * `clf.data['target']` is the z-scored target (datahub.py:64), so it is
        #     the ONLY array that still needs `inverse_transform`.
        # Applying inverse_transform to cv_pred a second time re-inflates it by
        # (mu + sd*x), which leaves ranks untouched (Spearman stays healthy) but
        # destroys the scale -> aggregate R² of -21 / -29. That was the bug here.
        #
        # SHAPE: `n_t` is 1 for every single-target dataset and 16 for MTR. The
        # arrays are kept 2-D as (n, n_t) rather than raveled to 1-D, because a
        # ravel on a 16-column array produces n*16 values that still pass a
        # "same length as the target" check against another raveled array -- so
        # every downstream index would pair a molecule with a different target's
        # prediction while all five fold R² values stayed plausible. `n_t == 1`
        # is what keeps every historical single-target path bit-for-bit identical:
        # column 0 of an (n, 1) array is the same sequence ravel() produced.
        scaler = clf.data["target_scaler"]
        y_pred_m = np.asarray(clf.cv_pred, dtype=float)
        y_true_m = scaler.inverse_transform(
            np.asarray(clf.data["target"], dtype=float))
        if y_pred_m.ndim == 1:
            y_pred_m = y_pred_m.reshape(-1, 1)
        if y_true_m.ndim == 1:
            y_true_m = y_true_m.reshape(-1, 1)
        n_t = y_true_m.shape[1]
        assert y_pred_m.shape == y_true_m.shape, (
            f"predictions are {y_pred_m.shape} but the target is {y_true_m.shape}")
        assert n_t == len(tcols), (
            f"{ds_key}: {n_t} output column(s) but {len(tcols)} target name(s) "
            f"({tcols}); the result JSON and the OOF dump are keyed by name, so a "
            f"mismatch here would mislabel every column.")

        # --es-val internal: undo MolTrain's fold bookkeeping.
        #
        # With the internal holdout, MolTrain wrote each fold's predictions for the
        # HOLDOUT into `y_pred[te_idx]` -- and those te_idx are now TRAIN row indices --
        # while every real test row kept the zeros it was allocated with. Taken at face
        # value that is not an obviously broken array: MolTrain inverse-transforms it and
        # hands it back as `cv_pred`, so each fold's R² would be computed against a
        # constant and would read as "this model does not work", not as "the array is
        # wired wrong". Rebuild it from the per-fold checkpoints instead.
        #
        # `folds` (the real partition), not `es_folds`: the whole point is to score the
        # molecules MolTrain was never allowed to look at.
        if args.es_val == "internal":
            raw = np.asarray(clf.cv_pred)
            if np.allclose(raw, 0.0):
                raise SystemExit(
                    "cv_pred is all zeros under --es-val internal; the injected folds "
                    "were not used, so nothing was trained and there is nothing to "
                    "re-predict. Refusing to write a result.")
            _re = repredict_real_folds(clf, folds, n_t)
            y_pred_m = scaler.inverse_transform(_re)
            assert y_pred_m.shape == y_true_m.shape, (
                f"re-predicted OOF is {y_pred_m.shape} but the target is "
                f"{y_true_m.shape}; refusing to write a mis-aligned prediction file")
            print(f"  [es-val internal] re-predicted the {len(folds)} real test folds "
                  f"from model_<k>.pth ({np.isfinite(y_pred_m).sum()}/{y_pred_m.size} "
                  f"finite)")

        # Persist the per-molecule out-of-fold predictions. Without this the run is
        # only a set of 5 fold R² values, and the downstream screening metrics
        # (EF@k, NDCG@k, pooled) -- which are per-molecule by construction and are
        # the LUMI paper's headline -- cannot be computed after the fact.
        #
        # ORDER: `clf.cv_pred` is NOT fold-major. nnmodel.py allocates
        # `y_pred = np.zeros((n, output_dim))` and assigns `y_pred[te_idx] = _y_pred`
        # per fold (nnmodel.py:250), so it is in ORIGINAL dataframe row order,
        # aligned with `clf.data['target']`. Both arrays are therefore indexed
        # identically and can be zipped against `df.iloc[i]` directly. (Checked,
        # not assumed -- a fold-major array here would silently pair every molecule
        # with another fold's prediction while all five fold R² values stayed correct.)
        #
        # Shape: the arrays are (n, n_t). `n_t == 1` is the single-target case and
        # is kept 2-D on purpose -- column 0 of an (n, 1) array is exactly the
        # sequence the old ravel() produced, so every historical path is unchanged.
        # The guard that used to live here refused multi-output outright, because
        # a ravel of (n, 16) gives 16n values that still satisfy a length check
        # against another raveled array, silently pairing molecules with the wrong
        # target's prediction. That is now handled by keeping the second axis
        # instead of forbidding it, so what is asserted is the alignment itself.
        assert np.asarray(clf.cv_pred).ndim == 2, (
            f"cv_pred has shape {np.asarray(clf.cv_pred).shape}; the OOF dump "
            f"assumes a 2-D (n, n_targets) array.")
        assert y_pred_m.shape == y_true_m.shape and \
            y_pred_m.shape[0] == len(df) and y_pred_m.shape[1] == n_t, (
            f"OOF arrays are {y_pred_m.shape}/{y_true_m.shape} but the dataset is "
            f"{len(df)} rows x {n_t} target(s); refusing to write a mis-aligned "
            f"prediction file")
        # The fold_of_row written below is derived from the REAL folds, but the
        # assertion has to check what was actually INJECTED -- under --es-val internal
        # that is es_folds (the train-internal holdouts), and comparing against `folds`
        # would fail on a run that is behaving correctly. Assert the injected partition
        # was used; if MolTrain's data layer kept a different one, the stored map would
        # describe a partition the predictions never came from, and downstream would
        # trust it.
        _injected = es_folds if args.es_val == "internal" else folds
        _used = clf.data.get("split_nfolds")
        if _used is None or [tuple(map(int, te)) for _tr, te in _used] != \
                [tuple(map(int, te)) for _tr, te in _injected]:
            raise SystemExit(
                "MolTrain's split_nfolds differ from the injected folds; "
                "the OOF dump would describe the wrong partition. Refusing.")
        # Name the dump by the FULL run identity, not just the protocol. It was
        # `oof_<ds>_<split>_seed<S>.npz` until 2026-09-18, which meant a stock-84M
        # run and a LipidFM-v5 run on the same (dataset, split, seed) wrote the
        # SAME file -- and `collect_neural` PREFERS the dump over `cv.data`, so
        # whichever arm finished last owned it and the other could never be
        # collected. Same failure class as 坑 #6: an identity that omits a
        # protocol dimension.
        #
        # `run_label` already carries the checkpoint directory plus the
        # freeze/adapter variables (see build_run_label), so it is exactly the
        # dimension that was missing. Slashes cannot appear in a filename, so they
        # are replaced; `collect_neural.legacy_dump_path` still finds the old names
        # for the pKa runs written before this change.
        safe_label = run_label.replace("/", "_")
        oof_path = (args.output / "oof"
                    / f"oof_{ds_key}_{args.split}_seed{args.seed}_{safe_label}.npz")
        oof_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            oof_path, y=y_true_m, oof=y_pred_m,
            row_index=np.arange(len(df)),
            smiles=np.asarray(df["SMILES"].astype(str)),
            fold_of_row=_fold_of_row(folds, len(df)),
            split=args.split, seed=args.seed, dataset=ds_key,
            target_scale=result_target_scale(ds_key),
            # The column NAMES, in head order. Without them a 16-column dump is
            # unreadable: the collector has to know which axis-1 slice is which
            # assay, and re-deriving that from ds_registry would silently mislabel
            # every column if the registry's order ever changed.
            targets=np.array(tcols, dtype=object),
        )
        print(f"  [oof] per-molecule predictions -> {oof_path}")

        from sklearn.metrics import r2_score, mean_squared_error
        from scipy.stats import spearmanr

        trainable = sum(p.numel() for p in clf.model.model.parameters()
                        if p.requires_grad)
        total_params = sum(p.numel() for p in clf.model.model.parameters())
        print(f"  [params] trainable {trainable:,} / {total_params:,} "
              f"({trainable/total_params*100:.3f}%)")

        # Both sheets are reported on their modelling scale (Sheet2=log2(RLU),
        # Sheet3=log10(RLU)) — the scale the model was trained and validated on,
        # and the one the frozen-feature benchmark reports, so the two are
        # directly comparable. Spearman is rank-based, hence identical whether
        # computed in log or raw units, so no exponentiation is needed.
        # PRIMARY metric = mean over folds. Pooling all 5 folds' predictions and
        # scoring once against the global mean is NOT a per-fold generalisation
        # estimate: under a scaffold split each fold has a different label mean,
        # so the pooled SST understates the within-fold error and the number is
        # neither comparable across splits nor to the literature.
        # Use the canonical cached folds (already leak-checked) rather than recomputing:
        # recomputation is exactly how a split can silently drift between bases.
        #
        # Under --es-val internal, `split_nfolds` is the train-internal holdout list, so
        # taking it here would score the HOMEWORK -- five R² values on the holdout the
        # early stopping just optimised against, which would look like a healthy model
        # and would be the exact opposite of what this control arm is for. The real
        # partition is what the arm is being measured on, so keep `folds`.
        if args.es_val != "internal":
            folds = clf.data.get("split_nfolds")

        def _per_target_block(j: int) -> dict:
            """Metrics for output column ``j``, aggregated over the real folds.

            A block, not a function of the whole array: R²/RMSE/Spearman are only
            defined per column, and scoring the (n, 16) arrays in one call would
            compare each molecule's prediction for one assay against its label for
            another. `n_t == 1` reproduces the historical scalar result exactly --
            column 0 of the old (n, 1) array is the same sequence ravel() produced.
            """
            yt, yp = y_true_m[:, j], y_pred_m[:, j]
            pf = []
            for k, (_tr_idx, te_idx) in enumerate(folds):
                te = np.asarray(te_idx, dtype=int)
                if te.size < 2:
                    continue
                pf.append({
                    "fold": k,
                    "n": int(te.size),
                    "r2": round(float(r2_score(yt[te], yp[te])), 4),
                    "rmse": round(float(np.sqrt(mean_squared_error(yt[te], yp[te]))), 4),
                    "spearman": round(float(spearmanr(yt[te], yp[te]).statistic), 4),
                    # LUMI's metric_for_best_model; their compute_metrics reports no R2 at all
                    "relaxed_spearman": round(
                        lumi_protocol.relaxed_spearman(yp[te], yt[te]), 4),
                })

            def _agg(key):
                vals = [f[key] for f in pf]
                return (round(float(np.mean(vals)), 4),
                        round(float(np.std(vals, ddof=1)), 4))

            r2_mean, r2_std = _agg("r2")
            rmse_mean, rmse_std = _agg("rmse")
            sp_mean, sp_std = _agg("spearman")
            rsp_mean, rsp_std = _agg("relaxed_spearman")
            return {
                # headline = fold-mean ± std
                "r2": r2_mean, "r2_std": r2_std,
                "rmse": rmse_mean, "rmse_std": rmse_std,
                "spearman": sp_mean, "spearman_std": sp_std,
                "relaxed_spearman": rsp_mean, "relaxed_spearman_std": rsp_std,
                # kept only as a secondary diagnostic; never use as the headline
                "r2_pooled": round(float(r2_score(yt, yp)), 4),
                "spearman_pooled": round(float(spearmanr(yt, yp).statistic), 4),
                "per_fold": pf,
                "n": int(len(df)),
                "split": args.split,
                "target_scale": result_target_scale(ds_key),
                "target": tcols[j],
            }

        # `per_target` keyed by column name is the schema `collect_neural` and
        # `run_matrix` already share, so MTR needs no second result format.
        # The single-target datasets keep their historical flat keys AS WELL --
        # every analysis script written before today reads `result["r2"]`, and
        # breaking them to unify a schema would invalidate the archived numbers.
        per_target = {tcols[j]: _per_target_block(j) for j in range(n_t)}
        result = dict(per_target[tcols[0]])
        result.update({
            "per_target": per_target,
            "targets": list(tcols),
            # What this run ACTUALLY trained. `requires_grad` is not recoverable from a
            # saved state_dict (frozen tensors are still stored), so the count has to be
            # captured here or it is lost -- and this is the only number that reveals an
            # arm whose freeze/adapter flag never reached the model.
            "trainable_params": trainable,
            "total_params": total_params,
            "trainable_prefixes": trainable_prefix_summary(clf.model.model),
        })
        if n_t > 1:
            result.pop("target", None)  # ambiguous with 16 columns; per_target has them
        # The 坑 #12 control arm must be self-describing. `es_val` says which protocol
        # this number is on, and `es_val_meta` records the actual holdout sizes -- the
        # val_frac floor changes the effective fraction, and a reader comparing two
        # datasets has to be able to see where that happened without re-deriving it.
        result["es_val"] = args.es_val
        if es_meta is not None:
            result["es_val_meta"] = {
                "val_frac_target": ES_VAL_FRAC, "val_frac_floor": ES_VAL_MIN,
                "seed": ES_VAL_SEED, "per_fold": es_meta,
                "note": ("Early stopping watched a train-internal holdout instead of "
                         "the test fold. The reported per_fold R² are on the REAL "
                         "folds, re-predicted from model_<k>.pth. Comparable to "
                         "downstream/baselines.py arms; NOT comparable to the "
                         "default (es_val=test) arm."),
            }
        if n_t == 1:
            # Read the numbers back OUT of the result dict instead of the names the
            # nested ``_per_target_block`` bound. Those were locals of that function,
            # so the console summary raised NameError after the whole 5-fold run had
            # already trained and saved its metrics -- 10 jobs' worth of compute lost
            # at the last statement (found 2026-09-22 on the stock-84M arm; every
            # previous post-2026-09-20 run happened to be multi-target, which took
            # the ``else`` branch and never touched these names). The dict is the
            # interface the collectors read, so printing from it is also the more
            # honest summary: the console can no longer disagree with the artifact.
            b = per_target[tcols[0]]
            print(f"  [MolTrain] fold-mean R²={b['r2']:.4f}±{b['r2_std']:.4f}  "
                  f"RMSE={b['rmse']:.4f}  Spearman={b['spearman']:.4f}±{b['spearman_std']:.4f}")
            print(f"             relaxed-Spearman={b['relaxed_spearman']:.4f}±{b['relaxed_spearman_std']:.4f} (LUMI headline)")
            print(f"             per-fold R²: {[f['r2'] for f in b['per_fold']]}")
            print(f"             (pooled R²={result['r2_pooled']:.4f} — diagnostic only)")
        else:
            # 16 lines, not a summary: with this many columns the only useful
            # console output is per-column, and a fold-mean over columns would be
            # a number that describes no assay (the 8 safety columns have a
            # different scale and sign convention from the 8 transfection ones).
            print(f"  [MolTrain] {n_t} target columns, fold-mean R² each:")
            for j, t in enumerate(tcols):
                b = per_target[t]
                flag = "  [near-constant: exclude from headline]" \
                    if _is_low_variance(ds_key, t) else ""
                print(f"             {t:34s} R²={b['r2']:+.4f}±{b['r2_std']:.4f}  "
                      f"Spearman={b['spearman']:+.4f}{flag}")
            print(f"             (pooled R² is per-column in per_target[t]['r2_pooled'])")
        all_results[ds_key] = result

    # Save results.
    out_path = args.output / f"moltrain_results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    # `output` is a Path; cast args to primitive types so json.dumps never chokes.
    args_sane = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    out_path.write_text(json.dumps({
        "checkpoint": str(CHECKPOINT),
        "run_label": run_label,
        "split": args.split,
        "model": ("stock-84M (official MolTrain load)" if USE_STOCK_84M
                  else f"LipidFM ({CHECKPOINT_VAR}={CHECKPOINT})"),
        "args": args_sane,
        "results": all_results,
    }, indent=2), encoding="utf-8")
    print(f"\nResults -> {out_path}")

    # Aggregate into a single latest-results file (one row per model:dataset),
    # merged with any previously recorded runs so the final metrics are never
    # lost to a timestamped-file housekeeping sweep.
    agg_path = args.output / "moltrain_results.json"
    agg = {}
    if agg_path.exists():
        try:
            agg = json.loads(agg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass  # corrupt/partial -> start fresh; the timestamped file still holds this run
    runs = agg.get("runs", {}) if isinstance(agg, dict) else {}
    # Carry `void` / `void_note` through the merge. This writer rebuilds the file
    # from `runs` alone, so before 2026-09-19 every merge silently DELETED them --
    # and the two quarantine scripts (`mark_void_sheet23_butina_in_aggregate.py`,
    # `mark_void_sheet3_in_aggregate.py`) both write into this same file, so their
    # work was erased by the next training run. That is exactly what happened: the
    # marks were written 2026-09-18 21:41 / 22:29, and evaluation 1's first run
    # (2026-09-19 00:52) wiped all 42 of them.
    #
    # The loss was invisible because BOTH scripts are idempotent and report
    # "to mark void: N" from the current file -- so the next --dry-run just shows
    # the same N again, which reads as "still pending", not "was destroyed".
    # A quarantine that a routine action silently undoes is worse than none: it
    # leaves the false confidence without the protection.
    #
    # Do not "clean up" stale entries here. A voided key that a later re-run
    # overwrites is still void until someone deliberately re-runs it -- deciding
    # that from the merge path (which cannot see the quarantine's own criteria)
    # is how the marks got lost in the first place.
    void = agg.get("void", {}) if isinstance(agg, dict) else {}
    void_note = agg.get("void_note") if isinstance(agg, dict) else None
    # Key on (checkpoint variant, split) so a v3 run can never clobber v2's entry.
    # Keep the provenance alongside the numbers so a stale entry is self-describing.
    # The dataset, split AND seed are all part of the evaluation protocol: results from
    # different protocols are not comparable, so they must not collide on one key.
    # The dataset MUST be in the key — a `--dataset Sheet3` run once clobbered the
    # Sheet2 entry that shared its base+split+seed (the same overwrite class this
    # refactor set out to kill, reintroduced one level down).
    # Every dimension that changes the result must be in the key: base, dataset,
    # split, seed AND the training hyperparameters. An lr sweep would otherwise
    # collapse all three lrs onto one entry (the same overwrite bug, third variant).
    # Conformer source is one of those dimensions: a 3D-conformer run is a different
    # experiment from a built-in-generator run on the same base/split/seed.
    conf_key = "|3dconf" if args.use_3d_conformers else ""
    # The early-stopping validation source is a protocol dimension too (坑 #12): the
    # control arm and the main arm are different experiments on the same folds. Without
    # this suffix the second one to finish would overwrite the first one's aggregate
    # entry -- the same class of collision conf_key exists to prevent.
    es_key = "|esval" if args.es_val == "internal" else ""
    for ds, r in all_results.items():
        runs[f"{run_label}|{ds}|{args.split}|seed{args.seed}|lr{args.lr}|ep{args.epochs}{conf_key}{es_key}"] = {
            "checkpoint": str(CHECKPOINT) if not USE_STOCK_84M else "stock-84M (official)",
            "dataset": ds,
            "split": args.split,
            "seed": args.seed,
            "lr": args.lr,
            "epochs": args.epochs,
            "kfold": args.kfold,
            "freeze_backbone": args.freeze_backbone,
            "adapter_dim": args.adapter_dim,
            "conformers": ("etkdgv3+mmff94s-3d" if args.use_3d_conformers
                           else "moltrain-builtin"),
            "es_val": args.es_val,
            "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": {ds: r},
        }
    out_agg = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runs": runs,
    }
    # Re-emit the quarantine blocks verbatim (see the comment at `runs` above).
    if void:
        out_agg["void"] = void
    if void_note:
        out_agg["void_note"] = void_note
    agg_path.write_text(json.dumps(out_agg, indent=2), encoding="utf-8")
    print(f"Aggregated   -> {agg_path}")


def build_and_save_splits(ds_key: str, df: pd.DataFrame, args: argparse.Namespace,
                          fold_splits: list) -> dict:
    """Write the canonical folds out as per-fold train/test CSVs for inspection.

    The folds are NOT computed here — they come from ``load_or_build_folds`` so that
    the CSVs on disk are guaranteed to be the same folds the model was scored on.
    """
    split_dir = args.output / "splits" / f"{ds_key}_{args.split}_seed{args.seed}"
    split_dir.mkdir(parents=True, exist_ok=True)

    smiles = df["SMILES"].astype(str).tolist()
    # Every label column, under its own name -- not just `TARGET`. MTR has no
    # `TARGET` column at all (its 16 columns keep their assay names), so the old
    # single-column read would raise KeyError there; and writing only the first
    # column would hand a reader of the split CSVs a 1/16th view of the labels
    # with nothing on the file saying so. Single-target datasets are unchanged:
    # their one column is literally named TARGET.
    tcols_here = [c for c in df.columns if c != "SMILES"]
    targets = {c: df[c].astype(float).tolist() for c in tcols_here}

    if args.split == "scaffold":
        labels = chem_splits.scaffold_group_ids(smiles)
    elif args.split == "butina":
        _fs, _fi, labels = chem_splits.butina_folds(
            smiles, args.kfold, cutoff=effective_butina_cutoff(ds_key, args), seed=args.seed)
    else:
        labels = np.arange(len(smiles))  # no meaningful chemical group for random

    summary = {"dataset": ds_key, "split": args.split, "seed": args.seed,
               "k": len(fold_splits), "fold_test_sizes": []}
    for k, (tr_idx, te_idx) in enumerate(fold_splits):
        tr_idx = np.asarray(tr_idx); te_idx = np.asarray(te_idx)
        pd.DataFrame({
            "row_index": tr_idx,
            "SMILES": [smiles[i] for i in tr_idx],
            **{c: [targets[c][i] for i in tr_idx] for c in tcols_here},
            "group": [labels[i] for i in tr_idx],
        }).to_csv(split_dir / f"{ds_key}_{args.split}_fold{k}_train.csv", index=False)
        pd.DataFrame({
            "row_index": te_idx,
            "SMILES": [smiles[i] for i in te_idx],
            **{c: [targets[c][i] for i in te_idx] for c in tcols_here},
            "group": [labels[i] for i in te_idx],
        }).to_csv(split_dir / f"{ds_key}_{args.split}_fold{k}_test.csv", index=False)
        summary["fold_test_sizes"].append(int(len(te_idx)))
        print(f"    split fold {k}: train={len(tr_idx)} test={len(te_idx)}")

    summary["n_unique_groups"] = int(len(np.unique(labels)))
    (split_dir / f"{ds_key}_{args.split}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    # Only the checkpoint loader is LipidFM-specific. The fold injection is installed
    # inside main() for EVERY base (stock included) so all of them are scored on the
    # identical cached folds.
    if not USE_STOCK_84M:
        _install_lipidfm_loader()
    main()
