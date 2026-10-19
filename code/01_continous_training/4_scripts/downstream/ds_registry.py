#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dataset registry for the three downstream evaluations.

Mirrors the ``DATASETS`` dict in ``finetune_moltrain.py`` so a new task plugs
into the same loader shape it already expects: a frame with ``SMILES`` plus one
or more target columns, SMILES canonicalised ONCE here so the split cache's
hash and the trained model are guaranteed to see the same ordering.

Two things this module deliberately does NOT do:

* It does not silently keep rows it cannot parse. Every drop is counted and
  reported, because this project has twice been burned by a filter that looked
  like a no-op.
* It does not merge quantities measured differently. The pKa workbook contains
  an ``apparent_pka`` sheet (566 rows, measured in lipid systems) and a ``pka``
  sheet (108 rows, true aqueous pKa). Those are different experiments; the plan
  locks evaluation 2 to ``apparent_pka`` only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRAINING_ROOT = Path(__file__).resolve().parents[2]
ROOT = TRAINING_ROOT.parents[1]
BENCHMARK_XLSX = ROOT / "benchmark/input_data/dataset.xlsx"
PKA_XLSX = ROOT / "pka_dataset/curate_v2.xlsx"
MTR_CSV = ROOT / "MTR_dataset/merged_dataset_addSmiles_v6.csv"

# --- pKa cleaning constants (plan §5.3) -----------------------------------
# Apparent pKa of an ionizable lipid amine sits in ~4-11. Values above this are
# not a different chemistry, they are a different unit or an unfitted reading:
# the four offenders are 13.2 / 12.8 / 13.7 / 76.7, and 76.7 in particular is
# physically impossible for an amine.
PKA_MAX_PLAUSIBLE = 12.0

# --- MTR target groups (plan §5.4, user decision #4) ----------------------
MTR_TRANSFECTION = [
    "log_IGROV1_transfection", "log_RAW264.7_transfection", "log_Hepa1_6_transfection",
    "log_liver_transfection", "log_spleen_transfection", "log_muscle_transfection_IM",
    "log_liver_transfection_IM", "log_lymphnode_transfection_IM",
]
MTR_SAFETY = ["IL-6_IV", "CXCL1_IV", "CCL2_IV", "IL-6_IM", "CXCL1_IM", "CCL2_IM",
              "BUN_IV", "WBC_IV"]

# Columns whose spread is too small to support a regression headline. Kept in
# the matrix but flagged, and excluded from per-target leaderboards: predicting
# a near-constant column yields an impressive-looking R2 that means nothing.
# Threshold is a third of the narrowest clearly-predictable transfection column.
MTR_LOW_VARIANCE = {
    "log_Hepa1_6_transfection": 0.302,
    "BUN_IV": 0.218,
    "WBC_IV": 0.256,
}
MTR_LOW_VARIANCE_STD_MAX = 0.35

MTR_TARGETS = MTR_TRANSFECTION + MTR_SAFETY


def canonicalize(smiles: str) -> str | None:
    """RDKit canonical SMILES, or None if the string will not parse.

    Canonicalising on the way in matters here: the split cache is keyed on a
    sha256 of the SMILES column, so two runs that differ only in how RDKit
    happened to write a string would silently build different folds.
    """
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol) if mol is not None else None


# --------------------------------------------------------------------------
# Evaluation 1: TE (Sheet2 / Sheet3)
# --------------------------------------------------------------------------
# These two loaders do NOT canonicalise, and that is the whole point of writing
# them out separately instead of reusing `canonicalize` above.
#
# Every other loader here canonicalises because its frozen split files were built
# over canonical SMILES. Sheet2/Sheet3's were not: their folds were frozen from
# the workbook strings verbatim (Sheet2 sha256 `d3a3687ce7c15f96`, Sheet3
# `08c177e083980cf9`, both re-verified 2026-09-18), and
# `finetune_moltrain.load_data` -- which builds the neural arm's folds -- reads
# those same raw strings. Canonicalising here would change the hash
# `split_files.load_folds` checks, so the neural arm and the baseline arm would
# refuse to share folds. `load_folds` is what keeps the two in step; it is not
# decoration.
#
# ROW ORDER is the other half of the contract: the frozen folds are integer
# indices into this frame, so an index that means molecule A for one arm and
# molecule B for the other is a silent corruption of every paired comparison.
# Both readers therefore do a bare `pd.read_excel(..., sheet_name=...)` with no
# filter, no sort and no dedup. If either side ever grows one, the SMILES hash
# check fires -- loudly, at load time, before any training.
#
# Sheet3's target column is accepted under either spelling because the 2026-09-13
# structure fix renamed it with an `IGROV1_` prefix and older copies of the
# workbook are still in circulation.
SHEET3_TARGET_COLS = ("IGROV1_transfection_efficiency", "transfection_efficiency")


def _load_benchmark_sheet(sheet_name: str, target_col, log10: bool) -> pd.DataFrame:
    """Read one benchmark sheet and reduce it to ``SMILES`` + ``TARGET``.

    Deliberately the same three operations as ``finetune_moltrain.load_data``'s
    workbook branch, in the same order, so the two cannot disagree about which
    row is which.
    """
    df = pd.read_excel(BENCHMARK_XLSX, sheet_name=sheet_name)
    if isinstance(target_col, tuple):
        col = next((c for c in target_col if c in df.columns), None)
        if col is None:
            raise SystemExit(f"{sheet_name}: none of {target_col} in {list(df.columns)}")
    else:
        col = target_col
    target = df[col].astype(float)
    if log10:
        # Sheet2's `RLU (log2)` is already log-scaled; Sheet3's raw RLU spans
        # 2e3..1e8, so it needs log10 to land on a comparable scale. Without
        # this the metrics would be dominated by the unit difference.
        target = np.log10(target)
    return pd.DataFrame({"SMILES": df["SMILES"].astype(str), "TARGET": target})


def load_sheet2(verbose: bool = True) -> pd.DataFrame:
    """Sheet2 (n=1920): the 4CR combinatorial library all main conclusions rest on."""
    out = _load_benchmark_sheet("Sheet2", "RLU (log2)", log10=False)
    if verbose:
        print(f"[Sheet2] workbook -> {len(out)} molecules, RAW SMILES (not canonicalised); "
              f"target=log2(RLU)  mean={out['TARGET'].mean():.4f} "
              f"std={out['TARGET'].std():.4f}")
    return out


def load_sheet3(verbose: bool = True) -> pd.DataFrame:
    """Sheet3 (n=768): the group's own series, structures corrected 2026-09-13."""
    out = _load_benchmark_sheet("Sheet3", SHEET3_TARGET_COLS, log10=True)
    if verbose:
        print(f"[Sheet3] workbook -> {len(out)} molecules, RAW SMILES (not canonicalised); "
              f"target=log10(RLU)  mean={out['TARGET'].mean():.4f} "
              f"std={out['TARGET'].std():.4f}")
    return out


# --------------------------------------------------------------------------
# Evaluation 2: pKa
# --------------------------------------------------------------------------
def load_pka(verbose: bool = True) -> pd.DataFrame:
    """Cleaned apparent-pKa frame: columns ``SMILES`` (canonical) + ``TARGET``.

    Cleaning order matters and is fixed: parse -> drop implausible -> dedup.
    Dedup runs LAST so that a duplicate pair where one entry is an outlier does
    not survive as the group mean.
    """
    raw = pd.read_excel(PKA_XLSX, sheet_name="merge")
    n0 = len(raw)

    canon = [canonicalize(s) for s in raw["smiles"]]
    n_bad = sum(c is None for c in canon)
    df = raw.assign(SMILES=canon).dropna(subset=["SMILES"])

    n_high = int((df["apparent_pka"] > PKA_MAX_PLAUSIBLE).sum())
    df = df[df["apparent_pka"] <= PKA_MAX_PLAUSIBLE]

    # Collapse replicate measurements of the same molecule. The within-group
    # spread is small (mean std 0.062) EXCEPT for a few groups, so keep the
    # count and the spread rather than only the mean -- they are the evidence
    # that averaging was defensible.
    grp = df.groupby("SMILES")["apparent_pka"]
    out = grp.mean().rename("TARGET").to_frame()
    out["n_measurements"] = grp.size()
    out["within_group_std"] = grp.std().fillna(0.0)
    out = out.reset_index()

    n_dup_groups = int((out["n_measurements"] > 1).sum())
    if verbose:
        print(f"[pKa] raw {n0} rows -> drop {n_bad} unparseable -> "
              f"drop {n_high} implausible (>{PKA_MAX_PLAUSIBLE}) -> "
              f"collapse {n_dup_groups} replicate groups -> {len(out)} unique molecules")
        worst = out.nlargest(3, "within_group_std")[["SMILES", "TARGET", "n_measurements",
                                                     "within_group_std"]]
        if len(worst) and worst["within_group_std"].max() > 0.2:
            print(f"[pKa] ⚠️ widest replicate spread (label noise on these molecules):")
            for _, r in worst.iterrows():
                print(f"       std={r.within_group_std:.3f} over {int(r.n_measurements)} "
                      f"measurements, mean={r.TARGET:.3f}  {r.SMILES[:60]}")
    return out[["SMILES", "TARGET", "n_measurements", "within_group_std"]]


# --------------------------------------------------------------------------
# Evaluation 3: MTR (multi-target)
# --------------------------------------------------------------------------
def load_mtr(verbose: bool = True) -> pd.DataFrame:
    """MTR frame: ``SMILES`` + the 16 requested targets + ``TARGET_pKa``.

    ``pKa`` is renamed with a prefix so it cannot collide with a target name
    from evaluation 2 when both land in the same results table.
    """
    raw = pd.read_csv(MTR_CSV)
    canon = [canonicalize(s) for s in raw["SMILES"]]
    n_bad = sum(c is None for c in canon)
    df = raw.assign(SMILES=canon).dropna(subset=["SMILES"])

    missing = [c for c in MTR_TARGETS + ["pKa"] if c not in df.columns]
    if missing:
        raise SystemExit(f"MTR is missing expected columns: {missing}")

    out = df[["SMILES"] + MTR_TARGETS + ["pKa"]].copy()
    out = out.rename(columns={"pKa": "TARGET_pKa"})
    out = out.drop_duplicates(subset=["SMILES"]).reset_index(drop=True)

    if verbose:
        print(f"[MTR] raw {len(raw)} rows -> drop {n_bad} unparseable -> "
              f"{len(out)} unique molecules x {len(MTR_TARGETS)} targets")
        low = {c: float(out[c].std()) for c in MTR_TARGETS
               if float(out[c].std()) < MTR_LOW_VARIANCE_STD_MAX}
        if low:
            print(f"[MTR] ⚠️ near-constant targets (<{MTR_LOW_VARIANCE_STD_MAX} std) — "
                  f"keep in matrix, EXCLUDE from headline numbers:")
            for c, s in sorted(low.items(), key=lambda kv: kv[1]):
                print(f"       {c:34s} std={s:.3f}")
    return out


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------
LOADERS = {
    # Evaluation 1 (TE). Raw workbook SMILES -- see the block comment above
    # `_load_benchmark_sheet` for why these two must not canonicalise.
    "Sheet2": (load_sheet2, ["TARGET"]),
    "Sheet3": (load_sheet3, ["TARGET"]),
    # Evaluation 2 / 3.
    "pKa": (load_pka, ["TARGET"]),
    "MTR": (load_mtr, MTR_TARGETS),
}


def load(ds_key: str, verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """Return ``(frame, target_columns)`` for a downstream dataset key."""
    if ds_key not in LOADERS:
        raise SystemExit(f"unknown downstream dataset {ds_key!r}; have {list(LOADERS)}")
    fn, targets = LOADERS[ds_key]
    return fn(verbose=verbose), list(targets)


def verify_against_moltrain(verbose: bool = True) -> dict:
    """Assert this module's frame IS ``finetune_moltrain.load_data``'s frame.

    The registry exists so the classical baselines can read the same data the
    neural arm trains on. That only holds if the two agree on **both** the
    contents and the order -- the frozen folds are integer indices, so a
    reordering would not raise anywhere, it would just score every baseline on a
    permutation of the truth and quietly produce a plausible-looking table.

    So this compares SMILES elementwise and raises on any difference, rather than
    comparing a summary (a mean and an std survive a permutation, which is
    exactly the failure mode being guarded). The workbook path is duplicated in
    the two modules by necessity -- one of them has to own the fold cache -- and
    this is the check that keeps the duplication honest.
    """
    import importlib
    import sys
    sys.path.insert(0, str(ROOT / "code/01_continous_training/4_scripts"))
    fm = importlib.import_module("finetune_moltrain")

    report = {}
    for key in ("Sheet2", "Sheet3"):
        ours, _ = load(key, verbose=False)
        theirs = fm.load_data(key)
        a, b = ours["SMILES"].astype(str).tolist(), theirs["SMILES"].astype(str).tolist()
        if a != b:
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
            raise SystemExit(
                f"{key}: ds_registry and finetune_moltrain.load_data disagree on the "
                f"SMILES column (n {len(a)} vs {len(b)}, first difference at row {first}). "
                f"The frozen folds are integer indices into this order -- a mismatch "
                f"would score every baseline against a permutation of the truth.")
        dta = ours["TARGET"].to_numpy(float) - theirs["TARGET"].to_numpy(float)
        if not np.allclose(dta, 0.0):
            raise SystemExit(f"{key}: targets differ, max |Δ|={np.abs(dta).max():.3g}")
        report[key] = {"n": len(a), "smiles_identical": True,
                       "max_abs_target_delta": float(np.abs(dta).max())}
        if verbose:
            print(f"[verify] {key}: {len(a)} rows identical to finetune_moltrain "
                  f"(order and values), max |Δtarget|={np.abs(dta).max():.3g}")
    return report


def main() -> None:
    """Print the cleaning report for every registered dataset."""
    for key in LOADERS:
        print("=" * 72)
        df, targets = load(key)
        print(f"  targets ({len(targets)}): {targets}")
        print(f"  shape after cleaning: {df.shape}")
        tcols = [t for t in targets if t in df.columns]
        print(df[tcols].describe().T[["mean", "std", "min", "max"]].round(3).to_string())
        print()


if __name__ == "__main__":
    main()
