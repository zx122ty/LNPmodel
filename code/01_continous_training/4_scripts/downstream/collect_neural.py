#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapter: MolTrain fine-tune output -> ``downstream_results/`` format.

The two halves of this evaluation write their results in different shapes, and
they must not stay that way: ``run_matrix.compare_runs`` refuses to pair runs
whose folds differ, and it reads ``per_target[<col>][f"{metric}_per_fold"]``. The
MolTrain script instead writes ``per_fold: [{fold, r2, ...}, ...]`` per dataset.

This module is the single translation point. It exists rather than teaching
``compare_runs`` a second format because a second format in the comparison code
is how a pairing bug hides: the numbers would still line up positionally even if
the fold lists did not.

**The one thing that can silently corrupt a pairing here** is MolTrain dropping a
fold. ``nnmodel.py`` skips any fold with fewer than 2 test rows, and the
downstream runner skips none. If a fold were dropped, the surviving per-fold
values would shift left and every subsequent pair would compare different folds.
That is checked explicitly below and refuses rather than warns.

Locating the run: MolTrain names its per-run file by wall-clock second
(``moltrain_results_YYYYmmdd_HHMMSS.json``), so a directory scan is required --
but a scan must not be allowed to guess. Every candidate is matched against the
caller's (dataset, split, seed, lr, epochs), and if two candidates survive the
match with DIFFERENT numbers the module refuses. Silently taking the newest would
make a re-run invisible: the report would quote the first run's R2 while the
bundle shipped the second run's predictions.

Usage:
    python 4_scripts/downstream/collect_neural.py \
        --dataset pKa --split butina --seed 42 --model lipidfm_v5 \
        --lr 5e-5 --epochs 40
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
sys.path.insert(0, str(ROOT / "4_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import split_files  # noqa: E402
from metrics import score_folds  # noqa: E402

RESULTS_DIR = ROOT / "5_outputs/downstream_results"
PRED_DIR = ROOT / "5_outputs/downstream_predictions"
MOLTRAIN_DIR = ROOT / "5_outputs/finetune_moltrain"
OOF_DIR = MOLTRAIN_DIR / "oof"


# The label MolTrain writes into ``run_label`` for each arm the project collects
# under. Keyed by the label ``collect_neural --model`` uses, because that is the
# name that appears in the result-cell filename -- and the cell filename is what
# the queue's idempotence guard tests. Three names for one thing (the queue's
# TAG, the cell's --model, MolTrain's run_label) is exactly the drift that makes
# a guard silently never fire, so the mapping lives here in one place.
MODEL_RUN_LABELS = {
    "lipidfm_v5": "unimol2_lipids_84m_v5/checkpoint_1_30000.pt",
    "stock_84m": "stock-84M",
    # The 坑 #12 control arm (finetune_moltrain --es-val internal). It needs its own
    # label for the same reason it needs its own aggregate key: it shares
    # (dataset, split, seed, lr, epochs) with the main arm and differs only in which
    # fold early stopping watched. Collecting it under "lipidfm_v5" would overwrite
    # the main arm's result cell -- the cell filename is the queue's idempotence
    # guard, so the control run would both destroy the main number and mark itself
    # as "already done" for it.
    "lipidfm_v5_esval": "unimol2_lipids_84m_v5/checkpoint_1_30000.pt_esval",
}


def _matches(d: dict, dataset: str, split: str, seed: int,
             lr: float | None, epochs: int | None,
             conformers: str | None = None,
             run_label: str | None = None,
             es_val: str | None = None) -> bool:
    a = d.get("args") or {}
    if dataset not in (d.get("results") or {}):
        return False
    if a.get("split") != split or a.get("seed") != seed:
        return False
    # Hyperparameters are part of the protocol (坑 #6): the same (split, seed) run
    # at a different lr is a different experiment and must not be substituted.
    if lr is not None and abs(float(a.get("lr", -1)) - lr) > 1e-12:
        return False
    if epochs is not None and a.get("epochs") != epochs:
        return False
    # Conformer source is a protocol dimension too, and it was MISSING here until
    # 2026-09-18. MolTrain's aggregate key already distinguishes it
    # (`...|ep40|3dconf`), but this finder did not -- so on Sheet2/butina/seed42
    # the 2D run (0.7652) and the 3dconf ablation (0.7106) were indistinguishable
    # candidates and `find_run` refused to choose between them.
    #
    # `None` means "the caller did not say", and is treated as "2D" because that
    # is what every existing invocation means: no caller passes
    # --use-3d-conformers through to collection, and the 3dconf runs are
    # ablations that are collected (if at all) by explicit filename.
    if conformers is not None:
        got = "3d" if a.get("use_3d_conformers") else "2d"
        if got != conformers:
            return False
    elif a.get("use_3d_conformers"):
        return False
    # WHICH MODEL is the same kind of protocol dimension, and it was missing too.
    # Found 2026-09-18 while preparing evaluation 1: on Sheet2/scaffold/seed42
    # this finder admitted **16** candidates spanning 14 distinct run_labels
    # (stock-84M, v2, v3, v4, v5, v5_60k, v5e, v6, and five freeze/adapter
    # ablations) because none of them is distinguished by
    # (dataset, split, seed, lr, epochs). Sheet2/scaffold is the protocol whose
    # folds never changed, so it is the one protocol where a dozen historical
    # runs legitimately share the key -- and collection there could never
    # succeed, at any seed.
    #
    # The refusal message is not a safe failure mode here either: it prints the
    # candidates with their R2, and the top of that menu (0.6892, a v5_60k run)
    # is not the arm the queue launched. A human picking from it would put a
    # different model's number under the ``lipidfm_v5`` cell name.
    #
    # `None` keeps the old behaviour for callers that predate this parameter
    # (the pKa cells were collected that way and are correct -- they had exactly
    # one candidate, so nothing was ambiguous).
    if run_label is not None and d.get("run_label") != run_label:
        return False
    # The early-stopping validation SOURCE is the same kind of protocol dimension,
    # and it was missing too (added 2026-09-19 with --es-val). The 坑 #12 control
    # arm and the main arm share (dataset, split, seed, lr, epochs, run_label) and
    # differ only in which fold early stopping watched -- so without this, one arm's
    # number lands under the other arm's cell name, which is precisely the failure
    # this finder was hardened against twice already.
    #
    # `None` means "the caller did not say", and resolves to "test" because that is
    # what every invocation before this parameter meant, and what every run recorded
    # before the aggregate key grew `|esval` means (they have no `es_val` field).
    #
    # WHERE the field lives is part of this fix, not an implementation detail.
    # `finetune_moltrain` records `es_val` INSIDE the per-dataset result block
    # (`results[<ds>]["es_val"]`) and does not echo it at the file's top level, so
    # `d.get("es_val")` here would read None on every real file and the dimension
    # would silently degrade to "matches anything" -- the exact failure mode this
    # parameter was added to prevent. The aggregate container does put it at the top
    # level of each run entry, so read there too: `d` is a timestamped result file OR
    # an aggregate entry depending on the caller, and the two layouts differ.
    if es_val is not None:
        got = (d.get("es_val")
               or ((d.get("results") or {}).get(dataset) or {}).get("es_val")
               or "test")
        if got != es_val:
            return False
    return True


def find_run(results_dir: Path, dataset: str, split: str, seed: int,
             lr: float | None = None, epochs: int | None = None,
             conformers: str | None = None,
             run_label: str | None = None,
             es_val: str | None = None,
             pick_newest: bool = False) -> tuple[dict, Path]:
    """Find the unique MolTrain run matching this protocol.

    Returns ``(result_dict_for_dataset, source_path)``. Refuses on zero matches
    (nothing was run) and on multiple matches with different numbers (a re-run,
    where picking either would silently misreport which one the bundle holds).

    ``pick_newest`` resolves that second case instead of refusing, by taking the
    most recently written result file. It is OFF by default and must be asked for
    explicitly, because "the newest file is the right one" is only true when the
    caller has just launched the run -- and even then it is a claim about
    wall-clock ordering, not about identity. When it is used, every candidate it
    passed over is printed, so the choice is auditable rather than invisible.
    This is the case for evaluation 1 on ``Sheet2/scaffold``: the folds never
    changed there, so a dozen historical runs legitimately share the protocol
    key, and only the one the queue just finished is the deliverable's.
    """
    cands = []
    for p in sorted(Path(results_dir).glob("moltrain_results_*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if _matches(d, dataset, split, seed, lr, epochs, conformers, run_label, es_val):
            cands.append((p, d))
    if not cands:
        raise SystemExit(
            f"no MolTrain run under {results_dir} matches "
            f"(dataset={dataset}, split={split}, seed={seed}, lr={lr}, epochs={epochs}, "
            f"conformers={conformers or '2d'}, run_label={run_label or 'any'}, "
            f"es_val={es_val or 'test'}). "
            f"Check the queue log -- the run may have failed, or the hyperparameters "
            f"may differ from what was launched.")
    sigs = {json.dumps(c[1]["results"][dataset], sort_keys=True) for c in cands}
    if len(sigs) > 1:
        if not pick_newest:
            raise SystemExit(
                f"{len(cands)} MolTrain runs match but disagree:\n" +
                "\n".join(f"  {p.name}  r2={c['results'][dataset]['r2']}"
                          for p, c in cands) +
                "\nPick the intended one explicitly; refusing to choose by "
                "timestamp. Pass --pick-newest if you just launched this run.")
        # MolTrain stamps its result filename with the wall-clock second it was
        # written, so the lexical sort below IS the write order.
        chosen = max(cands, key=lambda c: c[0].name)
        print(f"  [pick-newest] {len(cands)} candidates disagree; taking the most "
              f"recently written result file:")
        for p, c in sorted(cands, key=lambda c: c[0].name):
            mark = "  <-- taken" if p is chosen[0] else ""
            print(f"      {p.name}  r2={c['results'][dataset]['r2']}{mark}")
        cands = [chosen]
    p, d = cands[0]
    if len(cands) > 1:
        print(f"  [note] {len(cands)} matching runs, all identical -> using {p.name}")
    return d["results"][dataset], p


def to_downstream_format(res: dict, dataset: str, model: str, split: str, seed: int,
                         kfold: int, target_cols: list[str], smiles: list[str],
                         checkpoint: str = "", source: str = "",
                         conformers: str = "2d") -> dict:
    """Reshape one MolTrain result into the ``downstream_results`` schema."""
    folds = split_files.load_folds(dataset, split, seed, kfold, smiles=smiles)
    n_folds = len(folds)
    per_fold = res["per_fold"]

    # A dropped fold is the failure this adapter exists to catch: nnmodel.py skips
    # folds with <2 test rows, and the downstream runner does not, so the two
    # lists would still be the same LENGTH only if nothing was dropped. Compare
    # the fold ids rather than the counts -- a count can match after a drop+add.
    got_ids = [f["fold"] for f in per_fold]
    if got_ids != list(range(n_folds)):
        raise SystemExit(
            f"MolTrain reported folds {got_ids}, expected {list(range(n_folds))}. "
            f"A dropped fold shifts every subsequent value and the paired test "
            f"would compare different folds. Refusing to write a mismatched result.")
    for f, (_tr, te) in zip(per_fold, folds):
        if f["n"] != len(te):
            raise SystemExit(
                f"fold {f['fold']}: MolTrain scored {f['n']} rows, the frozen split "
                f"has {len(te)}. The model was not trained on these folds.")

    # MolTrain is single-target on this path, so every target column shares the
    # one set of per-fold values. Duplicating is correct here (the model really
    # did produce one prediction per molecule); it is NOT a placeholder.
    #
    # MTR is the exception, and the duplication above would be a LIE there: its 16
    # columns are 16 different assays with 16 different fold R² values, and copying
    # the first column's numbers across all 16 would produce a full, plausible,
    # completely wrong table -- the failure mode has no error, only a table. So when
    # the run reports `per_target` (finetune_moltrain does, for every dataset), that
    # is the source of truth and the loop below only fills gaps for a column it does
    # not name. Historical single-target runs predate `per_target` and take the
    # duplication branch unchanged.
    res_per_target = res.get("per_target") or {}
    per_target = {}
    for col in target_cols:
        b = res_per_target.get(col)
        if b is not None:
            pf = b["per_fold"]
            per_target[col] = {
                "r2": b["r2"], "r2_std": b["r2_std"],
                "r2_per_fold": [f["r2"] for f in pf],
                "rmse": b["rmse"], "rmse_per_fold": [f["rmse"] for f in pf],
                "spearman": b["spearman"],
                "spearman_per_fold": [f["spearman"] for f in pf],
                "relaxed_spearman": b.get("relaxed_spearman"),
                "relaxed_spearman_per_fold": [f.get("relaxed_spearman") for f in pf],
                # MolTrain reports R2 and Spearman but not the ranking metrics; those
                # come from the persisted OOF array below, not faked here.
                "r2_pooled": b.get("r2_pooled"),
                "source": "moltrain",
            }
            continue
        per_target[col] = {
            "r2": res["r2"], "r2_std": res["r2_std"],
            "r2_per_fold": [f["r2"] for f in per_fold],
            "rmse": res["rmse"], "rmse_per_fold": [f["rmse"] for f in per_fold],
            "spearman": res["spearman"],
            "spearman_per_fold": [f["spearman"] for f in per_fold],
            # MolTrain reports R2 and Spearman but not the ranking metrics; those
            # come from the persisted OOF array below, not faked here.
            "r2_pooled": res.get("r2_pooled"),
            "source": "moltrain",
        }

    return {
        "run_id": f"{dataset}|{model}|{split}|seed{seed}|k{kfold}",
        "dataset": dataset, "model": model, "split": split, "seed": seed,
        "kfold": kfold, "n": res["n"], "targets": target_cols,
        "checkpoint": checkpoint,
        "target_scale": res.get("target_scale"),
        "elapsed_sec": None,
        "_provenance": {
            "note": "imported from a MolTrain run by collect_neural.py",
            "source_result": source,
            "trainable_params": res.get("trainable_params"),
            "total_params": res.get("total_params"),
            "es_val": res.get("es_val") or "test",
            # The conformer source is a protocol dimension (it changes the model's
            # INPUT, not its hyperparameters). Recorded here as well as checked in
            # find_run, because `report_table` pairs arms by reading the CELLS, and
            # a cell that does not carry this dimension cannot be protected from
            # being paired against a 3dconf ablation.
            "conformers": conformers,
            # Two different protocols hide behind one field name, and which one a
            # number is on decides whether it may be quoted beside the classical arms.
            "early_stop_validation": (
                "the TEST fold (unimol_tools/models/nnmodel.py:188) -- 坑 #12. "
                "Strictly weaker protocol than the MLP baseline, which carves its "
                "early-stopping set out of train."
                if (res.get("es_val") or "test") == "test" else
                "a train-internal holdout carved out of the TRAIN fold "
                "(finetune_moltrain --es-val internal). Same protocol as "
                "downstream/baselines.py's MLP, so this number IS comparable to the "
                "classical arms; the real test folds were re-predicted from "
                "model_<k>.pth and MolTrain never saw them."),
            "es_val_meta": res.get("es_val_meta"),
        },
        "per_target": per_target,
    }


def dump_path(oof_dir: Path, dataset: str, split: str, seed: int,
              run_label: str | None = None, allow_legacy: bool = True) -> Path | None:
    """The OOF dump filename for this run, current naming preferred; None if absent.

    Current (2026-09-18 onward): ``oof_<ds>_<split>_seed<S>_<run_label>.npz``.
    The run_label was added because the protocol alone does not identify a run --
    stock-84M and LipidFM-v5 on the same (dataset, split, seed) shared a filename,
    and since collection PREFERS the dump, the arm that finished last silently
    owned it.

    Legacy: ``oof_<ds>_<split>_seed<S>.npz``. Kept as a fallback so the pKa runs
    written before the change stay collectable; it is only accepted when the
    current name is absent, and the caller still verifies the contents.

    ``allow_legacy=False`` for the 坑 #12 control arm: the legacy name carries no
    run identity at all, so falling back to it would silently hand the control arm
    the MAIN arm's predictions -- the exact substitution the run_label was added to
    stop, and one that would produce a plausible-looking number rather than an
    error.

    Returning None (rather than raising) is the caller's signal to fall back to
    ``cv.data``, which is a legitimate path for runs that predate the dump.
    """
    # run_label is a checkpoint path (``unimol2_lipids_84m_v5/checkpoint_1_30000.pt``),
    # so it must be flattened the same way finetune_moltrain flattens it for the
    # exp_tag -- a raw "/" here would silently become a directory separator and the
    # file would never be found.
    safe = run_label.replace("/", "_") if run_label else None
    cur = Path(oof_dir) / f"oof_{dataset}_{split}_seed{seed}_{safe}.npz" if safe else None
    legacy = Path(oof_dir) / f"oof_{dataset}_{split}_seed{seed}.npz"
    if cur is not None and cur.exists():
        return cur
    if allow_legacy and legacy.exists():
        return legacy
    return None


def load_oof_dump(oof_dir: Path, dataset: str, split: str, seed: int,
                  n_rows: int, run_label: str | None = None,
                  allow_legacy: bool = True,
                  target_cols: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, Path]:
    """Read the per-molecule OOF dump written by ``finetune_moltrain`` (§3.6).

    This is the PREFERRED source: unlike ``cv.data`` it is self-describing -- it
    carries ``fold_of_row`` and ``target_scale``, so the row alignment does not
    have to be re-derived, and it is written under a name that includes the
    dataset, split and seed rather than living in a save_path shared by every
    re-run of the same protocol.

    The ``.npz`` is only a cache of a computed quantity, so it is verified rather
    than trusted: its own ``fold_of_row`` must equal the frozen folds, and its
    ``y`` must equal the registry's target. A dump left over from a different
    dataset version would otherwise flow straight into the deliverable, since the
    file name alone cannot detect that.

    ``run_label`` selects the current per-run filename (see ``dump_path``). When
    it is given and the file's stored contents disagree with it in any way the
    caller can check, the caller's ``merge_screening`` still refuses -- the name
    is a convenience, the verification is the guarantee.
    """
    p = dump_path(oof_dir, dataset, split, seed, run_label, allow_legacy=allow_legacy)
    if p is None:
        raise SystemExit(f"no OOF dump for {dataset}/{split}/seed{seed} "
                         f"(run_label={run_label})")
    d = np.load(p, allow_pickle=True)
    # Keep (n, n_t): a ravel here is the exact mistake the shape guard upstream
    # forbids, because a raveled (83, 16) still has "n rows" if you multiply wrong
    # and every downstream index would pair a molecule with another assay's
    # prediction. n_t == 1 gives the same sequence the old ravel produced.
    y = np.asarray(d["y"], float)
    oof = np.asarray(d["oof"], float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if oof.ndim == 1:
        oof = oof.reshape(-1, 1)
    if y.shape[0] != n_rows or oof.shape[0] != n_rows:
        raise SystemExit(f"{p.name} holds {y.shape[0]}/{oof.shape[0]} rows, dataset has {n_rows}")
    if y.shape != oof.shape:
        raise SystemExit(f"{p.name}: y is {y.shape} but oof is {oof.shape}")
    # The stored column names are what makes a 16-wide array addressable. Verify
    # them against the caller's list rather than trusting the order: a dump from a
    # different registry version would otherwise relabel every assay silently.
    if "targets" in d.files:
        stored = [str(t) for t in d["targets"]]
        if stored != list(target_cols):
            raise SystemExit(
                f"{p.name}: its stored targets {stored} do not match this dataset's "
                f"{list(target_cols)}; the columns would be mislabelled.")
    if y.shape[1] != len(target_cols):
        raise SystemExit(f"{p.name} has {y.shape[1]} target column(s), the dataset "
                         f"declares {len(target_cols)}")
    if str(d["split"]) != split or int(d["seed"]) != seed:
        raise SystemExit(f"{p.name} declares split={d['split']!r} seed={int(d['seed'])}, "
                         f"asked for {split!r}/{seed}")
    return y, oof, p


def recover_oof_from_cv_data(results_dir: Path, dataset: str, split: str, seed: int,
                             lr: float, epochs: int, n_rows: int,
                             result_json: Path | None = None) -> tuple[np.ndarray, Path]:
    """Rebuild the OOF arrays from MolTrain's own saved artifacts.

    Runs launched before the OOF dump existed still left ``cv.data`` (the per-fold
    out-of-fold predictions, in the run's ``save_path``) and ``target_scaler.ss``
    beside it. Two facts make these usable, and both were checked rather than
    assumed:

    * ``cv.data`` is the OOF vector in ORIGINAL ROW ORDER, not fold-major:
      nnmodel.py does ``y_pred[te_idx] = _y_pred`` into a zeros(n) array.
    * ``cv.data`` is on the Z-SCORED scale. ``train.py`` un-standardises into
      ``clf.cv_pred`` but the dump happens first (nnmodel.py:257), so the file
      holds the raw modelling-scale values. Reading it without the scaler gives
      R2 of -50 to -120 -- which is how this was noticed.

    The caller MUST still verify the result: ``merge_screening`` re-derives the
    per-fold R2 and refuses if it does not match what MolTrain logged. This
    function therefore returns data, not trust.

    **Re-run clobber guard.** MolTrain's ``save_path`` is keyed only by the
    experiment tag -- ``<ds>_exp_<ckpt>_<split>_seed<S>_lr<L>_ep<E>`` -- with no
    timestamp, so launching the same protocol twice writes into the SAME
    directory and overwrites ``cv.data`` in place (train.py logs "Overwrite
    output directory" when it does). The result JSON, by contrast, is named by
    wall-clock second. So a ``cv.data`` newer than the result JSON it is being
    paired with cannot have produced it: the numbers in the JSON came from the
    first run and the predictions from the second, and both are internally
    consistent, so nothing downstream would notice. Refusing is the only safe
    response.

    This is not hypothetical. On 2026-09-18 a queue was re-launched for two pKa
    seeds that had already completed, and only an early kill left the original
    ``cv.data`` in place; a completed re-run would have silently substituted
    different predictions under the same run_id.
    """
    import joblib  # noqa: PLC0415

    pat = f"{dataset}_exp_*_{split}_seed{seed}_lr{lr}_ep{epochs}"
    cands = sorted(p for p in Path(results_dir).glob(pat) if (p / "cv.data").exists())
    if not cands:
        raise SystemExit(f"no OOF dump and no cv.data under {results_dir} matching "
                         f"{pat}; the run's artifacts are gone, re-run it")
    if len(cands) > 1:
        raise SystemExit(f"{len(cands)} candidate run dirs match {pat}: "
                         f"{[p.name for p in cands]}; refusing to guess")
    d = cands[0]

    if result_json is not None:
        # MolTrain names the result file moltrain_results_YYYYmmdd_HHMMSS.json.
        stamp = re.search(r"moltrain_results_(\d{8}_\d{6})\.json$", Path(result_json).name)
        if stamp is None:
            raise SystemExit(f"cannot read a wall-clock stamp from {result_json}; the "
                             f"clobber guard needs it. Pass the MolTrain result JSON.")
        t_result = datetime.strptime(stamp.group(1), "%Y%m%d_%H%M%S").timestamp()
        t_cv = (d / "cv.data").stat().st_mtime
        if t_cv > t_result + 1.0:
            raise SystemExit(
                f"{d.name}/cv.data is NEWER than {Path(result_json).name} "
                f"(cv.data {datetime.fromtimestamp(t_cv):%F %T} > result "
                f"{datetime.fromtimestamp(t_result):%F %T}). The run was re-launched "
                f"into the same save_path and overwrote the predictions, while the "
                f"result JSON still holds the FIRST run's numbers. Re-collect from a "
                f"run whose artifacts and JSON belong together.")

    scaler = joblib.load(d / "target_scaler.ss")
    cv = np.asarray(joblib.load(d / "cv.data"), float)
    if cv.ndim == 1:
        cv = cv.reshape(-1, 1)
    if cv.shape[0] != n_rows:
        raise SystemExit(f"{d}/cv.data has {cv.shape[0]} rows, dataset has {n_rows}")
    oof = np.asarray(scaler.inverse_transform(cv), float)
    if oof.ndim == 1:
        oof = oof.reshape(-1, 1)
    print(f"  [recover] rebuilt OOF from {d.name}/cv.data + target_scaler.ss "
          f"(pre-dump run; per-fold R2 verified below)")
    # y is not in cv.data; the caller supplies it from the registry. Returning a
    # placeholder here would be worse than returning None, so the signature keeps
    # y out and the caller passes df's target in.
    return oof, d


def merge_screening(out: dict, y: np.ndarray, oof: np.ndarray, dataset: str,
                    split: str, seed: int, kfold: int, target_cols: list[str],
                    smiles: list[str], origin: str,
                    fold_of_row_stored: np.ndarray | None = None) -> None:
    """Attach the ranking metrics computed from the run's per-molecule OOF array.

    **This is also where the OOF row order is verified.** MolTrain reports its own
    per-fold R² from the live ``cv_pred`` array; this function recomputes per-fold
    R² from the array handed to it. If the two disagree, the array is in a
    different order than MolTrain scored -- the fold means would still look
    plausible while every ranking metric was computed on scrambled pairs. That
    check is why this asserts instead of merging quietly.
    """
    folds = split_files.load_folds(dataset, split, seed, kfold, smiles=smiles)
    # 2-D from here on: (n, n_t). `n_t == 1` behaves exactly as the old ravel did,
    # because column 0 of an (n, 1) array is the same sequence. Anything wider MUST
    # stay column-indexed -- a ravel of (83, 16) gives 16n values, and scoring that
    # against another raveled array pairs each molecule with a different assay's
    # prediction while every fold R2 stays plausible.
    y = np.asarray(y, float)
    oof = np.asarray(oof, float)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if oof.ndim == 1:
        oof = oof.reshape(-1, 1)
    if y.shape != oof.shape:
        raise SystemExit(f"{origin}: y is {y.shape} but oof is {oof.shape}")
    if y.shape[1] != len(target_cols):
        raise SystemExit(
            f"{origin}: the OOF array has {y.shape[1]} column(s) but the dataset "
            f"declares {len(target_cols)} target(s) ({target_cols}); the per-target "
            f"table would be built from the wrong axis.")

    # Re-derive the fold-of-row map from the frozen folds and require it to match
    # any stored one. If these disagree, the OOF array was produced on a different
    # partition and every screening metric built from it is meaningless.
    if fold_of_row_stored is not None:
        fold_of_row = np.full(len(smiles), -1, dtype=int)
        for k, (_tr, te) in enumerate(folds):
            fold_of_row[np.asarray(te, int)] = k
        if not np.array_equal(fold_of_row, np.asarray(fold_of_row_stored, int)):
            raise SystemExit(f"{origin}: stored fold_of_row does not match the frozen "
                             f"folds -- predictions came from a different partition")

    for j, col in enumerate(target_cols):
        scored = score_folds(y[:, j], oof[:, j], folds)
        # MolTrain's per-fold values are round(...,4); recomputed ones are not.
        got = scored["r2_per_fold"]
        want = out["per_target"][col]["r2_per_fold"]
        if len(got) != len(want) or any(
                g is None or abs(g - w) > 5e-4 for g, w in zip(got, want)):
            raise SystemExit(
                f"{origin}: per-fold R2 recomputed from the OOF array "
                f"{got} does not match MolTrain's own {want}. The persisted "
                f"predictions are not aligned with the rows MolTrain scored; "
                f"refusing to attach ranking metrics computed from them.")
        # Keep every scored metric EXCEPT the per-fold r2/rmse/pearson/spearman
        # blocks, which MolTrain already reported from its live arrays and which
        # were just proven identical.
        out["per_target"][col].update(
            {k: v for k, v in scored.items()
             if not k.startswith(("r2", "rmse", "pearson", "spearman"))})
    out["_provenance"]["oof"] = origin
    out["_provenance"]["oof_verified"] = (
        "per-fold R2 recomputed from the OOF array matches MolTrain's own to 5e-4")

    # Write the .npz companion in run_matrix's exact format. This is not
    # bookkeeping: repro_bundle.build_bundle REFUSES to bundle a result whose
    # companion is missing ("a result without its per-sample predictions cannot
    # be audited"), so without this the neural arms would be the only ones in the
    # deliverable that a third party could not check. Same keys, same shapes --
    # (n, n_targets) even for one target -- so nothing downstream needs a branch.
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PRED_DIR / f"{out['run_id']}.npz",
        oof=oof,
        y=y,
        targets=np.array(target_cols, dtype=object),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--model", required=True,
                    help="label for this arm, e.g. lipidfm_v5 or stock_84m")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--lr", required=True, type=float,
                    help="the run's learning rate; part of the run's identity (坑 #6), "
                         "so it is required rather than defaulted")
    ap.add_argument("--epochs", required=True, type=int)
    ap.add_argument("--run-label", default=None,
                    help="MolTrain's run_label for this arm (default: looked up from "
                         "--model via MODEL_RUN_LABELS). Pass an explicit value for an "
                         "arm that is not in that table; pass '' to match any model, "
                         "which is only safe on a protocol where exactly one run "
                         "exists (see the model-dimension note on _matches).")
    ap.add_argument("--pick-newest", action="store_true",
                    help="when several matching runs disagree, take the most recently "
                         "written result file instead of refusing. Only correct right "
                         "after launching a run yourself; every candidate passed over "
                         "is printed.")
    ap.add_argument("--es-val", choices=["test", "internal"], default="test",
                    help="which early-stopping validation source this arm used. "
                         "'test' (default) is every run before 2026-09-19 and matches "
                         "runs whose aggregate entry has no es_val field; 'internal' is "
                         "the 坑 #12 control arm (finetune_moltrain --es-val internal). "
                         "Part of the run identity: the two arms share every other "
                         "dimension, so without this the control arm's number would be "
                         "collected into the main arm's cell.")
    ap.add_argument("--conformers", choices=["2d", "3d"], default="2d",
                    help="conformer source this arm was trained with (default '2d', "
                         "which is every run except the --use-3d-conformers ablation). "
                         "Part of the run identity in find_run, and recorded on the "
                         "cell so report_table can refuse to pair arms across it.")
    ap.add_argument("--results-dir", type=Path, default=MOLTRAIN_DIR)
    ap.add_argument("--oof-dir", type=Path, default=OOF_DIR,
                    help="where finetune_moltrain wrote its per-molecule dumps "
                         "(default: <results-dir>/oof). Exposed so a smoke run "
                         "written to a scratch output directory can be collected "
                         "without copying its dumps into the real store.")
    a = ap.parse_args()

    import ds_registry  # noqa: PLC0415
    df, target_cols = ds_registry.load(a.dataset, verbose=False)
    smiles = df["SMILES"].astype(str).tolist()

    if a.run_label is None:
        if a.model not in MODEL_RUN_LABELS:
            raise SystemExit(
                f"--model {a.model!r} has no entry in MODEL_RUN_LABELS, so the model "
                f"dimension of the run identity would go unchecked and find_run could "
                f"resolve to a different arm's run. Pass --run-label explicitly "
                f"(known: {sorted(MODEL_RUN_LABELS)}).")
        run_label = MODEL_RUN_LABELS[a.model]
    else:
        run_label = a.run_label or None

    res, src = find_run(a.results_dir, a.dataset, a.split, a.seed, a.lr, a.epochs,
                        conformers=a.conformers, run_label=run_label,
                        es_val=a.es_val, pick_newest=a.pick_newest)
    print(f"  [run] {src.name}  run_label={json.loads(src.read_text())['run_label']}")
    out = to_downstream_format(res, a.dataset, a.model, a.split, a.seed, a.kfold,
                               target_cols, smiles,
                               checkpoint=json.loads(src.read_text())["checkpoint"],
                               source=str(src), conformers=a.conformers)

    # OOF source: prefer the run's own dump (self-describing, and named by the
    # FULL run identity) and fall back to cv.data only for runs that predate it.
    # The two are NOT equivalent for the clobber case -- a dump is keyed by
    # (protocol, run_label) so a re-run of a different arm writes a different
    # file, while cv.data is keyed by save_path and a re-run overwrites in place.
    # Preferring the dump is therefore also the stronger guard; the fallback
    # carries its own (see recover_oof_from_cv_data).
    #
    # The run_label is passed so a stock-84M run cannot be collected against the
    # LipidFM-v5 dump for the same protocol. Before 2026-09-18 the dump name held
    # only the protocol, so the second arm of every pair would have read the
    # first arm's predictions -- and `merge_screening`'s per-fold R2 check is what
    # would have caught it, by refusing. Passing the label means it resolves
    # correctly instead of merely failing safely.
    dump = dump_path(a.oof_dir, a.dataset, a.split, a.seed, run_label,
                     allow_legacy=(a.es_val != "internal"))
    fold_of_row_stored = None
    # The registry's labels, as (n, n_t). `target_cols[0]` is NOT used: on MTR that
    # would verify column 0's labels against the dump's 16 columns and the shape
    # check would fail -- or worse, on a single-target dataset it would quietly
    # accept a dump whose other 15 columns came from somewhere else.
    y_registry = df[target_cols].to_numpy(float)
    if dump is not None:
        y_dump, oof, origin = load_oof_dump(a.oof_dir, a.dataset, a.split, a.seed, len(df),
                                            run_label=run_label,
                                            allow_legacy=(a.es_val != "internal"),
                                            target_cols=target_cols)
        # Tolerance, not exactness: the dump stores `y` as float32 (it comes from
        # MolTrain's `clf.data['target']`, which is float32), while the registry
        # holds float64. `np.allclose`'s default rtol=1e-5 is RELATIVE to b, so it
        # collapses to atol=1e-8 for a label that is exactly 0 -- and Sheet2 has
        # **10 exact-zero labels** (log2 RLU of 0). A float32 round-trip of 0.0
        # stores as 1.575e-07, which is > 1e-8 and therefore "not equal".
        #
        # This refused every Sheet2 collection on 2026-09-19 (all 10 jobs trained,
        # 0 collected) while Sheet3/pKa passed -- they have no zero labels, so
        # rtol*|b| stayed large enough to absorb float32. The check was right to
        # exist; the threshold was wrong for the scale it was comparing at.
        #
        # atol=1e-5 is ~1e-6 of the label std (3.96 on Sheet2) and still ~100x
        # tighter than the smallest difference that could mean "different dataset
        # version": a stale Sheet3 dump against Sheet2 differs by ~2.7 in the mean.
        # The two datasets' targets are not close in any coordinate, so this
        # loosening does not weaken the version check it is there to perform.
        if not np.allclose(y_dump, y_registry, atol=1e-5):
            _d = np.abs(y_dump - y_registry)
            raise SystemExit(
                f"{dump.name}: its stored y does not match the registry's target "
                f"(max|delta|={_d.max():.3e}, mean|delta|={_d.mean():.3e}, "
                f"n>1e-3={int((_d > 1e-3).sum())}); the dump is from a different "
                f"dataset version. Note the dump stores y as float32, so a max "
                f"delta of ~1e-6 is rounding, not a mismatch.")
        fold_of_row_stored = np.load(dump, allow_pickle=True)["fold_of_row"]
        origin = f"{origin} (finetune_moltrain OOF dump)"
    else:
        oof, src_dir = recover_oof_from_cv_data(
            a.results_dir, a.dataset, a.split, a.seed, a.lr, a.epochs, len(df),
            result_json=src)
        origin = str(src_dir / "cv.data") + " (unscaled via target_scaler.ss)"
    merge_screening(out, y_registry, oof, a.dataset, a.split,
                    a.seed, a.kfold, target_cols, smiles, origin=origin,
                    fold_of_row_stored=fold_of_row_stored)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / f"{out['run_id']}.json"
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"wrote {p}")
    # For MTR the flat `res['r2']` is column 0's number, so quoting it as THE
    # fold-mean R2 would read as a single headline for 16 assays. Print per column
    # instead, and flag the near-constant ones the same way the training run did.
    tcols = out["targets"]
    if len(tcols) > 1:
        print(f"  fold-mean R2 per target ({len(tcols)} columns):")
        for t in tcols:
            b = out["per_target"][t]
            flag = "  [near-constant: exclude from headline]" \
                if t in ds_registry.MTR_LOW_VARIANCE else ""
            print(f"    {t:34s} {b['r2']:+.4f} +- {b['r2_std']:.4f}{flag}")
    else:
        print(f"  fold-mean R2 = {res['r2']:.4f} +- {res['r2_std']:.4f}  "
              f"per-fold {[f['r2'] for f in res['per_fold']]}")


if __name__ == "__main__":
    main()
