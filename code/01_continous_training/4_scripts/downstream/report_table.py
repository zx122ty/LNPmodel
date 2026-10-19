#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Paired comparison table: every model on one (dataset, split), pooled over seeds.

This is the reporting step the plan's §8 calls "指标全表 + Wilcoxon 配对检验". It
reads the ``downstream_results/`` JSONs, pools the 5 seeds x 5 folds into one
25-fold vector per model, and runs the paired Wilcoxon against a reference arm.

Two things about the pooling, both of which change what the number means:

* **Fold order must be identical across models.** The seeds are iterated in
  ``PARTITION_SEEDS`` order for every model, so fold ``i`` of the pooled vector is
  the same (seed, fold) cell for every arm. ``paired_compare`` is a signed-rank
  test, so a permuted order would still produce a p-value -- a meaningless one.
  The order is therefore asserted, not assumed.

* **A missing seed is a hard error, not a shorter vector.** If one arm has 4 seeds
  and another has 5, the pooled vectors differ in length and the "pairing" is
  really a comparison of different fold sets. Refusing is the only safe response;
  ``paired_compare`` would silently truncate nothing but the caller's intent.

The 25-fold minimum is the project's own rule (CLAUDE.md 工作约定 #2): effects
below 0.04 need 15 folds, and the measured run-to-run noise is 0.0335. A 5-fold
table is printed too, but it is labelled as the underpowered thing it is.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
sys.path.insert(0, str(ROOT / "4_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import split_files  # noqa: E402
from metrics import benjamini_hochberg, paired_compare  # noqa: E402

RESULTS_DIR = ROOT / "5_outputs/downstream_results"

# MTR's screening metrics are reported POOLED, at k as a fraction of n, and this
# is not a stylistic choice. MTR's folds hold 11-32 molecules, so EF@50 and
# NDCG@50 are either NaN (k > n) or silently identical to EF@n / NDCG@n -- a
# "top-50" metric on a 17-molecule fold ranks the whole fold (坑 #11). Pooled
# metrics rank all 83 molecules using the out-of-fold prediction of each, so the
# k in the name is the k it claims to be. `ef@50_pooled` is still computed (83 >
# 50, so it is at least defined), but the ceiling moves with k, so the 1/5/10%
# family is the comparable one.
MTR_SCREENING = ["ef@1pct_pooled", "ef@5pct_pooled", "ef@10pct_pooled",
                 "ndcg@1pct_pooled", "ndcg@5pct_pooled", "ndcg@10pct_pooled"]


def load_cell(dataset: str, model: str, split: str, seed: int, kfold: int,
              target: str) -> dict:
    p = RESULTS_DIR / f"{dataset}|{model}|{split}|seed{seed}|k{kfold}.json"
    if not p.exists():
        raise SystemExit(f"missing run: {p.name} (model {model!r}, seed {seed})")
    d = json.loads(p.read_text(encoding="utf-8"))
    if target not in d["per_target"]:
        raise SystemExit(f"{p.name} has no target {target!r} "
                         f"(has {sorted(d['per_target'])})")
    return d


def pooled_vector(dataset: str, model: str, split: str, seeds: list[int], kfold: int,
                  target: str, metric: str,
                  model_meta: dict[str, dict] | None = None) -> np.ndarray:
    """Concatenate per-fold values across seeds, in PARTITION_SEEDS order.

    Each cell must have exactly ``kfold`` per-fold values; a cell with fewer means
    MolTrain dropped a fold and the concatenation would shift every later value.

    ``model_meta`` maps a model label to the (es_val, conformers) pair its cells
    were collected with. Every cell of a comparison must agree on both: they are
    protocol dimensions, and the two arms that differ in one of them share a
    ``run_id`` otherwise -- so pairing a default-protocol arm against an
    ``--es-val internal`` control arm by model name alone would silently compare
    two different experiments fold-by-fold and report the difference as a model
    difference. See ``resolve_arms``.
    """
    out = []
    for seed in seeds:
        d = load_cell(dataset, model, split, seed, kfold, target)
        if model_meta is not None:
            _check_protocol(d, model_meta[model])
        vals = d["per_target"][target][f"{metric}_per_fold"]
        if len(vals) != kfold or any(v is None for v in vals):
            raise SystemExit(
                f"{d['run_id']}: {len(vals)} per-fold values for kfold={kfold} "
                f"(values {vals}). A dropped or unscored fold shifts the pooled "
                f"vector and every paired comparison built on it.")
        out.extend(float(v) for v in vals)
    return np.asarray(out, float)


def _cell_protocol(d: dict) -> tuple[str, str]:
    """The (es_val, conformers) pair a collected cell was produced under.

    ``es_val`` is read from the provenance block -- ``collect_neural`` writes it
    there for neural cells, and it is the dimension that separates the 坑 #12
    control arm from the main arm. A cell with no provenance is ``("test", "2d")``,
    which is correct for both kinds of legacy cell: ``baselines`` never validates
    on a test fold and never sees conformers, and every neural cell collected
    before 2026-09-20 was collected from a default-protocol run.
    """
    prov = d.get("_provenance") or {}
    return (prov.get("es_val") or "test", prov.get("conformers") or "2d")


def _check_protocol(d: dict, want: tuple[str, str]) -> None:
    got = _cell_protocol(d)
    if got != want:
        raise SystemExit(
            f"{d['run_id']}: collected with es_val={got[0]!r}/conformers={got[1]!r} "
            f"but the requested arm is {want[0]!r}/{want[1]!r}. These are protocol "
            f"dimensions, not labels -- pairing across them would compare two "
            f"different experiments and report the difference as a model effect. "
            f"Pass --es-val/--conformers to match, or collect the missing cells.")


def expected_protocol(model: str) -> tuple[str, str]:
    """The (es_val, conformers) pair a model label promises.

    The 坑 #12 control arm is a SEPARATE arm with its own label, not a variant of
    the main one -- so a table can (and for MTR should) hold both, and the pairing
    between them is the measurement. Deriving the expectation from the label keeps
    the two from being confused: asking for ``lipidfm_v5`` and being handed the
    control arm's cells would put a different protocol's numbers under the main
    arm's name, which is exactly the substitution the queue's cell filename and
    ``collect_neural``'s run_label check exist to prevent.

    The convention is the one ``collect_neural.MODEL_RUN_LABELS`` already uses:
    a ``_esval`` suffix means the train-internal early-stopping arm.

    External arms (``EXTERNAL_ARMS``) carry the same ``_esval`` convention but are
    not necessarily ``2d``, so they are looked up explicitly rather than inferred
    from the suffix.
    """
    if model in EXTERNAL_ARMS:
        return EXTERNAL_ARMS[model]
    return ("internal" if model.endswith("_esval") else "test", "2d")


# Arms that come from another codebase and whose input representation is not the
# project's 2D/3D toggle. Declared explicitly rather than inferred, because the
# inference rule above would silently label them "2d".
#
# LUMI-lab (deploy/lumi_pipeline_adapter.py) is inherently 3D: its input is
# 11 conformers per molecule (10x ETKDGv3/MMFF94s + 1x 2D), with no 2D-only mode.
# Declaring "2d" to satisfy the default rule would be a lie that a reader would
# then act on -- e.g. by pairing a LUMI cell against a `--use-3d-conformers`
# ablation cell as if they shared an input representation.
#
# Both LUMI arms early-stop the way their own driver does by default (on the test
# fold, i.e. 坑 #12), which is why the plain arm is "test"; `_esval` is the
# matched-protocol control the deploy adapter also produces.
EXTERNAL_ARMS: dict[str, tuple[str, str]] = {
    "lumi_lab": ("test", "3d"),
    "lumi_lab_esval": ("internal", "3d"),
}


def resolve_arms(models: list[str]) -> dict[str, tuple[str, str]]:
    """``model -> expected protocol`` for every arm in the table."""
    return {m: expected_protocol(m) for m in models}


def pooled_screening(dataset: str, model: str, split: str, seeds: list[int], kfold: int,
                     target: str, key: str) -> float:
    """Mean of a pooled screening metric across seeds.

    Screening metrics are pooled WITHIN a seed (one ranking over all 574 molecules,
    built from that seed's 5 fold-models) and then averaged across seeds. They are
    deliberately NOT averaged over folds: EF@k on a 115-molecule fold has a
    different ceiling than on the whole set (坑 #11), so a fold-mean EF@25 would
    compare the same label across different quantities.
    """
    vals = []
    for seed in seeds:
        d = load_cell(dataset, model, split, seed, kfold, target)
        v = d["per_target"][target].get(key)
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def print_multi_target(a) -> None:
    """MTR: one row per ASSAY, and a BH-adjusted p per assay.

    A single fold-mean over 16 columns would be a number that describes no assay:
    the 8 transfection columns are log-scale signals and the 8 safety columns are
    cytokine/clinical readings on their own scales, so averaging them is arithmetic
    without a referent. The per-assay table is therefore the table.

    Multiple comparisons are handled explicitly. Each assay contributes one paired
    Wilcoxon per arm pair, so 16 tests at alpha=0.05 would be expected to produce
    ~0.8 false positives per arm pair even if every model were useless -- and the
    MTR arm's whole output is "which assays does this model predict", so an
    unadjusted table would be systematically over-optimistic. Both raw and
    BH-adjusted p are printed; the raw one is kept because the adjustment is a
    claim about the family of 16, and a reader comparing against a single-assay
    result from another dataset needs the unadjusted number.
    """
    import ds_registry  # noqa: PLC0415

    _df, target_cols = ds_registry.load(a.dataset, verbose=False)
    if a.target:
        target_cols = [a.target]
    arms = resolve_arms(a.models)
    n_planned = len(a.seeds) * a.kfold
    print(f"=== {a.dataset} / {a.split} / {len(target_cols)} targets / "
          f"{len(a.seeds)} seeds x {a.kfold} folds = {n_planned} folds per assay ===")
    print(f"    metric = fold-mean {a.metric.upper()}, per assay. Multiple comparison: "
          f"BH-FDR across the {len(target_cols)} assays, per arm pair.")
    print(f"    Screening metrics are POOLED within each seed then averaged over "
          f"seeds (MTR folds are 11-32 molecules; 坑 #11).")
    print(f"    Screening keys: {' '.join(MTR_SCREENING)}\n")

    ref = a.models[0]
    per_assay: dict[str, dict] = {}
    print(f"{'assay':<34} {'es_val':>8} {'fold-mean':>10} {'std':>8} "
          f"{'Δ vs ' + ref:>12} {'p_raw':>10} {'p_BH':>10} {'W/L/T':>10}")
    print("-" * 116)
    for t in target_cols:
        vecs = {m: pooled_vector(a.dataset, m, a.split, a.seeds, a.kfold, t,
                                 a.metric, model_meta=arms) for m in a.models}
        low = t in ds_registry.MTR_LOW_VARIANCE
        entry: dict = {
            "low_variance": low,
            "note": ("near-constant column (std < "
                     f"{ds_registry.MTR_LOW_VARIANCE_STD_MAX}); excluded from any "
                     "headline count of 'assays predicted'") if low else None,
            "arms": {m: {"fold_mean": float(np.mean(v)),
                         "fold_std": float(np.std(v, ddof=1)),
                         "per_seed_means": [float(np.mean(v[i * a.kfold:(i + 1) * a.kfold]))
                                            for i in range(len(a.seeds))]}
                     for m, v in vecs.items()},
            "screening_pooled": {
                m: {k: pooled_screening(a.dataset, m, a.split, a.seeds, a.kfold, t, k)
                    for k in MTR_SCREENING} for m in a.models},
            "paired": [],
        }
        raw_ps = []
        rows_tmp = []
        for m in a.models[1:]:
            r = paired_compare(vecs[ref], vecs[m], name=m)
            r["ref"] = ref
            r["metric"] = a.metric
            r["sign_normalised"] = 1.0
            r["target"] = t
            raw_ps.append(r["p"])
            rows_tmp.append(r)
        adj = benjamini_hochberg(raw_ps)
        for r, p_adj in zip(rows_tmp, adj):
            r["p_bh"] = p_adj
            r["n_tests_in_family"] = len(target_cols)
            entry["paired"].append(r)
            flag = "  [near-constant]" if low else ""
            print(f"{t:<34} {arms[ref][0]:>8} {np.mean(vecs[ref]):>10.4f} "
                  f"{np.std(vecs[ref], ddof=1):>8.4f} "
                  f"{r['delta']:>+12.4f} {r['p']:>10.3g} {p_adj:>10.3g} "
                  f"{r['wins']:>4}/{r['losses']:<4}/{r['ties']:<1}{flag}")
        per_assay[t] = entry

    # How many assays each arm pair separates on, before and after the adjustment.
    # This is the summary the per-assay p-values exist to support, and it is stated
    # with the near-constant columns excluded -- a "predicted" near-constant assay
    # is a high R2 on a column that barely moves.
    scorable = [t for t in target_cols if t not in ds_registry.MTR_LOW_VARIANCE]
    # The Wilcoxon signed-rank test's smallest attainable two-sided p at n pairs is
    # 2/2^n. At n=5 that is 0.0625 -- so on a 5-fold table NOTHING can reach 0.05,
    # and "0 assays separate" would read as a result about the models when it is a
    # property of the test. Say so rather than letting the column speak.
    p_floor = 2.0 / (2 ** n_planned)
    print(f"\n=== how many of the {len(scorable)} scorable assays separate "
          f"(near-constant columns excluded) ===")
    if p_floor > 0.05:
        print(f"  !! n={n_planned} pairs: the smallest attainable Wilcoxon p is "
              f"{p_floor:.4g} > 0.05, so NO assay can reach significance at this "
              f"fold count. These p-values cannot support a negative claim; run the "
              f"full 25-fold grid before reading them.")
    for m in a.models[1:]:
        raw_sig = sum(1 for t in scorable
                      if per_assay[t]["paired"][a.models.index(m) - 1]["p"] < 0.05)
        bh_sig = sum(1 for t in scorable
                     if (per_assay[t]["paired"][a.models.index(m) - 1]["p_bh"] or 1) < 0.05)
        wins = sum(1 for t in scorable
                   if per_assay[t]["paired"][a.models.index(m) - 1]["delta"] > 0)
        print(f"  {ref} vs {m:<16} raw p<0.05: {raw_sig:>2}/{len(scorable)}   "
              f"BH p<0.05: {bh_sig:>2}/{len(scorable)}   "
              f"direction: {wins}/{len(scorable)} favour {ref}")
    print(f"\n  Note: {len(target_cols) - len(scorable)} near-constant column(s) "
          f"({', '.join(sorted(ds_registry.MTR_LOW_VARIANCE))}) are kept in the matrix "
          f"and in the table, but excluded from these counts.")

    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps({
            "dataset": a.dataset, "split": a.split, "metric": a.metric,
            "seeds": a.seeds, "kfold": a.kfold, "n_folds": n_planned,
            "targets": target_cols,
            "low_variance_targets": sorted(ds_registry.MTR_LOW_VARIANCE),
            "screening_keys": MTR_SCREENING,
            "ref": ref, "arms": {m: list(v) for m, v in arms.items()},
            "per_assay": per_assay,
        }, indent=1), encoding="utf-8")
        print(f"\nwrote {a.json_out}")


def main() -> None:
    global RESULTS_DIR  # noqa: PLW0603 -- rebound by --results-dir; see below
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--models", nargs="+", required=True,
                    help="first model is the reference every other is paired against")
    ap.add_argument("--seeds", nargs="+", type=int, default=split_files.PARTITION_SEEDS)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--target", default=None)
    ap.add_argument("--metric", default="r2")
    ap.add_argument("--screening", nargs="*",
                    default=["ef@10pct_pooled", "ndcg@10pct_pooled"],
                    help="pooled screening keys to tabulate (see metrics.py)")
    ap.add_argument("--paired-metrics", nargs="*",
                    default=["r2", "rmse", "spearman"],
                    help="per-fold metrics to run the paired Wilcoxon on. The "
                         "headline R2 alone is not enough: pKa's R2 advantage over "
                         "ECFP4+RF disappears entirely on Spearman, and a table "
                         "that only shows R2 invites reading it as 'better at "
                         "ranking' when the model is better at CALIBRATION.")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR,
                    help="where the collected cells live (default: the real "
                         "downstream_results/). Exposed so a scratch table can be "
                         "built over smoke cells without touching the store.")
    a = ap.parse_args()
    RESULTS_DIR = a.results_dir

    import ds_registry  # noqa: PLC0415
    _df, target_cols = ds_registry.load(a.dataset, verbose=False)

    # Multi-target datasets get the per-assay table. It is not a variant of the
    # single-target one: `target = target_cols[0]` would silently report assay 0 as
    # "the" result for a 16-assay dataset, and a fold-mean over 16 assays would be a
    # number describing none of them. `--target` still selects one column for a
    # focused look.
    if a.target is None and len(target_cols) > 1:
        print_multi_target(a)
        return

    target = a.target or target_cols[0]
    arms = resolve_arms(a.models)

    n_planned = len(a.seeds) * a.kfold
    print(f"=== {a.dataset} / {a.split} / target={target} / "
          f"{len(a.seeds)} seeds x {a.kfold} folds = {n_planned} folds ===")
    print(f"    metric = fold-mean {a.metric.upper()}; "
          f"effect floor: 15 folds needed for <0.04, {n_planned} available"
          + ("" if n_planned >= 15 else "  *** BELOW THE 15-FOLD FLOOR ***") + "\n")

    vecs = {m: pooled_vector(a.dataset, m, a.split, a.seeds, a.kfold, target, a.metric,
                             model_meta=arms)
            for m in a.models}

    n = len(next(iter(vecs.values())))
    hdr = f"{'model':<14} {'fold-mean':>10} {'std':>8} {'per-seed means':<34}"
    for k in a.screening:
        hdr += f" {k:>18}"
    print(hdr)
    print("-" * len(hdr))
    for m in a.models:
        v = vecs[m]
        per_seed = [round(float(np.mean(v[i * a.kfold:(i + 1) * a.kfold])), 4)
                    for i in range(len(a.seeds))]
        row = (f"{m:<14} {np.mean(v):>10.4f} {np.std(v, ddof=1):>8.4f} "
               f"{str(per_seed):<34}")
        for k in a.screening:
            row += f" {pooled_screening(a.dataset, m, a.split, a.seeds, a.kfold, target, k):>18.4f}"
        print(row)

    ref = a.models[0]
    print(f"\n=== paired Wilcoxon vs {ref} (same {n} folds, same order) ===")
    rows = []
    for m in a.models[1:]:
        r = paired_compare(vecs[ref], vecs[m], name=m)
        r["ref"] = ref
        r["metric"] = a.metric
        r["sign_normalised"] = 1.0
        rows.append(r)
        print(f"  {ref} vs {m:<14} Δ={r['delta']:+.4f}  "
              f"W={r['w']:.1f}  p={r['p']:.4g}  r_rb={r['effect_r']:+.3f}  "
              f"W/L/T={r['wins']}/{r['losses']}/{r['ties']}  n={r['n_pairs']}")
        if np.isfinite(r["p"]) and r["p"] >= 0.05:
            print(f"      -> not significant at 0.05; "
                  f"|Δ|={abs(r['delta']):.4f} vs run-to-run noise 0.0335")

    # The same pairing on the other metrics. RMSE is signed so that a POSITIVE
    # delta always means "the reference model is better", matching R2/Spearman;
    # without that flip a reader comparing deltas down the column would read a
    # lower RMSE as a loss.
    extra = [k for k in a.paired_metrics if k != a.metric]
    if extra:
        vecs_extra = {(m, k): pooled_vector(a.dataset, m, a.split, a.seeds, a.kfold,
                                           target, k, model_meta=arms)
                      for m in a.models for k in extra}
        print(f"\n=== same pairing on {'/'.join(extra)} (sign-normalised: "
              f"+ always favours {ref}) ===")
        for k in extra:
            sign = -1.0 if k in ("rmse", "mae", "mse") else 1.0
            for m in a.models[1:]:
                r = paired_compare(sign * vecs_extra[(ref, k)],
                                   sign * vecs_extra[(m, k)], name=m)
                r["ref"] = ref
                r["metric"] = k
                r["sign_normalised"] = sign
                rows.append(r)
                # r["delta"] is already on the normalised scale (paired_compare
                # averages the difference of its two inputs), so it is printed
                # as-is: applying `sign` again would double-flip it and show a
                # win as a loss.
                print(f"  [{k:<8}] vs {m:<14} Δ={r['delta']:+.4f}  "
                      f"p={r['p']:.4g}  r_rb={r['effect_r']:+.3f}  "
                      f"W/L/T={r['wins']}/{r['losses']}/{r['ties']}")

    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps({
            "dataset": a.dataset, "split": a.split, "target": target,
            "metric": a.metric, "seeds": a.seeds, "kfold": a.kfold,
            "n_folds": n,
            "fold_mean": {m: float(np.mean(v)) for m, v in vecs.items()},
            "fold_std": {m: float(np.std(v, ddof=1)) for m, v in vecs.items()},
            "per_seed_means": {
                m: [float(np.mean(v[i * a.kfold:(i + 1) * a.kfold]))
                    for i in range(len(a.seeds))] for m, v in vecs.items()},
            "screening_pooled": {
                m: {k: pooled_screening(a.dataset, m, a.split, a.seeds, a.kfold,
                                        target, k) for k in a.screening}
                for m in a.models},
            "paired": rows,
            "per_fold": {m: [round(float(x), 4) for x in v] for m, v in vecs.items()},
        }, indent=1), encoding="utf-8")
        print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
