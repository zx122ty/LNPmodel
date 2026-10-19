#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared metrics for the three downstream evaluations (TE / pKa / MTR).

Every number that goes into a report is computed here, so that the three tasks
cannot drift apart in how they score. Two conventions are FROZEN by the
2026-09-16 evaluation plan and must not be changed without re-reporting every
table that used them:

1. **Ranking metrics use an ABSOLUTE k** (EF@25 / EF@50 / NDCG@50), not a
   percentage. On the project's fold sizes a percentage is meaningless:
   Sheet2 folds are 384 -> EF@1% is 3.84 molecules, Sheet3 folds are 148-175
   -> 1.5, and MTR folds go down to 11 -> 0.11. An expectation of zero is not
   a small number, it is a broken metric. See plan doc §3.1.

2. **fold-mean is the headline; pooled is a labelled diagnostic.** This mirrors
   ``finetune_moltrain.py`` so new numbers line up with every historical one.
   Never mix the two columns in one table.

``lumi_metrics`` is a verbatim port of ``diagnostics/top25_benchmark.py`` so the
LUMI-protocol column stays comparable with ``repr_benchmark_*.json``.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.metrics import mean_squared_error, r2_score

# "High efficiency" is defined exactly as LUMI defines it: the top 25% of the
# TRUE values within the set being scored. Fixed here rather than per-call so
# EF and the LUMI agreement metric cannot disagree about who is "top".
TOP_FRACTION = 0.75


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R2 / RMSE / Pearson / Spearman for one scored set."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if len(y_true) < 3 or np.std(y_true) == 0:
        # A constant truth column has no explainable variance: R2 is undefined,
        # not zero. Returning NaN keeps it out of the mean instead of dragging
        # a fold to 0.0 and quietly deflating the headline.
        return {"r2": np.nan, "rmse": np.nan, "pearson": np.nan, "spearman": np.nan}
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }


# --------------------------------------------------------------------------
# Ranking (virtual-screening value) -- absolute k, see module docstring
# --------------------------------------------------------------------------
def enrichment_factor(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """EF@k = (hits in top-k predicted / k) / (hits in whole set / n).

    A "hit" is a molecule in the top 25% of TRUE values (``TOP_FRACTION``).
    Returns NaN and says so, rather than clamping, when k exceeds the set --
    a silently clamped EF is exactly the kind of plausible-looking number this
    project has been burned by.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    if n == 0 or k <= 0 or k > n:
        return np.nan
    thr = np.quantile(y_true, TOP_FRACTION)
    hits_all = y_true >= thr
    if hits_all.sum() == 0:
        return np.nan
    top_k = np.argsort(-y_pred, kind="stable")[:k]
    hits_top = hits_all[top_k].sum()
    base_rate = hits_all.mean()
    return float((hits_top / k) / base_rate)


def ndcg_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """NDCG@k with the true value as graded relevance.

    Relevance is min-max normalised to [0, 1] before the 2^rel-1 gain so the
    exponential cannot blow up on log-scale targets (Sheet2 spans -5.5..16.2).
    Normalisation is a monotone transform, so it does not change the ideal
    ordering -- only the gain scale.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    if n == 0 or k <= 0:
        return np.nan
    k = min(k, n)  # NDCG is well defined for k=n; only EF needs k<=n
    span = y_true.max() - y_true.min()
    rel = (y_true - y_true.min()) / span if span > 0 else np.zeros(n)

    def dcg(order: np.ndarray) -> float:
        gains = (2.0 ** rel[order] - 1.0) / np.log2(np.arange(2, len(order) + 2))
        return float(gains.sum())

    pred_order = np.argsort(-y_pred, kind="stable")[:k]
    ideal_order = np.argsort(-rel, kind="stable")[:k]
    ideal = dcg(ideal_order)
    return float(dcg(pred_order) / ideal) if ideal > 0 else np.nan


def lumi_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Verbath port of ``diagnostics/top25_benchmark.py``'s ``lumi_metrics``.

    Kept byte-compatible with the existing ``repr_benchmark_*.json`` so a new
    model can be dropped into that table without a re-run. ``top25_*`` is
    top-25 PERCENT (480 of 1920), not top-25 molecules -- the 19x confusion is
    called out in the handoff doc and is the single easiest way to misquote
    this project's numbers.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    thr = np.quantile(y_true, TOP_FRACTION)
    m = y_true >= thr
    top_t, top_p = y_true[m], y_pred[m]
    pred_thr = np.quantile(y_pred, TOP_FRACTION)
    agree = len(set(np.nonzero(m)[0]) & set(np.nonzero(y_pred >= pred_thr)[0])) / max(m.sum(), 1)
    nz = y_true > 0
    return {
        "pearson": float(pearsonr(y_true, y_pred)[0]),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
        "top25_pearson": float(pearsonr(top_t, top_p)[0]),
        "top25_spearman": float(spearmanr(top_t, top_p).statistic),
        "top25_agreement": float(agree),
        "non_zero_pearson": float(pearsonr(y_true[nz], y_pred[nz])[0]) if nz.sum() > 2 else np.nan,
    }


# --------------------------------------------------------------------------
# Per-fold scoring
# --------------------------------------------------------------------------
def screening_metrics_pooled(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Virtual-screening metrics on ONE ranking of the whole dataset, at k = 1%/5%/10% of n.

    These are the numbers that answer "if I synthesise the top-k, how enriched am
    I" -- and they are **pooled by necessity, not by preference**. The fold-mean
    analogue is not available here: EF@25 needs a fold of at least 25 molecules,
    and MTR's folds are 11-32, so EF@25 is undefined on most of its folds while
    being well-defined on Sheet2's 384. Reporting the fold-mean would silently
    compare "EF@25" across datasets where it means different things.

    Pooling has its own caveat and it must travel with the number: a pooled OOF
    ranking is built from predictions made by 5 different fold-models, so it is
    not the ranking a single deployed model would produce. Use it as the
    screening headline, and the fold-mean metrics above as the accuracy headline.

    On Sheet2 (n=1920) k=1% is 19.2 -> 19 molecules, versus 3.84 if taken
    per-fold. That difference is exactly why the plan fixes an absolute k.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    n = len(y_true)
    out: dict[str, float] = {"n_pooled": n}
    for pct in (1, 5, 10):
        k = max(1, int(round(n * pct / 100.0)))
        out[f"ef@{pct}pct_pooled"] = enrichment_factor(y_true, y_pred, k)
        out[f"ndcg@{pct}pct_pooled"] = ndcg_at_k(y_true, y_pred, k)
        out[f"k_for_{pct}pct"] = k
    # How many labels sit at or below zero (assay detection limit). Sheet2 has
    # 16%; a molecule predicted into that band is not "confidently inactive",
    # its true value may simply be censored. Carried alongside the screening
    # numbers so a top-k list can be read with that in mind.
    out["frac_censored"] = float((y_true <= 0).mean())
    return out


def ef_ceiling(n_set: int, k: int) -> float:
    """Best EF@k any predictor can reach on a set of ``n_set`` molecules.

    EF is capped, not unbounded: a perfect ranker puts all ``top25%`` molecules
    first, so the ceiling is ``(min(k, 0.25n)/k)/0.25``. Two consequences worth
    stating in any table that shows EF:

    * the ceiling DEPENDS ON k -- EF@5 ceilings at 4.00 while EF@50 ceilings at
      2.30-2.96 on this project's fold sizes, so EF@50 will always look worse
      for reasons that have nothing to do with the model;
    * if k > n_set the metric is undefined, which is why ``enrichment_factor``
      returns NaN rather than clamping.

    Print this next to an EF column before concluding one model beats another.
    """
    if k > n_set or k <= 0 or n_set <= 0:
        return float("nan")
    # The number of hits in the whole set is (1 - TOP_FRACTION) * n_set, NOT
    # TOP_FRACTION * n_set: TOP_FRACTION is the QUANTILE that defines the
    # threshold, so the molecules at or above it are the upper tail, 25% of the
    # set. Using the quantile itself would inflate the ceiling 3x.
    #
    # The discrete ``enrichment_factor`` uses np.quantile, which lands the hit
    # count on an integer, so this continuous formula overshoots slightly on
    # small sets: at n=115 it returns 4.00 where a perfect predictor measures
    # 3.97 (0.8% high). Treat it as a guide to what is reachable, not as a
    # value to divide by.
    n_hits = (1 - TOP_FRACTION) * n_set
    return (min(k, n_hits) / k) / (n_hits / n_set)


def default_metric_set(k_ef: Sequence[int] = (25, 50), k_ndcg: Sequence[int] = (50,)) -> dict:
    """The frozen metric set. k values are molecule counts, not percents."""
    out: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {}
    for k in k_ef:
        out[f"ef@{k}"] = (lambda kk: lambda t, p: enrichment_factor(t, p, kk))(k)
    for k in k_ndcg:
        out[f"ndcg@{k}"] = (lambda kk: lambda t, p: ndcg_at_k(t, p, kk))(k)
    return out


def score_folds(y_true: np.ndarray, y_pred: np.ndarray, folds: Iterable,
                metric_set: dict | None = None) -> dict:
    """Score every fold; return per-fold values plus the mean over folds.

    ``folds`` is the project's ``[(train_idx, test_idx), ...]`` list. Only the
    test index is used -- train indices are ignored by design, because anything
    that reads them here would be a leak.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    metric_set = metric_set if metric_set is not None else default_metric_set()

    per_fold: dict[str, list[float]] = {k: [] for k in metric_set}
    per_fold.update({"r2": [], "rmse": [], "pearson": [], "spearman": [],
                     "top25_pearson": [], "top25_agreement": [], "n": []})

    for _tr, te in folds:
        te = np.asarray(te, int)
        per_fold["n"].append(int(len(te)))
        for name, fn in metric_set.items():
            per_fold[name].append(fn(y_true[te], y_pred[te]))
        reg = regression_metrics(y_true[te], y_pred[te])
        for name in ("r2", "rmse", "pearson", "spearman"):
            per_fold[name].append(reg[name])
        lumi = lumi_metrics(y_true[te], y_pred[te])
        per_fold["top25_pearson"].append(lumi["top25_pearson"])
        per_fold["top25_agreement"].append(lumi["top25_agreement"])

    pooled = {
        "r2_pooled": float(r2_score(y_true, y_pred)) if np.std(y_true) > 0 else np.nan,
        "pearson_pooled": float(pearsonr(y_true, y_pred)[0]),
        "spearman_pooled": float(spearmanr(y_true, y_pred).statistic),
    }
    for name, fn in metric_set.items():
        pooled[f"{name}_pooled"] = fn(y_true, y_pred)
    pooled.update(screening_metrics_pooled(y_true, y_pred))

    summary = {}
    for name, vals in per_fold.items():
        if name == "n":
            summary["n_test_per_fold"] = vals
            continue
        arr = np.asarray(vals, float)
        # nanmean with a count, so a fold that legitimately could not be scored
        # (e.g. k > fold size) does not silently become a zero contribution.
        n_ok = int(np.isfinite(arr).sum())
        summary[name] = float(np.nanmean(arr)) if n_ok else np.nan
        summary[f"{name}_std"] = float(np.nanstd(arr)) if n_ok else np.nan
        summary[f"{name}_n_scored"] = n_ok
        summary[f"{name}_per_fold"] = [None if not np.isfinite(v) else round(float(v), 4)
                                       for v in arr]
    summary.update(pooled)
    return summary


# --------------------------------------------------------------------------
# Statistics for "is this difference real?"
# --------------------------------------------------------------------------
def rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Matched-pairs rank-biserial correlation (Kerby 2014).

    r = (W+ - W-) / (W+ + W-) over the signed ranks of the differences. This is
    the effect size that pairs with the Wilcoxon signed-rank test the plan
    specifies; it is signed, so the direction of the difference is readable
    from the sign. |r| < 0.1 negligible, 0.1-0.3 small, 0.3-0.5 medium, >= 0.5 large.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    d = (x - y)[ok]
    d = d[d != 0]  # zero differences carry no rank information
    if len(d) < 2:
        return np.nan
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    total = w_pos + w_neg
    return float((w_pos - w_neg) / total) if total > 0 else np.nan


def paired_compare(ours: np.ndarray, baseline: np.ndarray,
                   name: str = "baseline") -> dict:
    """Wilcoxon signed-rank + rank-biserial, paired on the SAME folds.

    Pairing is the whole point: unpaired comparison of two 25-fold means is
    dominated by fold difficulty, which the project measured at 0.0335
    run-to-run. Both inputs must therefore come from an identical fold list,
    in identical fold order.
    """
    ours = np.asarray(ours, float)
    base = np.asarray(baseline, float)
    ok = np.isfinite(ours) & np.isfinite(base)
    a, b = ours[ok], base[ok]
    out = {
        "baseline": name, "n_pairs": int(len(a)),
        "ours_mean": float(np.mean(a)) if len(a) else np.nan,
        "baseline_mean": float(np.mean(b)) if len(b) else np.nan,
        "delta": float(np.mean(a - b)) if len(a) else np.nan,
        "wins": int((a > b).sum()), "losses": int((a < b).sum()),
        "ties": int((a == b).sum()),
    }
    if len(a) < 5 or np.allclose(a, b):
        # scipy's wilcoxon needs a handful of pairs, and an all-zero difference
        # vector has no rank information -- report that rather than a fake p=1.
        out.update({"w": np.nan, "p": np.nan, "effect_r": np.nan,
                    "note": "too few pairs or all differences zero"})
        return out
    try:
        stat = wilcoxon(a, b)
        out["w"] = float(stat.statistic)
        out["p"] = float(stat.pvalue)
    except ValueError as exc:
        out.update({"w": np.nan, "p": np.nan, "note": str(exc)})
    out["effect_r"] = rank_biserial(a, b)
    return out


def verdict(p: float) -> str:
    """Two-sided p at the project's 0.05 threshold, spelled out."""
    if not np.isfinite(p):
        return "n/a"
    if p < 0.001:
        return "p<0.001"
    return "significant" if p < 0.05 else "not significant"


def benjamini_hochberg(pvals: Sequence[float]) -> list[float]:
    """BH-FDR adjusted p-values, in the input's order (NaNs preserved).

    MTR's report runs 16 paired tests per (arm pair, metric) -- one per assay. At
    alpha=0.05 roughly one of sixteen is expected to come out "significant" from
    noise alone, and the whole point of the MTR arm is to say which assays a model
    predicts, so an unadjusted table is not a conservative reading of the data, it
    is a misleading one.

    Ties are handled by giving equal p-values equal adjusted values (the standard
    step-up uses the largest rank in the tie group, not an arbitrary one), because
    two assays that happen to land on the same p are not ordered by anything.

    Returns ``nan`` where the input was not finite, rather than dropping the row:
    the caller's row order must survive, or the adjusted value would be attached
    to a different assay.
    """
    p = np.asarray(pvals, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out.tolist()
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx], kind="stable")]
    m = len(order)
    # Step-up: multiply by m/rank, then enforce monotonicity from the largest p down.
    ranked = p[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.minimum(ranked, 1.0)
    # Equal p-values share the LARGEST adjusted value among them (they are
    # indistinguishable, so no arbitrary ordering may decide the outcome).
    for value in np.unique(p[order]):
        grp = order[p[order] == value]
        out[grp] = ranked[np.flatnonzero(p[order] == value)].max()
    return out.tolist()
