#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L3 of P1: if we could choose which lipids to assay next, which rule should we use?

L1 showed that sigma ranks molecules' ERRORS well (Spearman +0.21 / +0.16 against
two trivial controls that sit at zero) but that ranking by sigma is a bad way to
find the top-25 transfectors -- greedy on mu beat every mu+beta*sigma rule. L1 only
ever re-RANKED a fixed pool though. This script runs the experiment the acquisition
function is actually for: an iterative loop where each round's labels change the
next round's model.

Design
------
For each of the 15 cached folds, the fold's TEST set is locked away and never
labelled. The fold's TRAIN set (1536 molecules) is the universe the loop works in:
  round 0 : label a random 384 of them (25%)
  rounds 1..8 : label a further 96 (5% of the library) chosen by the strategy
The learner is retrained from scratch after every acquisition, and scored on the
locked test fold. 15 folds = 15 independent repetitions of the whole loop, which is
the project's hard floor and makes every comparison paired across folds.

The learner is a CPU arm ensemble, deliberately
------------------------------------------------
The deployed predictor is a 3-arm blend whose neural leg takes ~25 min per fold to
retrain; a 9-round loop would need 9x that per fold and cannot be done in the time
this layer is worth. Instead the learner is a 5-arm CPU ensemble (two fingerprint
families x three model families) rebuilt each round from cached features. What this
measures is the ACQUISITION FUNCTION's behaviour: every strategy faces the same
learner, the same folds and the same budget, so a ranking of strategies transfers
even though the absolute R2 is not the deployed model's. It is reported as such --
no number here is comparable to the 0.7143/0.8109 headline.

Metrics
-------
* R2 on the locked test fold (fold-mean over the 15 repetitions).
* recall@25 of the test fold's true top-25 -- the decision metric, matching L1.
* label efficiency: how many labels each strategy needs to reach the score that
  RANDOM reaches with its full budget. This is the number the project's stated gap
  ("which lipids do we make next") is really asking for.

Strategies
----------
  random      uniform without replacement -- the baseline every AL claim needs
  greedy      mu (exploit only) -- the L1 winner
  sigma       sigma (explore only)
  ucb_b*      mu + beta*sigma
  diverse     max Tanimoto distance to the labelled set (a strong, classical
              molecular-AL baseline; without it a win for sigma could just be
              "any non-greedy rule helps")
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import ttest_rel
from sklearn.metrics import r2_score

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
CACHE = ROOT / "5_outputs/active_learning_sim.json"

DS = "Sheet2"
PROTOCOLS = {"scaffold": (42, 1337, 2024), "butina": (42, 1337, 2024)}
INIT_FRAC, STEP_FRAC, N_ROUNDS = 0.25, 0.05, 8   # of the full library
BETAS = (0.5, 1.0)
SEED0 = 20260915                                   # initial-label draw per fold
N_JOBS = int(os.environ.get("AL_NJOBS", "6"))
TOP_K = 25


# ------------------------------------------------------------------- features
def build_features():
    """The same fixed-config arms arm_zoo.py screened, on cached fingerprints."""
    import sys
    sys.path.insert(0, str(ROOT / "4_scripts"))
    import arm_zoo

    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", "RLU (log2)"])
    smis = d["SMILES"].astype(str).str.strip().tolist()
    y = pd.to_numeric(d["RLU (log2)"]).values.astype(float)
    F = {
        "desc": arm_zoo._cache("rdkit_desc", lambda: arm_zoo.rdkit_descriptors(smis)),
        "morgan_r2_2048": arm_zoo._cache("morgan_r2_2048", lambda: arm_zoo.morgan(smis, 2, 2048)),
        "atompair": arm_zoo._cache("atompair", lambda: arm_zoo.atompair(smis)),
        "maccs": arm_zoo._cache("maccs", lambda: arm_zoo.maccs(smis)),
    }
    h = np.hstack
    # Four arms spanning both diversity axes; each keeps arm_zoo's fixed config so
    # nothing here is tuned on the folds it will be scored on.
    #
    # `svr_tanimoto` is deliberately ABSENT. It is a legitimate arm at n=1536 (0.5966
    # in arm_zoo's scaffold screen) but collapses at the small n that an AL loop
    # spends its budget in: measured on 480 labels it scores R2 = -56 against 0.44
    # to 0.57 for the arms below, because the precomputed-kernel SVR's epsilon-tube
    # degenerates to a near-constant predictor and then dominates the ensemble mean
    # (mean R2 = -1.56 at round 0). Including it would have made every learning
    # curve a plot of that one arm recovering, not of the acquisition function --
    # exactly the "silent number" failure this project keeps hitting. This is a
    # statement about the AL regime, not about the deployed ensemble, whose third
    # arm is xgb_atompair anyway.
    arms = {
        "svr_rbf_desc_morgan": ("svr", h([F["desc"], F["morgan_r2_2048"]])),
        "rf_desc_morgan": ("rf", h([F["desc"], F["morgan_r2_2048"]])),
        "xgb_atompair": ("xgb", F["atompair"]),
        "xgb_maccs": ("xgb", F["maccs"]),
    }
    return arms, F["morgan_r2_2048"], y, arm_zoo


def fit_arm(kind, X, y, tr, arm_zoo_mod):
    """Fit one arm on `tr` and return a predictor over all rows."""
    if kind == "svr_tanimoto":
        A = X[np.ix_(tr, tr)]
        K = arm_zoo_mod._tanimoto(A, A)
        from sklearn.svm import SVR
        m = SVR(kernel="precomputed", C=10.0, epsilon=0.1).fit(K, y[tr])
        return lambda rows: m.predict(arm_zoo_mod._tanimoto(X[rows], X[tr]))
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    if kind == "svr":
        est = Pipeline([("sc", StandardScaler()),
                        ("m", SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1,
                                  cache_size=1024))])
    elif kind == "rf":
        est = RandomForestRegressor(n_estimators=400, max_features=0.3, n_jobs=1,
                                    random_state=0)
    elif kind == "xgb":
        import xgboost as xgb
        est = xgb.XGBRegressor(tree_method="hist", n_jobs=1, random_state=0,
                               n_estimators=600, max_depth=6, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.4, reg_lambda=1.0)
    else:
        raise ValueError(kind)
    est.fit(X[tr], y[tr])
    return lambda rows: est.predict(X[rows])


def tanimoto_to_labelled(fp, labelled, pool):
    """Max Tanimoto similarity of each pool molecule to anything already labelled."""
    A = np.asarray(fp[pool], np.float32)
    B = np.asarray(fp[labelled], np.float32)
    inter = A @ B.T
    a = A.sum(1, keepdims=True)
    b = B.sum(1, keepdims=True)
    return (inter / np.maximum(a + b.T - inter, 1e-9)).max(1)


# ----------------------------------------------------------------------- loop
def acquire(strategy, mu, sigma, div, pool, n_take, rng):
    """Pick n_take pool positions. Returns positions INTO `pool`."""
    k = min(n_take, len(pool))
    if strategy == "random":
        return rng.choice(len(pool), k, replace=False)
    if strategy == "greedy":
        score = mu
    elif strategy == "sigma":
        score = sigma
    elif strategy.startswith("ucb_b"):
        beta = float(strategy.split("_b")[1])
        score = mu + beta * sigma
    elif strategy == "diverse":
        score = div
    else:
        raise ValueError(strategy)
    return np.argsort(-np.asarray(score), kind="stable")[:k]


def run_fold(fold_id, protocol, tr_idx, te_idx, y, arms, fp, arm_zoo_mod, strategies):
    fold_seed = SEED0 + fold_id
    rng = np.random.default_rng(fold_seed)
    n_init = int(round(INIT_FRAC * len(y)))
    n_step = int(round(STEP_FRAC * len(y)))
    init = rng.choice(tr_idx, min(n_init, len(tr_idx)), replace=False)

    truth_top = set(np.argsort(-y[te_idx], kind="stable")[:TOP_K].tolist())

    out = {}
    for strat in strategies:
        r = np.random.default_rng(fold_seed)      # same initial draw for every strategy
        labelled = list(r.choice(tr_idx, min(n_init, len(tr_idx)), replace=False))
        curve = []
        for rnd in range(N_ROUNDS + 1):
            lab = np.asarray(labelled, int)
            preds = []
            for name, (kind, X) in arms.items():
                f = fit_arm(kind, X, y, lab, arm_zoo_mod)
                preds.append(f(np.arange(len(y))))
            P = np.stack(preds, 1)
            mu_all = P.mean(1)
            sd_all = P.std(1, ddof=1)
            test_r2 = float(r2_score(y[te_idx], mu_all[te_idx]))
            order = np.argsort(-mu_all[te_idx], kind="stable")[:TOP_K]
            rec = len(set(order.tolist()) & truth_top) / TOP_K
            curve.append({"round": rnd, "n_labeled": int(len(lab)),
                          "r2": test_r2, "recall25": float(rec)})
            if rnd == N_ROUNDS:
                break
            pool = np.setdiff1d(tr_idx, lab, assume_unique=False)
            div = tanimoto_to_labelled(fp, lab, pool)
            take = acquire(strat, mu_all[pool], sd_all[pool], div, pool, n_step, r)
            labelled.extend(pool[take].tolist())
        out[strat] = curve
    return fold_id, out


def arm_health(y, tr_idx, te_idx, arms, arm_zoo_mod, y_full) -> dict:
    """Round-0 per-arm test R2. A single degenerate arm silently destroys any
    mean-ensemble learning curve, so this is checked before the loop is trusted."""
    lab = np.random.default_rng(0).choice(tr_idx, int(round(INIT_FRAC * len(y_full))),
                                          replace=False)
    out = {}
    for name, (kind, X) in arms.items():
        p = fit_arm(kind, X, y_full, lab, arm_zoo_mod)(np.arange(len(y_full)))
        out[name] = float(r2_score(y_full[te_idx], p[te_idx]))
    return out


def aggregate(curves, strategies, n_rounds=N_ROUNDS):
    """All the reported numbers, derived purely from the per-fold curves.

    Split out from main() so `--reanalyze` can recompute the report from the
    cached curves: the fit loop costs ~20 min per protocol, and a metric fix
    should not require re-running it.
    """
    rounds = list(range(n_rounds + 1))
    agg, tests = {}, {}
    for s in strategies:
        r2 = np.array([[curves[s][f][r]["r2"] for f in curves[s]] for r in rounds])
        rec = np.array([[curves[s][f][r]["recall25"] for f in curves[s]]
                        for r in rounds])
        agg[s] = {"r2_mean": r2.mean(1).tolist(), "r2_std": r2.std(1, ddof=1).tolist(),
                  "recall_mean": rec.mean(1).tolist(),
                  "recall_std": rec.std(1, ddof=1).tolist(),
                  "n_labeled": [curves[s][list(curves[s])[0]][r]["n_labeled"]
                                for r in rounds]}
    base = np.array([[curves["random"][f][r]["r2"] for f in curves["random"]]
                     for r in rounds])
    base_rec = np.array([[curves["random"][f][r]["recall25"]
                          for f in curves["random"]] for r in rounds])
    n_folds = len(curves["random"])
    for s in strategies:
        if s == "random":
            continue
        a = np.array([[curves[s][f][r]["r2"] for f in curves[s]] for r in rounds])
        ar = np.array([[curves[s][f][r]["recall25"] for f in curves[s]]
                       for r in rounds])
        tests[s] = {}
        for r in rounds:
            _, p = ttest_rel(a[r], base[r])
            _, pr = ttest_rel(ar[r], base_rec[r])
            tests[s][r] = {"delta_r2": float(a[r].mean() - base[r].mean()),
                           "p_r2": float(p), "wins_r2": int((a[r] > base[r]).sum()),
                           "delta_recall": float(ar[r].mean() - base_rec[r].mean()),
                           "p_recall": float(pr),
                           "wins_recall": int((ar[r] > base_rec[r]).sum())}
    return agg, tests, {"random_final_r2": float(base[-1].mean()), "n_folds": n_folds}


def efficiency(curve, target, n_labeled_col="n_labeled"):
    """Labels needed to reach `target`; NaN if never reached. Linear interpolation."""
    xs = np.array([c[n_labeled_col] for c in curve], float)
    ys = np.array([c["r2"] for c in curve], float)
    ok = ys >= target
    if not ok.any():
        return float("nan")
    i = int(np.argmax(ok))
    if i == 0:
        return float(xs[0])
    x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
    if y1 == y0:
        return float(x1)
    return float(x0 + (target - y0) / (y1 - y0) * (x1 - x0))


def label_efficiency(curves, strategies, metric="r2"):
    """Labels each strategy needs to match RANDOM's OWN final score, per fold.

    The threshold has to be random's final value ON THE SAME FOLD, not the
    cross-fold mean. With a mean threshold, between-fold variance alone makes
    random look like it beats itself -- the first version of this function
    reported random as saving 33% of its own budget, which is impossible and was
    the tell. Paired this way random needs its full budget by construction, so
    every other number is a genuine saving against a like-for-like baseline.
    """
    out = {}
    for s in strategies:
        folds = list(curves["random"])
        need, skipped = [], 0
        for f in folds:
            tgt = curves["random"][f][-1][metric]
            if not np.isfinite(tgt):
                skipped += 1
                continue
            if metric == "r2":
                need.append(efficiency(curves[s][f], tgt))
            else:
                need.append(efficiency_recall(curves[s][f], tgt))
        arr = np.asarray(need, float)
        full = curves[s][folds[0]][-1]["n_labeled"]
        out[s] = {"mean": float(np.nanmean(arr)),
                  "n_reached": int(np.isfinite(arr).sum()),
                  "n_folds": len(arr),
                  "full_budget": int(full),
                  "savings": float(1 - np.nanmean(arr) / full)}
    return out


def efficiency_recall(curve, target):
    """Same as `efficiency` but on the recall@25 curve."""
    xs = np.array([c["n_labeled"] for c in curve], float)
    ys = np.array([c["recall25"] for c in curve], float)
    ok = ys >= target
    if not ok.any():
        return float("nan")
    i = int(np.argmax(ok))
    return float(xs[i])


def reanalyze() -> int:
    """Recompute the report from cached curves, without refitting anything."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reanalyze", action="store_true",
                    help="rebuild the report from the curves already in "
                         "5_outputs/active_learning_sim.json")
    args = ap.parse_args()
    if not args.reanalyze:
        return main()
    cached = json.loads(CACHE.read_text())
    print("reanalyzing cached curves (no refitting)")
    for protocol, blk in cached["protocols"].items():
        strategies = list(blk["curves"])
        curves = {s: {int(f): blk["curves"][s][f] for f in blk["curves"][s]}
                  for s in strategies}
        agg, tests, meta = aggregate(curves, strategies)
        eff = label_efficiency(curves, strategies, "r2")
        eff_rec = label_efficiency(curves, strategies, "recall25")
        blk.update({"aggregate": agg, "paired_vs_random": tests,
                    "label_efficiency": eff, "label_efficiency_recall25": eff_rec,
                    "final_random_r2": meta["random_final_r2"]})
        print(f"\n  [{protocol}] label efficiency vs random's own final R2 on the "
              f"same fold (full budget {eff['random']['full_budget']})")
        print(f"    {'strategy':10s} {'R2 labels':>10} {'saving':>8} {'reached':>9}"
              f"   {'recall25 labels':>16} {'saving':>8}")
        for s in strategies:
            e, er = eff[s], eff_rec[s]
            print(f"    {s:10s} {e['mean']:10.0f} {e['savings']:+8.1%} "
                  f"{e['n_reached']:>4}/{e['n_folds']:<4}"
                  f"   {er['mean']:16.0f} {er['savings']:+8.1%}")
    CACHE.write_text(json.dumps(cached, indent=2))
    print(f"\nrewrote {CACHE}")
    return 0


def main() -> int:
    strategies = (["random", "greedy", "sigma", "diverse"]
                  + [f"ucb_b{b}" for b in BETAS])
    arms, fp, y, arm_zoo_mod = build_features()
    print(f"{DS}: n={len(y)}, {len(arms)} learner arms, "
          f"init {INIT_FRAC:.0%} + {N_ROUNDS} x {STEP_FRAC:.0%}, "
          f"{len(strategies)} strategies")

    report = {"dataset": DS, "n_arms": len(arms), "arm_names": list(arms),
              "init_frac": INIT_FRAC, "step_frac": STEP_FRAC, "n_rounds": N_ROUNDS,
              "betas": list(BETAS), "top_k": TOP_K, "strategies": strategies,
              "protocols": {}}

    for protocol, seeds in PROTOCOLS.items():
        jobs = []
        for f_i, s in enumerate(seeds):
            folds = json.loads(
                (ROOT / f"5_outputs/split_cache/{DS}_{protocol}_k5_seed{s}.json").read_text()
            )["folds"]
            for k, f in enumerate(folds):
                jobs.append((f"{s}_f{k}", protocol, np.asarray(f["train"], int),
                             np.asarray(f["test"], int)))
        print(f"\n{protocol}: {len(jobs)} folds x {N_ROUNDS+1} rounds x "
              f"{len(strategies)} strategies = "
              f"{len(jobs)*(N_ROUNDS+1)*len(strategies)*len(arms)} arm fits")
        # Guard before spending 30 min: if any arm is degenerate at the smallest
        # budget, every curve below would be measuring that arm's recovery.
        jid, p, tr0, te0 = jobs[0]
        health = arm_health(y, tr0, te0, arms, arm_zoo_mod, y)
        print(f"  arm health at {int(round(INIT_FRAC*len(y)))} labels "
              f"(fold {jid}): " + "  ".join(f"{k}={v:+.3f}" for k, v in health.items()))
        assert min(health.values()) > 0.2, (
            f"degenerate learner arm(s): {[k for k,v in health.items() if v<=0.2]} "
            f"-- fix the arm set before trusting any curve below")
        t0 = time.time()
        res = Parallel(n_jobs=N_JOBS, verbose=5)(
            delayed(run_fold)(i, p, tr, te, y, arms, fp, arm_zoo_mod, strategies)
            for i, (jid, p, tr, te) in enumerate(jobs))
        print(f"  done in {time.time()-t0:.0f}s", flush=True)

        curves = {s: {} for s in strategies}
        for fid, out in res:
            for s, c in out.items():
                curves[s][fid] = c

        # ---- aggregate per round ----
        rounds = list(range(N_ROUNDS + 1))
        agg, tests, _meta = aggregate(curves, strategies, N_ROUNDS)
        print(f"\n  {'strategy':10s} " + " ".join(
            f"{'r'+str(r):>16}" for r in rounds))
        for s in strategies:
            cells = " ".join(f"{agg[s]['r2_mean'][r]:>7.4f}/{agg[s]['recall_mean'][r]:>7.3f}"
                             for r in rounds)
            print(f"  {s:10s} {cells}")
        print("  (cells are fold-mean R2 / recall@25 on the locked test fold)")

        print(f"\n  paired vs random (15 folds), delta R2 / delta recall@25 / p")
        for s in strategies:
            if s == "random":
                continue
            last = tests[s][rounds[-1]]
            print(f"    {s:10s} final: dR2 {last['delta_r2']:+.4f} (p={last['p_r2']:.3f}, "
                  f"{last['wins_r2']}/15)   dRecall {last['delta_recall']:+.3f} "
                  f"(p={last['p_recall']:.3f}, {last['wins_recall']}/15)")

        # ---- label efficiency ----
        eff = label_efficiency(curves, strategies, "r2")
        eff_rec = label_efficiency(curves, strategies, "recall25")
        print(f"\n  label efficiency: labels needed to match RANDOM's OWN final score "
              f"on the same fold ({eff['random']['full_budget']} = full budget, "
              f"so random's saving is 0 by construction)")
        print(f"    {'strategy':10s} {'R2: labels':>12} {'saving':>8} "
              f"{'(reached)':>11} {'recall@25: labels':>18} {'saving':>8}")
        for s in strategies:
            e, er = eff[s], eff_rec[s]
            print(f"    {s:10s} {e['mean']:12.0f} {e['savings']:+8.1%} "
                  f"{e['n_reached']:>5}/{e['n_folds']:<5} "
                  f"{er['mean']:18.0f} {er['savings']:+8.1%}")

        report["protocols"][protocol] = {
            "n_folds": len(curves["random"]), "aggregate": agg,
            "paired_vs_random": tests, "label_efficiency": eff,
            "label_efficiency_recall25": eff_rec,
            "final_random_r2": float(base[-1].mean()),
            "curves": {s: {f: curves[s][f] for f in curves[s]} for s in strategies},
        }

    CACHE.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {CACHE}")
    return 0


if __name__ == "__main__":
    import sys as _sys
    raise SystemExit(reanalyze() if "--reanalyze" in _sys.argv else main())
