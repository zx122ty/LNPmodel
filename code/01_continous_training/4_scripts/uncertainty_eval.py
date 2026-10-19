#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""L1 of P1: does the ensemble know where it is wrong, and can that be acted on?

This layer trains nothing. It takes the deployed 3-arm NNLS blend whose per-fold
OOF predictions were dumped by `ensemble_oof_dump.py` and attaches an uncertainty
to each molecule, then asks the only question that matters for the project's
stated gap ("which lipids should we make next?"): does ranking by that uncertainty
put more true high-transfectors at the top of the list?

The uncertainty is a MODEL-DIRECTION bootstrap, not a partition-direction one
------------------------------------------------------------------------------
Read this before changing `sigma_variants`; the choice of axis is half the signal.

The arm panel gives 12 arms x 3 partitions = 36 predictions per molecule. There
are two ways to turn that into a spread, and they are NOT interchangeable:

    by MODEL      : 12 groups of 3 (a group = one arm averaged over partitions)
    by PARTITION  :  3 groups of 12 (a group = one partition averaged over arms)

The two sigma vectors correlate only ~0.2 with each other, because they measure
different things: the partition direction varies the TRAINING DATA under a fixed
model, and `07-冻结层数扫描与集成入臂.md` already showed that same-model variants
have residual correlations of 0.80+, so that direction carries little information.
The model direction varies the INDUCTIVE BIAS, which is where the diversity is.

The handoff proposed retraining neural_2d on three different 80% subsets to get an
ensemble whose mean is the deployed predictor. That is unnecessary: averaging each
arm over the partitions and taking the std across arms is the same estimator up to
Monte Carlo error (the model-direction sigma correlates 0.98+ with the naive
pooled-36 version), it costs seconds instead of ~45 min of GPU on a card that has
dropped off the bus three times, and it keeps mu bit-identical to the published
0.7143 / 0.8109, so sigma cannot be accused of riding on a changed predictor. The
partition-direction version is reported alongside as the control documenting why
it is the wrong axis.

Two metric families, and only the second is a deliverable
---------------------------------------------------------
* risk-coverage / Spearman are DIAGNOSTICS: they use the test labels, and
  "abstain" is not an action anyone can take in a lab -- you cannot un-synthesise
  a molecule. They are kept because they are the standard evidence that the signal
  is real, and because the handoff already quotes them.
* precision@k / recall@k / enrichment are the DECISION metrics: "if we made the
  top k of this list, how many of the truly good lipids would we have found".
  Labels are used only to SCORE the ranking, never to build it -- which is the
  real deployment setting.

The decision metrics are the headline. Both "top-25" conventions the handoff flags
are reported: top-25% (480 molecules) is what `repr_ml.json` reports, so it is
comparable to existing numbers; the true top-25 (25 molecules, 1.3%) is what a
"next synthesis batch" actually looks like. The 25-molecule target gets a
bootstrap, because a hit rate over 25 molecules is noisy.

Calibration
-----------
The plan's "唯一的坑" is that fitting and evaluating calibration on the same folds
understates ECE. In this dataset every partition covers every molecule, so
splitting by PARTITION does not separate anything -- the split has to be by
MOLECULE, and it is taken along seed 42's folds (calibrate on the molecules of
folds 0-2, evaluate on the disjoint molecules of folds 3-4). The same-molecule
version is also computed, purely so the size of the optimism it would have
introduced is on the record rather than assumed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr, t as tdist
from sklearn.metrics import r2_score

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")

DS = "Sheet2"
PROTOCOLS = {"scaffold": (42, 1337, 2024), "butina": (42, 1337, 2024)}
SPLIT_SEED = 42          # partition whose folds define the calibration molecule split
CAL_FOLDS = (0, 1, 2)    # folds whose test molecules calibrate; the rest evaluate
K_LIST = (25, 48, 96, 192, 480)     # 1.3% / 2.5% / 5% / 10% / 25% of the library
BETA_LIST = (0.25, 0.5, 1.0, 2.0)   # mu + beta * sigma_s, sigma_s in label units
N_BOOT = 2000


# --------------------------------------------------------------------- loading
def load(protocol: str):
    z = np.load(ROOT / "5_outputs/ensemble_oof.npz")
    rep = json.loads((ROOT / "5_outputs/ensemble_oof.json").read_text())
    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", "RLU (log2)"])
    y = pd.to_numeric(d["RLU (log2)"]).values.astype(float)
    seeds = PROTOCOLS[protocol]
    folds = {s: json.loads(
        (ROOT / f"5_outputs/split_cache/{DS}_{protocol}_k5_seed{s}.json").read_text()
    )["folds"] for s in seeds}
    return z, rep["protocols"][protocol]["p1_equal_folds"], y, folds, seeds


def fold_of(y, folds, seeds):
    """For each molecule and partition, which fold it was a test member of."""
    owner = {}
    for s in seeds:
        o = np.full(len(y), -1)
        for k, f in enumerate(folds[s]):
            o[np.asarray(f["test"], int)] = k
        assert (o >= 0).all(), f"partition {s} does not cover every molecule"
        owner[s] = o
    return owner


def train_mean_per_molecule(y, folds, seeds):
    """Each molecule's training-fold mean, averaged over partitions -- the honest
    constant predictor (a global mean would leak test labels into the baseline)."""
    acc = np.zeros(len(y))
    for s in seeds:
        for f in folds[s]:
            acc[np.asarray(f["test"], int)] += y[np.asarray(f["train"], int)].mean()
    return acc / len(seeds)


def fold_mean_r2(pred, y, folds, seeds, subset=None):
    """Project-consistent metric: per-fold R2, then mean over folds.

    Pooled R2 is not a generalisation estimate here (folds differ in label mean,
    so pooling inflates SST). When `subset` is given, only the folds' retained
    molecules count and folds with <10 kept molecules are dropped.
    """
    vals = []
    for s in seeds:
        for f in folds[s]:
            te = np.asarray(f["test"], int)
            if subset is not None:
                te = te[subset[te]]
                if te.size < 10:
                    continue
            vals.append(r2_score(y[te], pred[s][te]))
    return float(np.mean(vals)), vals


# ------------------------------------------------------------------- sigma, mu
def sigma_variants(panel: np.ndarray) -> dict[str, np.ndarray]:
    """panel: (n_arms, n_partitions, n_molecules) -> four spread estimators."""
    arms_mean = panel.mean(axis=1)                 # (n_arms, n)
    parts_mean = panel.mean(axis=0)                # (n_partitions, n)
    return {
        # headline: one group per ARM, averaged over partitions
        "by_model": arms_mean.std(axis=0, ddof=1),
        # control: one group per PARTITION, averaged over arms
        "by_partition": parts_mean.std(axis=0, ddof=1),
        # control: all arms x partitions pooled into one flat sample
        "pooled": panel.reshape(-1, panel.shape[-1]).std(axis=0, ddof=1),
        # control: spread ACROSS ARMS within each partition, then averaged
        "within_partition": panel.std(axis=0, ddof=1).mean(axis=0),
    }


def partial_spearman(x, y_, z_):
    """Spearman of x vs y_ controlling for z_, on ranks.

    The control matters: sigma is a spread and |mu - ybar| is also a spread, so a
    raw correlation could be riding on the trivial statistic rather than on
    anything model-specific.
    """
    rx, ry, rz = (pd.Series(v).rank().values for v in (x, y_, z_))
    r_xy, r_xz, r_yz = (np.corrcoef(a, b)[0, 1] for a, b in
                        ((rx, ry), (rx, rz), (ry, rz)))
    val = (r_xy - r_xz * r_yz) / np.sqrt((1 - r_xz ** 2) * (1 - r_yz ** 2))
    df = len(rx) - 3
    t = val * np.sqrt(df / max(1 - val ** 2, 1e-12))
    return float(val), float(2 * tdist.sf(abs(t), df))


def risk_coverage(score, preds, y, folds, seeds, fracs=(0.0, 0.1, 0.2, 0.3, 0.5, 0.7)):
    """Fold-mean R2 over the molecules retained after abstaining on the top-scoring ones."""
    order = np.argsort(score, kind="stable")       # lowest score first -> kept
    n = len(y)
    rows = []
    for frac in fracs:
        k = int(round(frac * n))
        keep = np.zeros(n, bool)
        keep[order[: n - k] if k else order] = True
        m, _ = fold_mean_r2(preds, y, folds, seeds, subset=keep)
        rows.append({"abstain": frac, "n_kept": int(keep.sum()), "r2": m})
    xs = np.array([1 - r["abstain"] for r in rows])[::-1]   # increasing
    ys = np.array([r["r2"] for r in rows])[::-1]
    trapz = getattr(np, "trapezoid", np.trapz)
    return rows, float(trapz(ys, xs) / (xs.max() - xs.min()))


# --------------------------------------------------------------- calibration
def calibration(y, mu, sigma, folds, seeds, fracs=(0.5, 0.8, 0.9, 0.95)):
    """Conformal-style interval calibration on molecule-disjoint halves.

    For nominal coverage p, q = the empirical p-quantile of |y-mu|/sigma on the
    calibration molecules; coverage is then how often |y-mu| <= q*sigma holds on
    the held-out molecules. ECE is the mean absolute gap. `naive` fits q and
    measures coverage on the SAME molecules -- by construction its ECE is ~0
    regardless of how bad sigma is, which is exactly the trap the plan names.
    """
    z = np.abs(y - mu) / sigma
    n = len(y)
    cal_mask = np.zeros(n, bool)
    for k in CAL_FOLDS:
        cal_mask[np.asarray(folds[SPLIT_SEED][k]["test"], int)] = True
    ev_mask = ~cal_mask
    assert cal_mask.sum() + ev_mask.sum() == n

    out = {"n_cal": int(cal_mask.sum()), "n_eval": int(ev_mask.sum()),
           "z_stats": {}, "rows": []}
    out["z_stats"] = {
        "mean": float(z.mean()), "std": float(z.std(ddof=1)),
        "median": float(np.median(z)),
        "frac_within_1sigma": float((z <= 1).mean()),
        "frac_within_1.645sigma": float((z <= 1.645).mean()),
        "frac_within_1.96sigma": float((z <= 1.96).mean()),
        "gaussian_reference": {"1sigma": 0.6827, "1.645sigma": 0.90, "1.96sigma": 0.95},
        # sigma only needs to RANK errors; a scalar fixes the scale. This is that
        # scalar: the inflation that would make 80% intervals honest.
        "sigma_inflation_for_80pct": float(
            np.quantile(z[cal_mask], 0.80) / norm.ppf(0.90)),
    }
    ece_naive, ece_held = [], []
    for p in fracs:
        q = float(np.quantile(z[cal_mask], p))
        cov_held = float((z[ev_mask] <= q).mean())
        cov_naive = float((z[cal_mask] <= q).mean())
        ece_naive.append(abs(p - cov_naive))
        ece_held.append(abs(p - cov_held))
        out["rows"].append({"nominal": p, "q": q,
                            "coverage_naive_same_molecules": cov_naive,
                            "coverage_held_out": cov_held})
    out["ece_naive"] = float(np.mean(ece_naive))
    out["ece_held_out"] = float(np.mean(ece_held))
    return out


# ------------------------------------------------------------- decision layer
def decision_metrics(y, mu, sigma_s, k_list=K_LIST, betas=BETA_LIST, n_boot=N_BOOT,
                     seed=0):
    """precision@k / recall@k / enrichment for each ranking strategy.

    Targets are the two conventions the handoff flags: the true top-25 molecules
    (a real synthesis batch) and the true top-25% / 480 (repr_ml.json's口径).
    Bootstrapping resamples MOLECULES with replacement and redefines the target on
    the resample; every strategy sees the same resample, so the comparison against
    `random` stays paired.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    n25 = 25
    tgt = {"top25": set(np.argsort(-y, kind="stable")[:n25].tolist()),
           "top25pct": set(np.argsort(-y, kind="stable")[:int(round(0.25 * n))].tolist())}
    strategies = ([("greedy", 0.0), ("sigma", 0.0)]
                  + [(f"ucb_b{b}", b) for b in betas] + [("random", 0.0)])

    def rank_scores(name, beta, idx):
        if name == "random":
            return rng.random(len(idx))
        if name == "greedy":
            return mu[idx]
        if name == "sigma":
            return sigma_s[idx]
        return mu[idx] + beta * sigma_s[idx]

    kmax = max(k_list)
    out = {}
    rng_r = np.random.default_rng(12345)

    def recall_at(order, k, target):
        return len(set(order[:k].tolist()) & target) / len(target)

    for name, beta in strategies:
        if name == "random":
            # A single random draw is not the baseline -- at k=25 the chance level is
            # 1.3%, so one draw lands on 0 or 0.04 by luck. Average over many draws.
            out[name] = {}
            for k in k_list:
                vals = [recall_at(np.argsort(-rng_r.random(n), kind="stable"), k, s)
                        for _ in range(400) for s in [tgt["top25"]]]
                out[name][k] = {"top25": {"recall": float(np.mean(vals)),
                                          "precision": float(np.mean(vals) * 25 / k),
                                          "enrichment": 1.0},
                                "top25pct": {"recall": k / n, "precision": k / n,
                                             "enrichment": 1.0}}
            continue
        order = np.argsort(-rank_scores(name, beta, np.arange(n)), kind="stable")
        out[name] = {}
        for k in k_list:
            sel = set(order[:k].tolist())
            out[name][k] = {}
            for t, s in tgt.items():
                hit = len(sel & s)
                out[name][k][t] = {"recall": hit / len(s), "precision": hit / k,
                                   "enrichment": (hit / len(s)) / (k / n)}

    # Bootstrap over LIBRARY COMPOSITION: resample molecules with replacement and
    # redefine the true top-25 on the resample. The ranking is held fixed, so this
    # answers "would this ranking still find the good ones in a slightly different
    # library", not "would the model retrain differently". Every strategy sees the
    # same resample, which keeps the vs-greedy comparison paired.
    boot = {name: {k: [] for k in k_list} for name, _ in strategies}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        tgt_r = set(np.argsort(-y[idx], kind="stable")[:n25].tolist())
        for name, beta in strategies:
            order = np.argsort(-rank_scores(name, beta, idx), kind="stable")
            for k in k_list:
                boot[name][k].append(len(set(order[:k].tolist()) & tgt_r) / n25)
    for name in boot:
        for k in k_list:
            v = np.asarray(boot[name][k])
            out[name][k]["recall_boot_mean"] = float(v.mean())
            out[name][k]["recall_ci"] = [float(np.quantile(v, 0.025)),
                                         float(np.quantile(v, 0.975))]
        # paired bootstrap against plain greedy, at the decision-relevant k=25
        g = np.asarray(boot["greedy"][25])
        d = np.asarray(boot[name][25]) - g
        out[name]["vs_greedy_at_k25"] = {"delta": float(d.mean()),
                                         "p_gt": float((d > 0).mean()),
                                         "p_lt": float((d < 0).mean())}
    return out


# --------------------------------------------------------- handoff replication
def replicate_handoff(panel_all: np.ndarray, y: np.ndarray) -> dict:
    """Reproduce the handoff's probe table verbatim, so the two reports tie out.

    That probe used all 13x3 = 39 predictions as a flat deep ensemble, mu = their
    mean, and a global label mean. Same numbers here, same estimator definitions,
    no sigma-binning choice in between.
    """
    P = panel_all.reshape(-1, len(y))
    mu = P.mean(axis=0)
    sig = P.std(axis=0, ddof=1)
    err = np.abs(y - mu)
    ybar = y.mean()
    return {
        "spearman_sigma": float(spearmanr(sig, err).statistic),
        "control_trivial": float(spearmanr(np.abs(mu - ybar), err).statistic),
        "control_oracle": float(spearmanr(np.abs(y - ybar), err).statistic),
        "n_predictions": int(P.shape[0]),
    }


def main() -> int:
    report = {"dataset": DS, "k_list": list(K_LIST), "beta_list": list(BETA_LIST),
              "n_boot": N_BOOT, "split_seed": SPLIT_SEED, "cal_folds": list(CAL_FOLDS),
              "protocols": {}}

    for protocol, seeds in PROTOCOLS.items():
        z, blend, y, folds, seeds = load(protocol)
        panel = z[f"{protocol}__panel"]            # (n_arms, n_partitions, n)
        arms = blend["panel"]
        preds = np.stack([z[f"{protocol}__pred__seed{s}"] for s in seeds])  # (P, n)
        preds_by_seed = {s: z[f"{protocol}__pred__seed{s}"] for s in seeds}

        print(f"\n{'='*80}\n  {protocol}  ({len(seeds)} partitions, {len(arms)} arms)"
              f"   per-partition blend={blend['nnls3_mean']:.4f}\n{'='*80}")

        variants = sigma_variants(panel)
        mu = preds.mean(axis=0)                    # partition-averaged blend (bagged)
        ybar = train_mean_per_molecule(y, folds, seeds)
        sig = variants["by_model"]
        trivial = np.abs(mu - ybar)

        bagged, _ = fold_mean_r2(preds_by_seed, y, folds, seeds)
        print(f"  per-partition blend fold-mean R2 {blend['nnls3_mean']:.4f}; "
              f"same on the partition-averaged (bagged) mu {bagged:.4f} "
              f"-- sigma is computed from mu, so this ties the two together")

        # ---- (0) tie out with the handoff's probe ----
        rep0 = replicate_handoff(panel, y)
        print(f"\n[0] handoff probe replication "
              f"({rep0['n_predictions']} preds, mu = their mean)")
        print(f"    Spearman(sigma,|err|) {rep0['spearman_sigma']:+.4f}   "
              f"trivial {rep0['control_trivial']:+.4f}   "
              f"oracle {rep0['control_oracle']:+.4f}")

        # ---- (A) is sigma informative? ----
        print("\n[A] Spearman(sigma, |error|)   [mu = deployed bagged blend]")
        sens = {}
        for vname, s in variants.items():
            rho = float(spearmanr(s, np.abs(y - mu)).statistic)
            sens[vname] = {
                "rho": rho,
                "rho_vs_partition_mu": float(spearmanr(s, np.abs(y - preds.mean(0))).statistic),
                "corr_with_by_model": float(np.corrcoef(s, sig)[0, 1]),
                "rho_handoff_mu": float(spearmanr(s, np.abs(y - panel.reshape(
                    -1, len(y)).mean(0))).statistic)}
            print(f"    {vname:18s} rho={rho:+.4f}   "
                  f"(corr with by_model {sens[vname]['corr_with_by_model']:+.3f}, "
                  f"rho vs handoff mu {sens[vname]['rho_handoff_mu']:+.4f})")
        par, par_p = partial_spearman(sig, np.abs(y - mu), trivial)
        oracle = float(spearmanr(np.abs(y - ybar), np.abs(y - mu)).statistic)
        controls = {"trivial_|mu-ybar|": float(spearmanr(trivial, np.abs(y - mu)).statistic),
                    "oracle_|y-ybar|": oracle,
                    "partial_spearman_controlling_trivial": par, "partial_p": par_p}
        print(f"    {'trivial |mu-ybar|':18s} rho={controls['trivial_|mu-ybar|']:+.4f}"
              f"   <- baseline to beat")
        print(f"    {'oracle |y-ybar|':18s} rho={oracle:+.4f}"
              f"   <- label-using upper bound")
        print(f"    partial (controlling trivial) rho={par:+.4f}  p={par_p:.2e}")

        # ---- (B) risk-coverage (diagnostic) ----
        print("\n[B] risk-coverage  (fold-mean R2 of what is kept after abstaining)")
        rows_sig, auc_sig = risk_coverage(sig, preds_by_seed, y, folds, seeds)
        rows_triv, auc_triv = risk_coverage(trivial, preds_by_seed, y, folds, seeds)
        rng = np.random.default_rng(0)
        mc = [risk_coverage(rng.random(len(y)), preds_by_seed, y, folds, seeds)[1]
              for _ in range(20)]
        rows_rand = risk_coverage(rng.random(len(y)), preds_by_seed, y, folds, seeds)[0]
        print(f"    {'abstain':>8} {'by sigma':>10} {'by |mu-ybar|':>13} {'random':>9}")
        for i, r in enumerate(rows_sig):
            print(f"    {r['abstain']:>8.0%} {r['r2']:>10.4f} {rows_triv[i]['r2']:>13.4f}"
                  f" {rows_rand[i]['r2']:>9.4f}")
        print(f"    AUC: sigma {auc_sig:.4f}  trivial {auc_triv:.4f}  "
              f"random {np.mean(mc):.4f}")

        # ---- (C) calibration ----
        print(f"\n[C] calibration (molecules of {protocol} fold 0-2 of seed {SPLIT_SEED} "
              f"vs the disjoint rest)")
        cal = calibration(y, mu, sig, folds, seeds)
        zs = cal["z_stats"]
        print(f"    n_cal={cal['n_cal']} n_eval={cal['n_eval']}")
        print(f"    z=(y-mu)/sigma: mean {zs['mean']:+.3f}  std {zs['std']:.3f}   "
              f"|z|<1 {zs['frac_within_1sigma']:.3f} (.683)  "
              f"|z|<1.96 {zs['frac_within_1.96sigma']:.3f} (.95)")
        for r in cal["rows"]:
            print(f"    nominal {r['nominal']:>4.0%}  q={r['q']:5.3f}  "
                  f"held-out coverage {r['coverage_held_out']:.3f}   "
                  f"(same-molecule {r['coverage_naive_same_molecules']:.3f})")
        print(f"    ECE held-out {cal['ece_held_out']:.4f}   "
              f"ECE same-molecule {cal['ece_naive']:.4f}")

        # ---- (D) decision layer ----
        print("\n[D] decision metrics vs the TRUE top-25 molecules (1.3% of the library)")
        sigma_s = sig * (np.std(y - mu) / sig.std(ddof=1))
        dec = decision_metrics(y, mu, sigma_s)
        hdr = " ".join(f"{'k='+str(k):>13}" for k in K_LIST)
        print(f"    {'strategy':10s} {hdr}")
        for name, _ in ([("random", 0), ("greedy", 0), ("sigma", 0)]
                        + [(f"ucb_b{b}", b) for b in BETA_LIST]):
            cells = " ".join(f"{dec[name][k]['top25']['recall']:>13.3f}" for k in K_LIST)
            print(f"    {name:10s} {cells}")
        print(f"    (chance at k=25: {25/len(y):.4f})")
        print("\n    same, against the top-25% / 480-molecule convention that "
              "repr_ml.json reports")
        for name, _ in [("random", 0), ("greedy", 0), ("sigma", 0)]:
            cells = " ".join(f"{dec[name][k]['top25pct']['recall']:>13.3f}"
                             for k in K_LIST)
            print(f"    {name:10s} {cells}")
        print("    bootstrap (2000 resamples of library composition), recall@25 per k:")
        print(f"      {'strategy':10s} " + " ".join(f"{'k='+str(k):>20}" for k in K_LIST))
        for name, _ in ([("random", 0), ("greedy", 0), ("sigma", 0)]
                        + [(f"ucb_b{b}", b) for b in BETA_LIST]):
            cells = []
            for k in K_LIST:
                b = dec[name][k]
                cells.append(f"{b['recall_boot_mean']:>7.3f}"
                             f"[{b['recall_ci'][0]:.3f},{b['recall_ci'][1]:.3f}]")
            v = dec[name].get("vs_greedy_at_k25")
            extra = (f"   vs greedy@25: {v['delta']:+.3f} (P(>)={v['p_gt']:.2f})"
                     if v is not None else "")
            print(f"      {name:10s} " + " ".join(f"{c:>20}" for c in cells) + extra)

        report["protocols"][protocol] = {
            "n_folds": blend["n_folds"], "per_partition_blend_mean": blend["nnls3_mean"],
            "bagged_blend_mean": bagged, "panel": arms, "weights": blend["weights"],
            "handoff_replication": rep0, "sigma_sensitivity": sens, "controls": controls,
            "risk_coverage": {"by_sigma": rows_sig, "by_trivial": rows_triv,
                              "random_curve": rows_rand, "auc_sigma": auc_sig,
                              "auc_trivial": auc_triv,
                              "auc_random_mean": float(np.mean(mc))},
            "calibration": cal, "decision": dec,
        }

    out = ROOT / "5_outputs/uncertainty_eval.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
