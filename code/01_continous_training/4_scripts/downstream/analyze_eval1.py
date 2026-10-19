#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation-1 (TE) analysis: every number quoted in ``prompt/26-09-18/08-*.md``.

Run it and diff against the report; if a number moves, the report is stale. That
is the point -- 教训 7 of ``06-*`` is that a prose claim of "verified" expires
silently while a command does not.

Sections:
  1. four paired tables (fold-mean R2, Wilcoxon vs v5)
  2. mean-shift decomposition -- the 坑 #12 diagnostic
  3. 坑 #12 leak test: paired |pred_mu - test_mu| against a train-only arm
  4. does the raw-R2 gap open on high-mean-shift folds? (falsifiable form)
  5. constant-predictor reference
  6. 7-seed vs 5-seed stability
  7. fold concentration on Sheet3/scaffold (the 04-* re-analysis) + 坑 #17 dedup
  8. NDCG@50 -- the ranking metric that is immune to the prediction's mean
  9. 坑 #12 control arm: same model, early stopping on a TRAIN-INTERNAL holdout
 10. how much each arm loses to the fold mean (centered minus raw)

Section 8 exists because §2/§3 of the report rest on it: a reader who doubts
that the 坑 #12 mean-shift is doing the work needs a metric where the mean
*cannot* do any work. NDCG@50 is that metric, and it is already stored per fold
by ``run_matrix``, so it costs nothing to report.

Section 9 is the control arm the report's §3.5 calls option B: ``--es-val internal``
trains the identical model on the identical folds with the identical budget, and
changes only the source of the early-stopping signal. Its raw R2 is therefore on
the classical arms' protocol, so it is the arm a like-for-like comparison must
use, and ``default - control`` measures the 坑 #12 advantage directly instead of
inferring it from a mean-shift decomposition.

Usage:
    python 4_scripts/downstream/analyze_eval1.py
    python 4_scripts/downstream/analyze_eval1.py > 5_outputs/report_tables/eval1_TE_20260919/analysis.txt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, pearsonr, spearmanr, wilcoxon
from sklearn.metrics import r2_score

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
sys.path.insert(0, str(ROOT / "4_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ds_registry  # noqa: E402
import split_files  # noqa: E402

SEEDS = (42, 123, 456, 789, 1011)
SEEDS7 = SEEDS + (1337, 2024)
ARMS = ("lipidfm_v5", "tuned_gbm", "ecfp4_rf", "ecfp4_xgb", "mlp_morgan")
# The 坑 #12 control arm: same model, same folds, same budget -- early stopping
# watches a train-internal 15% holdout instead of the test fold, and the real test
# folds are re-predicted from model_<k>.pth. Its raw R2 is on the SAME protocol as
# the classical arms, so it is the arm a like-for-like comparison must use.
CTRL = "lipidfm_v5_esval"
COMBOS = [("Sheet2", "scaffold"), ("Sheet2", "butina"),
          ("Sheet3", "scaffold"), ("Sheet3", "butina")]
OOF_DIR = ROOT / "5_outputs/finetune_moltrain/oof"
PRED_DIR = ROOT / "5_outputs/downstream_predictions"
RES_DIR = ROOT / "5_outputs/downstream_results"


def cell(ds, arm, sp, s):
    return RES_DIR / f"{ds}|{arm}|{sp}|seed{s}|k5.json"


def load_oof(ds, arm, sp, s):
    """Out-of-fold predictions for one cell, from wherever that arm stores them."""
    if arm in ("lipidfm_v5", CTRL):
        # The control arm's run_label carries the `_esval` suffix, so its dump is a
        # different file. Reading the main arm's dump for the control cell would
        # silently produce the main arm's numbers under the control's name.
        lab = "unimol2_lipids_84m_v5_checkpoint_1_30000.pt" + (
            "_esval" if arm == CTRL else "")
        p = OOF_DIR / f"oof_{ds}_{sp}_seed{s}_{lab}.npz"
    else:
        p = PRED_DIR / f"{ds}|{arm}|{sp}|seed{s}|k5.npz"
    return np.asarray(np.load(p, allow_pickle=True)["oof"], float).ravel()


def per_fold(ds, arm, sp, seeds=SEEDS, ndcg_key="r2_per_fold"):
    """Per-fold metric vector for one cell. ``ndcg_key`` selects another stored
    per-fold metric (e.g. ``ndcg@50_per_fold``) without a second code path."""
    out = []
    for s in seeds:
        c = json.loads(cell(ds, arm, sp, s).read_text())
        out += c["per_target"]["TARGET"][ndcg_key]
    return np.array(out)


def hdr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    data = {c: ds_registry.load(c[0], verbose=False) for c in COMBOS}
    ys = {k: v[0]["TARGET"].to_numpy(float) for k, v in data.items()}
    smis = {k: v[0]["SMILES"].astype(str).tolist() for k, v in data.items()}
    folds = {(ds, sp, s): split_files.load_folds(ds, sp, s, 5, smiles=smis[(ds, sp)])
             for ds, sp in COMBOS for s in SEEDS}

    hdr("1. four paired tables -- fold-mean R2, 25 folds (plan grid)")
    for ds, sp in COMBOS:
        print(f"\n--- {ds}/{sp}")
        table = {}
        for a in ARMS:
            if not all(cell(ds, a, sp, s).exists() for s in SEEDS):
                print(f"    {a:11s} MISSING")
                continue
            v = per_fold(ds, a, sp)
            table[a] = v
            print(f"    {a:11s} {v.mean():+.4f}  (std {v.std(ddof=1):.4f})")
        if "lipidfm_v5" not in table:
            continue
        for a in ARMS:
            if a in ("lipidfm_v5",) or a not in table:
                continue
            d = table["lipidfm_v5"] - table[a]
            w = wilcoxon(table["lipidfm_v5"], table[a])
            print(f"      v5 vs {a:11s} d={d.mean():+.4f}  p={w.pvalue:.3g}  "
                  f"W/L/T={int((d>0).sum())}/{int((d<0).sum())}/{int((d==0).sum())}")

    hdr("2. mean-shift decomposition: R2 vs centered R2 (Pearson^2)")
    print("centered R2 ignores the prediction's mean, so it is immune to a model")
    print("that lands its fold-mean near the test fold's mean (坑 #12).")
    for ds, sp in COMBOS:
        print(f"\n--- {ds}/{sp}")
        raw, cen = {}, {}
        for a in ARMS:
            if not all(cell(ds, a, sp, s).exists() for s in SEEDS):
                continue
            R, C = [], []
            for s in SEEDS:
                oof = load_oof(ds, a, sp, s)
                y = ys[(ds, sp)]
                for _tr, te in folds[(ds, sp, s)]:
                    te = np.asarray(te)
                    R.append(r2_score(y[te], oof[te]))
                    C.append(pearsonr(oof[te], y[te]).statistic ** 2)
            raw[a], cen[a] = np.mean(R), np.mean(C)
            print(f"    {a:11s} raw={raw[a]:+.4f}  centered={cen[a]:+.4f}")
        best = max((a for a in raw if a != "lipidfm_v5"), key=lambda a: raw[a])
        bestc = max((a for a in cen if a != "lipidfm_v5"), key=lambda a: cen[a])
        bestu = max((a for a in cen if a not in ("lipidfm_v5", "tuned_gbm")),
                    key=lambda a: cen[a])
        print(f"    v5 advantage: raw {raw['lipidfm_v5']-raw[best]:+.4f} vs {best}"
              f" | centered {cen['lipidfm_v5']-cen[bestc]:+.4f} vs {bestc}")
        # The headline table's Sheet3/scaffold cell quotes the centered gap against
        # the best UNTUNED arm (+0.17), because the tuned arm only exists on this
        # grid and is the arm a reader will actually object to. Both gaps are
        # printed, and both are paired, so neither is a hand-picked number.
        for tag, ref in (("centered vs best untuned", bestu),):
            R, C = [], []
            for s in SEEDS:
                oa, ob = load_oof(ds, "lipidfm_v5", sp, s), load_oof(ds, ref, sp, s)
                y = ys[(ds, sp)]
                for _tr, te in folds[(ds, sp, s)]:
                    te = np.asarray(te)
                    R.append(r2_score(y[te], oa[te]) - r2_score(y[te], ob[te]))
                    C.append(pearsonr(oa[te], y[te]).statistic ** 2
                             - pearsonr(ob[te], y[te]).statistic ** 2)
            w = wilcoxon(C)
            print(f"    {tag} ({ref}): raw {np.mean(R):+.4f} | centered {np.mean(C):+.4f} "
                  f"p={w.pvalue:.3g} W/L={int((np.array(C)>0).sum())}/{int((np.array(C)<0).sum())}")

    hdr("3. 坑 #12 leak test: is v5's fold-mean closer to the TEST mean than a")
    print("   train-only arm's is? Paired over the same folds.")
    for ds, sp in COMBOS:
        print(f"\n--- {ds}/{sp}")
        for other in ("mlp_morgan", "tuned_gbm"):
            if not all(cell(ds, other, sp, s).exists() for s in SEEDS):
                continue
            a_e, b_e, shift = [], [], []
            for s in SEEDS:
                ya = load_oof(ds, "lipidfm_v5", sp, s)
                yb = load_oof(ds, other, sp, s)
                y = ys[(ds, sp)]
                for tr, te in folds[(ds, sp, s)]:
                    tr, te = np.asarray(tr), np.asarray(te)
                    a_e.append(abs(ya[te].mean() - y[te].mean()))
                    b_e.append(abs(yb[te].mean() - y[te].mean()))
                    shift.append(abs(y[te].mean() - y[tr].mean()))
            a_e, b_e, shift = map(np.array, (a_e, b_e, shift))
            w = wilcoxon(a_e, b_e)
            hi = shift >= np.median(shift)
            print(f"    v5={a_e.mean():.3f}  {other}={b_e.mean():.3f}  p={w.pvalue:.3g}  "
                  f"W/L={int((a_e<b_e).sum())}/{int((a_e>b_e).sum())}")
            print(f"      high-shift folds: v5={a_e[hi].mean():.3f} {other}={b_e[hi].mean():.3f}"
                  f" | low-shift: v5={a_e[~hi].mean():.3f} {other}={b_e[~hi].mean():.3f}")

    hdr("4. does the raw-R2 gap OPEN on high-mean-shift folds? (falsifiable form)")
    for ds, sp in COMBOS:
        rows = []
        for s in SEEDS:
            oofs = {a: load_oof(ds, a, sp, s) for a in ARMS if cell(ds, a, sp, s).exists()}
            y = ys[(ds, sp)]
            for tr, te in folds[(ds, sp, s)]:
                tr, te = np.asarray(tr), np.asarray(te)
                r = {a: r2_score(y[te], o[te]) for a, o in oofs.items()}
                rows.append((abs(y[te].mean() - y[tr].mean()),
                             r["lipidfm_v5"] - max(r[a] for a in r if a != "lipidfm_v5")))
        sh = np.array([r[0] for r in rows])
        g = np.array([r[1] for r in rows])
        hi = sh >= np.median(sh)
        print(f"  {ds}/{sp:8s} high-shift gap={g[hi].mean():+.4f} (n={hi.sum()})  "
              f"low-shift gap={g[~hi].mean():+.4f} (n={(~hi).sum()})  "
              f"p={mannwhitneyu(g[hi], g[~hi]).pvalue:.3g}")

    hdr("5. constant (train-mean) predictor, 25 folds")
    for ds, sp in COMBOS:
        r = []
        for s in SEEDS:
            y = ys[(ds, sp)]
            for tr, te in folds[(ds, sp, s)]:
                tr, te = np.asarray(tr), np.asarray(te)
                r.append(r2_score(y[te], np.full(len(te), y[tr].mean())))
        r = np.array(r)
        print(f"  {ds}/{sp:8s} {r.mean():+.4f}  (worst {r.min():+.4f}, n<0 {int((r<0).sum())}/25)")

    hdr("6. 5-seed (plan) vs 7-seed stability, classical arms")
    for ds, sp in COMBOS:
        for a in ARMS:
            if a == "lipidfm_v5" or not all(cell(ds, a, sp, s).exists() for s in SEEDS7):
                continue
            a5, a7 = per_fold(ds, a, sp, SEEDS), per_fold(ds, a, sp, SEEDS7)
            print(f"  {ds}/{sp:8s} {a:11s} 25f={a5.mean():+.4f} 35f={a7.mean():+.4f} "
                  f"d={a7.mean()-a5.mean():+.4f}")

    hdr("7. Sheet3/scaffold: fold concentration vs arm behaviour (re-analysis of 04-*)")
    import chem_splits
    for ds, sp in [("Sheet3", "scaffold")]:
        labs = chem_splits.scaffold_group_ids(smis[(ds, sp)])
        y = ys[(ds, sp)]
        rows = []
        for s in SEEDS:
            oofs = {a: load_oof(ds, a, sp, s) for a in ARMS if cell(ds, a, sp, s).exists()}
            for tr, te in folds[(ds, sp, s)]:
                tr, te = np.asarray(tr), np.asarray(te)
                _v, c = np.unique(labs[te], return_counts=True)
                rows.append(dict(
                    share=c.max() / len(te),
                    **{a: r2_score(y[te], o[te]) for a, o in oofs.items()},
                    const=r2_score(y[te], np.full(len(te), y[tr].mean()))))
        sh = np.array([r["share"] for r in rows])
        for lo, hi_, name in ((0, .5, "healthy (<50%)"), (.5, .99, "concentrated 50-99%"),
                              (.99, 1.01, "single-family (=100%)")):
            m = (sh >= lo) & (sh < hi_)
            if not m.sum():
                continue
            parts = "  ".join(f"{a}={np.mean([r[a] for r, k in zip(rows, m) if k]):+.3f}"
                              for a in ("lipidfm_v5", "ecfp4_rf", "const"))
            print(f"  {name:22s} n={m.sum():2d}  {parts}")
        v5 = np.array([r["lipidfm_v5"] for r in rows])
        cn = np.array([r["const"] for r in rows])
        print(f"  v5 beats the constant predictor in {int((v5>cn).sum())}/{len(rows)} folds")
        print("  per-fold family shares (all 5 seeds):")
        for s in SEEDS:
            lab = [round(float(np.unique(labs[np.asarray(te)], return_counts=True)[1].max()
                               / len(te)), 2) for _tr, te in folds[("Sheet3", "scaffold", s)]]
            print(f"    seed{s:5d}: {lab}")
        # The 25 fold-slots are NOT 25 distinct test sets, and the three buckets
        # above therefore overstate their own n. Deduplicate by test-set identity
        # and print both counts, because "single-family n=5" is the number a
        # reader carries into §4.1 and it is the one that is most wrong: the
        # 175-molecule piperidine fold is the SAME SET in all 5 seeds.
        seen: dict = {}
        for s in SEEDS:
            for tr, te in folds[("Sheet3", "scaffold", s)]:
                te = np.asarray(te)
                _v, c = np.unique(labs[te], return_counts=True)
                seen.setdefault(frozenset(te.tolist()), []).append(c.max() / len(te))
        dedup = {"healthy (<50%)": 0, "concentrated 50-99%": 0, "single-family (=100%)": 0}
        for k, shares in seen.items():
            sh = shares[0]
            b = ("healthy (<50%)" if sh < .5 else
                 "concentrated 50-99%" if sh < .99 else "single-family (=100%)")
            dedup[b] += 1
        print(f"  unique test sets: {len(seen)} (25 fold-slots); "
              + "  ".join(f"{b}={n}" for b, n in dedup.items()))
        print(f"  v5 beats the constant predictor on {int((v5>cn).sum())}/{len(rows)} "
              f"fold-slots -- but the independent n is {len(seen)}, not {len(rows)}")

    # 坑 #17 generalised: the fold-slot count overstates the independent n by a
    # DIFFERENT factor in each combination, and the discriminator the pit text
    # gives (max group / mean fold size >= 1.0) does not cover the worst case.
    # Sheet3/butina sits at 0.99 -- just under the threshold -- yet 25 fold-slots
    # are only 11 distinct test sets, because `split_clusters_to_folds` places
    # clusters by DESCENDING size and only breaks ties with a 1e-9 jitter, so the
    # assignment is largely seed-independent. Print the ratio next to the count so
    # the two are never read apart.
    print("\n  坑 #17 across all four combinations (dedup by frozenset(test_idx)):")
    for ds, sp in COMBOS:
        seen: dict = {}
        for s in SEEDS:
            for _tr, te in folds[(ds, sp, s)]:
                seen.setdefault(frozenset(np.asarray(te).tolist()), []).append(s)
        mult = {}
        for v in seen.values():
            mult[len(v)] = mult.get(len(v), 0) + 1
        audit = json.loads((ROOT / "5_outputs/downstream_splits"
                            / f"{ds}_{sp}_k5_seed42.json").read_text())["audit"]
        fam = audit.get("family", {})
        print(f"    {ds}/{sp:8s} {len(seen):2d} distinct / 25 slots  "
              f"multiplicity {dict(sorted(mult.items()))}  "
              f"ratio={fam.get('max_group_over_fold_size')} "
              f"n_groups={audit.get('n_groups')}")

    hdr("8. NDCG@50 -- the ranking metric that cannot see the prediction's mean")
    print("If the 坑 #12 advantage were purely a mean shift, it would vanish here.")
    print("It does not vanish everywhere: it shrinks to ~1/16 on Sheet3/scaffold.")
    for ds, sp in COMBOS:
        print(f"\n--- {ds}/{sp}")
        t = {}
        for a in ARMS:
            if not all(cell(ds, a, sp, s).exists() for s in SEEDS):
                continue
            t[a] = per_fold(ds, a, sp, ndcg_key="ndcg@50_per_fold")
            print(f"    {a:11s} {t[a].mean():.4f}  (std {t[a].std(ddof=1):.4f})")
        if "lipidfm_v5" not in t:
            continue
        best = max((a for a in t if a != "lipidfm_v5"), key=lambda a: t[a].mean())
        d = t["lipidfm_v5"] - t[best]
        w = wilcoxon(t["lipidfm_v5"], t[best])
        print(f"    v5 vs best baseline ({best}): d={d.mean():+.4f}  p={w.pvalue:.4g}  "
              f"W/L/T={int((d>0).sum())}/{int((d<0).sum())}/{int((d==0).sum())}")


    hdr("9. 坑 #12 control arm: same model, early stopping on a TRAIN-INTERNAL holdout")
    print("`lipidfm_v5_esval` = identical model, folds and budget; the only change is")
    print("that early stopping watches a 15% holdout carved out of TRAIN, and the real")
    print("test folds are re-predicted from model_<k>.pth. Its raw R2 is therefore on")
    print("the SAME protocol as the classical arms. Two things follow: the raw gap")
    print("(default - control) measures the 坑 #12 advantage directly, and the control")
    print("is the arm a like-for-like comparison must use.")
    for ds, sp in COMBOS:
        if not all(cell(ds, CTRL, sp, s).exists() for s in SEEDS):
            print(f"\n--- {ds}/{sp}: control arm MISSING")
            continue
        print(f"\n--- {ds}/{sp}")
        a = per_fold(ds, "lipidfm_v5", sp)
        b = per_fold(ds, CTRL, sp)
        d = a - b
        w = wilcoxon(a, b)
        print(f"    default(es_val=test) {a.mean():+.4f}   control(train-internal) "
              f"{b.mean():+.4f}   raw gap {d.mean():+.4f}  p={w.pvalue:.3g}  "
              f"W/L/T={int((d>0).sum())}/{int((d<0).sum())}/{int((d==0).sum())}")
        # The same gap on the mean-immune scale. If the raw gap were ENTIRELY the
        # mean shift, this would be zero: the control cannot see the test fold's
        # mean, so whatever it still loses on centered R2 is a genuine cost of
        # losing the test fold as the early-stopping signal (a noisier stopping
        # point), not a leak.
        ca, cb = [], []
        for s in SEEDS:
            oa, ob = load_oof(ds, "lipidfm_v5", sp, s), load_oof(ds, CTRL, sp, s)
            y = ys[(ds, sp)]
            for _tr, te in folds[(ds, sp, s)]:
                te = np.asarray(te)
                ca.append(pearsonr(oa[te], y[te]).statistic ** 2)
                cb.append(pearsonr(ob[te], y[te]).statistic ** 2)
        ca, cb = np.array(ca), np.array(cb)
        print(f"    centered: default {ca.mean():+.4f}  control {cb.mean():+.4f}  "
              f"gap {ca.mean()-cb.mean():+.4f}  p={wilcoxon(ca, cb).pvalue:.3g}  "
              f"-> {(ca.mean()-cb.mean())/(d.mean())*100:.0f}% of the raw gap survives "
              f"on the mean-immune scale")
        # Per-fold: the gap should be small where the fold mean barely moves and
        # large where it moves a lot -- that is the leak's signature, and here it is
        # measured within one model instead of across two model families.
        shift, gap = [], []
        for s in SEEDS:
            oa, ob = load_oof(ds, "lipidfm_v5", sp, s), load_oof(ds, CTRL, sp, s)
            y = ys[(ds, sp)]
            for tr, te in folds[(ds, sp, s)]:
                tr, te = np.asarray(tr), np.asarray(te)
                shift.append(abs(y[te].mean() - y[tr].mean()))
                gap.append(r2_score(y[te], oa[te]) - r2_score(y[te], ob[te]))
        shift, gap = np.array(shift), np.array(gap)
        hi = shift >= np.median(shift)
        rho = spearmanr(shift, gap)
        print(f"    gap high-shift {gap[hi].mean():+.4f} (n={hi.sum()}) | low-shift "
              f"{gap[~hi].mean():+.4f} (n={(~hi).sum()})  p={mannwhitneyu(gap[hi], gap[~hi]).pvalue:.3g}")
        print(f"    Spearman(|fold mean shift|, gap) = {rho.statistic:+.3f} p={rho.pvalue:.3g}")
        # The leak statistic itself, paired WITHIN one model: |pred_mu - test_mu|.
        # §3 runs this across two model families (v5 vs mlp_morgan), where the
        # difference could in principle be a model property rather than a protocol
        # one. Here the two arms are the same model, same folds, same budget, and
        # only the early-stopping validation source differs -- so a difference is
        # attributable to the protocol alone. The control is also compared to the
        # classical arms, which is the check that it really is on their protocol.
        pa, pb = [], []
        for s in SEEDS:
            oa, ob = load_oof(ds, "lipidfm_v5", sp, s), load_oof(ds, CTRL, sp, s)
            y = ys[(ds, sp)]
            for _tr, te in folds[(ds, sp, s)]:
                te = np.asarray(te)
                pa.append(abs(oa[te].mean() - y[te].mean()))
                pb.append(abs(ob[te].mean() - y[te].mean()))
        pa, pb = np.array(pa), np.array(pb)
        wp = wilcoxon(pa, pb)
        print(f"    |pred_mu-test_mu|: default {pa.mean():.3f}  control {pb.mean():.3f}  "
              f"ratio {pa.mean()/pb.mean():.2f}x  p={wp.pvalue:.3g}  "
              f"W/L={int((pa<pb).sum())}/{int((pa>pb).sum())}")
        for ref in ("tuned_gbm", "mlp_morgan"):
            if not all(cell(ds, ref, sp, s).exists() for s in SEEDS):
                continue
            pc = []
            for s in SEEDS:
                oc = load_oof(ds, ref, sp, s)
                y = ys[(ds, sp)]
                for _tr, te in folds[(ds, sp, s)]:
                    te = np.asarray(te)
                    pc.append(abs(oc[te].mean() - y[te].mean()))
            pc = np.array(pc)
            wc = wilcoxon(pb, pc)
            print(f"      control vs {ref:11s} {pb.mean():.3f} vs {pc.mean():.3f}  "
                  f"ratio {pb.mean()/pc.mean():.2f}x  p={wc.pvalue:.3g}")
        # The control arm must still be a real model, not a constant predictor.
        marg = []
        for s in SEEDS:
            ob = load_oof(ds, CTRL, sp, s)
            y = ys[(ds, sp)]
            for tr, te in folds[(ds, sp, s)]:
                tr, te = np.asarray(tr), np.asarray(te)
                marg.append(r2_score(y[te], ob[te])
                            - r2_score(y[te], np.full(len(te), y[tr].mean())))
        marg = np.array(marg)
        # 坑 #17: count distinct test sets, not fold-slots. On Sheet2 the folds are
        # 25 distinct sets, but on Sheet3/scaffold 25 slots are only 21 sets, so a
        # slot count would overstate the independent n.
        sets: dict = {}
        j = 0
        for s in SEEDS:
            for _tr, te in folds[(ds, sp, s)]:
                sets.setdefault(frozenset(np.asarray(te).tolist()), []).append(marg[j])
                j += 1
        won = sum(1 for v in sets.values() if np.mean(v) > 0)
        print(f"    control beats the train-mean predictor on {int((marg>0).sum())}/{len(marg)} "
              f"fold-slots = {won}/{len(sets)} distinct test sets")
        # Report the losses by identity too: if several losing slots were the SAME
        # test set, quoting the slot count would inflate how many chemical spaces
        # the control actually fails on.
        if won < len(sets):
            lost = {k: v for k, v in sets.items() if np.mean(v) <= 0}
            print(f"      losses: {len(lost)} distinct test set(s), margins "
                  + " ".join(f"{np.mean(v):+.4f}" for v in lost.values()))
        # Paired against the classical arms, on the SAME protocol -- this is the
        # honest form of the "neural beats traditional" claim.
        for ref in ("tuned_gbm", "ecfp4_rf", "ecfp4_xgb", "mlp_morgan"):
            if not all(cell(ds, ref, sp, s).exists() for s in SEEDS):
                continue
            R, C = [], []
            for s in SEEDS:
                oa, ob = load_oof(ds, CTRL, sp, s), load_oof(ds, ref, sp, s)
                y = ys[(ds, sp)]
                for _tr, te in folds[(ds, sp, s)]:
                    te = np.asarray(te)
                    R.append(r2_score(y[te], oa[te]) - r2_score(y[te], ob[te]))
                    C.append(pearsonr(oa[te], y[te]).statistic ** 2
                             - pearsonr(ob[te], y[te]).statistic ** 2)
            R, C = np.array(R), np.array(C)
            print(f"      control vs {ref:11s} raw {R.mean():+.4f} p={wilcoxon(R).pvalue:.3g} "
                  f"W/L={int((R>0).sum())}/{int((R<0).sum())}  |  centered {C.mean():+.4f} "
                  f"p={wilcoxon(C).pvalue:.3g} W/L={int((C>0).sum())}/{int((C<0).sum())}")
            # 坑 #17: the 25 fold-slots are not 25 independent samples, and the
            # dedup factor differs per combination (Sheet3/butina is 11/25). The
            # paired test above weights a repeated test set by how many seeds
            # happened to land on it, which is not a property of the chemistry.
            # Re-run it on one value per distinct test set -- the independent n --
            # and print both, because the two can disagree in the headline.
            per_set: dict = {}
            j = 0
            for s in SEEDS:
                for _tr, te in folds[(ds, sp, s)]:
                    per_set.setdefault(frozenset(np.asarray(te).tolist()), []).append(R[j])
                    j += 1
            if len(per_set) < len(R):
                ded = np.array([np.mean(v) for v in per_set.values()])
                print(f"        [坑 #17 dedup] {len(per_set)} distinct test sets "
                      f"(not {len(R)} slots): raw {ded.mean():+.4f} "
                      f"p={wilcoxon(ded).pvalue:.3g} W/L="
                      f"{int((ded>0).sum())}/{int((ded<0).sum())}")

    hdr("10. how much each arm loses to the fold mean (centered minus raw)")
    print("A larger drop means the arm's fold-level predictions lean harder on the test")
    print("fold's own mean. Under the default protocol the neural arm can see that mean")
    print("through early stopping (坑 #12); the control arm cannot.")
    for ds, sp in COMBOS:
        print(f"\n--- {ds}/{sp}")
        for a in ARMS + (CTRL,):
            if not all(cell(ds, a, sp, s).exists() for s in SEEDS):
                continue
            R, C = [], []
            for s in SEEDS:
                oof = load_oof(ds, a, sp, s)
                y = ys[(ds, sp)]
                for _tr, te in folds[(ds, sp, s)]:
                    te = np.asarray(te)
                    R.append(r2_score(y[te], oof[te]))
                    C.append(pearsonr(oof[te], y[te]).statistic ** 2)
            print(f"    {a:16s} drop={np.mean(C)-np.mean(R):+.4f}")


if __name__ == "__main__":
    main()
