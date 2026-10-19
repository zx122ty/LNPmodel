#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arm zoo: can a THIRD, decorrelated arm improve the ensemble on both protocols?

Why this exists
---------------
`11-最终结论汇总.md` established the project's only protocol-invariant result:
averaging the neural fine-tune with the tuned traditional model beats either one
(scaffold 0.7037 vs 0.6807, p=0.0003; butina 0.8062 vs 0.7908, p=0.0044), while
the RANKING of single models reverses between protocols.

So the lever with the best evidence is not another architecture -- it is more
diversity in the pool. With only two arms the achievable variance reduction is
capped. This script asks: is there a third arm that is (a) strong enough to
matter and (b) decorrelated enough to help?

Selection rule
--------------
Not "pick the strongest independent arm" -- that is how you end up re-adding
something the neural net already knows. The criterion is deliberately two-sided:

    strong enough : independent fold-mean R2 >= 0.55
    diverse enough: mean residual correlation vs the existing arms <= 0.70

Both are checked, then the actual ensemble gain is measured. An arm that fails
either bar is reported as a failure -- that is a result too (see the project's
convention of reporting negatives).

Anti-overfitting discipline
---------------------------
Arms are NOT hyperparameter-tuned here: each gets one sensible fixed
configuration, so no arm selection ever touches a test fold. Ensemble
composition is chosen greedily on partitions {1337, 2024} and then confirmed on
the untouched seed-42 partition. That is stricter than the existing baseline,
whose hyperparameters were selected on seed 42 (see traditional_baseline_tuned.py
honesty note).

Everything is paired: all arms share the cached folds in 5_outputs/split_cache/,
so every comparison is a within-fold paired t-test.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import DataConversionWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")
# KNeighborsRegressor's jaccard metric coerces float bitsets to bool (which is
# what Tanimoto wants); the notice is noise here.
warnings.filterwarnings("ignore", category=DataConversionWarning)

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
FEAT_CACHE = ROOT / "5_outputs/feat_cache"

DS = os.environ.get("AZ_DS", "Sheet2")

# Per-dataset spec. Sheet3 is the lab's own assay; its target gained an IGROV1_
# prefix when the structures were corrected on 2026-09-13.
DS_SPEC = {
    "Sheet2": {"target": "RLU (log2)", "log10": False,
               "seeds": {"scaffold": (42, 1337, 2024), "butina": (42, 1337)},
               "select": (1337, 2024)},
    "Sheet3": {"target": ("IGROV1_transfection_efficiency", "transfection_efficiency"),
               "log10": True,
               "seeds": {"scaffold": (42, 1337, 2024), "butina": (42, 1337)},
               "select": (1337, 2024)},
}
SPEC = DS_SPEC[DS]
TARGET_COL = SPEC["target"]
LOG10 = SPEC["log10"]
SEEDS = SPEC["seeds"]
SELECT_SEEDS = SPEC["select"]
ALL_SEEDS = SEEDS["scaffold"]

OUT = ROOT / f"5_outputs/arm_zoo{'' if DS == 'Sheet2' else '_' + DS}.json"

N_WORKERS = int(os.environ.get("AZ_NJOBS", "12"))

# Existing arms, keyed by the name used in ensemble_eval.json.
BASE_ARMS = ("neural_2d", "traditional")
# Which partitions the 3D-conformer arm was actually run on. Sheet3 has only the
# 2D arm on disk, so the 3D arm simply never enters its menu.
NEURAL_3D_SEEDS = {"Sheet2": {"scaffold": (42, 1337, 2024), "butina": (42,)},
                   "Sheet3": {"scaffold": (), "butina": ()}}[DS]

# --------------------------------------------------------------------- features
def _cache(name, fn):
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    p = FEAT_CACHE / f"{DS}_{name}.npy"
    if p.exists():
        return np.load(p)
    X = fn()
    np.save(p, X)
    return X


def _mols(smis):
    return [Chem.MolFromSmiles(s) for s in smis]


def rdkit_descriptors(smis):
    keys, rows = None, []
    for s in smis:
        v = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(s))
        if keys is None:
            keys = list(v.keys())
        rows.append([v[k] if v[k] is not None else np.nan for k in keys])
    D = np.asarray(rows, dtype=np.float64)
    D[~np.isfinite(D)] = np.nan
    med = np.nanmedian(D, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    idx = np.where(~np.isfinite(D))
    D[idx] = np.take(med, idx[1])
    D = D[:, D.std(axis=0) > 1e-8]
    D = D[:, np.abs(D).max(axis=0) <= 1e6]
    return D.astype(np.float32)


def morgan(smis, radius, nbits):
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=nbits)
    mols = _mols(smis)
    return np.asarray([np.asarray(gen.GetFingerprint(m), dtype=np.float32)
                       for m in mols], dtype=np.float32)


def maccs(smis):
    return np.asarray([np.asarray(AllChem.GetMACCSKeysFingerprint(m), dtype=np.float32)
                       for m in _mols(smis)], dtype=np.float32)


def torsion(smis, nbits=1024):
    gen = AllChem.GetTopologicalTorsionGenerator(fpSize=nbits)
    return np.asarray([np.asarray(gen.GetFingerprint(m), dtype=np.float32)
                       for m in _mols(smis)], dtype=np.float32)


def atompair(smis, nbits=1024):
    gen = AllChem.GetAtomPairGenerator(fpSize=nbits)
    return np.asarray([np.asarray(gen.GetFingerprint(m), dtype=np.float32)
                       for m in _mols(smis)], dtype=np.float32)


# ----------------------------------------------------------------------- folds
def folds_for(seed, split="scaffold", k=5):
    p = ROOT / f"5_outputs/split_cache/{DS}_{split}_k{k}_seed{seed}.json"
    d = json.loads(p.read_text())
    return [(np.asarray(f["train"], int), np.asarray(f["test"], int)) for f in d["folds"]]


def oof(kind, X, y, folds, params=None):
    """Out-of-fold predictions. Scalers live inside the fold via Pipeline, so no
    preprocessing statistic ever sees a test molecule."""
    params = params or {}
    if kind == "svr":
        def make():
            return Pipeline([("sc", StandardScaler()),
                             ("m", SVR(kernel="rbf", C=params.get("C", 10.0),
                                       gamma=params.get("gamma", "scale"),
                                       epsilon=0.1, cache_size=1024))])
    elif kind == "krr":
        def make():
            return Pipeline([("sc", StandardScaler()),
                             ("m", KernelRidge(kernel="rbf", alpha=params.get("alpha", 1.0),
                                               gamma=params.get("gamma", None)))])
    elif kind == "rf":
        def make():
            return RandomForestRegressor(n_estimators=params.get("n_estimators", 400),
                                         max_features=params.get("max_features", 0.3),
                                         n_jobs=1, random_state=0)
    elif kind == "xgb":
        import xgboost as xgb
        def make():
            return xgb.XGBRegressor(tree_method="hist", n_jobs=1, random_state=0,
                                    n_estimators=600, max_depth=6, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.4, reg_lambda=1.0)
    elif kind == "knn_tanimoto":
        def make():
            return KNeighborsRegressor(n_neighbors=params.get("k", 5),
                                       metric="jaccard", weights="distance", n_jobs=1)
    elif kind == "svr_tanimoto":
        def make():
            return SVR(kernel="precomputed", C=params.get("C", 10.0), epsilon=0.1)
    else:
        raise ValueError(kind)

    pred = np.zeros(len(y))
    per_fold = []
    for tr, te in folds:
        if kind == "svr_tanimoto":
            # Tanimoto kernel = |A n B| / (|A| + |B| - |A n B|) on binary rows.
            A, B = X[np.ix_(tr, tr)], X[np.ix_(te, tr)]
            Ktr = _tanimoto(A, A)
            Kte = _tanimoto(B, A)
            m = make().fit(Ktr, y[tr])
            pred[te] = m.predict(Kte)
        else:
            m = make().fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        per_fold.append(float(r2_score(y[te], pred[te])))
    return float(np.mean(per_fold)), per_fold, pred


def _tanimoto(A, B):
    inter = A @ B.T
    a = A.sum(axis=1, keepdims=True)
    b = B.sum(axis=1, keepdims=True)
    return inter / np.maximum(a + b.T - inter, 1e-9)


# --------------------------------------------------- existing arms (for pairing)
def load_neural(seed, split, arm3d=False):
    exp = (f"{DS}_exp_unimol2_lipids_84m_v5_checkpoint_1_30000.pt_{split}"
           f"_seed{seed}_lr5e-05_ep40")
    p = ROOT / "5_outputs/finetune_moltrain" / (exp + ("_3dconf" if arm3d else ""))
    if not (p / "cv.data").exists():
        return None
    z = np.asarray(joblib.load(p / "cv.data")).ravel()
    return joblib.load(p / "target_scaler.ss").inverse_transform(z.reshape(-1, 1)).ravel()


def load_traditional(seed, split):
    """Per-seed OOF. Sheet2 butina's seed-42 vector lives in the first npz and seed
    1337 in the '_more' one -- verified against the recorded per-fold R2 (0.7887 /
    0.7928, mean 0.7908 = the headline number). Sheet3 has no saved traditional
    arm, so it is recomputed live in build_arms (see 'traditional_recomp')."""
    cv = ROOT / f"5_outputs/traditional{'' if DS == 'Sheet2' else '_' + DS}_oof.npz"
    if cv.exists():
        d = np.load(cv)
        return d[f"pred_{seed}"] if f"pred_{seed}" in d else None
    if DS != "Sheet2":
        return None
    if split == "scaffold":
        p = ROOT / "5_outputs/traditional_oof.npz"
        d = np.load(p)
        return d[f"pred_{seed}"] if f"pred_{seed}" in d else None
    a = ROOT / "5_outputs/traditional_butina_oof.npz"
    b = ROOT / "5_outputs/traditional_butina_oof_more.npz"
    if seed == 42 and a.exists():
        return np.load(a)["pred"]
    if b.exists():
        d = np.load(b)
        return d[f"pred_{seed}"] if f"pred_{seed}" in d else None
    return None


# ------------------------------------------------------------------------- main
def build_arms(y, smis):
    """Fixed, untuned configurations spanning two axes of diversity: fingerprint
    family (what the model sees) and model family (how it fits)."""
    print("computing features...", flush=True)
    F = {
        "desc": _cache("rdkit_desc", lambda: rdkit_descriptors(smis)),
        "morgan_r2_2048": _cache("morgan_r2_2048", lambda: morgan(smis, 2, 2048)),
        "maccs": _cache("maccs", lambda: maccs(smis)),
        "torsion": _cache("torsion", lambda: torsion(smis)),
        "atompair": _cache("atompair", lambda: atompair(smis)),
    }
    for k, v in F.items():
        print(f"  {k:16s} {v.shape}", flush=True)

    h = np.hstack
    arms = {
        # model-family diversity on the known-good feature set
        "svr_rbf_desc_morgan": ("svr", h([F["desc"], F["morgan_r2_2048"]])),
        "krr_rbf_desc_morgan": ("krr", h([F["desc"], F["morgan_r2_2048"]])),
        "rf_desc_morgan": ("rf", h([F["desc"], F["morgan_r2_2048"]])),
        # similarity-kernel family: Tanimoto on ECFP4 -- maximally different
        # inductive bias from a boosted-tree model on the same bits
        "svr_tanimoto": ("svr_tanimoto", F["morgan_r2_2048"]),
        "knn_tanimoto": ("knn_tanimoto", F["morgan_r2_2048"]),
        # fingerprint-family diversity under a fixed model
        "xgb_maccs": ("xgb", F["maccs"]),
        "xgb_torsion": ("xgb", F["torsion"]),
        "xgb_atompair": ("xgb", F["atompair"]),
        # cross-combinations
        "svr_rbf_desc_maccs": ("svr", h([F["desc"], F["maccs"]])),
        "krr_rbf_torsion_atompair": ("krr", h([F["torsion"], F["atompair"]])),
    }
    # The 'traditional' incumbent has a saved OOF only for Sheet2. For any other
    # dataset, rebuild it with the EXACT config that won Sheet2's sweep
    # (desc+morgan_r3_2048 | xgb, n=1200/depth=8/lr=0.03/...). Reusing the frozen
    # config rather than re-tuning keeps this arm identical in kind across
    # datasets, and avoids tuning on folds we later score.
    if DS != "Sheet2":
        F["morgan_r3_2048"] = _cache("morgan_r3_2048", lambda: morgan(smis, 3, 2048))
        arms["traditional_recomp"] = ("xgb", h([F["desc"], F["morgan_r3_2048"]]))
    return arms


def evaluate(protocol):
    split = protocol
    seeds = (42, 1337) if protocol == "butina" else ALL_SEEDS
    # Composition is chosen on `sel` and confirmed on `cnf`, which is untouched by
    # the choice. Butina only has two partitions, so its confirm set is one
    # partition -- reported as a limitation rather than papered over.
    sel = [s for s in seeds if s in SELECT_SEEDS]
    cnf = [s for s in seeds if s not in SELECT_SEEDS]
    d = pd.read_excel(XLSX, sheet_name=DS)
    col = TARGET_COL
    if isinstance(col, tuple):
        col = next((c for c in col if c in d.columns), None)
        if col is None:
            raise SystemExit(f"{DS}: none of {TARGET_COL} in {list(d.columns)}")
    d = d.dropna(subset=["SMILES", col])
    smis = d["SMILES"].astype(str).str.strip().tolist()
    y = pd.to_numeric(d[col]).values.astype(float)
    if LOG10:
        y = np.log10(y)

    arms = build_arms(y, smis)

    # ---- per-seed OOF for every arm, plus the two incumbent arms ----
    # Cached to npz: computing the zoo costs ~6 min per protocol and the
    # analysis below is cheap to iterate on, and the OOF vectors are the actual
    # evidence behind any ensemble claim.
    cache_p = ROOT / f"5_outputs/arm_zoo_oof_{protocol}{'' if DS == 'Sheet2' else '_' + DS}.npz"
    store = {}
    t0 = time.time()
    if cache_p.exists():
        z = np.load(cache_p)
        for seed in seeds:
            store[seed] = {k[len(f"{seed}__"):]: z[k] for k in z.files
                           if k.startswith(f"{seed}__")}
        print(f"  loaded cached OOF ({cache_p.name})", flush=True)
    else:
        for seed in seeds:
            folds = folds_for(seed, split)
            cur = {}
            for name, (kind, X) in arms.items():
                _, pf, pred = oof(kind, X, y, folds)
                cur[name] = pred
                print(f"  [{protocol} seed{seed}] {name:24s} {np.mean(pf):+.4f}", flush=True)
            # neural_2d is required; 'traditional' may come from a saved OOF
            # (Sheet2) or from the recomputed arm (other datasets).
            p = load_neural(seed, split)
            if p is None:
                return None
            cur["neural_2d"] = p
            trad = load_traditional(seed, split)
            if trad is None:
                trad = cur.pop("traditional_recomp", None)
            if trad is None:
                return None
            cur["traditional"] = trad
            if seed in NEURAL_3D_SEEDS[split]:
                p3 = load_neural(seed, split, arm3d=True)
                if p3 is not None:
                    cur["neural_3d"] = p3
            store[seed] = cur
        np.savez(cache_p, **{f"{s}__{n}": v for s, d_ in store.items()
                             for n, v in d_.items()})
        print(f"  arm OOF done in {time.time()-t0:.0f}s -> {cache_p.name}", flush=True)

    # ---- independent strength + residual correlation, per arm ----
    # Only arms present on EVERY partition can be scored or ensembled; that is
    # what makes neural_3d ineligible under butina (only seed 42 was ever run).
    shared = [n for n in store[seeds[0]] if all(n in store[s] for s in seeds)]
    stats = {}
    for name in shared:
        rf = [r2_score(y[te], store[s][name][te])
              for s in seeds for _, te in folds_for(s, split)]
        rc = []
        for s in seeds:
            for base in BASE_ARMS:
                if base in store[s] and name != base:
                    rc.append(float(pearsonr(y - store[s][name], y - store[s][base])[0]))
        stats[name] = dict(r2=float(np.mean(rf)), r2_std=float(np.std(rf, ddof=1)),
                           n_folds=len(rf), resid_corr=float(np.mean(rc)) if rc else None)

    # a-priori admissibility: strong enough on its own AND decorrelated from the
    # incumbents. Both bars must clear -- either one alone is gameable.
    MIN_R2, MAX_CORR = 0.55, 0.70
    eligible = [n for n in shared
                if n not in BASE_ARMS and n != "neural_3d"
                and stats[n]["r2"] >= MIN_R2
                and stats[n]["resid_corr"] is not None
                and stats[n]["resid_corr"] <= MAX_CORR]
    eligible.sort(key=lambda n: stats[n]["resid_corr"])

    # ---- candidate ensembles: a SHORT menu, not a search ----
    # Greedy over a combinatorial space on 10 folds would fit the selection folds
    # rather than the problem. These five are declared up front instead.
    menu = {"base(2d+trad)": list(BASE_ARMS)}
    if eligible:
        menu["base+best_decorrelated"] = list(BASE_ARMS) + [eligible[0]]
        menu["base+all_eligible"] = list(BASE_ARMS) + eligible
    # neural_3d exists only on the partitions it was actually run on (all three
    # under scaffold, seed 42 only under butina), so it enters the menu only when
    # it is available everywhere -- otherwise the ensemble is not comparable.
    if all("neural_3d" in store[s] for s in seeds):
        menu["base+neural_3d"] = list(BASE_ARMS) + ["neural_3d"]

    def ens_fold_r2(names, seed):
        pred = np.mean([store[seed][n] for n in names], axis=0)
        return [r2_score(y[te], pred[te]) for _, te in folds_for(seed, split)]

    def folds_of(sset, names):
        return np.asarray([v for s in sset for v in ens_fold_r2(names, s)])

    # ---- report: selection folds and held-out confirm folds kept separate ----
    report = {"protocol": protocol, "seeds": list(seeds),
              "select_seeds": sel, "confirm_seeds": cnf,
              "min_r2": MIN_R2, "max_corr": MAX_CORR,
              "eligible": eligible, "arms": stats, "ensembles": {}}

    ref_sel = folds_of(sel, list(BASE_ARMS))
    ref_cnf = folds_of(cnf, list(BASE_ARMS)) if cnf else None
    for nm, names in menu.items():
        if not all(n in store[seeds[0]] for n in names):
            continue
        a_sel = folds_of(sel, names)
        e = {"arms": names, "select_mean": float(a_sel.mean()),
             "select_std": float(a_sel.std(ddof=1)),
             "per_fold": [round(v, 4) for v in folds_of(seeds, names)]}
        if nm != "base(2d+trad)":
            t, p = ttest_rel(a_sel, ref_sel)
            e.update(sel_delta=float(a_sel.mean() - ref_sel.mean()),
                     sel_p=float(p), sel_wins=int((a_sel > ref_sel).sum()),
                     sel_n=int(len(a_sel)))
        if cnf:
            a_cnf = folds_of(cnf, names)
            e.update(confirm_mean=float(a_cnf.mean()), confirm_std=float(a_cnf.std(ddof=1)))
            if nm != "base(2d+trad)" and ref_cnf is not None:
                t, p = ttest_rel(a_cnf, ref_cnf)
                e.update(confirm_delta=float(a_cnf.mean() - ref_cnf.mean()),
                         confirm_p=float(p), confirm_wins=int((a_cnf > ref_cnf).sum()),
                         confirm_n=int(len(a_cnf)))
        report["ensembles"][nm] = e
    return report


def main():
    out = {"datasets": DS, "select_seeds": list(SELECT_SEEDS),
           "confirm_seed": CONFIRM_SEED, "protocols": {}}
    for protocol in ("scaffold", "butina"):
        print(f"\n===== {protocol} =====", flush=True)
        r = evaluate(protocol)
        if r is None:
            print(f"  {protocol}: missing incumbent OOF, skipped")
            continue
        out["protocols"][protocol] = r

        print(f"\n  {'arm':24s} {'R2':>8} {'std':>7} {'resid_corr':>11}")
        for n, s in sorted(r["arms"].items(), key=lambda kv: -kv[1]["r2"]):
            rc = f"{s['resid_corr']:.4f}" if s["resid_corr"] is not None else "-"
            flag = "  <- eligible" if n in r["eligible"] else ""
            print(f"  {n:24s} {s['r2']:>8.4f} {s['r2_std']:>7.4f} {rc:>11}{flag}")
        print(f"\n  admissible: R2 >= {r['min_r2']} and resid_corr <= {r['max_corr']}")
        print(f"\n  {'ensemble':24s} {'select':>8} {'confirm':>8}  {'sel_delta':>9} "
              f"{'sel_p':>7}  {'cnf_delta':>9} {'cnf_p':>7}")
        for nm, e in r["ensembles"].items():
            cm = f"{e['confirm_mean']:.4f}" if "confirm_mean" in e else "-"
            sd = f"{e['sel_delta']:+.4f}" if "sel_delta" in e else "-"
            sp = f"{e['sel_p']:.4f}" if "sel_p" in e else "-"
            cd = f"{e['confirm_delta']:+.4f}" if "confirm_delta" in e else "-"
            cp = f"{e['confirm_p']:.4f}" if "confirm_p" in e else "-"
            print(f"  {nm:24s} {e['select_mean']:>8.4f} {cm:>8}  {sd:>9} {sp:>7}  "
                  f"{cd:>9} {cp:>7}")
        for nm, e in r["ensembles"].items():
            print(f"    {nm:24s} {e['arms']}")

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
