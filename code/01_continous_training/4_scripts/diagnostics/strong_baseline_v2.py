"""经典强基线 on Sheet3（不覆盖 Sheet2 结果）。log10 标准化 target。输出新文件。"""
import sys, json, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
import rdkit.Chem.rdFingerprintGenerator as rfg
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
RDLogger.DisableLog('rdApp.*')

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / '4_scripts'))
import lumi_protocol

# ---- Sheet3：raw transfection_efficiency -> log10 标准化 ----
df = pd.read_excel('/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx', sheet_name='Sheet3')
smiles = df['SMILES'].astype(str).tolist()
y = np.log10(df['transfection_efficiency'].astype(float).values)   # 先标准化
print('Sheet3 n=%d  target=log10(efficiency) sd=%.4f' % (len(y), y.std()))


def make_folds(split):
    if split == 'sequential':
        return lumi_protocol.sequential_folds(len(y), 5)
    cached = json.load(open(REPO / '5_outputs/split_cache' / 'Sheet3_scaffold_k5_seed42.json'))
    return [(np.array(f['train'], int), np.array(f['test'], int)) for f in cached['folds']]


mols = [Chem.MolFromSmiles(s) for s in smiles]
desc_fn = [(n, f) for n, f in Descriptors._descList]
D_desc = []
for m in mols:
    row = []
    for _n, f in desc_fn:
        try:
            row.append(f(m))
        except Exception:
            row.append(np.nan)
    D_desc.append(row)
D_desc = np.array(D_desc, float)
gen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
D_ecfp = np.array([np.array(gen.GetFingerprint(m)) for m in mols], np.float32)


def finite_fill(X):
    X = np.asarray(X, float)
    colmed = np.nanmedian(X, axis=0)
    colmed = np.where(np.isfinite(colmed), colmed, 0.0)
    X = np.where(np.isfinite(X), X, colmed)
    # clamp pathological magnitudes (e.g. Ipc ~1e55 on these highly-branched lipids)
    for j in range(X.shape[1]):
        lo, hi = np.nanpercentile(X[:, j], 1), np.nanpercentile(X[:, j], 99)
        X[:, j] = np.clip(X[:, j], lo - 5 * (hi - lo + 1e-12), hi + 5 * (hi - lo + 1e-12))
    X = np.where(np.isfinite(X), X, 0.0)
    return X.astype(np.float32)


D_desc = finite_fill(D_desc)
D_ecfp = finite_fill(D_ecfp)
FEAT = {'ECFP4': D_ecfp, 'desc217': D_desc, 'ECFP4+desc217': finite_fill(np.hstack([D_ecfp, D_desc]))}


def metrics(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    thr = np.quantile(t, 0.75); sel = t >= thr; pth = np.quantile(p, 0.75)
    agree = len(set(np.nonzero(sel)[0]) & set(np.nonzero(p >= pth)[0])) / max(sel.sum(), 1)
    return {'pearson': pearsonr(t, p)[0], 'top25_pearson': pearsonr(t[sel], p[sel])[0],
            'top25_agreement': float(agree), 'spearman': spearmanr(t, p).statistic}


def cv_eval(X, folds, make, tag):
    oof = np.zeros(len(y))
    for tr, te in folds:
        oof[te] = make().fit(X[tr], y[tr]).predict(X[te])
    per = [metrics(y[te], oof[te]) for _, te in folds]
    m = {k: float(np.mean([d[k] for d in per])) for k in per[0]}
    m['folds'] = [round(d['top25_agreement'], 3) for d in per]
    print(f'  {tag:30} PCC={m["pearson"]:.4f} top25P={m["top25_pearson"]:.4f} agree={m["top25_agreement"]:.4f} rho={m["spearman"]:.4f}')
    return m


xgb_grid = {'eta': [0.05, 0.1], 'max_depth': [4], 'colsample_bytree': [0.6, 0.9], 'subsample': [0.8]}
results = {}
for split_name in ['sequential', 'scaffold']:
    folds = make_folds(split_name)
    print(f'\n=== Sheet3 {split_name} 经典强基线 ===')
    for fn, X in FEAT.items():
        best_m, best = None, None
        for e, d, c in itertools.product(xgb_grid['eta'], xgb_grid['max_depth'], xgb_grid['colsample_bytree']):
            def make(e=e, d=d, c=c):
                return xgb.XGBRegressor(n_estimators=400, learning_rate=e, max_depth=d,
                                        subsample=0.8, colsample_bytree=c, random_state=42,
                                        n_jobs=8, verbosity=0)
            m = cv_eval(X, folds, make, f'{fn} XGB{e,d,c}')
            if best_m is None or m['top25_agreement'] > best_m['top25_agreement']:
                best_m, best = m, (e, d, c)
        results[f'{split_name}|{fn}'] = best_m
        results[f'{split_name}|{fn}_cfg'] = best
        print(f'    best: {best} agree={best_m["top25_agreement"]:.4f}')

for split_name in ['sequential', 'scaffold']:
    folds = make_folds(split_name)
    print(f'\n=== Sheet3 {split_name} RF ===')
    for fn, X in FEAT.items():
        m = cv_eval(X, folds, lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                            random_state=42, n_jobs=8), f'{fn} RF')
        results[f'{split_name}|{fn}_RF'] = m

out = REPO / '5_outputs/repr_benchmark' / 'strong_baseline_SHEET3.json'
out.write_text(json.dumps({k: (v if not isinstance(v, tuple) else list(v)) for k, v in results.items()}, indent=1))
print('\nsaved ->', out)