"""强基线上限：全套 RDKit 描述符 + ECFP4，XGBoost/RF + 有限超参调优。

回答：经典基线（含完整 RDKit 描述符、CPFP4、XGBoost/RF）调优后能到多高？预训练还剩多少净增益？
协议 = LUMI sequential 5-fold (Sheet2 = 4CR-1920)，指标 = LUMI 口径（PCC / top25 PCC / top25 agree / rho）。
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
import rdkit.Chem.rdFingerprintGenerator as rfg
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid
import xgboost as xgb
RDLogger.DisableLog('rdApp.*')

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / '4_scripts'))
import lumi_protocol

df = pd.read_excel('/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx', sheet_name='Sheet2')
smiles = df['SMILES'].astype(str).tolist()
y = df['RLU (log2)'].astype(float).values
folds = lumi_protocol.sequential_folds(len(y), 5)

mols = [Chem.MolFromSmiles(s) for s in smiles]

# 全部 RDKit 描述符
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
D_desc = np.nan_to_num(np.array(D_desc, float), nan=np.nanmedian(np.array(D_desc, float), axis=0))
print('RDKit descriptors:', D_desc.shape)

gen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
D_ecfp = np.array([np.array(gen.GetFingerprint(m)) for m in mols], np.float32)
print('ECFP4:', D_ecfp.shape)

# 三个特征集
FEAT = {
    'ECFP4':          D_ecfp,
    'desc217':        D_desc,
    'ECFP4+desc217':  np.hstack([D_ecfp, D_desc]),
}

def metrics(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    thr = np.quantile(t, 0.75); sel = t >= thr
    pred_thr = np.quantile(p, 0.75)
    agree = len(set(np.nonzero(sel)[0]) & set(np.nonzero(p >= pred_thr)[0])) / max(sel.sum(), 1)
    return {'pearson': pearsonr(t, p)[0],
            'top25_pearson': pearsonr(t[sel], p[sel])[0],
            'top25_agreement': float(agree),
            'spearman': spearmanr(t, p).statistic}

def cv_eval(X, make, tag):
    oof = np.zeros(len(y))
    for tr, te in folds:
        oof[te] = make().fit(X[tr], y[tr]).predict(X[te])
    per = [metrics(y[te], oof[te]) for _, te in folds]
    m = {k: float(np.mean([d[k] for d in per])) for k in per[0]}
    m['per_fold_agree'] = [round(d['top25_agreement'], 3) for d in per]
    print(f'  {tag:38} PCC={m["pearson"]:.4f}  top25_P={m["top25_pearson"]:.4f}  '
          f'agree={m["top25_agreement"]:.4f}  rho={m["spearman"]:.4f}  folds={m["per_fold_agree"]}')
    return m

results = {}
print('\n=== 强基线上限（Sheet2=LUMI 4CR-1920, sequential 5-fold）===\n')

# ---- XGBoost：各特征集 × 若干超参组合，按 top25_agreement 选最佳 ----
xgb_grid = {'eta': [0.05, 0.1], 'max_depth': [4, 6],
            'colsample_bytree': [0.6, 0.9], 'subsample': [0.8],
            'gamma': [0, 0.1]}
print('--- XGBoost ---')
for fname, X in FEAT.items():
    best_m, best_cfg = None, None
    for params in ParameterGrid(xgb_grid):
        def make(p=params):
            return xgb.XGBRegressor(n_estimators=400, learning_rate=p['eta'],
                                    max_depth=p['max_depth'], subsample=p['subsample'],
                                    colsample_bytree=p['colsample_bytree'], gamma=p['gamma'],
                                    random_state=42, n_jobs=8, verbosity=0)
        m = cv_eval(X, make, f'{fname} XGB {params}')
        if best_m is None or m['top25_agreement'] > best_m['top25_agreement']:
            best_m, best_cfg = m, params
    results[f'XGB_{fname}'] = best_m
    results[f'XGB_{fname}_cfg'] = best_cfg
    print(f'    best: {best_cfg} -> agree={best_m["top25_agreement"]:.4f}\n')

# ---- RandomForest ----
rf_grids = {'n': [300], 'depth': [None, 20], 'leaf': [1, 2], 'feat': [None, 0.5]}
print('--- RandomForest ---')
for fname, X in FEAT.items():
    best_m, best_cfg = None, None
    for params in ParameterGrid(rf_grids):
        def make(p=params):
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(n_estimators=p['n'], max_depth=p['depth'],
                                         min_samples_leaf=p['leaf'],
                                         max_features=p['feat'], random_state=42, n_jobs=8)
        m = cv_eval(X, make, f'{fname} RF {params}')
        if best_m is None or m['top25_agreement'] > best_m['top25_agreement']:
            best_m, best_cfg = m, params
    results[f'RF_{fname}'] = best_m
    results[f'RF_{fname}_cfg'] = best_cfg
    print(f'    best: {best_cfg} -> agree={best_m["top25_agreement"]:.4f}\n')

print('\n=== 最终汇总（最佳 top25_agreement，均已调参）===')
for k in results:
    if isinstance(results[k], dict) and 'per_fold_agree' in results and k.endswith('_cfg') is False:
        pass
print('  表见上方；每个 best 均已打印')
out = REPO / '5_outputs/repr_benchmark' / 'strong_baseline.json'
out.write_text(json.dumps(results, indent=1))
print('\nsaved ->', out)