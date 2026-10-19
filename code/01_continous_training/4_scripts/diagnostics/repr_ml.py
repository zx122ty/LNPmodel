"""大模型 [CLS] 表征 + 常规 ML (XGBoost/RF) 混合，验证是否带来提升。

这个其实是"用大模型抽特征的 OpenReview 必考题"：
* 把 4 个底座的冻结 [CLS] 768-dim 特征直接喂给 XGBoost/RF（不用 learned head）
* 对照：a) ECFP4 基线（我们已证明是最强经典）；b) 端到端全参微调数字
* 也测"ECFP4 + CLS"拼接，看大模型表征是否携带互补信息
* 协议 = LUMI sequential 5-fold，指标 = PCC / top25_PCC / top25 agreement
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as rfg
RDLogger.DisableLog('rdApp.*')

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / '4_scripts'))
import lumi_protocol

df = pd.read_excel('/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx', sheet_name='Sheet2')
smiles = df['SMILES'].astype(str).tolist()
y = df['RLU (log2)'].astype(float).values
folds = lumi_protocol.sequential_folds(len(y), 5)

FEAT = REPO / '5_outputs/repr_benchmark/features'
BASES = ['stock-84M', 'v2', 'v3', 'v4']

def metrics(t, p):
    t, p = np.asarray(t,float), np.asarray(p,float)
    thr=np.quantile(t,0.75); m=t>=thr; pth=np.quantile(p,0.75)
    agree=len(set(np.nonzero(m)[0])&set(np.nonzero(p>=pth)[0]))/max(m.sum(),1)
    return {'pearson':pearsonr(t,p)[0],'top25_pearson':pearsonr(t[m],p[m])[0],
            'top25_agreement':float(agree),'spearman':spearmanr(t,p).statistic}

def cv(X, y, folds, make):
    oof=np.zeros(len(y))
    for tr,te in folds: oof[te]=make().fit(X[tr],y[tr]).predict(X[te])
    per=[metrics(y[te],oof[te]) for _,te in folds]
    m={k:float(np.mean([d[k] for d in per])) for k in per[0]}
    m['folds']=[round(d['top25_agreement'],3) for d in per]
    return m

def xgb_default(**kw):
    return lambda: xgb.XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                                    subsample=0.8, colsample_bytree=0.6, random_state=42,
                                    n_jobs=8, verbosity=0, **kw)
def rf_default(**kw):
    return lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=8, **kw)

# ECFP4 (reference classic baseline)
mols=[Chem.MolFromSmiles(s) for s in smiles]
gen=rfg.GetMorganGenerator(radius=2,fpSize=2048)
FP=np.array([np.array(gen.GetFingerprint(m)) for m in mols],np.float32)

print('=== 大模型[CLS]表征 + 常规ML vs 基线（Sheet2=LUMI 4CR-1920, sequential）===\n')
ref_xg=cv(FP, y, folds, xgb_default()); print(f'  ECFP4-XGB(参考)         PCC={ref_xg["pearson"]:.4f} top25P={ref_xg["top25_pearson"]:.4f} agree={ref_xg["top25_agreement"]:.4f}')
print()

print('=== 大模型[CLS]表征 + 常规ML vs 基线（Sheet2=LUMI 4CR-1920, sequential）===\n')
ref_xg=cv(FP, y, folds, xgb_default()); print(f'  ECFP4-XGB(参考)         PCC={ref_xg["pearson"]:.4f} top25P={ref_xg["top25_pearson"]:.4f} agree={ref_xg["top25_agreement"]:.4f}')
print()

# 每个底座 CLS 单独 + 与 ECFP4 拼接
per_cls={}
for b in BASES:
    cls=np.load(FEAT/f'{b}_Sheet2_d3a3687c.npy').astype(np.float32)
    for X, tag in [(cls, f'{b}-CLS+XGB'), (np.hstack([FP, cls]), f'{b}-CLS+ECFP4-XGB')]:
        m=cv(X, y, folds, xgb_default())
        per_cls[tag]=m
        print(f'  {tag:28} PCC={m["pearson"]:.4f} top25P={m["top25_pearson"]:.4f} agree={m["top25_agreement"]:.4f}')

import pathlib
json.dump({k:{kk:vv for kk,vv in v.items()} for k,v in per_cls.items()},
          open(REPO/'5_outputs/repr_benchmark/repr_ml.json','w'), indent=1)
print('\ndone -> 5_outputs/repr_benchmark/repr_ml.json')