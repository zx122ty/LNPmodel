"""LUMI's headline comparison, reproduced on our bases.

The paper's claim is NOT that big models win overall -- it is:
  "XGBoost and the MLP on fingerprints achieved moderate performance overall but
   dropped sharply on the top 25% subset, underscoring their limited ability to
   prioritize high-performing candidates."           (paper, lines 260-263)

So the discriminating metric is top25_pearson, not full-set Pearson. Metric
definitions are transcribed from LUMI hg_mterics.py:88-126:
  * top-25% subset  = rows with target >= target.quantile(0.75)   (:91-92)
  * top25_pearson   = pearsonr on that subset                     (:118)
  * top25_agreement = |top25_by_target & top25_by_pred| / |top25_by_target|  (:97-101)
  * non_zero_*      = rows with target > 0                        (:103)
Aggregation follows LUMI: per-fold metric, then arithmetic mean over folds
(unimol_huggingface_kfold.py:370-382). Protocol = their sequential split.
"""
import sys, json, joblib
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator as rfg
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
RDLogger.DisableLog('rdApp.*')

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / '4_scripts'))
import lumi_protocol

df = pd.read_excel('/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx', sheet_name='Sheet2')
smiles = df['SMILES'].astype(str).tolist()
y = df['RLU (log2)'].astype(float).values
folds = lumi_protocol.sequential_folds(len(y), 5)

def lumi_metrics(t, p):
    """Verbatim port of the subset logic in hg_mterics.py:88-126."""
    t, p = np.asarray(t, float), np.asarray(p, float)
    thr = np.quantile(t, 0.75)
    m = t >= thr
    top_t, top_p = t[m], p[m]
    pred_thr = np.quantile(p, 0.75)
    agree = len(set(np.nonzero(m)[0]) & set(np.nonzero(p >= pred_thr)[0])) / max(m.sum(), 1)
    nz = t > 0
    return {
        'pearson': pearsonr(t, p)[0],
        'spearman': spearmanr(t, p).statistic,
        'top25_pearson': pearsonr(top_t, top_p)[0],
        'top25_spearman': spearmanr(top_t, top_p).statistic,
        'top25_agreement': agree,
        'non_zero_pearson': pearsonr(t[nz], p[nz])[0] if nz.sum() > 2 else np.nan,
    }

def eval_oof(pred, tag, store):
    per = [lumi_metrics(y[te], pred[te]) for _, te in folds]
    store[tag] = {k: float(np.mean([d[k] for d in per])) for k in per[0]}
    store[tag]['_top25_per_fold'] = [round(d['top25_pearson'], 3) for d in per]

# ---- neural bases: reuse saved CV predictions (z-scored; Pearson is affine-invariant)
DIRS = {'stock-84M':'Sheet2_exp_stock-84M_sequential_seed42_lr5e-05_ep40',
        'v2':'Sheet2_exp_unimol2_lipids_84m_v2_checkpoint_best.pt_sequential_seed42_lr5e-05_ep40',
        'v3':'Sheet2_exp_unimol2_lipids_84m_v3_checkpoint_best.pt_sequential_seed42_lr5e-05_ep40',
        'v4':'Sheet2_exp_unimol2_lipids_84m_v4_checkpoint_best.pt_sequential_seed42_lr5e-05_ep40'}
R = {}
for b, d in DIRS.items():
    eval_oof(np.asarray(joblib.load(REPO/'5_outputs/finetune_moltrain'/d/'cv.data')).ravel().astype(float), b, R)

# ---- classical baselines (LUMI's own comparison set)
mols = [Chem.MolFromSmiles(s) for s in smiles]
FP = np.array([np.array(rfg.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(m)) for m in mols], np.float32)
DFN = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, Descriptors.NumRotatableBonds,
       Descriptors.NumHDonors, Descriptors.NumHAcceptors, Descriptors.RingCount,
       Descriptors.FractionCSP3, Descriptors.HeavyAtomCount, Descriptors.NHOHCount, Descriptors.NOCount]
DESC = np.array([[f(m) for f in DFN] for m in mols], np.float32)

def run(make, X, tag):
    oof = np.zeros(len(y))
    for tr, te in folds:
        oof[te] = make().fit(X[tr], y[tr]).predict(X[te])
    eval_oof(oof, tag, R)

run(lambda: RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=8), np.hstack([FP, DESC]), 'ECFP4+desc RF')
run(lambda: xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             n_jobs=8, verbosity=0), DESC, 'XGBoost/desc (LUMI)')
def mlp_fp():
    from sklearn.pipeline import make_pipeline
    return make_pipeline(StandardScaler(with_mean=False),
                         MLPRegressor(hidden_layer_sizes=(512, 256), max_iter=400,
                                      early_stopping=True, random_state=42))
run(mlp_fp, FP, 'MLP/Morgan (LUMI)')

order = ['stock-84M','v2','v3','v4','ECFP4+desc RF','XGBoost/desc (LUMI)','MLP/Morgan (LUMI)']
print('='*94)
print('  LUMI 协议 (sequential 5-fold, fold-mean)，Sheet2 = LUMI 的 4CR-1920')
print('='*94)
print(f'  {"model":22} {"Pearson":>9} {"top25_P":>9} {"top25_rho":>10} {"top25_agree":>12} {"nonzero_P":>10}')
for k in order:
    m = R[k]
    print(f'  {k:22} {m["pearson"]:9.4f} {m["top25_pearson"]:9.4f} {m["top25_spearman"]:10.4f} '
          f'{m["top25_agreement"]:12.4f} {m["non_zero_pearson"]:10.4f}')
print()
print('  LUMI 论文声称: 指纹/XGBoost 全集"moderate"，但在 top-25% 上 "dropped sharply"')
nn = max(R[b]['top25_pearson'] for b in ['stock-84M','v2','v3','v4'])
cl = max(R[b]['top25_pearson'] for b in ['ECFP4+desc RF','XGBoost/desc (LUMI)','MLP/Morgan (LUMI)'])
print(f'  实测 top25_pearson: 最佳神经={nn:.4f}   最佳经典={cl:.4f}   差={nn-cl:+.4f}')
print(f'  v4 vs stock (top25): {R["v4"]["top25_pearson"]-R["stock-84M"]["top25_pearson"]:+.4f}')
print(f'  v4 per-fold top25 = {R["v4"]["_top25_per_fold"]}')
print(f'  stock per-fold top25 = {R["stock-84M"]["_top25_per_fold"]}')
json.dump(R, open(REPO/'5_outputs/repr_benchmark/top25_benchmark.json','w'), indent=1)
print('\nsaved -> 5_outputs/repr_benchmark/top25_benchmark.json')
