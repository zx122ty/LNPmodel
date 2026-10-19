"""Does the pretrained representation carry signal ECFP4 lacks (and vice versa)?

If concat(ECFP4, LipidFM) > both alone, the pretrained model has learned
something complementary -- the value is real but the [CLS]-only readout was
throwing it away.
"""
import json, hashlib, numpy as np, pandas as pd
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as rfg
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
RDLogger.DisableLog('rdApp.*')

REPO=Path('/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training')
df=pd.read_excel('/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx',sheet_name='Sheet2')
smiles=df['SMILES'].astype(str).tolist(); y=df['RLU (log2)'].astype(float).values
cached=json.loads((REPO/'5_outputs/split_cache/Sheet2_scaffold_k5_seed42.json').read_text())
assert cached['smiles_sha256']==hashlib.sha256("\n".join(smiles).encode()).hexdigest()[:16]
folds=[(np.array(f['train'],int),np.array(f['test'],int)) for f in cached['folds']]

gen=rfg.GetMorganGenerator(radius=2,fpSize=2048)
FP=np.array([np.array(gen.GetFingerprint(Chem.MolFromSmiles(s))) for s in smiles],dtype=np.float32)
C=Path('/tmp/atom_pool_cache')

def ridge(X,tag):
    r2s,sps=[],[]
    for tr,te in folds:
        sc=StandardScaler().fit(X[tr])
        m=RidgeCV(alphas=np.logspace(-2,4,13)).fit(sc.transform(X[tr]),y[tr])
        p=m.predict(sc.transform(X[te]))
        r2s.append(r2_score(y[te],p)); sps.append(spearmanr(y[te],p).statistic)
    print(f'  {tag:34} R2={np.mean(r2s):+.4f} +/- {np.std(r2s,ddof=1):.3f}  rho={np.mean(sps):.4f}')
    return float(np.mean(r2s)), r2s

print('=== Ridge, identical scaffold folds, Sheet2 (n=1920) ===')
base_r2,base_folds=ridge(FP,'ECFP4 alone')
res={}
for b in ['stock-84M','v2','v3','v4']:
    cls=np.load(C/f'{b}_cls.npy'); mean=np.load(C/f'{b}_mean.npy')
    print(f'--- {b} ---')
    ridge(cls,f'{b} CLS alone')
    both=np.hstack([cls,mean])
    r_both,_=ridge(both,f'{b} CLS+mean')
    r_fuse,f_folds=ridge(np.hstack([FP,cls,mean]),f'ECFP4 + {b} (CLS+mean)')
    res[b]=(r_fuse,f_folds)

print('\n=== fusion gain over ECFP4-alone (paired, per fold) ===')
from scipy.stats import ttest_rel
for b,(r,ff) in res.items():
    d=np.array(ff)-np.array(base_folds)
    t,p=ttest_rel(ff,base_folds)
    print(f'  ECFP4+{b:10} {r:+.4f}  gain={r-base_r2:+.4f}  per-fold delta={np.round(d,3)}  paired p={p:.3f}')
