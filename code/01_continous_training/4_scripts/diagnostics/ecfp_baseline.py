"""ECFP4 + descriptor baseline on the SAME cached scaffold folds."""
import json, hashlib, numpy as np, pandas as pd
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdFingerprintGenerator as rfg
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
RDLogger.DisableLog('rdApp.*')

XLSX='/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx'
df=pd.read_excel(XLSX,sheet_name='Sheet2')
smiles=df['SMILES'].astype(str).tolist(); y=df['RLU (log2)'].astype(float).values
cached=json.loads(Path('5_outputs/split_cache/Sheet2_scaffold_k5_seed42.json').read_text())
assert cached['smiles_sha256']==hashlib.sha256("\n".join(smiles).encode()).hexdigest()[:16]
folds=[(np.array(f['train'],int),np.array(f['test'],int)) for f in cached['folds']]

mols=[Chem.MolFromSmiles(s) for s in smiles]
gen=rfg.GetMorganGenerator(radius=2,fpSize=2048)
X_fp=np.array([np.array(gen.GetFingerprint(m)) for m in mols],dtype=np.float32)
desc_fns=[Descriptors.MolWt,Descriptors.MolLogP,Descriptors.TPSA,Descriptors.NumRotatableBonds,
          Descriptors.NumHDonors,Descriptors.NumHAcceptors,Descriptors.RingCount,
          Descriptors.FractionCSP3,Descriptors.HeavyAtomCount,Descriptors.NHOHCount,Descriptors.NOCount]
X_d=np.array([[f(m) for f in desc_fns] for m in mols],dtype=np.float32)
X_cat=np.hstack([X_fp,X_d])

def ev(X,name):
    for head in ('ridge','rf'):
        r2s,sps=[],[]
        for tr,te in folds:
            if head=='ridge':
                sc=StandardScaler().fit(X[tr])
                m=RidgeCV(alphas=np.logspace(-2,4,13)).fit(sc.transform(X[tr]),y[tr])
                p=m.predict(sc.transform(X[te]))
            else:
                m=RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=8).fit(X[tr],y[tr])
                p=m.predict(X[te])
            r2s.append(r2_score(y[te],p)); sps.append(spearmanr(y[te],p).statistic)
        print(f'{name:22} {head:6} R2={np.mean(r2s):+.4f} +/- {np.std(r2s,ddof=1):.3f}  rho={np.mean(sps):.4f}  per-fold={np.round(r2s,3)}')
print('=== ECFP4 / descriptor baselines, SAME scaffold folds, Sheet2 ===')
ev(X_fp,'ECFP4-2048'); ev(X_d,'11 descriptors'); ev(X_cat,'ECFP4+desc')
