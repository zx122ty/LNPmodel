import json, hashlib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

REPO=Path('/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training')
df=pd.read_excel('/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx',sheet_name='Sheet2')
smiles=df['SMILES'].astype(str).tolist(); y=df['RLU (log2)'].astype(float).values
cached=json.loads((REPO/'5_outputs/split_cache/Sheet2_scaffold_k5_seed42.json').read_text())
assert cached['smiles_sha256']==hashlib.sha256("\n".join(smiles).encode()).hexdigest()[:16]
folds=[(np.array(f['train'],int),np.array(f['test'],int)) for f in cached['folds']]
C=Path('/tmp/atom_pool_cache')
er=lambda X:(lambda e: float(np.exp(-(e*np.log(e+1e-12)).sum())))((lambda s: s/s.sum())(np.linalg.svd(X-X.mean(0),compute_uv=False)**2))

def ev(X):
    o={}
    for head in ('ridge','rf'):
        r2s,sps=[],[]
        for tr,te in folds:
            if head=='ridge':
                sc=StandardScaler().fit(X[tr]); m=RidgeCV(alphas=np.logspace(-2,4,13)).fit(sc.transform(X[tr]),y[tr]); p=m.predict(sc.transform(X[te]))
            else:
                m=RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=4).fit(X[tr],y[tr]); p=m.predict(X[te])
            r2s.append(r2_score(y[te],p)); sps.append(spearmanr(y[te],p).statistic)
        o[head]=(float(np.mean(r2s)),float(np.std(r2s,ddof=1)),float(np.mean(sps)))
    return o

bases=['stock-84M','v2','v3','v4']; S={}
print('=== effective rank (768-dim space) ===')
for b in bases:
    cls=np.load(C/f'{b}_cls.npy'); mean=np.load(C/f'{b}_mean.npy')
    print(f'  {b:10}  CLS={er(cls):5.2f}   mean-atom={er(mean):6.2f}')
    S[b]={'CLS':ev(cls),'mean-atom':ev(mean)}
print('\n=== fold-mean R2 on identical scaffold folds (Sheet2) ===')
print(f'{"base":10} | {"CLS ridge":>10} {"CLS rf":>8} | {"MEAN ridge":>11} {"MEAN rf":>9}')
print('-'*58)
for b in bases:
    c,m=S[b]['CLS'],S[b]['mean-atom']
    print(f'{b:10} | {c["ridge"][0]:+10.4f} {c["rf"][0]:+8.4f} | {m["ridge"][0]:+11.4f} {m["rf"][0]:+9.4f}')
print('\n=== Spearman rho ===')
for b in bases:
    c,m=S[b]['CLS'],S[b]['mean-atom']
    print(f'{b:10} | CLS ridge {c["ridge"][2]:.4f}  rf {c["rf"][2]:.4f} | MEAN ridge {m["ridge"][2]:.4f}  rf {m["rf"][2]:.4f}')
print('\n=== best mean-atom config per base (with std) ===')
for b in bases:
    m=S[b]['mean-atom']
    bh='ridge' if m['ridge'][0]>=m['rf'][0] else 'rf'
    print(f'  {b:10} {bh:6} R2={m[bh][0]:+.4f} +/- {m[bh][1]:.3f}')
json.dump(S,open('/tmp/pool_results.json','w'),indent=1)
