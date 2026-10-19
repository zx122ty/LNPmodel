"""Two tests:
(A) Paired significance on mean-pooled ATOM reprs (the ones that DO receive
    pretraining gradients), stock vs each continued-pretrain base.
(B) Weight drift from stock -- direct magnitude of catastrophic forgetting.
"""
import json, hashlib, numpy as np, pandas as pd, torch
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import ttest_rel

REPO=Path('/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training')
df=pd.read_excel('/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx',sheet_name='Sheet2')
smiles=df['SMILES'].astype(str).tolist(); y=df['RLU (log2)'].astype(float).values
cached=json.loads((REPO/'5_outputs/split_cache/Sheet2_scaffold_k5_seed42.json').read_text())
folds=[(np.array(f['train'],int),np.array(f['test'],int)) for f in cached['folds']]
C=Path('/tmp/atom_pool_cache')

def perfold(X):
    r2s=[]
    for tr,te in folds:
        sc=StandardScaler().fit(X[tr])
        m=RidgeCV(alphas=np.logspace(-2,4,13)).fit(sc.transform(X[tr]),y[tr])
        r2s.append(r2_score(y[te],m.predict(sc.transform(X[te]))))
    return np.array(r2s)

print('=== (A) mean-pooled ATOM repr, Ridge, paired vs stock ===')
ref=perfold(np.load(C/'stock-84M_mean.npy'))
print(f'  stock-84M  R2={ref.mean():+.4f}  per-fold={np.round(ref,3)}')
for b in ['v2','v3','v4']:
    r=perfold(np.load(C/f'{b}_mean.npy'))
    t,p=ttest_rel(r,ref)
    print(f'  {b:10} R2={r.mean():+.4f}  per-fold={np.round(r,3)}  delta={r.mean()-ref.mean():+.4f}  paired p={p:.4f}')

print('\n=== (B) weight drift from stock (relative L2 per param group) ===')
STOCK=Path('/home/data/zhengxiao/disk1/my_project/Uni-Mol2_Toxicity_Prediction/4_scripts/Uni-Mol/unimol_tools/unimol_tools/weights/modelzoo/84M/checkpoint.pt')
sd0=torch.load(STOCK,map_location='cpu',weights_only=False); sd0=sd0.get('model',sd0)
CK={b:REPO/f'2_models/continued_pretrain/unimol2_lipids_84m_{b}/checkpoints/checkpoint_best.pt' for b in ['v2','v3','v4']}
print(f'{"base":6} {"global":>9} {"embed":>9} {"encoder":>9} {"head/pred":>10}')
for b,p in CK.items():
    if not p.exists(): print(f'  {b}: MISSING {p}'); continue
    sd=torch.load(p,map_location='cpu',weights_only=False); sd=sd.get('model',sd)
    num=den=0.0; grp={}
    for k,v in sd0.items():
        if k not in sd or not torch.is_floating_point(v): continue
        d=(sd[k].float()-v.float()).norm().item()**2; n=v.float().norm().item()**2
        num+=d; den+=n
        g=('embed' if 'embed' in k else 'encoder' if k.startswith('encoder') else 'head/pred')
        a,bb=grp.get(g,(0.0,0.0)); grp[g]=(a+d,bb+n)
    f=lambda g:(np.sqrt(grp[g][0]/grp[g][1])*100 if g in grp else float('nan'))
    print(f'{b:6} {np.sqrt(num/den)*100:8.2f}% {f("embed"):8.2f}% {f("encoder"):8.2f}% {f("head/pred"):9.2f}%')
