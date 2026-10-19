"""Is [CLS] the bottleneck? Compare [CLS] vs mean-pooled ATOM representations.

Rationale: unimol2.py's loss has three terms (masked_token / coord / dist), all
computed on ATOM positions. The [CLS] token appears in the loss file only at
line 206, inside logging. It therefore receives no direct pretraining gradient,
while atom representations do. If mean-pooled atoms beat [CLS], the downstream
readout -- not the pretraining corpus -- is what caps performance.
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, torch

REPO = Path('/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training')
sys.path.insert(0, str(REPO / '4_scripts/Uni-Mol/unimol_tools'))
import unimol_tools.models.unimolv2 as v2mod
from unimol_tools import UniMolRepr
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

BASES = {
 'stock-84M': None,
 'v2': REPO/'2_models/continued_pretrain/unimol2_lipids_84m_v2/checkpoints/checkpoint_best.pt',
 'v3': REPO/'2_models/continued_pretrain/unimol2_lipids_84m_v3/checkpoints/checkpoint_best.pt',
 'v4': REPO/'2_models/continued_pretrain/unimol2_lipids_84m_v4/checkpoints/checkpoint_best.pt',
}
_STOCK = v2mod.UniMolV2Model.load_pretrained_weights
def install(ckpt):
    if ckpt is None:
        v2mod.UniMolV2Model.load_pretrained_weights = _STOCK; return
    def _load(self, path=None, strict=False):
        sd = torch.load(ckpt, map_location='cpu', weights_only=False)
        sd = sd.get('model', sd)
        missing,_ = self.load_state_dict(sd, strict=False)
        bad=[k for k in missing if not k.startswith('classification_head')]
        assert not bad, f'{len(bad)} backbone keys missing'
    v2mod.UniMolV2Model.load_pretrained_weights = _load

df = pd.read_excel('/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx', sheet_name='Sheet2')
smiles = df['SMILES'].astype(str).tolist(); y = df['RLU (log2)'].astype(float).values
cached = json.loads((REPO/'5_outputs/split_cache/Sheet2_scaffold_k5_seed42.json').read_text())
assert cached['smiles_sha256'] == hashlib.sha256("\n".join(smiles).encode()).hexdigest()[:16]
folds = [(np.array(f['train'],int), np.array(f['test'],int)) for f in cached['folds']]

CACHE = Path('/tmp/atom_pool_cache'); CACHE.mkdir(exist_ok=True)

def get_feats(base):
    fc, fm = CACHE/f'{base}_cls.npy', CACHE/f'{base}_mean.npy'
    if fc.exists() and fm.exists():
        return np.load(fc), np.load(fm)
    install(BASES[base])
    m = UniMolRepr(data_type='molecule', model_name='unimolv2', model_size='84m',
                   remove_hs=False, use_cuda=torch.cuda.is_available())
    m.model.eval()
    with torch.no_grad():
        out = m.get_repr(smiles, return_atomic_reprs=True)
    cls = np.asarray(out['cls_repr'], dtype=np.float32)
    mean = np.stack([np.asarray(a, dtype=np.float32).mean(0) for a in out['atomic_reprs']])
    np.save(fc, cls); np.save(fm, mean)
    return cls, mean

def ev(X, tag):
    res={}
    for head in ('ridge','rf'):
        r2s,sps=[],[]
        for tr,te in folds:
            if head=='ridge':
                sc=StandardScaler().fit(X[tr])
                mo=RidgeCV(alphas=np.logspace(-2,4,13)).fit(sc.transform(X[tr]),y[tr])
                p=mo.predict(sc.transform(X[te]))
            else:
                mo=RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=8).fit(X[tr],y[tr])
                p=mo.predict(X[te])
            r2s.append(r2_score(y[te],p)); sps.append(spearmanr(y[te],p).statistic)
        res[head]=(np.mean(r2s),np.std(r2s,ddof=1),np.mean(sps))
        print(f'  {tag:22} {head:6} R2={np.mean(r2s):+.4f} +/- {np.std(r2s,ddof=1):.3f}  rho={np.mean(sps):.4f}')
    return res

print('=== [CLS] vs mean-pooled ATOM reprs, same scaffold folds, Sheet2 ===')
summary={}
for b in BASES:
    print(f'\n--- {b} ---')
    cls, mean = get_feats(b)
    Xc = cls - cls.mean(0); ev_c = np.linalg.svd(Xc, compute_uv=False)**2; ev_c/=ev_c.sum()
    Xm = mean - mean.mean(0); ev_m = np.linalg.svd(Xm, compute_uv=False)**2; ev_m/=ev_m.sum()
    er = lambda e: float(np.exp(-(e*np.log(e+1e-12)).sum()))
    print(f'  eff_rank: CLS={er(ev_c):.2f}  mean-atom={er(ev_m):.2f}')
    summary[b]={'cls':ev(cls,'CLS'),'mean':ev(mean,'mean-atom')}
print('\n=== SUMMARY (Ridge R2) ===')
print(f'{"base":10} {"CLS":>9} {"mean-atom":>11} {"delta":>8}')
for b,v in summary.items():
    c,m=v['cls']['ridge'][0],v['mean']['ridge'][0]
    print(f'{b:10} {c:+9.4f} {m:+11.4f} {m-c:+8.4f}')
print(f'\n{"base":10} {"CLS-rf":>9} {"mean-rf":>11} {"delta":>8}')
for b,v in summary.items():
    c,m=v['cls']['rf'][0],v['mean']['rf'][0]
    print(f'{b:10} {c:+9.4f} {m:+11.4f} {m-c:+8.4f}')
