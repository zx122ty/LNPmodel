"""Head-to-head: does the SAME base rank differently under the two protocols?

Paired per-fold tests (each base vs stock, on identical folds) under
  (a) scaffold  = extrapolation to unseen Bemis-Murcko scaffolds  (ours)
  (b) sequential = LUMI's --kfold-mode default = combinatorial interpolation
Frozen [CLS] features, cached; heads = RidgeCV / RF / LUMI-style 3-layer MLP.
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, ttest_rel, wilcoxon

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / '4_scripts'))
import lumi_protocol

FEAT = REPO / '5_outputs/repr_benchmark/features'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
BASES = ['stock-84M', 'v2', 'v3', 'v4']

df = pd.read_excel('/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx', sheet_name='Sheet2')
smiles = df['SMILES'].astype(str).tolist()
y = df['RLU (log2)'].astype(float).values.astype(np.float32)
cached = json.loads((REPO/'5_outputs/split_cache/Sheet2_scaffold_k5_seed42.json').read_text())
assert cached['smiles_sha256'] == hashlib.sha256("\n".join(smiles).encode()).hexdigest()[:16]
PROTO = {
    'scaffold':   [(np.array(f['train'], int), np.array(f['test'], int)) for f in cached['folds']],
    'sequential': lumi_protocol.sequential_folds(len(smiles), 5),
}

def ridge_pf(X, folds):
    o = []
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), y[tr])
        o.append(r2_score(y[te], m.predict(sc.transform(X[te]))))
    return np.array(o)

def rf_pf(X, folds):
    o = []
    for tr, te in folds:
        m = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=8).fit(X[tr], y[tr])
        o.append(r2_score(y[te], m.predict(X[te])))
    return np.array(o)

def mlp_pf(X, folds, seeds=(0, 1, 2)):
    """LUMI's frozen-backbone head: 3-layer MLP, inner_dim 2048 (hf_unimol.py:113-116)."""
    out = []
    for tr, te in folds:
        preds = []
        for sd in seeds:
            rng = np.random.RandomState(sd); perm = rng.permutation(len(tr))
            nv = int(len(tr) * 0.15); va_i, tr_i = tr[perm[:nv]], tr[perm[nv:]]
            sc = StandardScaler().fit(X[tr_i]); mu, sg = y[tr_i].mean(), y[tr_i].std() + 1e-8
            T = lambda i: torch.tensor(sc.transform(X[i]), dtype=torch.float32, device=DEV)
            Xtr, Xva, Xte = T(tr_i), T(va_i), T(te)
            ytr = torch.tensor((y[tr_i]-mu)/sg, dtype=torch.float32, device=DEV).unsqueeze(1)
            yva = torch.tensor((y[va_i]-mu)/sg, dtype=torch.float32, device=DEV).unsqueeze(1)
            torch.manual_seed(sd)
            net = nn.Sequential(nn.Linear(X.shape[1], 2048), nn.GELU(),
                                nn.Linear(2048, 2048), nn.GELU(), nn.Linear(2048, 1)).to(DEV)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-2)
            best, bstate, wait = 1e9, None, 0
            for _ in range(300):
                net.train(); opt.zero_grad()
                nn.functional.mse_loss(net(Xtr), ytr).backward(); opt.step()
                net.eval()
                with torch.no_grad(): v = nn.functional.mse_loss(net(Xva), yva).item()
                if v < best - 1e-5:
                    best, wait = v, 0
                    bstate = {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= 40: break
            net.load_state_dict(bstate); net.eval()
            with torch.no_grad(): preds.append(net(Xte).cpu().numpy().ravel()*sg + mu)
        out.append(r2_score(y[te], np.mean(preds, axis=0)))
    return np.array(out)

HEADS = {'ridge': ridge_pf, 'rf': rf_pf, 'mlp': mlp_pf}
R = {}
for pname, folds in PROTO.items():
    for hname, fn in HEADS.items():
        for b in BASES:
            R[(pname, hname, b)] = fn(np.load(FEAT/f'{b}_Sheet2_d3a3687c.npy'), folds)

for pname in PROTO:
    print(f'\n{"="*74}\n  {pname.upper()}  (frozen [CLS], Sheet2, fold-mean R²)\n{"="*74}')
    for hname in HEADS:
        print(f'  -- head={hname}')
        ref = R[(pname, hname, 'stock-84M')]
        print(f'     {"stock-84M":10} {ref.mean():+.4f} ± {ref.std(ddof=1):.3f}   folds={np.round(ref,3)}')
        for b in BASES[1:]:
            r = R[(pname, hname, b)]
            d = r - ref
            p = ttest_rel(r, ref)[1]
            star = '  <-- beats stock' if d.mean() > 0 and p < 0.05 else ('  (beats stock, n.s.)' if d.mean() > 0 else '')
            print(f'     {b:10} {r.mean():+.4f} ± {r.std(ddof=1):.3f}   '
                  f'delta={d.mean():+.4f}  paired p={p:.3f}{star}')

print(f'\n{"="*74}\n  KEY CONTRAST: v4 - stock under each protocol/head\n{"="*74}')
for hname in HEADS:
    for pname in PROTO:
        d = R[(pname, hname, 'v4')] - R[(pname, hname, 'stock-84M')]
        print(f'  {hname:6} {pname:11} v4-stock = {d.mean():+.4f}  '
              f'per-fold={np.round(d,3)}  p={ttest_rel(R[(pname,hname,"v4")],R[(pname,hname,"stock-84M")])[1]:.3f}')
json.dump({f'{k[0]}|{k[1]}|{k[2]}': v.tolist() for k, v in R.items()},
          open(REPO/'5_outputs/repr_benchmark/protocol_contrast.json', 'w'), indent=1)
print('\nsaved -> 5_outputs/repr_benchmark/protocol_contrast.json')
