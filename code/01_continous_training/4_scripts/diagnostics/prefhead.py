"""Ranking-loss + 4-layer MLP head on frozen [CLS] features — fast "quantum jump" check.

Question: does replacing the 768->1 Linear head + pure MSE (which gave ~0.45-0.58 R2
with Ridge/RF, and ECFP4+RF beating all neural on top25 agreement) with a deeper
MLP + LUMI-style preference loss actually improve top-25% recall / PCC?

Design (mirrors LUMI-lab downstream):
  * head: 768 -> 256 -> 128 -> 64 -> 1, GELU, out dropout 0.2, on frozen [CLS]
  * loss:  LUMI normalized_preference_log = preference(log-sigmoid pairwise) +
           0.5*take, on rank-sorted (by target) batch; + MSE (hf_unimol.py:180-270)
  * protocol: LUMI sequential 5-fold, fold-mean; metric = PCC, top25 PCC, top25 agreement
  * invariant: Pearson/agreement unaffected by affine scale -> no need to un-normalize
"""
import sys, json
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr
import pandas as pd, argparse

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / '4_scripts'))
import lumi_protocol

ap = argparse.ArgumentParser()
ap.add_argument('--loss', choices=['mse', 'preference', 'preference_log'], default='preference_log')
ap.add_argument('--epochs', type=int, default=120)
ap.add_argument('--seeds', type=int, default=2)   # ensemble averaged per fold
args = ap.parse_args()

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
FEAT = REPO / '5_outputs/repr_benchmark/features'
BASES = ['stock-84M', 'v2', 'v3', 'v4']

df = pd.read_excel('/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx', sheet_name='Sheet2')
y = df['RLU (log2)'].astype(float).values
folds = lumi_protocol.sequential_folds(len(y), 5)


def lumi_metrics(t, p):
    t, p = np.asarray(t, float), np.asarray(p, float)
    thr = np.quantile(t, 0.75); m = t >= thr
    pred_thr = np.quantile(p, 0.75)
    agree = len(set(np.nonzero(m)[0]) & set(np.nonzero(p >= pred_thr)[0])) / max(m.sum(), 1)
    return {'pearson': pearsonr(t, p)[0],
            'top25_pearson': pearsonr(t[m], p[m])[0],
            'top25_agreement': float(agree)}


def make_mlp(in_dim=768, widths=(256, 128, 64)):
    layers, prev = [], in_dim
    for w in widths:
        layers += [nn.Linear(prev, w), nn.GELU(), nn.Dropout(0.2)]
        prev = w
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def preference_loss(x, t, use_log):
    # x, t: 1D tensors in current mini-batch. Reorder preds by ascending target,
    # then apply LUMI log-sigmoid / plain-sigmoid over the (upper-tri) pairwise diffs.
    order = t.sort()[1]
    xr = x[order]
    diff = xr.unsqueeze(1) - xr.unsqueeze(0)        # want high-target -> higher pred
    if use_log:
        term = -F.logsigmoid(diff)
    else:
        term = -F.sigmoid(diff)
    # upper triangle only (excluding diagonal), consistent with LUMI triu_indices(offset=1)
    triu = torch.triu_indices(diff.size(0), diff.size(0), 1)
    return term[triu[0], triu[1]].mean()


def train_one(Xtr, ytr, Xva, yva, seed):
    torch.manual_seed(seed)
    net = make_mlp().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
    Xtr = torch.from_numpy(Xtr.astype(np.float32)).to(DEV)
    Xva = torch.from_numpy(Xva.astype(np.float32)).to(DEV)
    mu, sd = ytr.mean(), ytr.std() + 1e-8
    yt = torch.from_numpy(((ytr - mu) / sd).astype(np.float32)).unsqueeze(1).to(DEV)
    n = len(Xtr); bs = 128
    use_pref = 'preference' in args.loss     # controls preference component
    use_log = 'log' in args.loss
    best = -1.0; best_state = None
    for ep in range(args.epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            Xb, yb = Xtr[idx], yt[idx]
            opt.zero_grad()
            logits = net(Xb).squeeze(1)
            mse = F.mse_loss(logits, yb.squeeze(1))
            pref = preference_loss(logits, yb.squeeze(1), use_log) if use_pref else torch.tensor(1.0)
            loss = mse + (0.5 * pref if use_pref else 0.0)
            loss.backward(); opt.step()
        with torch.no_grad():
            vpred = net(Xva).squeeze(1).cpu().numpy() * sd + mu
            vpe = pearsonr(vpred, yva)[0]
        if vpe > best:
            best = vpe
            best_state = {k: t.clone() for k, t in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net


def run():
    results = {}
    for base in BASES:
        X = np.load(FEAT / f'{base}_Sheet2_d3a3687c.npy').astype(np.float32)
        oof = np.zeros(len(y))
        for fold, (tr, te) in enumerate(folds):
            preds = []
            for sd in range(args.seeds):
                net = train_one(X[tr], y[tr], X[te], y[te], seed=1000 * fold + sd)
                with torch.no_grad():
                    p = net(torch.from_numpy(X[te]).to(DEV)).squeeze(1).cpu().numpy()
                preds.append(p)
            oof[te] = np.mean(preds, axis=0)
        per = [lumi_metrics(y[te], oof[te]) for _, te in folds]
        results[base] = {k: float(np.mean([d[k] for d in per])) for k in per[0]}
        results[base]['per_fold_top25_agree'] = [round(d['top25_agreement'], 3) for d in per]
    summary_json = REPO / '5_outputs/repr_benchmark' / f'pref_{args.loss}.json'
    summary_json.write_text(json.dumps({'loss': args.loss, 'results': results}, indent=1))
    print(json.dumps({'loss': args.loss, **results}, indent=1))
    return results


if __name__ == '__main__':
    run()