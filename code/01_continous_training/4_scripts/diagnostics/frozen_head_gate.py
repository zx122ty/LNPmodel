"""关卡 B：冻结 backbone + 非线性 head —— 预训练表征到底有没有可提取的增益？

为什么不跑 MolTrain --freeze-layers encoder：
  unimolv2 的 classification_head 是单个 Linear(768,1)（769 参数），
  且回归任务强制 pooler_dropout=0（nnmodel.py:126-128）、
  微调期每分子只有 1 个固定构象且每 epoch 不重采样（conformer.py:231-265）。
  → 冻结 backbone 后特征完全确定，该配置在数学上就是线性探针，
    等价于已有的 RidgeCV 结果（v4 +0.4518）。跑一小时只会复现它。

有信息量的是 LUMI 冻结时用的 3 层 MLP / inner_dim 2048（hf_unimol.py:113-116）。
本脚本在缓存特征上直接跑该 head，分钟级给出答案。

判据：
  若 v4 冻结+MLP 显著 > stock 冻结+MLP → 预训练有效，是端到端遮盖了它。
  若各底座齐平且全部 < ECFP4+RF(0.5532) → 预训练表征对本任务确实弱于指纹。
"""
import json, hashlib, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as rfg
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, ttest_rel
RDLogger.DisableLog('rdApp.*')

REPO = Path('/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training')
CACHE = Path('/tmp/atom_pool_cache')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_excel('/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx', sheet_name='Sheet2')
smiles = df['SMILES'].astype(str).tolist()
y = df['RLU (log2)'].astype(float).values.astype(np.float32)
cached = json.loads((REPO/'5_outputs/split_cache/Sheet2_scaffold_k5_seed42.json').read_text())
assert cached['smiles_sha256'] == hashlib.sha256("\n".join(smiles).encode()).hexdigest()[:16], 'fold cache mismatch'
folds = [(np.array(f['train'], int), np.array(f['test'], int)) for f in cached['folds']]


def mlp_probe(X, y, folds, seeds=(0, 1, 2), epochs=300, inner_val=0.15):
    """LUMI 冻结口径的 head: Linear->GELU->Linear->GELU->Linear, inner_dim=2048.
    每折用 15% 训练集做 inner-val 早停，多 seed 平均以压噪声。"""
    per_fold = []
    for tr, te in folds:
        preds = []
        for sd in seeds:
            rng = np.random.RandomState(sd)
            perm = rng.permutation(len(tr))
            n_val = int(len(tr) * inner_val)
            va_i, tr_i = tr[perm[:n_val]], tr[perm[n_val:]]
            sc = StandardScaler().fit(X[tr_i])
            ys = y[tr_i].mean(), y[tr_i].std() + 1e-8
            def prep(idx):
                return torch.tensor(sc.transform(X[idx]), dtype=torch.float32, device=DEV)
            Xtr, Xva, Xte = prep(tr_i), prep(va_i), prep(te)
            ytr = torch.tensor((y[tr_i]-ys[0])/ys[1], dtype=torch.float32, device=DEV).unsqueeze(1)
            yva = torch.tensor((y[va_i]-ys[0])/ys[1], dtype=torch.float32, device=DEV).unsqueeze(1)
            torch.manual_seed(sd)
            net = nn.Sequential(nn.Linear(X.shape[1], 2048), nn.GELU(),
                                nn.Linear(2048, 2048), nn.GELU(),
                                nn.Linear(2048, 1)).to(DEV)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-2)
            best, best_state, wait = 1e9, None, 0
            for ep in range(epochs):
                net.train(); opt.zero_grad()
                loss = nn.functional.mse_loss(net(Xtr), ytr)
                loss.backward(); opt.step()
                net.eval()
                with torch.no_grad():
                    v = nn.functional.mse_loss(net(Xva), yva).item()
                if v < best - 1e-5:
                    best, wait = v, 0
                    best_state = {k: t.clone() for k, t in net.state_dict().items()}
                else:
                    wait += 1
                    if wait >= 40: break
            net.load_state_dict(best_state); net.eval()
            with torch.no_grad():
                preds.append(net(Xte).cpu().numpy().ravel()*ys[1] + ys[0])
        per_fold.append(r2_score(y[te], np.mean(preds, axis=0)))
    return np.array(per_fold)


gen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
FP = np.array([np.array(gen.GetFingerprint(Chem.MolFromSmiles(s))) for s in smiles], dtype=np.float32)

print('=== 关卡 B：冻结 backbone + LUMI 式 3 层 MLP head (768->2048->2048->1) ===')
print('    同一份 Sheet2 scaffold seed42 缓存折；3 seed 平均；inner-val 早停\n')
res = {}
for b in ['stock-84M', 'v2', 'v3', 'v4']:
    r = mlp_probe(np.load(CACHE/f'{b}_cls.npy'), y, folds)
    res[b] = r
    print(f'  {b:10} [CLS]+MLP  R2={r.mean():+.4f} +/- {r.std(ddof=1):.3f}  per-fold={np.round(r,3)}')

print()
r_fp = mlp_probe(FP, y, folds); res['ECFP4'] = r_fp
print(f'  {"ECFP4":10} +MLP       R2={r_fp.mean():+.4f} +/- {r_fp.std(ddof=1):.3f}  per-fold={np.round(r_fp,3)}')

print('\n=== 配对检验（同折逐折做差）===')
ref = res['stock-84M']
for b in ['v2', 'v3', 'v4']:
    d = res[b] - ref
    t, p = ttest_rel(res[b], ref)
    print(f'  {b:6} - stock = {d.mean():+.4f}  per-fold={np.round(d,3)}  paired p={p:.4f}')
d = res['v4'] - r_fp; t, p = ttest_rel(res['v4'], r_fp)
print(f'  v4 - ECFP4  = {d.mean():+.4f}  per-fold={np.round(d,3)}  paired p={p:.4f}')

print('\n=== 汇总（全部为 Sheet2 scaffold fold-mean R2）===')
print(f'  {"ECFP4+RF (已有)":28} +0.5532')
print(f'  {"ECFP4+MLP":28} {r_fp.mean():+.4f}')
for b in ['stock-84M', 'v2', 'v3', 'v4']:
    print(f'  {b+" [CLS]+MLP":28} {res[b].mean():+.4f}')
print(f'  {"v4 [CLS]+Ridge (已有)":28} +0.4518')
print(f'  {"端到端全参微调 (已有)":28} stock 0.6624 / v4 0.6403')
json.dump({k: v.tolist() for k, v in res.items()}, open('/tmp/gateB_results.json', 'w'), indent=1)
