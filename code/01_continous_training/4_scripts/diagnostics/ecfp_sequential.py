"""ECFP4 baseline under BOTH protocols — the reference line every neural base must clear.

Sheet2 is LUMI's own 4CR-1920 library (verified 1920/1920 canonical-SMILES overlap),
so running their sequential split here is a like-for-like comparison, not an analogy.
"""
import sys, json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator as rfg
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
RDLogger.DisableLog('rdApp.*')

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / '4_scripts'))
import lumi_protocol

df = pd.read_excel('/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx', sheet_name='Sheet2')
smiles = df['SMILES'].astype(str).tolist()
y = df['RLU (log2)'].astype(float).values
mols = [Chem.MolFromSmiles(s) for s in smiles]

gen = rfg.GetMorganGenerator(radius=2, fpSize=2048)
FP = np.array([np.array(gen.GetFingerprint(m)) for m in mols], dtype=np.float32)
dfns = [Descriptors.MolWt, Descriptors.MolLogP, Descriptors.TPSA, Descriptors.NumRotatableBonds,
        Descriptors.NumHDonors, Descriptors.NumHAcceptors, Descriptors.RingCount,
        Descriptors.FractionCSP3, Descriptors.HeavyAtomCount, Descriptors.NHOHCount,
        Descriptors.NOCount]
D = np.array([[f(m) for f in dfns] for m in mols], dtype=np.float32)
CAT = np.hstack([FP, D])

cached = json.loads((REPO/'5_outputs/split_cache/Sheet2_scaffold_k5_seed42.json').read_text())
assert cached['smiles_sha256'] == hashlib.sha256("\n".join(smiles).encode()).hexdigest()[:16]
PROTOCOLS = {
    'scaffold (ours, extrapolation)':
        [(np.array(f['train'], int), np.array(f['test'], int)) for f in cached['folds']],
    'sequential (LUMI, interpolation)':
        lumi_protocol.sequential_folds(len(smiles), 5),
}

def ev(X, folds, tag):
    for head in ('ridge', 'rf'):
        r2s, sps, rsps = [], [], []
        for tr, te in folds:
            if head == 'ridge':
                sc = StandardScaler().fit(X[tr])
                m = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(sc.transform(X[tr]), y[tr])
                p = m.predict(sc.transform(X[te]))
            else:
                m = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=8).fit(X[tr], y[tr])
                p = m.predict(X[te])
            r2s.append(r2_score(y[te], p)); sps.append(spearmanr(y[te], p).statistic)
            rsps.append(lumi_protocol.relaxed_spearman(p, y[te]))
        print(f'  {tag:18} {head:6} R2={np.mean(r2s):+.4f}±{np.std(r2s,ddof=1):.3f}  '
              f'rho={np.mean(sps):.4f}  relaxSp={np.mean(rsps):.4f}  folds={np.round(r2s,3)}')

for pname, folds in PROTOCOLS.items():
    print(f'\n=== {pname} ===  test sizes={[len(t) for _, t in folds]}')
    ev(FP, folds, 'ECFP4-2048'); ev(CAT, folds, 'ECFP4+desc')
