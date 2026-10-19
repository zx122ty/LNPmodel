"""
Utility functions for LNP transfection efficiency prediction.
Accepts a config object for flexible data paths.
"""
import os, sys
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import warnings
warnings.filterwarnings('ignore')


# ============== Config  ==============

class Config:
    """Configuration object, can be loaded from a dict or module."""
    def __init__(self, **kwargs):
        # Defaults
        self.SMILES_COL = "SMILES"
        self.TARGET_COL = "TARGET"
        self.TARGET_LOG = False
        self.TEST_SIZE = 0.2
        self.RANDOM_STATE = 42
        self.PRETRAINED_DIR = "/home/data/zhengxiao/disk1/my_project/Uni-Mol2_Toxicity_Prediction/2_models/pretrained"
        self.UNIMOL_MODEL_SIZE = "84m"
        self.UNIMOL_BATCH_SIZE = 4
        self.UNIMOL_EPOCHS = 50
        self.UNIMOL_LEARNING_RATE = 1e-4
        self.UNIMOL_EARLY_STOPPING = 10
        self.INPUT_FILE = None
        self.MOL_DIR = None
        self.OUTPUT_DIR = None
        self.DATA_DIR = None

        # Override with kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_args(cls, args):
        """Create config from argparse namespace."""
        kwargs = {}
        for key in dir(args):
            if key.startswith('_'):
                continue
            kwargs[key.upper()] = getattr(args, key)
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# ============== Data Loading ==============

def load_data(cfg):
    """Load data from CSV or Excel based on config."""
    if cfg.INPUT_FILE is None:
        raise ValueError("INPUT_FILE not set in config")

    ext = os.path.splitext(cfg.INPUT_FILE)[1].lower()
    if ext == '.xlsx':
        df = pd.read_excel(cfg.INPUT_FILE)
    elif ext == '.csv':
        df = pd.read_csv(cfg.INPUT_FILE)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    # Rename columns
    df = df.rename(columns={cfg.SMILES_COL: 'SMILES', cfg.TARGET_COL: 'TARGET'})

    # Drop rows without target
    df = df.dropna(subset=['TARGET']).copy()

    # Log transform if needed
    if cfg.TARGET_LOG:
        df['TARGET_log10'] = np.log10(df['TARGET'].values)
    else:
        df['TARGET_log10'] = df['TARGET'].values.copy()

    return df


def load_mol_smiles(cfg):
    """Load all mol files and convert to SMILES."""
    if cfg.MOL_DIR is None:
        return {}
    mol_files = sorted([f for f in os.listdir(cfg.MOL_DIR) if f.endswith('.mol')])
    name_to_smiles = {}
    failed = []
    for fname in mol_files:
        name = fname.replace('.mol', '')
        mol_path = os.path.join(cfg.MOL_DIR, fname)
        mol = Chem.MolFromMolFile(mol_path)
        if mol is None:
            with open(mol_path, 'r') as f:
                content = f.read()
            mol = Chem.MolFromMolBlock(content)
        if mol is not None:
            try:
                smiles = Chem.MolToSmiles(mol)
                name_to_smiles[name] = smiles
            except:
                failed.append(name)
        else:
            failed.append(name)
    print(f"Loaded {len(name_to_smiles)} mol files, {len(failed)} failed")
    if failed:
        print(f"Failed: {failed[:10]}")
    return name_to_smiles


def merge_data(df, name_to_smiles, name_col='name_ionziable_lipid'):
    """Merge metadata with SMILES from mol files."""
    df['SMILES'] = df[name_col].map(name_to_smiles)
    missing = df['SMILES'].isna().sum()
    if missing > 0:
        print(f"Warning: {missing} molecules have no SMILES, dropping them")
        df = df.dropna(subset=['SMILES']).copy()
    return df


# ============== 1D-QSAR (Molecular Descriptors) ==============

def calculate_1dqsar_repr(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumSaturatedRings(mol),
        Descriptors.NumHeteroatoms(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.NumRadicalElectrons(mol),
        Descriptors.qed(mol),
    ]


# ============== 2D-QSAR (Morgan Fingerprints) ==============

def calculate_2dqsar_repr(smiles, nbits=512, radius=3):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)


# ============== UniMol2 Representation ==============

def get_unimol2_repr(smiles_list, cfg):
    """Get UniMol2 molecular representations. Returns (n, hidden_dim) array."""
    from unimol_tools import UniMolRepr
    size = cfg.UNIMOL_MODEL_SIZE
    # e.g. '84m' -> checkpoint_84M.pt, '1.1b' -> checkpoint_1.1B.pt
    model_path = f'{cfg.PRETRAINED_DIR}/checkpoint_{size.upper()}.pt'
    dict_path = f'{cfg.PRETRAINED_DIR}/dict.txt'

    clf = UniMolRepr(
        data_type='molecule',
        remove_hs=False,
        model_name='unimolv2',
        model_size=size,
        batch_size=cfg.UNIMOL_BATCH_SIZE,
        max_atoms=256,
        use_cuda=True,
        pretrained_model_path=model_path,
        pretrained_dict_path=dict_path,
    )
    repr_result = clf.get_repr(smiles_list, return_atomic_reprs=False)
    return np.array(repr_result)


# ============== Evaluation Metrics ==============

def evaluate(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import spearmanr
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    spearman, _ = spearmanr(y_true, y_pred)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'Spearman': spearman}