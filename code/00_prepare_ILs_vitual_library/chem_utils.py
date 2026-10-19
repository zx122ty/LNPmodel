"""
分子标准化与可电离脂质判定。

标准化管线：去盐（取最大片段）→ 中性化 → 规范化 SMILES → InChIKey
InChIKey 作为跨源去重的唯一键。
"""

import collections

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

import config as C

RDLogger.DisableLog("rdApp.*")

_UNCHARGER = rdMolStandardize.Uncharger()
_LARGEST_FRAG = rdMolStandardize.LargestFragmentChooser()

PAT_BASIC_AMINE = Chem.MolFromSmarts(C.SMARTS_BASIC_AMINE)
PAT_QUAT_N = Chem.MolFromSmarts(C.SMARTS_QUAT_N)
PAT_PHOSPHATE = Chem.MolFromSmarts(C.SMARTS_PHOSPHATE)


def standardize(smiles):
    """返回 (mol, canonical_smiles, inchikey)，失败时返回 (None, None, None)。"""
    if smiles is None:
        return None, None, None
    # LNPDB 的试剂列存在数据缺陷：309 条 head SMILES 带尾随下划线，
    # 另有制表符/空格混入。RDKit 对这些会静默解析失败，须先剥离。
    raw = str(smiles).strip().strip("_").strip()
    raw = raw.replace("\t", "").replace("\n", "").replace(" ", "")
    if not raw or raw in ("NA", "nan", "None", "-"):
        return None, None, None
    try:
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            return None, None, None
        mol = _LARGEST_FRAG.choose(mol)   # 去盐/去反离子
        mol = _UNCHARGER.uncharge(mol)    # 中性化，统一质子化态
        Chem.SanitizeMol(mol)
        return mol, Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol)
    except Exception:
        return None, None, None


def n_aliphatic_carbons(mol):
    """总脂肪碳数——衡量疏水质量。多臂脂质每条臂都短，但总量大。"""
    return sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C" and not a.GetIsAromatic())


def longest_carbon_chain(mol):
    """最长连续非环碳链长度（BFS 近似，用于排除完全无脂链的分子）。"""
    idx = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C" and not a.IsInRing()]
    if not idx:
        return 0
    pool, best = set(idx), 0
    for start in idx:
        queue, seen = collections.deque([(start, 1)]), {start}
        while queue:
            node, depth = queue.popleft()
            best = max(best, depth)
            for nb in mol.GetAtomWithIdx(node).GetNeighbors():
                j = nb.GetIdx()
                if j in pool and j not in seen:
                    seen.add(j)
                    queue.append((j, depth + 1))
    return best


def classify_ionizable(mol):
    """
    判定是否为可电离脂质，返回 (bool, reason)。

    判据在 LNPDB 12,844 个已知真实 IL 上标定：召回 88.33%，
    对非脂质负样本（MTCS target=0）假阳率 3.10%。
    """
    if mol is None:
        return False, "unparseable"
    if mol.HasSubstructMatch(PAT_PHOSPHATE):
        return False, "phospholipid"          # DOPE 类 helper lipid，非可电离脂质
    if not mol.HasSubstructMatch(PAT_BASIC_AMINE):
        if mol.HasSubstructMatch(PAT_QUAT_N):
            return False, "permanent_quat"    # 永久带电，无 pH 响应
        return False, "no_basic_amine"
    if Crippen.MolLogP(mol) < C.MIN_LOGP:
        return False, "too_hydrophilic"
    if n_aliphatic_carbons(mol) < C.MIN_SP3_CARBONS:
        return False, "too_small_hydrophobe"
    if longest_carbon_chain(mol) < C.MIN_CHAIN_LEN:
        return False, "no_lipid_tail"
    return True, "ok"


def compute_descriptors(mol):
    """基础理化描述符 + Uni-Mol2 相关的原子计数。"""
    n_heavy = mol.GetNumHeavyAtoms()
    return {
        "mw": round(Descriptors.MolWt(mol), 3),
        "logp": round(Crippen.MolLogP(mol), 3),
        "tpsa": round(rdMolDescriptors.CalcTPSA(mol), 2),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "n_rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "n_rings": rdMolDescriptors.CalcNumRings(mol),
        "n_arom_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "n_heavy_atoms": n_heavy,
        "n_atoms_with_h": Chem.AddHs(mol).GetNumAtoms(),
        "n_N": sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N"),
        "n_basic_amines": len(mol.GetSubstructMatches(PAT_BASIC_AMINE)),
        "n_aliphatic_C": n_aliphatic_carbons(mol),
        "longest_chain": longest_carbon_chain(mol),
        # remove_hs=True 时 Uni-Mol2 按重原子计预算
        "fits_unimol": n_heavy <= C.UNIMOL_MAX_ATOMS,
    }


def passes_property_filter(desc):
    """性质窗口过滤，返回 (bool, reason)。窗口取自真实 IL 的 [p1, p99]。"""
    if not (C.MW_RANGE[0] <= desc["mw"] <= C.MW_RANGE[1]):
        return False, "mw_out_of_range"
    if not (C.LOGP_RANGE[0] <= desc["logp"] <= C.LOGP_RANGE[1]):
        return False, "logp_out_of_range"
    if not (C.HEAVY_ATOM_RANGE[0] <= desc["n_heavy_atoms"] <= C.HEAVY_ATOM_RANGE[1]):
        return False, "heavy_atoms_out_of_range"
    if not desc["fits_unimol"]:
        return False, "exceeds_unimol_max_atoms"
    return True, "ok"
