"""
02_augment_reagents.py —— 尾巴试剂规则化扩增

真实尾巴库仅 168 个，不足以支撑百万级枚举。观察发现尾巴均为
「反应性弹头 + 直链烷基」结构（如 C=CC(=O)O-C14、C8-环氧、C10-醛），
故沿碳链长度维度做规则化衍生，弹头官能团保持不变。

扩增轴：
  1. 链长 C6~C20（覆盖真实分布 C4~C20）
  2. 单顺式双键位置迁移（模拟油酸/亚油酸型不饱和尾）
  3. 甲基支化（模拟支化脂尾）

不做的事：随机官能团替换——会破坏反应模板匹配并降低可合成性。

输入：output/tails.parquet
输出：output/tails_augmented.parquet
"""

import re
import sys
from collections import Counter

import pandas as pd
from rdkit import Chem

import chem_utils as CU
import config as C
import reaction_templates as RT

# 链长扩增范围：真实尾巴为 C4~C20，略微外扩以增加多样性
CHAIN_LENGTHS = range(6, 21)

# 双键位置：从链末端起第 n 个碳（9-顺式对应油酸，最常见）
DBOND_POSITIONS = (9, 11, 6)

# 稀缺反应类型的加强扩增。原始尾巴数极不均衡（醛 97 个 vs 丙烯酰胺 5 个），
# 若一视同仁地扩增，枚举产物会被还原胺化主导（实测占 76%），
# 下游模型将过拟合单一化学。故对稀缺弹头启用更多双键位置与支化位点。
SCARCE_REACTIONS = {
    "acrylate_michael", "acrylamide_michael", "epoxide_opening", "n_alkylation",
}
DBOND_POSITIONS_SCARCE = (3, 5, 6, 7, 9, 11, 13)

# 匹配 SMILES 中连续 6 个以上的 'C'（直链烷基片段）
_RUN_OF_C = re.compile(r"C{6,}")


def _variants_by_chain_length(smiles):
    """把最长的直链烷基片段替换成 C6~C20，生成同系物。"""
    m = _RUN_OF_C.search(smiles)
    if m is None:
        return []
    start, end = m.span()
    return [smiles[:start] + "C" * n + smiles[end:] for n in CHAIN_LENGTHS]


def _variants_with_double_bond(smiles, positions=DBOND_POSITIONS):
    """在直链上引入单个顺式双键，模拟不饱和脂尾。"""
    m = _RUN_OF_C.search(smiles)
    if m is None:
        return []
    start, end = m.span()
    n = end - start
    out = []
    for pos in positions:
        if not 2 <= pos < n - 1:
            continue
        chain = "C" * pos + "/C=C\\" + "C" * (n - pos - 2)
        out.append(smiles[:start] + chain + smiles[end:])
    return out


def _variants_with_branch(smiles, extra=False):
    """在链中部插入甲基支化。extra=True 时启用更多支化位点。"""
    m = _RUN_OF_C.search(smiles)
    if m is None:
        return []
    start, end = m.span()
    n = end - start
    if n < 8:
        return []
    spots = (n // 2, n - 3) if not extra else (2, n // 3, n // 2, 2 * n // 3, n - 3)
    out = []
    for pos in spots:
        if not 2 <= pos < n - 1:
            continue
        chain = "C" * pos + "C(C)" + "C" * (n - pos - 1)
        out.append(smiles[:start] + chain + smiles[end:])
    return out


def augment_one(tail_row):
    """
    对单个尾巴试剂生成所有衍生物。

    每个衍生物必须：可解析、保留原反应类型、脂肪碳数合理。
    保留原反应类型是硬约束——弹头被破坏的衍生物无法参与枚举。
    """
    base_smiles = tail_row.smiles_std
    base_rxns = set(tail_row.rxn_types.split(","))
    scarce = bool(base_rxns & SCARCE_REACTIONS)
    positions = DBOND_POSITIONS_SCARCE if scarce else DBOND_POSITIONS

    candidates = {base_smiles}
    candidates.update(_variants_by_chain_length(base_smiles))
    candidates.update(_variants_with_double_bond(base_smiles, positions))
    candidates.update(_variants_with_branch(base_smiles, extra=scarce))
    # 双键/支化也施加到各链长同系物上，扩大覆盖
    for v in list(candidates):
        candidates.update(_variants_with_double_bond(v, positions))
        candidates.update(_variants_with_branch(v, extra=scarce))

    out = []
    for smi in candidates:
        mol, std, ikey = CU.standardize(smi)
        if mol is None:
            continue
        if not base_rxns.issubset(set(RT.classify_electrophile(mol))):
            continue          # 弹头被破坏，丢弃
        out.append({
            "smiles_std": std,
            "inchikey": ikey,
            "rxn_types": tail_row.rxn_types,
            "parent_tail_id": tail_row.tail_id,
            "is_original": std == base_smiles,
            "n_aliphatic_C": CU.n_aliphatic_carbons(mol),
            "longest_chain": CU.longest_carbon_chain(mol),
            "n_heavy_atoms": mol.GetNumHeavyAtoms(),
        })
    return out


def main():
    print("=" * 72)
    print("阶段 1-02：尾巴试剂扩增")
    print("=" * 72)

    tails = pd.read_parquet(C.OUT_DIR / "tails.parquet")
    print(f"  原始尾巴库：{len(tails)}")

    rows = []
    for r in tails.itertuples(index=False):
        rows.extend(augment_one(r))

    aug = pd.DataFrame(rows).drop_duplicates(subset="inchikey").reset_index(drop=True)
    # is_original 需按全库判定：A 的某个衍生物可能恰好等于原始尾巴 B，
    # 逐行标记会因 drop_duplicates 的保留顺序而误判。
    original_keys = set(tails.inchikey)
    aug["is_original"] = aug.inchikey.isin(original_keys)
    aug.insert(0, "tail_id", [f"TA{i:05d}" for i in range(len(aug))])
    aug.to_parquet(C.OUT_DIR / "tails_augmented.parquet", index=False)

    print(f"  扩增后：{len(aug)}  （放大 {len(aug) / len(tails):.1f}×）")
    print(f"  其中原始尾巴保留：{int(aug.is_original.sum())}/{len(tails)}")

    print("\n  各反应类型尾巴数：")
    counter = Counter()
    for s in aug.rxn_types:
        for r in s.split(","):
            counter[r] += 1
    for r, n in counter.most_common():
        print(f"    {r:24s} {n:6d}")

    print(f"\n  链长分布：{aug.longest_chain.min()}~{aug.longest_chain.max()}"
          f"  中位数 {aug.longest_chain.median():.0f}")

    heads = pd.read_parquet(C.OUT_DIR / "heads.parquet")
    n_pairs = sum(len(heads) * n for n in counter.values())
    print(f"\n  可枚举组合数：{n_pairs:,}  （头基 {len(heads)} × 尾巴-反应配对）")
    print(f"  输出：{C.OUT_DIR}/tails_augmented.parquet")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
