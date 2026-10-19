"""
01_build_reagent_library.py —— 构建组合枚举用的试剂库

LNPDB 的 head/tail 列是**合成试剂**（多胺前体 + 亲电试剂），可直接复用，
无需逆向拆解。本脚本提取并按官能团给每个 tail 打上可参与的反应类型标签。

输入：LNPDB.csv、output/reagents_ugi4cr.parquet（阶段 0 产物）
输出：output/heads.parquet、output/tails.parquet
"""

import sys
from collections import Counter

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

import chem_utils as CU
import config as C
import reaction_templates as RT


def load_lnpdb_reagents():
    """从 LNPDB 的试剂列提取 head / tail 原始 SMILES。"""
    df = pd.read_csv(C.SRC_LNPDB, encoding=C.ENCODINGS["LNPDB"], low_memory=False)
    heads = df["IL_head_SMILES"].dropna().astype(str)
    tails = pd.concat([df[f"IL_tail{i}_SMILES"].dropna().astype(str) for i in (1, 2)])
    return heads.tolist(), tails.tolist()


def build_heads(raw_heads, ugi_amines):
    """
    头基库：多胺前体，必须含至少一个可反应的 N-H 位点。

    无 N-H 的（如全叔胺）无法参与枚举，直接剔除。
    """
    rows, seen, rejects = [], set(), Counter()
    for smi in list(raw_heads) + list(ugi_amines):
        mol, std, ikey = CU.standardize(smi)
        if mol is None:
            continue
        if ikey in seen:
            continue
        seen.add(ikey)               # 先登记，确保拒绝原因按唯一分子计数
        n_sites = RT.count_nh_sites(mol)
        if n_sites == 0:
            rejects["no_reactive_NH"] += 1
            continue
        rows.append({
            "head_id": f"H{len(rows):04d}",
            "smiles_std": std,
            "inchikey": ikey,
            "n_nh_sites": n_sites,
            "n_basic_amines": len(mol.GetSubstructMatches(CU.PAT_BASIC_AMINE)),
            "mw": round(Descriptors.MolWt(mol), 2),
            "n_heavy_atoms": mol.GetNumHeavyAtoms(),
        })
    return pd.DataFrame(rows), rejects


def build_tails(raw_tails, ugi_tails):
    """
    尾巴库：亲电试剂，按官能团自动标注可参与的反应类型。

    不匹配任何模板的（硫醇/羧酸/异腈等多组分反应试剂）剔除——
    它们需要第三个组分，不在本二组分枚举范围内。
    """
    rows, seen, rejects = [], set(), Counter()
    for smi in list(raw_tails) + list(ugi_tails):
        mol, std, ikey = CU.standardize(smi)
        if mol is None:
            continue
        if ikey in seen:
            continue
        seen.add(ikey)               # 先登记，确保拒绝原因按唯一分子计数
        rxn_types = RT.classify_electrophile(mol)
        if not rxn_types:
            rejects["no_matching_reaction"] += 1
            continue
        rows.append({
            "tail_id": f"T{len(rows):04d}",
            "smiles_std": std,
            "inchikey": ikey,
            "rxn_types": ",".join(sorted(rxn_types)),
            "n_aliphatic_C": CU.n_aliphatic_carbons(mol),
            "longest_chain": CU.longest_carbon_chain(mol),
            "mw": round(Descriptors.MolWt(mol), 2),
            "n_heavy_atoms": mol.GetNumHeavyAtoms(),
        })
    return pd.DataFrame(rows), rejects


def main():
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("阶段 1-01：构建试剂库")
    print("=" * 72)

    raw_heads, raw_tails = load_lnpdb_reagents()
    print(f"  LNPDB 原始 head 记录：{len(raw_heads)}")
    print(f"  LNPDB 原始 tail 记录：{len(raw_tails)}")

    # Ugi 砌块：胺类补入 head，醛类补入 tail（还原胺化）
    ugi_path = C.OUT_DIR / "reagents_ugi4cr.parquet"
    ugi_amines, ugi_tails = [], []
    if ugi_path.exists():
        ugi = pd.read_parquet(ugi_path)
        ugi_amines = ugi[ugi.category == "Amines"].smiles_std.tolist()
        ugi_tails = ugi[ugi.category == "Lipid Aldehydes"].smiles_std.tolist()
        print(f"  Ugi 砌块补充：胺 {len(ugi_amines)}、醛 {len(ugi_tails)}")
    else:
        print(f"  [警告] 未找到 {ugi_path}，跳过 Ugi 砌块补充")

    heads, h_rej = build_heads(raw_heads, ugi_amines)
    tails, t_rej = build_tails(raw_tails, ugi_tails)

    heads.to_parquet(C.OUT_DIR / "heads.parquet", index=False)
    tails.to_parquet(C.OUT_DIR / "tails.parquet", index=False)

    print(f"\n  头基库：{len(heads)} 个")
    print(f"    剔除：{dict(h_rej)}")
    print(f"    N-H 位点分布：{dict(sorted(Counter(heads.n_nh_sites).items()))}")
    print(f"\n  尾巴库：{len(tails)} 个")
    print(f"    剔除：{dict(t_rej)}")
    print("    反应类型分布：")
    rxn_counter = Counter()
    for s in tails.rxn_types:
        for r in s.split(","):
            rxn_counter[r] += 1
    for r, n in rxn_counter.most_common():
        print(f"      {r:24s} {n:5d}")

    # 枚举规模预估：只有反应类型匹配的 (head, tail) 组合才会产出
    n_pairs = sum(len(heads) * n for n in rxn_counter.values())
    print(f"\n  可枚举组合数（head × tail-rxn 配对）：{n_pairs:,}")
    print(f"  输出：{C.OUT_DIR}/heads.parquet, tails.parquet")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
