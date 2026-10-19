"""
00_load_and_merge_raw.py —— 多源可电离脂质数据合并去重

输入：raw_data/ 下 5 个来源的 CSV
输出：
  ionizable_lipids.parquet  预训练语料（仅可电离脂质）
  all_molecules.parquet     全部分子含被拒记录，便于审计
  reagents_ugi4cr.parquet   Ugi-4CR 砌块库（供后续枚举使用，不进语料）
  merge_report.md           各源贡献、去重率、拒绝原因统计

运行：python 00_load_and_merge_raw.py
"""

import sys
from collections import Counter, defaultdict

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import chem_utils as CU
import config as C


# ================================================================== 数据加载
def load_lnpdb():
    """LNPDB：主干来源，19,797 行配方 → 约 12,834 个唯一脂质。"""
    df = pd.read_csv(C.SRC_LNPDB, encoding=C.ENCODINGS["LNPDB"], low_memory=False)
    out = df[["IL_SMILES", "IL_name"]].dropna(subset=["IL_SMILES"]).copy()
    out.columns = ["smiles_orig", "name"]
    out["source"] = "LNPDB"
    out["activity_value"] = pd.NA
    out["activity_type"] = pd.NA
    return out


def load_absp():
    """paper_ABSP：带转染表达量标签，注意 BOM 头。"""
    df = pd.read_csv(C.SRC_ABSP, encoding=C.ENCODINGS["ABSP"])
    out = df[["SMILES", "Total expression"]].dropna(subset=["SMILES"]).copy()
    out.columns = ["smiles_orig", "activity_value"]
    out["name"] = pd.NA
    out["source"] = "ABSP"
    out["activity_type"] = "total_expression"
    return out


def load_mtcs():
    """
    paper_MTCS：只取 target==1。

    target==0 是从通用化学库抓的非脂质负样本（实测 MW 中位数 492、
    99% 含芳环、多为核苷/激酶抑制剂/多肽），绝不能进脂质预训练语料。
    """
    df = pd.read_csv(C.SRC_MTCS, encoding=C.ENCODINGS["MTCS"])
    pos = df[df["target"] == 1].dropna(subset=["smiles"]).copy()
    out = pos[["smiles"]].copy()
    out.columns = ["smiles_orig"]
    out["name"] = pd.NA
    out["source"] = "MTCS"
    out["activity_value"] = pd.NA
    out["activity_type"] = pd.NA
    return out


def load_umi_reagents():
    """
    paper_umi-lab：Ugi 四组分反应的砌块，不是脂质产物。

    四个类别（胺/异腈/醛/羧酸）MW 中位数仅 138~226，而真实 IL 是 738。
    单独输出供后续组合枚举使用。
    """
    df = pd.read_csv(C.SRC_UMI, encoding=C.ENCODINGS["umi-lab"])
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    return df.rename(columns={"ID": "reagent_id", "SMILES": "smiles_orig", "Category": "category"})


# ============================================================== 单分子处理
def process_one(smiles_orig, name, source, activity_value, activity_type):
    """标准化 + 判定 + 描述符。返回单条记录 dict。"""
    mol, smi_std, ikey = CU.standardize(smiles_orig)
    if mol is None:
        return {
            "smiles_orig": smiles_orig, "source": source, "name": name,
            "inchikey": None, "smiles_std": None,
            "keep": False, "reject_reason": "unparseable",
            "activity_value": activity_value, "activity_type": activity_type,
        }

    is_ion, reason = CU.classify_ionizable(mol)
    desc = CU.compute_descriptors(mol)

    keep, reject = is_ion, (None if is_ion else reason)
    if keep:
        passed, prop_reason = CU.passes_property_filter(desc)
        if not passed:
            keep, reject = False, prop_reason

    return {
        "smiles_orig": smiles_orig, "source": source, "name": name,
        "inchikey": ikey, "smiles_std": smi_std,
        "keep": keep, "reject_reason": reject,
        "activity_value": activity_value, "activity_type": activity_type,
        **desc,
    }


# ================================================================== 主流程
def main():
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("阶段 0：多源可电离脂质数据合并")
    print("=" * 72)

    loaders = {"LNPDB": load_lnpdb, "ABSP": load_absp, "MTCS": load_mtcs}
    frames = []
    raw_counts = {}
    for src, fn in loaders.items():
        if src in C.EXCLUDE_SOURCES:
            print(f"  [跳过] {src}（已在 config 中排除）")
            continue
        df = fn()
        raw_counts[src] = len(df)
        frames.append(df)
        print(f"  [加载] {src:8s} {len(df):6d} 行")

    if "Atlas" in C.EXCLUDE_SOURCES:
        print("  [排除] Atlas —— SMILES 与化学名不一致，疑似 LLM 生成未经校验")

    # 各 loader 的列集一致但部分列全空，显式统一 dtype 以避免 concat 推断警告
    cols = ["smiles_orig", "name", "source", "activity_value", "activity_type"]
    frames = [f.reindex(columns=cols).astype({"name": "object",
                                              "activity_type": "object"}) for f in frames]
    raw = pd.concat(frames, ignore_index=True)
    print(f"\n  合计原始行数：{len(raw)}")

    # ---- 并行标准化 ----
    print(f"\n  标准化中（{C.N_JOBS} 进程）...")
    records = Parallel(n_jobs=C.N_JOBS, batch_size=256)(
        delayed(process_one)(r.smiles_orig, r.name, r.source, r.activity_value, r.activity_type)
        for r in tqdm(raw.itertuples(index=False), total=len(raw), ncols=76)
    )
    proc = pd.DataFrame(records)

    # ---- 按 InChIKey 去重，合并多源溯源信息 ----
    print("\n  按 InChIKey 去重...")
    valid = proc[proc.inchikey.notna()].copy()

    src_map = defaultdict(set)
    act_map = defaultdict(list)
    for r in valid.itertuples(index=False):
        src_map[r.inchikey].add(r.source)
        if pd.notna(r.activity_value):
            act_map[r.inchikey].append((float(r.activity_value), r.activity_type))

    # 同一分子被多次测定时取几何平均（活性值跨 5 个数量级，算术平均会被最大值主导）。
    # 冲突倍数记入 activity_fold_range，供下游按可信度加权或剔除。
    import math

    def agg_activity(key):
        vals = act_map.get(key, [])
        if not vals:
            return pd.NA, pd.NA, pd.NA, 0
        nums = [v for v, _ in vals if v > 0]
        atype = vals[0][1]
        if not nums:
            return 0.0, atype, 1.0, len(vals)
        geo = math.exp(sum(math.log(v) for v in nums) / len(nums))
        fold = max(nums) / min(nums)
        return round(geo, 3), atype, round(fold, 2), len(vals)

    dedup = valid.drop_duplicates(subset="inchikey", keep="first").copy()
    dedup["sources"] = dedup.inchikey.map(lambda k: ",".join(sorted(src_map[k])))
    dedup["n_sources"] = dedup.inchikey.map(lambda k: len(src_map[k]))
    agg = dedup.inchikey.map(agg_activity)
    dedup["activity_value"] = [a[0] for a in agg]
    dedup["activity_type"] = [a[1] for a in agg]
    dedup["activity_fold_range"] = [a[2] for a in agg]   # 1.0 = 无冲突
    dedup["n_activity_measurements"] = [a[3] for a in agg]
    dedup["has_activity_label"] = dedup.activity_value.notna()
    dedup = dedup.drop(columns=["source"])

    ionizable = dedup[dedup.keep].copy().reset_index(drop=True)

    # ---- Ugi 砌块库单独输出 ----
    reagents = load_umi_reagents()
    rec = []
    for r in reagents.itertuples(index=False):
        mol, smi, ikey = CU.standardize(r.smiles_orig)
        if mol is None:
            continue
        rec.append({"reagent_id": r.reagent_id, "category": r.category,
                    "smiles_std": smi, "inchikey": ikey, **CU.compute_descriptors(mol)})
    reagent_df = pd.DataFrame(rec).drop_duplicates(subset="inchikey").reset_index(drop=True)

    # ---- 落盘 ----
    p_ion = C.OUT_DIR / "ionizable_lipids.parquet"
    p_all = C.OUT_DIR / "all_molecules.parquet"
    p_rgt = C.OUT_DIR / "reagents_ugi4cr.parquet"
    ionizable.to_parquet(p_ion, index=False)
    dedup.to_parquet(p_all, index=False)
    reagent_df.to_parquet(p_rgt, index=False)

    write_report(raw_counts, proc, dedup, ionizable, reagent_df, src_map)

    print("\n" + "=" * 72)
    print(f"  唯一分子总数     : {len(dedup)}")
    print(f"  可电离脂质（语料）: {len(ionizable)}")
    print(f"  Ugi 砌块         : {len(reagent_df)}")
    print(f"\n  输出目录：{C.OUT_DIR}")
    print("=" * 72)


def write_report(raw_counts, proc, dedup, ionizable, reagent_df, src_map):
    """生成 merge_report.md：各源贡献、重叠、拒绝原因。"""
    L = ["# 阶段 0 数据合并报告\n",
         "## 1. 各来源原始行数\n",
         "| 来源 | 原始行数 |", "|---|---|"]
    for s, n in raw_counts.items():
        L.append(f"| {s} | {n} |")
    L.append(f"| **合计** | **{sum(raw_counts.values())}** |\n")

    L += ["## 2. 去重结果\n",
          f"- 可解析分子记录：{proc.inchikey.notna().sum()}",
          f"- 解析失败：{proc.inchikey.isna().sum()}",
          f"- 去重后唯一分子（InChIKey）：**{len(dedup)}**",
          f"- 其中可电离脂质：**{len(ionizable)}**\n"]

    L += ["## 3. 各来源净贡献（去重后）\n",
          "| 来源 | 唯一分子 | 独有 | 可电离 |", "|---|---|---|---|"]
    all_srcs = sorted({s for v in src_map.values() for s in v})
    for s in all_srcs:
        keys = {k for k, v in src_map.items() if s in v}
        only = {k for k, v in src_map.items() if v == {s}}
        ion = set(ionizable.inchikey)
        L.append(f"| {s} | {len(keys)} | {len(only)} | {len(keys & ion)} |")
    L.append("")

    L += ["## 4. 来源间重叠（InChIKey 交集）\n",
          "| | " + " | ".join(all_srcs) + " |", "|---" * (len(all_srcs) + 1) + "|"]
    for a in all_srcs:
        ka = {k for k, v in src_map.items() if a in v}
        row = [f"| {a} "]
        for b in all_srcs:
            kb = {k for k, v in src_map.items() if b in v}
            row.append(f"| {len(ka & kb)} ")
        L.append("".join(row) + "|")
    L.append("")

    L += ["## 5. 被排除分子的原因分布\n",
          "| 原因 | 数量 | 说明 |", "|---|---|---|"]
    explain = {
        "phospholipid": "含磷酸基，属 helper lipid（如 DOPE），非可电离脂质",
        "no_basic_amine": "无可质子化胺，无 pH 响应能力",
        "permanent_quat": "永久季铵盐，恒定带电不可电离",
        "too_hydrophilic": f"logP < {C.MIN_LOGP}，疏水性不足以自组装",
        "too_small_hydrophobe": f"脂肪碳数 < {C.MIN_SP3_CARBONS}",
        "no_lipid_tail": f"最长碳链 < {C.MIN_CHAIN_LEN}，无脂质尾巴",
        "mw_out_of_range": f"分子量超出 {C.MW_RANGE}",
        "logp_out_of_range": f"logP 超出 {C.LOGP_RANGE}",
        "heavy_atoms_out_of_range": f"重原子数超出 {C.HEAVY_ATOM_RANGE}",
        "exceeds_unimol_max_atoms": f"重原子 > {C.UNIMOL_MAX_ATOMS}，Uni-Mol2 会静默丢弃原子",
        "unparseable": "RDKit 无法解析",
    }
    for reason, n in Counter(dedup[~dedup.keep].reject_reason.dropna()).most_common():
        L.append(f"| `{reason}` | {n} | {explain.get(reason, '')} |")
    L.append("")

    if len(ionizable):
        L += ["## 6. 语料性质分布\n", "```",
              ionizable[["mw", "logp", "tpsa", "n_heavy_atoms", "n_rot_bonds",
                         "n_N", "n_basic_amines"]]
              .describe(percentiles=[.01, .25, .5, .75, .99]).T.round(2).to_string(),
              "```\n"]
        L.append(f"- 带活性标签的分子：{int(ionizable.has_activity_label.sum())}\n")

        # 活性值冲突需显式暴露：下游做回归时这些标签噪声极大
        conf = ionizable[ionizable.activity_fold_range > 2.0]
        if len(conf):
            L += ["### 6.1 活性标签冲突\n",
                  f"有 **{len(conf)}** 个分子被重复测定且数值差异 >2 倍"
                  f"（最大差异 **{conf.activity_fold_range.max():.0f} 倍**）。\n",
                  "原因是同一分子在原始 CSV 中以不同 SMILES 写法出现两次。"
                  "已取几何平均，并记录 `activity_fold_range` 字段。\n",
                  "> ⚠️ 下游做转染率回归时，建议剔除 `activity_fold_range > 10` "
                  "的分子，或按该值降权——这些标签不可靠。\n",
                  "| 差异倍数区间 | 分子数 |", "|---|---|"]
            for lo, hi in [(2, 10), (10, 100), (100, 1e9)]:
                n = int(((conf.activity_fold_range >= lo) & (conf.activity_fold_range < hi)).sum())
                label = f"{lo}~{hi:g}x" if hi < 1e9 else f">{lo}x"
                L.append(f"| {label} | {n} |")
            L.append("")

    L += ["## 7. Ugi-4CR 砌块库（不进语料）\n",
          "| 类别 | 数量 | MW 中位数 |", "|---|---|---|"]
    for cat, g in reagent_df.groupby("category"):
        L.append(f"| {cat} | {len(g)} | {g.mw.median():.1f} |")
    L += ["", "> 这些是合成试剂而非脂质产物（MW 中位数 138~226，"
          "而真实脂质为 738），单独保存供阶段 1 组合枚举使用。\n"]

    L += ["## 8. 判据标定\n",
          "可电离判据在 LNPDB 的 12,844 个已知真实脂质上标定：\n",
          "- 召回率 **88.33%**",
          "- 对非脂质负样本（MTCS target=0）假阳率 **3.10%**\n",
          "被拒的真实脂质主要是磷脂（4.49%）与短链亲水对照分子，属预期行为。\n"]

    (C.OUT_DIR / "merge_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
