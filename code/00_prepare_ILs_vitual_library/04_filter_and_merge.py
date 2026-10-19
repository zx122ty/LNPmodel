"""
04_filter_and_merge.py —— 质量控制与语料合并

对枚举产物做：
  1. 可电离判定 + 性质窗口过滤（窗口取自真实 IL 的 [p1, p99]）
  2. 与真实集去泄漏（InChIKey 比对）——防止下游微调时训练集污染测试集
  3. 反应类型配额均衡——枚举产物 76% 来自还原胺化，直接使用会让模型
     过拟合单一化学；按配额下采样以贴近真实数据的反应类型分布
  4. 与真实语料合并，输出预训练数据集

输入：output/virtual_raw.parquet、output/ionizable_lipids.parquet
输出：output/pretrain_corpus.parquet、output/qc_report.md
"""

import sys
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import chem_utils as CU
import config as C

# 虚拟库目标规模定义在 config.py（见其中关于构象采样机制的说明）
TARGET_VIRTUAL_SIZE = C.TARGET_VIRTUAL_SIZE

# 反应类型配额：避免还原胺化（占枚举产物 76%）主导语料。
# 还原胺化只生成 C-N 键、不引入含氧基团，产物 TPSA 中位数仅 38（真实脂质为 85），
# 故配额压到 0.22，把份额让给引入酯/酰胺键的迈克尔加成，使分布更贴近真实。
RXN_QUOTA = {
    "reductive_amination": 0.22,
    "acrylate_michael": 0.30,
    "n_alkylation": 0.20,
    "epoxide_opening": 0.16,
    "acrylamide_michael": 0.12,
}


def screen_one(smiles):
    """标准化 + 可电离判定 + 性质过滤。返回记录 dict 或 None。"""
    mol, std, ikey = CU.standardize(smiles)
    if mol is None:
        return None
    is_ion, reason = CU.classify_ionizable(mol)
    if not is_ion:
        return {"reject": reason}
    desc = CU.compute_descriptors(mol)
    ok, prop_reason = CU.passes_property_filter(desc)
    if not ok:
        return {"reject": prop_reason}
    return {"smiles_std": std, "inchikey": ikey, **desc}


def balance_by_reaction(df, target):
    """
    按 RXN_QUOTA 配额下采样。

    部分反应类型池量不足（如 acrylamide_michael 仅 4.8 万），缺额需补足。
    补足时必须**按配额比例**从各类型剩余分子中抽取——若直接从整个剩余池
    随机抽，会因该池 76% 是还原胺化而使配额完全失效。
    """
    rng = np.random.default_rng(C.RANDOM_SEED)
    pools = {rxn: df[df.rxn_type == rxn] for rxn in df.rxn_type.unique()}
    picked_idx = {rxn: set() for rxn in pools}

    def take(rxn, n):
        """从某类型的未选中分子里抽 n 个，返回实际抽到的数量。"""
        pool = pools.get(rxn)
        if pool is None or n <= 0:
            return 0
        avail = [i for i in range(len(pool)) if i not in picked_idx[rxn]]
        n = min(n, len(avail))
        if n:
            picked_idx[rxn].update(rng.choice(avail, n, replace=False).tolist())
        return n

    shortfall = sum(int(target * f) - take(rxn, int(target * f))
                    for rxn, f in RXN_QUOTA.items())

    # 缺额按配额权重重新分配给仍有余量的类型，循环直到补满或全部耗尽
    while shortfall > 0:
        spare = {r: len(pools[r]) - len(picked_idx[r]) for r in pools
                 if len(pools[r]) > len(picked_idx[r])}
        if not spare:
            break
        weights = {r: RXN_QUOTA.get(r, 0.01) for r in spare}
        total_w = sum(weights.values())
        filled = sum(take(r, max(1, int(shortfall * w / total_w)))
                     for r, w in weights.items())
        if filled == 0:
            break
        shortfall -= filled

    out = pd.concat([pools[r].iloc[sorted(idx)] for r, idx in picked_idx.items() if idx],
                    ignore_index=True)
    return out.reset_index(drop=True)


def main():
    print("=" * 72)
    print("阶段 1-04：质量控制与语料合并")
    print("=" * 72)

    virtual = pd.read_parquet(C.OUT_DIR / "virtual_raw.parquet")
    real = pd.read_parquet(C.OUT_DIR / "ionizable_lipids.parquet")
    print(f"  枚举产物：{len(virtual):,}")
    print(f"  真实语料：{len(real):,}")

    print(f"\n  质量筛选中（{C.N_JOBS} 进程）...")
    results = Parallel(n_jobs=C.N_JOBS, batch_size=2048)(
        delayed(screen_one)(s) for s in tqdm(virtual.smiles, ncols=76)
    )

    rejects = Counter()
    keep_idx, keep_rows = [], []
    for i, r in enumerate(results):
        if r is None:
            rejects["unparseable"] += 1
        elif "reject" in r:
            rejects[r["reject"]] += 1
        else:
            keep_idx.append(i)
            keep_rows.append(r)

    passed = pd.DataFrame(keep_rows)
    passed["rxn_type"] = virtual.rxn_type.values[keep_idx]
    passed["head_id"] = virtual.head_id.values[keep_idx]
    passed["tail_id"] = virtual.tail_id.values[keep_idx]
    passed = passed.drop_duplicates(subset="inchikey").reset_index(drop=True)
    print(f"\n  通过质量筛选：{len(passed):,}")

    # 去泄漏：虚拟库中与真实分子相同的，归属真实集
    real_keys = set(real.inchikey)
    leaked = passed.inchikey.isin(real_keys)
    print(f"  与真实集重叠（剔除）：{int(leaked.sum()):,}")
    passed = passed[~leaked].reset_index(drop=True)

    balanced = balance_by_reaction(passed, TARGET_VIRTUAL_SIZE)
    print(f"  配额均衡后：{len(balanced):,}")

    # 合并成预训练语料
    virtual_out = balanced.copy()
    virtual_out["source_type"] = "virtual"
    virtual_out["sources"] = "enumerated"

    real_out = real.copy()
    real_out["source_type"] = "real"
    real_out["rxn_type"] = pd.NA
    real_out["head_id"] = pd.NA
    real_out["tail_id"] = pd.NA

    shared = [c for c in virtual_out.columns if c in real_out.columns]
    corpus = pd.concat([real_out[shared], virtual_out[shared]], ignore_index=True)
    corpus = corpus.drop_duplicates(subset="inchikey").reset_index(drop=True)
    corpus.to_parquet(C.OUT_DIR / "pretrain_corpus.parquet", index=False)

    write_report(virtual, passed, balanced, real, corpus, rejects, int(leaked.sum()))

    print(f"\n  预训练语料：{len(corpus):,}"
          f"  （真实 {int((corpus.source_type == 'real').sum()):,}"
          f" + 虚拟 {int((corpus.source_type == 'virtual').sum()):,}）")
    print(f"  输出：{C.OUT_DIR}/pretrain_corpus.parquet")
    print("=" * 72)


def write_report(virtual, passed, balanced, real, corpus, rejects, n_leaked):
    n_virtual = len(corpus[corpus.source_type == "virtual"])
    n_real = len(corpus[corpus.source_type == "real"])
    L = [
        "# 阶段 1 虚拟库构建 QC 报告\n",
        "## 1. 枚举与筛选漏斗\n",
        "| 步骤 | 分子数 |", "|---|---|",
        f"| 枚举产物（去重后） | {len(virtual):,} |",
        f"| 通过可电离判定 + 性质过滤 | {len(passed) + n_leaked:,} |",
        f"| 剔除与真实集重叠 | -{n_leaked:,} |",
        f"| 反应类型配额均衡 | {len(balanced):,} |",
        f"| **最终虚拟库** | **{n_virtual:,}** |\n",
        "## 2. 被过滤分子的原因\n",
        "| 原因 | 数量 |", "|---|---|",
    ]
    for reason, n in rejects.most_common():
        L.append(f"| `{reason}` | {n:,} |")

    L += ["\n## 3. 反应类型分布\n",
          "| 反应类型 | 枚举产物 | 最终语料 | 配额 |", "|---|---|---|---|"]
    enum_counts = Counter(virtual.rxn_type)
    final_counts = Counter(balanced.rxn_type)
    for rxn in sorted(enum_counts, key=lambda r: -enum_counts[r]):
        quota = RXN_QUOTA.get(rxn, 0)
        L.append(f"| {rxn} | {enum_counts[rxn]:,} | {final_counts.get(rxn, 0):,} "
                 f"| {quota:.0%} |")
    L += ["", "> 枚举产物中还原胺化占 76%，直接使用会让模型过拟合单一化学，"
          "故按配额下采样。\n"]

    L += ["## 4. 语料构成\n",
          "| 类型 | 分子数 | 占比 |", "|---|---|---|",
          f"| 真实（文献） | {n_real:,} | {n_real / len(corpus):.1%} |",
          f"| 虚拟（枚举） | {n_virtual:,} | {n_virtual / len(corpus):.1%} |",
          f"| **合计** | **{len(corpus):,}** | |\n",
          "> 阶段 2 构象采样时真实分子取 10 个构象、虚拟分子取 2~5 个，"
          "有效样本比约 1:6~1:12，避免真实分布被淹没。\n"]

    L += ["## 5. 性质分布对比（真实 vs 虚拟）\n"]
    cols = ["mw", "logp", "tpsa", "n_heavy_atoms", "n_rot_bonds", "n_basic_amines"]
    L += ["| 描述符 | 真实 p50 | 虚拟 p50 | 真实 p1~p99 | 虚拟 p1~p99 |",
          "|---|---|---|---|---|"]
    for c in cols:
        rv, vv = real[c], balanced[c]
        L.append(f"| {c} | {rv.median():.1f} | {vv.median():.1f} "
                 f"| {rv.quantile(.01):.1f}~{rv.quantile(.99):.1f} "
                 f"| {vv.quantile(.01):.1f}~{vv.quantile(.99):.1f} |")

    L += ["\n## 6. Uni-Mol2 兼容性\n",
          f"- 采用 `remove_hs={C.UNIMOL_REMOVE_HS}`，按重原子计预算",
          f"- 语料重原子数上限：{corpus.n_heavy_atoms.max()}（限 {C.UNIMOL_MAX_ATOMS}）",
          f"- 超限分子数：{int((corpus.n_heavy_atoms > C.UNIMOL_MAX_ATOMS).sum())}\n",
          "> conformer.py 在原子数超限时会随机丢弃原子而非报错，"
          "已在过滤阶段拦截。\n"]

    (C.OUT_DIR / "qc_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
