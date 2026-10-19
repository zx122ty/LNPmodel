"""
03_enumerate_library.py —— 组合化学枚举

对每个 (head, tail, rxn_type) 三元组做穷尽取代，只保留完全取代产物。
每条产物记录溯源三元组，便于后续虚拟筛选时回溯合成路线。

性能：实测 ~1100 组合/秒/核，全量约 196 万组合，20 核约 2 分钟。

输入：output/heads.parquet、output/tails_augmented.parquet
输出：output/virtual_raw.parquet
"""

import sys
from collections import Counter

import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import config as C
import reaction_templates as RT


def enumerate_for_head(head_id, head_smiles, tail_records):
    """
    单个头基 × 全部尾巴。以头基为并行单元，避免向每个 worker 重复传输尾巴库。

    返回 (产物SMILES, head_id, tail_id, rxn_type) 元组列表。
    """
    out = []
    for tail_id, tail_smiles, rxn_types in tail_records:
        for rxn in rxn_types.split(","):
            for product in RT.enumerate_exhaustive(head_smiles, tail_smiles, rxn):
                out.append((product, head_id, tail_id, rxn))
    return out


def main():
    print("=" * 72)
    print("阶段 1-03：组合化学枚举")
    print("=" * 72)

    heads = pd.read_parquet(C.OUT_DIR / "heads.parquet")
    tails = pd.read_parquet(C.OUT_DIR / "tails_augmented.parquet")
    tail_records = list(zip(tails.tail_id, tails.smiles_std, tails.rxn_types))

    n_combos = sum(len(heads) * len(r[2].split(",")) for r in tail_records)
    print(f"  头基 {len(heads)} × 尾巴 {len(tails)}")
    print(f"  待枚举组合：{n_combos:,}")
    print(f"  并行度：{C.N_JOBS}\n")

    batches = Parallel(n_jobs=C.N_JOBS, batch_size=1)(
        delayed(enumerate_for_head)(h.head_id, h.smiles_std, tail_records)
        for h in tqdm(heads.itertuples(index=False), total=len(heads), ncols=76)
    )

    print("\n  汇总去重中...")
    seen = {}
    rxn_counter = Counter()
    for batch in batches:
        for smiles, head_id, tail_id, rxn in batch:
            rxn_counter[rxn] += 1
            if smiles not in seen:      # 同一产物可由多条路线得到，保留第一条
                seen[smiles] = (head_id, tail_id, rxn)

    virtual = pd.DataFrame(
        [{"smiles": s, "head_id": h, "tail_id": t, "rxn_type": r}
         for s, (h, t, r) in seen.items()]
    )
    virtual.to_parquet(C.OUT_DIR / "virtual_raw.parquet", index=False)

    total = sum(rxn_counter.values())
    print(f"\n  产物总数（含重复路线）：{total:,}")
    print(f"  去重后唯一分子：{len(virtual):,}")
    print(f"  重复率：{(1 - len(virtual) / max(total, 1)) * 100:.1f}%")
    print("\n  按反应类型：")
    for rxn, n in rxn_counter.most_common():
        print(f"    {rxn:24s} {n:9,}")
    print(f"\n  输出：{C.OUT_DIR}/virtual_raw.parquet")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
