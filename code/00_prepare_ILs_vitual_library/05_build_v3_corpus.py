"""
05_build_v3_corpus.py —— 构建 LipidFM v3 / v4 预训练语料（scaling-law step 1）

把新生成的虚拟库 CSV（other_model/csv_output/generated_virtual_library_processed_pretraininig_data.csv，
15,491,072 行，仅 smiles 一列）按指定**采样策略**抽取候选，与既有 v2 语料
pretrain_corpus.parquet（real 12,381 + virtual 100,000）按 InChIKey **去重/去泄漏**后合并。

--------------------------------------------------------------------------
## ⚠️ v3 的失败原因（2026-08-19 定位，勿重犯）

v3 用 `--strategy prefix --n-rows 120000`，即 `pd.read_csv(nrows=120000)` 取**前 12 万行**。
但该 CSV 是**按头基分块排序**的，前 12 万行落在同一个块里：

    行 2         …N(CCN1CCN(CCN)CC1)…   氨乙基哌嗪
    行 120,000   …N(CCN1CCN(CCN)CC1)…   还是它
    行 3,000,000 …N(CCC1CCN(CCCO)CC1)…  换头基了
    行 15,000,000…N(Cc1nccn1C)…         又换了

后果实测（ECFP4 / Murcko）：

| 指标                        | 真实脂质 | v2 虚拟 | v3 新增 | 全库随机 |
|-----------------------------|---------|--------|--------|---------|
| ECFP4 内部多样性             | 0.727   | 0.726  | 0.515  | 0.685   |
| → Sheet2 下游 NN 中位相似度   | 0.623   | 0.477  | 0.521  | 0.642   |
| 含氨乙基哌嗪头基             | 2.9%    | 0.7%   | 100%   | 12.7%   |
| 唯一 Murcko 骨架（采样 4 万） | 294     | 314    | 12     | —       |

分子数 +119%，唯一骨架只 +11 个 → 语料量翻倍全押在一个头基上，
**scaling law 没有失效，是这批数据根本不构成 scaling 的自变量。**

## 本版修复

新增 `--strategy scaffold_quota`（默认）：对全库 1549 万行做 Murcko 骨架分层 + 配额采样
（waterfilling：小骨架全取、大骨架截断，让配额尽量均摊到所有骨架）。
全库骨架扫描实测 20 进程约 2.4 分钟，无需子采样。

但**纯均摊会矫枉过正**：全库 1303 个骨架各分 92 个，会把稀有多环骨架抬得过高——
真实脂质 89.5% 是 0~1 环，纯均摊产出只剩 1.0%（n_rings 中位 4）；可旋转键 ≥30 的
真实占 77.8%、纯均摊仅 23.2%。那等于把「更多样」换成「另一类分子」，重蹈 OOD。

故默认再叠加 `--match-real-profile`（物理画像重加权）：
  1. 按 **base 中真实脂质**的 n_rings 边缘分布，把配额拆到各「骨架环数」层；
  2. 层内对骨架做 waterfilling 均摊；
  3. `--max-frac-per-scaffold`（默认 0.15）给单骨架设占比上限 —— 全库无环分子
     166 万条**全部来自同一个骨架**，不设限会直接退化成 v3 的单骨架坍缩。
--------------------------------------------------------------------------

设计要点（与 04_filter_and_merge.py 同口径，避免引入额外变量）：
  - 新增行只做「标准化 + 可电离/性质过滤 + 与 base 去重」，复用 chem_utils / config 的判据，
    与阶段 0 完全一致。
  - 新增行全部标 source_type='virtual'（与 v2 既有 virtual 同构象档，保证唯一变量是语料）。
  - 防泄漏：新增分子若 InChIKey 已存在于 base（real 或 virtual），一律剔除。
  - 新增行内部按 InChIKey 去重，避免同一分子多行重复进预训练。
  - 不覆盖既有 pretrain_corpus.parquet / pretrain_corpus_v3.parquet（v2/v3 可复现）。
  - **默认 --target-new 119230 = v3 的有效新增数**，使 v4 与 v3 分子数完全相同，
    则 v3 vs v4 的唯一变量是**多样性**而非规模。

用法：
    conda activate unimol2

    # v4（推荐）：全库骨架分层配额，规模对齐 v3
    python 05_build_v3_corpus.py --version v4 --strategy scaffold_quota

    # 全库均匀随机（对照）
    python 05_build_v3_corpus.py --version v4rand --strategy random

    # 复现旧 v3 的错误口径（仅供审计，勿用于训练）
    python 05_build_v3_corpus.py --version v3repro --strategy prefix --n-rows 120000

产出（output/）：
    pretrain_corpus_<version>.parquet          # 完整语料 = base + 新增
    pretrain_corpus_<version>_newonly.parquet  # 仅新增分子（阶段 1 只对增量生成构象）
    <version>_dedup_report.md                  # 去重漏斗 + 采样多样性统计
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from tqdm import tqdm

import chem_utils as CU
import config as C

# 新虚拟库 CSV 路径（本 prompt 的关键输入）
NEW_CSV = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/other_model/csv_output/"
               "generated_virtual_library_processed_pretraininig_data.csv")

# v3 的有效新增数。v4 默认对齐此值，使 v3/v4 规模相同、唯一变量为多样性。
DEFAULT_TARGET_NEW = 119_230

# 过采样系数：筛完再精确裁剪到 target。
# 实测通过率随策略变化——prefix/random 约 99%，但画像重加权会多取无环层分子，
# 那里 no_basic_amine 过滤率高得多（8.7% → 17.8%），故留 1.35 的余量。
OVERSAMPLE = 1.35

# 骨架扫描的分块大小（兼顾 joblib 负载均衡与 pickle 峰值）
SCAN_CHUNK = 50_000


def _embeddable(mol):
    """
    排除 ETKDG 几乎必然嵌合失败的桥环/笼状手性分子。

    实测（2026-08-19）：v4 语料首次跑构象时卡死在 4%，元凶是奎宁环
    `C1C[N@]2CC[C@@H]1CC2` 与金刚烷这类**桥头带手性**的笼状头基。
    ETKDGv3 默认 maxIterations=0（无限重试），配合 enforceChirality=True 会永久空转；
    即便把 maxIterations 限到 200，实测 8/8 全部失败、每个仍耗 80~170 秒。
    这类分子占 v4 新增约 4.4%，留着只会空耗 22 worker × ~7 小时且零产出，
    故在语料阶段就剔除 —— 它们本来也进不了训练集。

    判据：存在两个环共享 >= 2 个原子（稠合/桥环），且共享原子中有手性中心。
    只共享 2 个原子的简单稠环（如萘）不受影响，除非桥头带手性。
    """
    ri = mol.GetRingInfo()
    rings = [set(r) for r in ri.AtomRings()]
    if len(rings) < 2:
        return True
    chiral = {i for i, _ in Chem.FindMolChiralCenters(
        mol, includeUnassigned=True, useLegacyImplementation=False)}
    if not chiral:
        return True
    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            shared = rings[i] & rings[j]
            if len(shared) >= 2 and (shared & chiral):
                return False
    return True


def screen_one(smiles):
    """标准化 + 可电离判定 + 性质过滤。返回记录 dict 或 None（与 04 一致）。"""
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
    if not _embeddable(mol):
        return {"reject": "bridged_chiral_unembeddable"}
    return {"smiles_std": std, "inchikey": ikey, **desc}


# --------------------------------------------------------------------------
# 采样策略
# --------------------------------------------------------------------------

def _scaffold_chunk(smiles_chunk):
    """对一批 SMILES 求 Murcko 骨架字符串；解析失败返回 None（后续丢弃）。"""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in smiles_chunk:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            out.append(None)
            continue
        try:
            out.append(MurckoScaffold.MurckoScaffoldSmiles(mol=mol))
        except Exception:
            out.append(None)
    return out


def read_all_smiles(csv_path):
    """流式读入全部 SMILES（CSV 含 UTF-8 BOM，须 utf-8-sig）。"""
    print(f"  读取全库 SMILES：{csv_path.name} ...")
    df = pd.read_csv(csv_path, encoding="utf-8-sig", usecols=[0])
    col = df.columns[0]
    if col.strip("﻿").lower() != "smiles":
        sys.exit(f"CSV 首列不是 smiles，实际列名：{col!r}")
    smiles = df[col].astype(str).to_numpy()
    print(f"  全库行数：{len(smiles):,}")
    return smiles


def scan_scaffolds(smiles, n_jobs):
    """全库 Murcko 骨架扫描 → (codes:int32 数组, 骨架名数组)。失败行 code = -1。"""
    chunks = [smiles[i:i + SCAN_CHUNK] for i in range(0, len(smiles), SCAN_CHUNK)]
    print(f"  全库 Murcko 骨架扫描（{n_jobs} 进程 × {len(chunks)} 块）...")
    results = Parallel(n_jobs=n_jobs, batch_size=1)(
        delayed(_scaffold_chunk)(list(ch)) for ch in tqdm(chunks, ncols=76)
    )
    flat = [s for ch in results for s in ch]
    codes, names = pd.factorize(pd.Series(flat, dtype="object"), use_na_sentinel=True)
    codes = codes.astype(np.int64)
    n_fail = int((codes < 0).sum())
    print(f"  唯一 Murcko 骨架数：{len(names):,}（解析失败 {n_fail:,} 行，已排除）")
    return codes, names


def waterfill_quota(group_sizes, target, rng):
    """
    配额分配（waterfilling）：小组全取、大组截断，尽量把 target 均摊到所有组。

    先按组大小升序给每组分 remaining // groups_left，取不满的组把余额让给后面的组；
    最后若仍有余额，随机打散地分给还有余量的组（避免总是偏向同一批大组）。
    """
    group_sizes = np.asarray(group_sizes, dtype=np.int64)
    n = len(group_sizes)
    total = int(group_sizes.sum())
    if target >= total:
        return group_sizes.copy()

    quota = np.zeros(n, dtype=np.int64)
    order = np.argsort(group_sizes, kind="stable")   # 升序：小组优先吃满
    remaining = int(target)
    groups_left = n
    for gi in order:
        share = remaining // groups_left
        take = min(int(group_sizes[gi]), share)
        quota[gi] = take
        remaining -= take
        groups_left -= 1

    # 整除余数：随机打散分给仍有余量的组，每组每轮 +1
    while remaining > 0:
        spare = np.flatnonzero(quota < group_sizes)
        if spare.size == 0:
            break
        rng.shuffle(spare)
        take = spare[:remaining]
        quota[take] += 1
        remaining -= len(take)
    return quota


def scaffold_ring_counts(names):
    """每个 Murcko 骨架自身的环数（1303 个骨架，秒级）。空骨架（无环脂质）= 0。"""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    out = []
    for s in names:
        if not s:
            out.append(0)
            continue
        mol = Chem.MolFromSmiles(s)
        out.append(0 if mol is None else mol.GetRingInfo().NumRings())
    return np.asarray(out, dtype=np.int64)


def profile_layer_targets(target, ring_profile, avail_by_ring):
    """
    按真实脂质的 n_rings 边缘分布，把 target 拆到各「骨架环数」层。

    源库供给有限的层（如 0 环只占全库 10.7%）先吃满自身上限，
    缺口按剩余各层的**目标权重**再分配，避免整体规模缩水。
    """
    layers = sorted(avail_by_ring)
    want = {r: int(round(target * ring_profile.get(r, 0.0))) for r in layers}

    # 第一轮：不超过各层可用量
    alloc = {r: min(want[r], avail_by_ring[r]) for r in layers}
    deficit = target - sum(alloc.values())

    # 缺口按仍有余量的层的目标权重再分配（多轮直到无处可放）
    while deficit > 0:
        room = {r: avail_by_ring[r] - alloc[r] for r in layers if avail_by_ring[r] > alloc[r]}
        if not room:
            break
        wsum = sum(ring_profile.get(r, 0.0) for r in room) or float(len(room))
        added = 0
        for r in sorted(room, key=lambda x: -room[x]):
            w = ring_profile.get(r, 0.0) / wsum if wsum else 1.0 / len(room)
            take = min(room[r], max(1, int(deficit * w)))
            alloc[r] += take
            added += take
            if sum(alloc.values()) >= target:
                break
        if added == 0:
            break
        deficit = target - sum(alloc.values())
    return alloc


def sample_scaffold_quota(smiles, n_take, seed, n_jobs, ring_profile=None,
                          max_frac_per_scaffold=None):
    """
    全库骨架分层配额采样 → (选中行索引, 诊断信息 dict)。

    ring_profile 非空时启用**物理画像重加权**：先按真实脂质的 n_rings 边缘分布
    把配额拆到各「骨架环数」层，层内再对骨架做 waterfilling 均摊。
    max_frac_per_scaffold 给单个骨架设占比上限，避免无环层（全库仅 1 个骨架）
    把整批新增分子压回单一骨架，重蹈 v3 的覆辙。
    """
    codes, names = scan_scaffolds(smiles, n_jobs)
    valid = np.flatnonzero(codes >= 0)
    vcodes = codes[valid]

    if ring_profile:
        return _sample_profile_weighted(
            valid, vcodes, names, n_take, seed, ring_profile, max_frac_per_scaffold)

    # 按骨架分组：排序后用 searchsorted 切片，避免建 15M 个 python list
    order = np.argsort(vcodes, kind="stable")
    sorted_codes = vcodes[order]
    uniq, starts, counts = np.unique(sorted_codes, return_index=True, return_counts=True)

    rng = np.random.default_rng(seed)
    quota = waterfill_quota(counts, n_take, rng)

    picked = []
    for gi in range(len(uniq)):
        q = int(quota[gi])
        if q <= 0:
            continue
        rows = valid[order[starts[gi]:starts[gi] + counts[gi]]]
        picked.append(rows if q >= len(rows) else rng.choice(rows, q, replace=False))
    idx = np.sort(np.concatenate(picked))

    diag = {
        "strategy": "scaffold_quota",
        "library_scaffolds": int(len(uniq)),
        "sampled_scaffolds": int((quota > 0).sum()),
        "max_per_scaffold": int(quota.max()),
        "median_per_scaffold": float(np.median(quota[quota > 0])),
        "largest_group_size": int(counts.max()),
    }
    print(f"  骨架配额：覆盖 {diag['sampled_scaffolds']:,}/{len(uniq):,} 个骨架，"
          f"每骨架最多 {diag['max_per_scaffold']}、中位 {diag['median_per_scaffold']:.0f}")
    return idx, diag


def _sample_profile_weighted(valid, vcodes, names, n_take, seed,
                             ring_profile, max_frac_per_scaffold):
    """
    骨架分层 + 物理画像重加权。

    纯均摊配额（1303 个骨架各 92 个）会把稀有多环骨架抬得过高：真实脂质 89.5% 是
    0~1 环，纯均摊产出只剩 1.0%，等于把「更多样」变成「另一类分子」。
    这里先按真实 n_rings 边缘分布把配额分到各「骨架环数」层，层内再 waterfilling，
    并对单骨架设占比上限（无环层全库只有 1 个骨架，不设限会退化成 v3 的单骨架坍缩）。
    """
    rng = np.random.default_rng(seed)
    order = np.argsort(vcodes, kind="stable")
    sorted_codes = vcodes[order]
    uniq, starts, counts = np.unique(sorted_codes, return_index=True, return_counts=True)

    sc_rings = scaffold_ring_counts(names)[uniq]          # 各骨架自身环数
    cap = int(n_take * max_frac_per_scaffold) if max_frac_per_scaffold else None

    # 单骨架上限同时约束「可用量」，让层目标分配基于真实可取数
    eff_counts = counts if cap is None else np.minimum(counts, cap)
    avail_by_ring = {int(r): int(eff_counts[sc_rings == r].sum())
                     for r in np.unique(sc_rings)}
    layer_target = profile_layer_targets(n_take, ring_profile, avail_by_ring)

    picked, layer_rows = [], {}
    for r, tgt in sorted(layer_target.items()):
        if tgt <= 0:
            continue
        gidx = np.flatnonzero(sc_rings == r)
        quota = waterfill_quota(eff_counts[gidx], tgt, rng)
        got = 0
        for j, gi in enumerate(gidx):
            q = int(quota[j])
            if q <= 0:
                continue
            rows = valid[order[starts[gi]:starts[gi] + counts[gi]]]
            picked.append(rows if q >= len(rows) else rng.choice(rows, q, replace=False))
            got += min(q, len(rows))
        layer_rows[r] = got

    idx = np.sort(np.concatenate(picked))
    n_scaf_used = int(len(np.unique(vcodes[np.isin(valid, idx)])))
    diag = {
        "strategy": "scaffold_quota+profile",
        "library_scaffolds": int(len(uniq)),
        "sampled_scaffolds": n_scaf_used,
        "max_per_scaffold_cap": cap if cap else -1,
        "layer_alloc": str({k: layer_rows.get(k, 0) for k in sorted(layer_rows)}),
        "largest_group_size": int(counts.max()),
    }
    print(f"  画像重加权分层（按真实脂质 n_rings 分布）：")
    for r in sorted(layer_rows):
        print(f"    骨架环数={r}: {layer_rows[r]:,} 个分子"
              f"（目标占比 {100*ring_profile.get(r,0):.1f}%，"
              f"实得 {100*layer_rows[r]/max(1,len(idx)):.1f}%）")
    print(f"  覆盖骨架 {n_scaf_used:,}/{len(uniq):,}，单骨架上限 {cap if cap else '无'}")
    return idx, diag


def sample_random(smiles, n_take, seed):
    """全库均匀随机采样。"""
    rng = np.random.default_rng(seed)
    n_take = min(n_take, len(smiles))
    idx = np.sort(rng.choice(len(smiles), n_take, replace=False))
    print(f"  全库均匀随机采样 {n_take:,} 行")
    return idx, {"strategy": "random"}


def sample_prefix(smiles, n_take):
    """取前 N 行 —— v3 的旧口径，已知与 CSV 的头基排序耦合，仅供审计复现。"""
    n_take = min(n_take, len(smiles))
    print(f"  ⚠️  prefix 口径：取前 {n_take:,} 行（该 CSV 按头基排序，会导致头基坍缩）")
    return np.arange(n_take), {"strategy": "prefix"}


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v4",
                    help="产物版本号，决定输出文件名（默认 v4）")
    ap.add_argument("--strategy", default="scaffold_quota",
                    choices=["scaffold_quota", "random", "prefix"],
                    help="采样策略（默认 scaffold_quota）")
    ap.add_argument("--target-new", type=int, default=DEFAULT_TARGET_NEW,
                    help=f"目标有效新增分子数（默认 {DEFAULT_TARGET_NEW}，= v3 的有效新增数）")
    ap.add_argument("--n-rows", type=int, default=None,
                    help="仅 --strategy prefix 使用：取 CSV 前 N 行")
    ap.add_argument("--seed", type=int, default=C.RANDOM_SEED,
                    help=f"采样随机种子（默认 {C.RANDOM_SEED}）")
    ap.add_argument("--match-real-profile", action="store_true", default=True,
                    help="按真实脂质的 n_rings 边缘分布重加权（默认开启，仅 scaffold_quota 生效）")
    ap.add_argument("--no-match-real-profile", dest="match_real_profile",
                    action="store_false", help="关闭画像重加权，退回纯骨架均摊")
    ap.add_argument("--max-frac-per-scaffold", type=float, default=0.15,
                    help="单个骨架占新增分子的比例上限（默认 0.15，防单骨架坍缩）")
    args = ap.parse_args()

    OUT_DIR = C.OUT_DIR
    base_path = OUT_DIR / "pretrain_corpus.parquet"
    v_path = OUT_DIR / f"pretrain_corpus_{args.version}.parquet"
    newonly_path = OUT_DIR / f"pretrain_corpus_{args.version}_newonly.parquet"
    report_path = OUT_DIR / f"{args.version}_dedup_report.md"

    print("=" * 72)
    print(f"构建 Lipid {args.version} 预训练语料（strategy={args.strategy}）")
    print("=" * 72)

    # 1) 读既有 v2 语料（base）
    base = pd.read_parquet(base_path)
    base_keys = set(base.inchikey)
    print(f"  既有 base 语料：{len(base):,} 分子"
          f"（real {int((base.source_type == 'real').sum()):,}"
          f" + virtual {int((base.source_type == 'virtual').sum()):,}）")

    # 2) 读全库 SMILES + 按策略采样候选（过采样，筛完再精确裁剪）
    all_smiles = read_all_smiles(NEW_CSV)
    n_cand = int(args.target_new * OVERSAMPLE)

    if args.strategy == "scaffold_quota":
        ring_profile = None
        if args.match_real_profile:
            # 目标分布取自 base 里的**真实脂质**（不是虚拟），这是下游任务的物理画像
            rp = base[base.source_type == "real"].n_rings.value_counts(normalize=True)
            ring_profile = {int(k): float(v) for k, v in rp.items()}
            top = ", ".join(f"{k}环={100*v:.1f}%" for k, v in
                            sorted(ring_profile.items())[:5])
            print(f"  真实脂质 n_rings 目标分布：{top}")
        idx, samp_diag = sample_scaffold_quota(
            all_smiles, n_cand, args.seed, C.N_JOBS,
            ring_profile=ring_profile,
            max_frac_per_scaffold=args.max_frac_per_scaffold)
    elif args.strategy == "random":
        idx, samp_diag = sample_random(all_smiles, n_cand, args.seed)
    else:
        idx, samp_diag = sample_prefix(all_smiles, args.n_rows or n_cand)

    new_smiles = all_smiles[idx].tolist()
    cand_scaf_code = None
    if args.strategy == "scaffold_quota":
        # 保留候选行的骨架分组，供筛后精确裁剪时二次配额（保持分层）
        cand_scaf_code = _scaffold_chunk(new_smiles)
    del all_smiles
    print(f"  候选行数：{len(new_smiles):,}（目标有效新增 {args.target_new:,}）")

    # 3) 标准化 + 可电离/性质过滤（与阶段 0 完全一致）
    print(f"\n  标准化 + 质量筛选（{C.N_JOBS} 进程）...")
    results = Parallel(n_jobs=C.N_JOBS, batch_size=4096)(
        delayed(screen_one)(s) for s in tqdm(new_smiles, ncols=76)
    )

    rejects = {}
    keep_rows, keep_scaf = [], []
    for i, r in enumerate(results):
        if r is None:
            rejects["unparseable"] = rejects.get("unparseable", 0) + 1
        elif "reject" in r:
            rejects[r["reject"]] = rejects.get(r["reject"], 0) + 1
        else:
            keep_rows.append(r)
            keep_scaf.append(cand_scaf_code[i] if cand_scaf_code else None)

    passed = pd.DataFrame(keep_rows)
    passed["_scaf"] = keep_scaf
    print(f"\n  通过质量筛选：{len(passed):,}")
    if rejects:
        print("  被过滤（top）："
              + ", ".join(f"{k}={v:,}" for k, v in sorted(rejects.items(), key=lambda x: -x[1])[:8]))

    # 4) 新增行内部按 InChIKey 去重
    n_dup_internal = int(passed.duplicated(subset="inchikey").sum())
    passed = passed.drop_duplicates(subset="inchikey").reset_index(drop=True)
    print(f"  新增行内部 InChIKey 重复：{n_dup_internal:,}（剔除）")

    # 5) 与 base 去泄漏：剔除已存在于 base 的分子
    leaked = passed.inchikey.isin(base_keys)
    n_leaked = int(leaked.sum())
    passed = passed[~leaked].reset_index(drop=True)
    print(f"  与 base 重叠（去泄漏剔除）：{n_leaked:,}")

    # 6) 精确裁剪到 target —— 分层策略下用二次配额裁剪，保持骨架均摊
    n_before_trim = len(passed)
    if len(passed) > args.target_new:
        if args.strategy == "scaffold_quota":
            # 二次配额裁剪：按 (骨架, 分子实际环数) 联合分组，
            # 既保住骨架均摊，也保住上一步按真实分布调好的环数画像。
            rng = np.random.default_rng(args.seed + 1)
            key = passed["_scaf"].astype(str) + "|" + passed["n_rings"].astype(str)
            codes, _ = pd.factorize(key, use_na_sentinel=False)
            uniq, counts = np.unique(codes, return_counts=True)
            quota = waterfill_quota(counts, args.target_new, rng)
            take = []
            for gi, g in enumerate(uniq):
                q = int(quota[gi])
                if q <= 0:
                    continue
                rows = np.flatnonzero(codes == g)
                take.append(rows if q >= len(rows) else rng.choice(rows, q, replace=False))
            passed = passed.iloc[np.sort(np.concatenate(take))].reset_index(drop=True)
        else:
            passed = passed.sample(args.target_new, random_state=args.seed).reset_index(drop=True)
        print(f"  裁剪到目标规模：{n_before_trim:,} → {len(passed):,}")
    elif len(passed) < args.target_new:
        print(f"  ⚠️  有效新增 {len(passed):,} < 目标 {args.target_new:,}，"
              f"如需补足请增大 --target-new 对应的过采样或放宽策略")
    print(f"  ▶ 有效新增（去重+去泄漏+裁剪后）：{len(passed):,}")

    passed = passed.drop(columns=["_scaf"])

    # 7) 补列，与 base 对齐，标 virtual
    passed["source_type"] = "virtual"
    passed["sources"] = "generated_virtual"
    passed["rxn_type"] = pd.NA
    passed["head_id"] = pd.NA
    passed["tail_id"] = pd.NA

    # 只保留 base 中实际存在的列，保证 concat 不产生 NaN 空列
    base_cols = set(base.columns)
    new_cols = [c for c in passed.columns if c in base_cols]
    base_use = base[list(base.columns)].copy()
    passed_use = passed[new_cols].copy()

    # 顺序对齐 base 列顺序
    col_order = list(base.columns)
    passed_use = passed_use[col_order]

    corpus = pd.concat([base_use, passed_use], ignore_index=True)
    corpus = corpus.drop_duplicates(subset="inchikey").reset_index(drop=True)

    corpus.to_parquet(v_path, index=False)
    passed_use.to_parquet(newonly_path, index=False)
    print(f"\n  {args.version} 语料写入：{v_path}")
    print(f"  仅新增分子写入：{newonly_path}（阶段 1 只对增量生成构象）")
    print(f"  规模：{len(corpus):,}"
          f"（real {int((corpus.source_type == 'real').sum()):,}"
          f" + virtual {int((corpus.source_type == 'virtual').sum()):,}）")

    # 8) 去重统计报告
    _write_report(base, passed_use, len(new_smiles), len(passed_use), n_leaked,
                  n_dup_internal, rejects, report_path, v_path, args, samp_diag)
    print(f"  去重报告写入：{report_path}")
    print("=" * 72)
    return 0


def _write_report(base, new_passed, n_read, n_kept, n_leaked, n_internal_dup,
                  rejects, path, v_path, args, samp_diag):
    """写去重统计报告（采样策略 + 重叠数 → 有效新增行数）。"""
    L = [
        f"# {args.version} 语料去重统计报告\n",
        "## 0. 采样策略\n",
        "| 项 | 值 |", "|---|---|",
        f"| strategy | `{args.strategy}` |",
        f"| 目标有效新增 | {args.target_new:,} |",
        f"| 随机种子 | {args.seed} |",
    ]
    for k, v in samp_diag.items():
        if k != "strategy":
            L.append(f"| {k} | {v:,} |" if isinstance(v, int) else f"| {k} | {v} |")
    if args.strategy == "prefix":
        L.append("|  ⚠️ | prefix 口径与 CSV 的头基排序耦合，会导致头基坍缩，勿用于训练 |")

    L += [
        "", "## 1. 输入\n",
        "| 项 | 数值 |", "|---|---|",
        f"| 候选行数（过采样后） | {n_read:,} |",
        f"| 既有 base 语料分子数 | {len(base):,} |",
        f"| 有效新增（最终） | {n_kept:,} |\n",
        "## 2. 被过滤原因（候选 SMILES）\n",
        "| 原因 | 数量 |", "|---|---|",
    ]
    for k, v in sorted(rejects.items(), key=lambda x: -x[1]):
        L.append(f"| `{k}` | {v:,} |")

    L += ["", "## 3. 去重漏斗\n",
          "| 步骤 | 数量 |", "|---|---|",
          f"| 候选行数 | {n_read:,} |",
          f"| 新增行内部 InChIKey 去重 | -{n_internal_dup:,} |",
          f"| 与既有语料 InChIKey 重叠（去泄漏） | -{n_leaked:,} |",
          f"| **有效新增分子（裁剪到目标后）** | **{n_kept:,}** |\n",
          "## 4. 新增分子性质分布（中位数）\n",
          "| 描述符 | 中位数 |", "|---|---|"]
    for c in ["mw", "logp", "n_rot_bonds", "n_heavy_atoms", "n_basic_amines", "n_rings"]:
        if c in new_passed.columns:
            L.append(f"| {c} | {new_passed[c].median():.2f} |")

    L += ["", "## 5. 结论\n",
          f"- **有效新增 {n_kept:,} 行**。",
          f"- {args.version} 语料 = 既有 {len(base):,} + 新增 {n_kept:,} = "
          f"**{len(base) + n_kept:,}** 分子。",
          f"- 写入：{v_path}\n"]
    path.write_text("\n".join(L), encoding="utf-8")
    return "\n".join(L)


if __name__ == "__main__":
    sys.exit(main())
