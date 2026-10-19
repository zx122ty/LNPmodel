#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
scaffold_balanced_sampling.py

针对可电离脂质分子库，基于 Murcko scaffold 做平衡采样。
目标：用于 Uni-Mol2 域适应微调前，对输入分子库进行精简，平衡高频和低频 scaffold。

输入:
    一个 CSV 文件，其中包含 SMILES 列，默认列名为 SMILES。

输出:
    精简后的 CSV 文件。

默认采样策略:
    q_i ∝ N_i^tau

其中:
    N_i 是第 i 个 scaffold 的分子数。
    tau 越小，分布越平滑，低频 scaffold 越容易被保留。
    tau = 1 时，等价于按原始 scaffold 频率采样。
    tau = 0 时，等价于按 scaffold 均匀采样。

python scaffold_balanced_sampling.py \
    --input ionizable_lipids.csv \
    --output ionizable_lipids_balanced.csv \
    --smiles-col SMILES \
    --n-samples 200000 \
    --method power \
    --alpha 0.3 \
    --min-per-scaffold 1
"""

import argparse
import logging

import numpy as np
import pandas as pd

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles


RDLogger.DisableLog("rdApp.*")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def safe_murcko_scaffold(smiles: str):
    """
    安全计算 Murcko scaffold。
    如果 SMILES 无效或无法解析，返回 None。
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        scaffold = MurckoScaffoldSmiles(mol=mol)

        if not scaffold:
            return None

        return scaffold

    except Exception:
        return None


def compute_scaffold_probs(
    counts: pd.Series,
    method: str = "power",
    tau: float = 0.005,
    alpha: float = None,
) -> pd.Series:
    """
    计算每个 scaffold 的采样概率。

    参数:
        counts:
            index 为 scaffold，value 为该 scaffold 的分子数。
        method:
            "power": 推荐使用，q_i ∝ N_i^alpha。
                     如果 alpha 未指定，则使用 tau 作为 alpha。
            "paper": 按论文字面公式，softmax(P_i / tau)。
                     注意：该模式不一定能平衡高低频。
            "uniform": 每个 scaffold 等概率采样。
        tau:
            温度/平滑系数。
            在 power 模式下，tau 默认作为指数使用。
        alpha:
            power 模式下显式指定指数。
            如果指定，会覆盖 tau。

    返回:
        每个 scaffold 的采样概率。
    """
    n = counts.to_numpy(dtype=np.float64)

    if n.sum() <= 0:
        raise ValueError("没有有效的 scaffold 统计结果。")

    if method == "power":
        exponent = alpha if alpha is not None else tau

        if exponent < 0:
            raise ValueError("power 模式下 tau / alpha 不能小于 0。")

        # softmax(alpha * log(N_i)) 等价于 q_i ∝ N_i^alpha
        logits = exponent * np.log(n + 1e-12)

    elif method == "paper":
        if tau <= 0:
            raise ValueError("paper 模式下 tau 必须大于 0。")

        # P_i = N_i / sum_j N_j
        p = n / n.sum()

        # softmax(P_i / tau)
        logits = p / tau

    elif method == "uniform":
        logits = np.zeros_like(n)

    else:
        raise ValueError(f"未知 method: {method}")

    # 数值稳定
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)

    return pd.Series(probs, index=counts.index, name="scaffold_prob")


def allocate_samples_no_replacement(
    counts: pd.Series,
    probs: pd.Series,
    n_total: int,
    min_per_scaffold: int = 1,
) -> np.ndarray:
    """
    无放回采样时，为每个 scaffold 分配要采样的分子数。

    逻辑:
        1. 如果 min_per_scaffold > 0，并且目标样本数足够，
           则先给每个 scaffold 至少分配 min_per_scaffold 条。
        2. 剩余样本数按照 scaffold 概率 probs 分配。
        3. 每个 scaffold 的采样数不能超过其原始分子数。

    返回:
        与 counts 顺序一致的采样数量数组。
    """
    counts_arr = counts.to_numpy(dtype=np.int64)
    probs_arr = probs.reindex(counts.index).to_numpy(dtype=np.float64)

    total_available = int(counts_arr.sum())
    n_total = int(min(n_total, total_available))

    alloc = np.zeros_like(counts_arr, dtype=np.int64)

    # 尽量保证每个 scaffold 至少保留 min_per_scaffold 条
    if min_per_scaffold > 0:
        base = np.minimum(counts_arr, min_per_scaffold).astype(np.int64)

        if n_total >= int(base.sum()):
            alloc += base
        else:
            logger.warning(
                f"目标采样数 {n_total} 小于唯一 scaffold 数附近的最小保留需求 {int(base.sum())}，"
                "无法满足所有 scaffold 至少保留指定数量，将改为按概率采样。"
            )

    remaining = n_total - int(alloc.sum())

    if remaining <= 0:
        return alloc

    active = (counts_arr - alloc) > 0

    while remaining > 0 and active.any():
        active_idx = np.where(active)[0]
        p_active = probs_arr[active_idx]

        p_sum = p_active.sum()
        if not np.isfinite(p_sum) or p_sum <= 0:
            p_active = np.ones_like(p_active) / len(p_active)
        else:
            p_active = p_active / p_sum

        avail = counts_arr[active_idx] - alloc[active_idx]

        # 期望继续分配的连续数量
        desired = p_active * remaining

        # 如果某个 scaffold 的需求已经达到其最大可用量，则先封顶
        capped = desired >= avail

        if capped.any():
            capped_idx = active_idx[capped]
            alloc[capped_idx] = counts_arr[capped_idx]

            remaining = n_total - int(alloc.sum())
            active = (counts_arr - alloc) > 0
            continue

        floors = np.floor(desired).astype(np.int64)
        floors = np.minimum(floors, avail)

        rem = remaining - int(floors.sum())

        if rem > 0:
            can_inc = (avail - floors) > 0

            if not can_inc.any():
                # 理论上不应发生，作为安全兜底
                alloc[active_idx] = counts_arr[active_idx]
                remaining = n_total - int(alloc.sum())
                active = (counts_arr - alloc) > 0
                continue

            if rem > int(can_inc.sum()):
                # 可加位置不够，全部加 1 后继续循环
                inc_idx = np.where(can_inc)[0]
                floors[inc_idx] += 1
            else:
                # 按小数部分从大到小补齐剩余样本数
                frac = desired - floors
                frac_masked = np.where(can_inc, frac, -np.inf)
                top = np.argsort(-frac_masked)[:rem]
                floors[top] += 1

        alloc[active_idx] += floors
        alloc = np.minimum(alloc, counts_arr)

        remaining = n_total - int(alloc.sum())
        active = (counts_arr - alloc) > 0

    return alloc


def sample_by_scaffold_balanced(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    n_samples: int = None,
    method: str = "power",
    tau: float = 0.005,
    alpha: float = None,
    replace: bool = False,
    min_per_scaffold: int = 1,
    seed: int = 42,
    keep_unparseable: bool = False,
) -> pd.DataFrame:
    """
    根据 Murcko scaffold 做平衡采样。

    参数:
        df:
            输入 DataFrame。
        smiles_col:
            SMILES 列名。
        n_samples:
            目标输出行数。
        method:
            "power": q_i ∝ N_i^alpha，推荐。
            "paper": 按论文字面公式。
            "uniform": scaffold 均匀。
        tau:
            平滑系数。
            power 模式下，如果 alpha 未指定，则使用 tau 作为指数。
        alpha:
            power 模式下的指数，指定后覆盖 tau。
        replace:
            True: 有放回采样，可能出现重复行。
            False: 无放回采样，输出不重复。
        min_per_scaffold:
            无放回模式下，尽量保证每个 scaffold 至少保留的条数。
        seed:
            随机种子。
        keep_unparseable:
            是否保留无法解析的 SMILES。
            False: 丢弃。
            True: 归入 __UNPARSEABLE__ scaffold。

    返回:
        采样后的 DataFrame。
    """
    if smiles_col not in df.columns:
        raise ValueError(f"输入 CSV 中未找到列: {smiles_col}")

    if n_samples is None:
        raise ValueError("请指定 n_samples，即目标输出行数。")

    if n_samples <= 0:
        raise ValueError("n_samples 必须大于 0。")

    df = df.copy().reset_index(drop=True)

    logger.info("开始计算 Murcko scaffold ...")

    df["_scaffold"] = df[smiles_col].map(safe_murcko_scaffold)

    n_failed = int(df["_scaffold"].isna().sum())
    if n_failed > 0:
        logger.warning(
            f"共有 {n_failed} 行 SMILES 无法解析或无法生成 Murcko scaffold。"
        )

    if keep_unparseable:
        df["_scaffold"] = df["_scaffold"].fillna("__UNPARSEABLE__")
    else:
        df = df.dropna(subset=["_scaffold"]).copy().reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("处理后可用分子数为 0，请检查输入数据。")

    scaffold_counts = df["_scaffold"].value_counts()

    logger.info(f"有效分子数: {len(df)}")
    logger.info(f"唯一 scaffold 数: {len(scaffold_counts)}")

    scaffold_probs = compute_scaffold_probs(
        counts=scaffold_counts,
        method=method,
        tau=tau,
        alpha=alpha,
    )

    logger.info(f"采样模式: {method}")
    if method == "power":
        effective_alpha = alpha if alpha is not None else tau
        logger.info(f"power 模式实际指数: {effective_alpha}")
    elif method == "paper":
        logger.info(f"paper 模式 tau: {tau}")

    if not replace and n_samples > len(df):
        logger.warning(
            f"无放回采样时目标行数 {n_samples} 超过有效分子数 {len(df)}，"
            f"输出将最多为 {len(df)}。"
        )

    group_indices = df.groupby("_scaffold").indices
    rng = np.random.default_rng(seed)

    if replace:
        # 有放回采样：适合模拟训练采样，但输出可能重复
        scaffold_labels = scaffold_counts.index.to_numpy()
        probs_arr = scaffold_probs.to_numpy(dtype=np.float64)

        chosen_positions = rng.choice(
            len(scaffold_labels),
            size=n_samples,
            replace=True,
            p=probs_arr,
        )

        chosen_scaffolds = scaffold_labels[chosen_positions]
        uniq_scaffolds, uniq_counts = np.unique(chosen_scaffolds, return_counts=True)

        chosen_indices = []

        for scaffold, num in zip(uniq_scaffolds, uniq_counts):
            idx = np.array(group_indices[scaffold], dtype=np.int64)
            chosen = rng.choice(idx, size=int(num), replace=True)
            chosen_indices.append(chosen)

        if chosen_indices:
            chosen_indices = np.concatenate(chosen_indices)
            rng.shuffle(chosen_indices)
        else:
            chosen_indices = np.array([], dtype=np.int64)

        selected = df.loc[chosen_indices].reset_index(drop=True)

    else:
        # 无放回采样：适合生成精简后的分子库，输出不重复
        alloc = allocate_samples_no_replacement(
            counts=scaffold_counts,
            probs=scaffold_probs,
            n_total=n_samples,
            min_per_scaffold=min_per_scaffold,
        )

        chosen_indices = []

        for scaffold, take in zip(scaffold_counts.index, alloc):
            take = int(take)

            if take <= 0:
                continue

            idx = np.array(group_indices[scaffold], dtype=np.int64)
            take = min(take, len(idx))

            chosen = rng.choice(idx, size=take, replace=False)
            chosen_indices.append(chosen)

        if chosen_indices:
            chosen_indices = np.concatenate(chosen_indices)
            rng.shuffle(chosen_indices)
        else:
            chosen_indices = np.array([], dtype=np.int64)

        selected = df.loc[chosen_indices].reset_index(drop=True)

    # 删除辅助列
    selected = selected.drop(columns=["_scaffold"], errors="ignore")

    return selected


def main():
    parser = argparse.ArgumentParser(
        description="基于 Murcko scaffold 的平衡采样工具，适用于可电离脂质分子库精简。"
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入 CSV 文件路径。"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出 CSV 文件路径。"
    )

    parser.add_argument(
        "--smiles-col",
        type=str,
        default="SMILES",
        help="SMILES 列名，默认为 SMILES。"
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        required=True,
        help="精简后的目标分子数量，即输出 CSV 行数。"
    )

    parser.add_argument(
        "--method",
        type=str,
        default="power",
        choices=["power", "paper", "uniform"],
        help=(
            "采样概率计算方式。"
            "power: 推荐，q_i ∝ N_i^alpha，可用于平衡高低频；"
            "paper: 按论文字面公式 softmax(P_i / tau)；"
            "uniform: 每个 scaffold 等概率。"
        )
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="温度/平滑系数。power 模式下若未指定 alpha，则 tau 作为指数使用。"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="power 模式下显式指定指数。设置后会覆盖 tau。"
    )

    parser.add_argument(
        "--min-per-scaffold",
        type=int,
        default=1,
        help="无放回模式下，尽量保证每个 scaffold 至少保留的条数，默认为 1。"
    )

    parser.add_argument(
        "--replace",
        action="store_true",
        help="是否有放回采样。默认无放回。"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42。"
    )

    parser.add_argument(
        "--add-scaffold",
        action="store_true",
        help="是否在输出中保留/添加 MurckoScaffold 列。"
    )

    parser.add_argument(
        "--keep-unparseable",
        action="store_true",
        help="是否保留无法解析的 SMILES。默认丢弃。"
    )

    args = parser.parse_args()

    logger.info(f"读取输入文件: {args.input}")

    df = pd.read_csv(args.input)

    selected = sample_by_scaffold_balanced(
        df=df,
        smiles_col=args.smiles_col,
        n_samples=args.n_samples,
        method=args.method,
        tau=args.tau,
        alpha=args.alpha,
        replace=args.replace,
        min_per_scaffold=args.min_per_scaffold,
        seed=args.seed,
        keep_unparseable=args.keep_unparseable,
    )

    # 如果用户要求输出 scaffold 列，需要重新计算一次
    if args.add_scaffold:
        selected["MurckoScaffold"] = selected[args.smiles_col].map(
            safe_murcko_scaffold
        )

    selected.to_csv(args.output, index=False)

    logger.info(f"采样完成，输出文件: {args.output}")
    logger.info(f"输出行数: {len(selected)}")


if __name__ == "__main__":
    main()