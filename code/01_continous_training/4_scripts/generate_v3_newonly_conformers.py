"""
generate_v3_newonly_conformers.py —— 只对 v3 语料中的【新增】脂质生成构象。

背景/思路
========
v2 语料（real 12,381 + virtual 100,000 = 112,381）的 3D 构象已经在
`unimolv2_3d/lipids_unimolv2_3d.sdf`（615,361 个 conformer）。v3 只是在语料里新增了
119,230 个全新 virtual 脂质。**base 分子无需重新生成构象**——直接复用 v2 的 SDF 即可，
避免 ~47h 的重复计算。

本脚本：
  1. 读 `pretrain_corpus_v3_newonly.parquet`（仅 119,230 个新增 virtual 分子）。
  2. 用与 prepare_lipids_unimol_dataset.py 相同的构象管线（ETKDG randomCoords + MMFF94s 2000it，
     virtual 4->3）生成每个分子的 3D 构象。
  3. 输出到独立 SDF `unimolv2_3d_v3_newonly/`，**绝不触碰/覆盖 v2 的 SDF**。
  4. 后续用合并步骤：v2 SDF + newonly SDF → `unimolv2_3d_v3/lipids_unimolv2_3d.sdf`（v3 完整）。

ROW_INDEX 唯一性：
  - base 语料行号占 0..112380。新增脂质在 v3 全语料 `pretrain_corpus_v3.parquet` 中的行号是
    112381..231610。本脚本直接采用 **v3 全语料行号**作为 ROW_INDEX，与后续 LMDB 建库按行对齐、
    不与 base 冲突。

用法：
    conda activate unimol2
    python generate_v3_newonly_conformers.py [--n-workers N] [--num-molecules M]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_lipids_unimol_dataset as P  # noqa: E402

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[1]

# 版本相关路径由 --version 决定（默认 v3，保持既有调用兼容）。
RAW_DIR = REPO_ROOT / "1_data/raw/ionizable_lipids"
PROC_DIR = REPO_ROOT / "1_data/processed/ionizable_lipids"

V3_CORPUS = RAW_DIR / "pretrain_corpus_v3.parquet"
NEWONLY_CORPUS = RAW_DIR / "pretrain_corpus_v3_newonly.parquet"
OUT_DIR = PROC_DIR / "unimolv2_3d_v3_newonly"


def set_version(version: str):
    """把全局路径切到指定语料版本（v3 / v4 / ...）。"""
    global V3_CORPUS, NEWONLY_CORPUS, OUT_DIR
    V3_CORPUS = RAW_DIR / f"pretrain_corpus_{version}.parquet"
    NEWONLY_CORPUS = RAW_DIR / f"pretrain_corpus_{version}_newonly.parquet"
    OUT_DIR = PROC_DIR / f"unimolv2_3d_{version}_newonly"
    for p in (V3_CORPUS, NEWONLY_CORPUS):
        if not p.exists():
            sys.exit(f"语料不存在：{p}")

CONFIG = {
    "num_conformers_real": 24,
    "keep_conformers_real": 20,
    "num_conformers_virtual": 4,   # 与 v2 的 virtual 一致
    "keep_conformers_virtual": 3,
    "max_iterations": 2000,
    "max_atoms": 256,
    "random_coords": True,
    "seed": 20260805,
}

# v2 语料行号占 0..112380，新增脂质从 112381 起（ROW_INDEX 直接采用 v3 全语料行号）


def build_tasks(num_molecules: int | None) -> list[tuple[int, dict, dict]]:
    """构造 (row_index, row_dict, config) 任务列表，row_index 用 v3 全量行号。

    v3 全语料 `pretrain_corpus_v3.parquet` 的行号就是我们要用的 ROW_INDEX：
    - base 语料（前 112,381 行）占 0..112380；
    - 新增脂质占 112381..231610。
    因此 ROW_INDEX = 在 v3 全语料中的行号本身，不要再加额外偏移（否则会双重移位）。
    """
    full = pd.read_parquet(V3_CORPUS, columns=["inchikey"])
    ik2row = {k: i for i, k in enumerate(full.inchikey)}   # 直接用 v3 行号，不 +偏移
    newonly = pd.read_parquet(NEWONLY_CORPUS)
    tasks = []
    for _, r in newonly.iterrows():
        ik = r.inchikey
        row_index = ik2row[ik]
        row = {"mol_id": ik, "smiles": r.smiles_std, "source_type": r.source_type}
        tasks.append((row_index, row, CONFIG))
    if num_molecules:
        tasks = tasks[:num_molecules]
    return tasks


def iter_results(tasks, n_workers):
    """并行跑 prepare_molecule，逐条产出结果（与既有脚本一致）。"""
    if n_workers <= 1:
        for t in tasks:
            yield P.prepare_molecule(t)
        return
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(P.prepare_molecule, t): t for t in tasks}
        for fut in as_completed(futures):
            yield fut.result()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v3",
                    help="语料版本：v3 / v4 ...（决定输入 parquet 与输出目录）")
    ap.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--num-molecules", type=int, default=None, help="只处理前 N 个（冒烟）")
    args = ap.parse_args()

    set_version(args.version)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sdf_path = OUT_DIR / "lipids_unimolv2_3d.sdf"
    failure_path = OUT_DIR / "lipids_unimolv2_failures.csv"

    tasks = build_tasks(args.num_molecules)
    print(f"语料版本    ：{args.version}")
    print(f"新增脂质任务数：{len(tasks):,}")
    print(f"输出 SDF    ：{sdf_path}")

    # 写失败表头
    if not failure_path.exists():
        with failure_path.open("w", encoding="utf-8") as fh:
            import csv
            csv.DictWriter(fh, fieldnames=P.FAILURE_FIELDS).writeheader()

    n_success = n_fail = written = 0
    reasons: dict[str, int] = {}
    with sdf_path.open("w", encoding="utf-8") as fh:
        for result in iter_results(tasks, args.n_workers):
            if result.status == "success":
                n_success += 1
                written += _append_success(fh, result)
            else:
                n_fail += 1
                with failure_path.open("a", encoding="utf-8") as ff:
                    P.append_failure(ff, result)  # 用既有 append_failure
                reasons[result.failure_reason] = reasons.get(result.failure_reason, 0) + 1

    summary = {
        "processed_rows": len(tasks),
        "successful_molecules": n_success,
        "failed_molecules": n_fail,
        "written_conformers": written,
        "failure_reasons": reasons,
        "config": CONFIG,
    }
    (OUT_DIR / "lipids_unimolv2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _append_success(handle, result):
    """写一个成功分子的所有 conformer 到 SDF（复用既有 sdf_record）。"""
    assert result.selected_conformers is not None
    for conf in result.selected_conformers:
        handle.write(P.sdf_record(conf.mol_block, {
            "MOL_ID": result.mol_id,
            "ROW_INDEX": result.row_index,
            "SMILES": result.smiles,
            "CANONICAL_SMILES": result.canonical_smiles or "",
            "SOURCE_TYPE": result.source_type,
            "HEAVY_ATOM_COUNT": result.heavy_atom_count or "",
            "TOTAL_ATOM_COUNT": result.total_atom_count or "",
            "CONFORMER_RANK": conf.conformer_rank,
            "CONFORMER_ENERGY": f"{conf.conformer_energy:.8f}",
        }))
    return len(result.selected_conformers)


if __name__ == "__main__":
    sys.exit(main())