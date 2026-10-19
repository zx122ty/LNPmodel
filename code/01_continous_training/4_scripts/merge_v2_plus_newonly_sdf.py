"""
merge_v2_plus_newonly_sdf.py —— 合并 v2 完整 SDF 与新增脂质 SDF → v3 完整 SDF。

v3 语料 = 既有 v2 语料（112,381 分子，构象在 v2 SDF）+ 新增 119,230 个 virtual 脂质
（构象由 generate_v3_newonly_conformers.py 生成在 newonly SDF）。

本脚本把两个 SDF 字节级拼接成 v3 完整训练 SDF（每个 SDF block 以 `$$$$` 结尾，拼接合法），
并写入：
    1_data/processed/ionizable_lipids/unimolv2_3d_v3/lipids_unimolv2_3d.sdf

绝不覆盖 v2 的 SDF（v2 保持可复现）。

用法：
    conda activate unimol2
    python merge_v2_plus_newonly_sdf.py
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = REPO_ROOT / "1_data/processed/ionizable_lipids"

V2_SDF = PROC_DIR / "unimolv2_3d/lipids_unimolv2_3d.sdf"
NEWONLY_SDF = PROC_DIR / "unimolv2_3d_v3_newonly/lipids_unimolv2_3d.sdf"
OUT_DIR = PROC_DIR / "unimolv2_3d_v3"
OUT_SDF = OUT_DIR / "lipids_unimolv2_3d.sdf"


def set_version(version: str):
    """把输入/输出路径切到指定语料版本（v3 / v4 / ...）。v2 SDF 始终作为 base 复用。"""
    global NEWONLY_SDF, OUT_DIR, OUT_SDF
    NEWONLY_SDF = PROC_DIR / f"unimolv2_3d_{version}_newonly/lipids_unimolv2_3d.sdf"
    OUT_DIR = PROC_DIR / f"unimolv2_3d_{version}"
    OUT_SDF = OUT_DIR / "lipids_unimolv2_3d.sdf"


def count_blocks(sdf_path: Path) -> int:
    n = 0
    with sdf_path.open("rb") as fh:
        for line in fh:
            if line.strip() == b"$$$$":
                n += 1
    return n


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v3",
                    help="语料版本：v3 / v4 ...（决定 newonly 输入与合并输出目录）")
    args = ap.parse_args()
    set_version(args.version)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in (V2_SDF, NEWONLY_SDF):
        if not p.exists():
            sys.exit(f"缺少输入 SDF：{p}")

    n_v2 = count_blocks(V2_SDF)
    n_new = count_blocks(NEWONLY_SDF)
    print(f"语料版本                 : {args.version}")
    print(f"v2 SDF conformer blocks : {n_v2:,}")
    print(f"新增 SDF conformer blocks: {n_new:,}")
    print(f"合并后 total            : {n_v2 + n_new:,}")

    # 逐块拷贝合并（避免一次性载入 8GB 内存）
    with OUT_SDF.open("wb") as out:
        for src in (V2_SDF, NEWONLY_SDF):
            with src.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    out.write(chunk)

    print(f"\n合并完成，写入：{OUT_SDF}")
    print(f"校验块数：{count_blocks(OUT_SDF):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())