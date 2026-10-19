#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_component.py — PubChem SDF → 轻量候选池 文本级流式提取（Ugi-4CR 底物库通用 Stage1）
=========================================================================================
为四大 Ugi-4CR 底物子库（Head / 醛酮 / Acid / 异腈）从 PubChem SDF 分片构建**通用候选池**。

为什么不用 RDKit 解析 SDF 分子块
-------------------------------
359 个分片约 1.8 亿条记录、109 GB。RDKit 逐条解析分子块成本高。
PubChem SDF 自带预生成 SMILES（PUBCHEM_SMILES，pKa 质子化态、含 [+]/[-]），
因此 Stage1 只做"文本级"字段抽取 + 轻量过滤，把体积压缩 5–15 倍，
各子库的 RDKit 精筛（Stage2）在候选池上进行，两者解耦、互可重跑。

输出字段（每分片一份 .tsv.gz，制表符分隔，表头 = COLS）：
    cid, smiles, inchikey, total_charge, mw, component_count, tpsa, rotatable_bonds

内置粗滤（可参数微调，宁多勿漏）：
    * SMILES 含指定子串（默认 N 或 n）
    * PUBCHEM_TOTAL_CHARGE == 0  （整分子净电荷 0；含盐/两性分子留待 Stage2 去盐/电离）
    * 10 <= MW <= 1200          （宽窗口；各子库 MW 窗在 Stage2 收紧）
    * component_count <= 3      （排除明显混合物/聚合物）
    * 可选：排除卤素（F/Cl/Br/I）、常见金属离子

增量 / 断点续跑 / 并行
----------------------
* 以"分片文件"为粒度：完成的分片路径写入 --done-file（每行一个绝对路径）。
* 启动时自动跳过已完成分片；输出经 .tmp → 原子 rename，不会留下半截文件。
* 多进程启动多个实例共享同一 --done-file，fcntl 文件锁保证追加不冲突。
* --split N 只跑第 N 个分片（用于单分片验证）。

用法示例
--------
    # 全库多实例并行
    python scripts/main_component.py --out-dir data/head_v0.1/candidates -t 12 &
    python scripts/main_component.py --out-dir data/head_v0.1/candidates -t 12 &
    # 单分片验证
    python scripts/main_component.py --split 1 -t 4

环境：python3.10，仅标准库（gzip / concurrent.futures / fcntl）。
"""

import argparse
import fcntl
import gzip
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

COLS = ["cid", "smiles", "inchikey", "total_charge", "mw",
        "component_count", "tpsa", "rotatable_bonds"]

# SDF 数据项名 → 输出字段名
FIELD_NAME = {
    "cid": "PUBCHEM_COMPOUND_CID",
    "smiles": "PUBCHEM_SMILES",
    "inchikey": "PUBCHEM_IUPAC_INCHIKEY",
    "total_charge": "PUBCHEM_TOTAL_CHARGE",
    "mw": "PUBCHEM_MOLECULAR_WEIGHT",
    "component_count": "PUBCHEM_COMPONENT_COUNT",
    "tpsa": "PUBCHEM_CACTVS_TPSA",
    "rotatable_bonds": "PUBCHEM_CACTVS_ROTATABLE_BOND",
}

DEFAULT_MW_LO, DEFAULT_MW_HI = 10.0, 1200.0
DEFAULT_ATOM_FILTER = "N,n"   # SMILES 值须包含这些子串之一（区分大小写）

# SDF 标签行正则（处理多余空格）
_TAG_RE = re.compile(r">\s*<(.+?)>")

# 卤素检测正则（“裸符号紧贴裸符号”即代表卤素原子与其它原子成键的位置；
# 排除 'F', 'Cl', 'Br', 'I' 作为非卤素元素符号后缀的情况，如 [Ti] 中的 'i'；
# 兼容 '/' '\\' 前导（烯丙基/偕卤等写法）与括号后）
_HALOGEN_RE = re.compile(
    r"\[(?:F|Cl|Br|I)[+\-0-9]*\]|(?:(?<=^)|(?<=[A-Z()%0-9./\\=+\[\]-]))(?:F|Cl|Br|I)"
)

# 常见金属离子/原子正则（粗筛，减少下游负载）
_METAL_RE = re.compile(
    r"\[(?:Li|Be|Na|Mg|Al|K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|"
    r"Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|"
    r"Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|"
    r"Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|"
    r"Fr|Ra|Ac|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr)[+\-]?[0-9]?\]"
)


def _pf(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _pi(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _has_halogen(smiles):
    """文本级卤素检测"""
    return bool(_HALOGEN_RE.search(smiles))


def _has_metal(smiles):
    """文本级金属检测"""
    return bool(_METAL_RE.search(smiles))


def _parse_file(sdf_path: str, cfg: dict) -> dict:
    """解析单个 .sdf.gz 分片 → 写出 <basename>.tsv.gz → 返回统计。"""
    t0 = time.time()
    st = {"path": sdf_path, "records": 0, "kept": 0,
          "missing_smiles": 0, "seconds": None}

    out_path = os.path.join(cfg["out_dir"], os.path.basename(sdf_path)
                            .replace(".sdf.gz", "") + ".tsv.gz")
    tmp_path = out_path + ".tmp"
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # 若已存在输出文件则直接跳过（断点续跑，避免同分片重复执行）
    if os.path.exists(out_path) and not cfg.get("force", False):
        try:
            with gzip.open(out_path, "rt") as _chk:
                _chk.readline()
            st.update({"records": -1, "kept": -1, "missing_smiles": -1,
                       "seconds": 0.0, "skipped_existing": True})
            return st
        except Exception:
            os.remove(out_path)  # 损坏则重跑

    need_atoms = tuple(cfg["atom_filter"].split(","))
    mw_lo, mw_hi = cfg["mw_lo"], cfg["mw_hi"]
    max_comp = cfg["max_components"]
    exclude_halogens = cfg.get("exclude_halogens", True)
    exclude_metals = cfg.get("exclude_metals", True)

    n_records = 0
    n_kept = 0
    n_missing = 0

    try:
        with gzip.open(sdf_path, "rt", encoding="utf-8", errors="replace") as fh, \
             gzip.open(tmp_path, "wt") as out:
            out.write("\t".join(COLS) + "\n")

            rec = {}
            cur_field = None
            expect_value = False

            for line in fh:
                s = line.rstrip("\n").strip()

                if s == "$$$$":
                    # —— 记录结束 ——
                    smi = rec.get("PUBCHEM_SMILES", "").strip()
                    if smi:
                        ch = _pi(rec.get("PUBCHEM_TOTAL_CHARGE", "x"))
                        mw = _pf(rec.get("PUBCHEM_MOLECULAR_WEIGHT", "x"))
                        comp = _pi(rec.get("PUBCHEM_COMPONENT_COUNT", "x"))

                        # 核心过滤逻辑
                        # 注意：移除了 "[+" not in smi and "[-" not in smi 条件
                        # 允许两性离子/盐类进入候选池，由 Stage2 处理去盐/中性化
                        if (
                            ch == 0
                            and any(tok in smi for tok in need_atoms)
                            and mw is not None and mw_lo <= mw <= mw_hi
                            and (comp or 1) <= max_comp
                        ):
                            # 可选：卤素/金属排除
                            if exclude_halogens and _has_halogen(smi):
                                pass  # skip
                            elif exclude_metals and _has_metal(smi):
                                pass  # skip
                            else:
                                row = [
                                    rec.get("PUBCHEM_COMPOUND_CID", "").strip(),
                                    smi,
                                    rec.get("PUBCHEM_IUPAC_INCHIKEY", "").strip(),
                                    str(ch),
                                    format(mw, ".2f"),
                                    str(comp or ""),
                                    rec.get("PUBCHEM_CACTVS_TPSA", "").strip(),
                                    rec.get("PUBCHEM_CACTVS_ROTATABLE_BOND", "").strip(),
                                ]
                                out.write("\t".join(row) + "\n")
                                n_kept += 1
                    else:
                        n_missing += 1
                    n_records += 1
                    rec = {}
                    cur_field = None
                    expect_value = False
                    continue

                # 使用正则匹配标签行（更鲁棒）
                m = _TAG_RE.match(s)
                if m:
                    tag = m.group(1).strip()
                    if tag in FIELD_NAME.values():
                        cur_field = tag
                    else:
                        cur_field = None
                    expect_value = True
                    continue

                if expect_value:
                    expect_value = False
                    if cur_field is not None:
                        rec[cur_field] = s
                    continue

    except Exception as e:
        print(f"[error] 解析 {sdf_path} 失败: {e!r}", file=sys.stderr)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        st.update({"records": n_records, "kept": n_kept,
                   "missing_smiles": n_missing, "seconds": round(time.time() - t0, 1),
                   "error": str(e)})
        return st

    os.replace(tmp_path, out_path)
    st.update({"records": n_records, "kept": n_kept,
               "missing_smiles": n_missing, "seconds": round(time.time() - t0, 1)})
    return st


def _worker(args):
    path, cfg = args
    return _parse_file(path, cfg)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pubchem-dir", default="/home/zx/df/disk1/resources/pubchem")
    ap.add_argument("--out-dir", default="data/head_v0.1/candidates")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--split", type=int, default=None, help="仅处理第 N 个分片（1 起）")
    ap.add_argument("--filelist", default=None,
                    help="可选：只处理列出的绝对路径 SDF 文件（每行一个），用于分区并行/跳过损坏片")
    ap.add_argument("--force", action="store_true", help="忽略 existing 输出 强制重跑")
    ap.add_argument("--done-file", default="data/head_v0.1/.done_files_pc.txt")
    ap.add_argument("--min-mw", type=float, default=DEFAULT_MW_LO)
    ap.add_argument("--max-mw", type=float, default=DEFAULT_MW_HI)
    ap.add_argument("--atom-filter", default=DEFAULT_ATOM_FILTER,
                    help="SMILES 须含这些子串之一（逗号分隔，区分大小写）")
    ap.add_argument("--max-components", type=int, default=3)
    ap.add_argument("--allow-halogens", action="store_true",
                    help="允许含卤素分子进入候选池（默认排除）")
    ap.add_argument("--allow-metals", action="store_true",
                    help="允许含金属分子进入候选池（默认排除）")
    args = ap.parse_args()

    pubchem_dir = Path(args.pubchem_dir)
    sdf_files = sorted(pubchem_dir.glob("Compound_*.sdf.gz"))
    if args.filelist:
        with open(args.filelist) as f:
            listed = [ln.strip() for ln in f if ln.strip()]
        sdf_files = [Path(p) if str(p).startswith("/") else pubchem_dir / p for p in listed]
        sdf_files = [p for p in sdf_files if p.exists()]
    if not sdf_files:
        print(f"[error] {pubchem_dir} 下没有 Compound_*.sdf.gz 或 filelist 无效", file=sys.stderr)
        sys.exit(1)

    cfg = {
        "out_dir": os.path.abspath(args.out_dir),
        "atom_filter": args.atom_filter,
        "mw_lo": args.min_mw,
        "mw_hi": args.max_mw,
        "max_components": args.max_components,
        "exclude_halogens": not args.allow_halogens,
        "exclude_metals": not args.allow_metals,
        "force": args.force,
    }
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # 输出存在且可读 → 跳过（除非 --force）
    def _out_ok(p):
        if args.force:
            return False
        out = os.path.join(cfg["out_dir"], os.path.basename(p).replace(".sdf.gz", "") + ".tsv.gz")
        if not os.path.exists(out):
            return False
        try:
            with gzip.open(out, "rt") as f:
                f.readline()
            return True
        except Exception:
            return False

    todo = [str(p) for p in sdf_files if not _out_ok(p)]
    if args.split is not None:
        idx = args.split - 1
        if not (0 <= idx < len(sdf_files)):
            print(f"[abort] --split {args.split} 超出范围 1..{len(sdf_files)}", file=sys.stderr)
            sys.exit(1)
        todo = [str(sdf_files[idx])]
    if not todo:
        print(f"[info] 所有候选分片已有可读输出（{len(sdf_files)} 片），退出", file=sys.stderr)
        return

    print(f"[info] 分片总数 {len(sdf_files)}，本轮待处理 {len(todo)}，跳过 {len(sdf_files) - len(todo)}", flush=True)
    print(f"[info] 输出 {cfg['out_dir']}，线程 {min(args.threads, len(todo))}", flush=True)
    print(f"[info] 卤素排除: {cfg['exclude_halogens']}, 金属排除: {cfg['exclude_metals']}", flush=True)

    summary = []
    with ProcessPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(_worker, (p, cfg)): p for p in todo}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"[error] {p}: {e!r}", file=sys.stderr)
                continue
            summary.append(r)
            if r.get("error"):
                print(f"  [err] {os.path.basename(p)}: {r['error']}", file=sys.stderr)
            else:
                print(f"  [ok] {os.path.basename(p)}: records={r['records']:,} kept={r['kept']:,} "
                      f"{r['seconds']}s", flush=True)

    # ---- 写 done-file（fcntl 锁原子追加） --------------------------------
    if summary:
        with open(args.done_file, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                for r in summary:
                    if not r.get("error"):
                        f.write(r["path"] + "\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    print(json.dumps({
        "files_done": len(summary),
        "files_failed": len(todo) - len(summary),
        "records_total": sum(r["records"] for r in summary if r["records"] > 0),
        "kept_total": sum(r["kept"] for r in summary if r["kept"] > 0),
        "wall_seconds": round(sum(r["seconds"] for r in summary if r["seconds"]), 1),
    }, indent=2))


if __name__ == "__main__":
    main()