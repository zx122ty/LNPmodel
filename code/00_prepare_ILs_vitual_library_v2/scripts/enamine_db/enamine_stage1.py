#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
enamine_stage1.py — Enamine Building Blocks SDF → Ugi-4CR 四库候选池（Stage1）
=============================================================================
v2/main_component.py 的 Enamine 适配版。

关键差异（vs PubChem 版）：
  * Enamine SDF 是「单文件纯文本」5.5 GB（非 .sdf.gz 分片），无 PUBCHEM_* 标签：
        >  <Catalog_ID>   EN300-07843          → 用作 cid 列
        >  <smiles>       CCOC=1C=C(C=O)C=CC1OC(C)C(=O)N(C)C
  * 无预生成 MW / InChIKey / TPSA / 旋转键 / 净电荷 → 全部用 RDKit 现算
    （2.83M 条，24 核字节分片并行，完全可承受）。
  * 库内子池仍为「文本级原子过滤 + 宁多勿漏」，与 v2 对齐。

输出（按库、按字节范围分片，每片一个 .tsv.gz）:
    data/enamine_db/<lib>/candidates/part_<start>_<end>.tsv.gz
    列: cid, smiles, inchikey, total_charge, mw, component_count, tpsa, rotatable_bonds

原子过滤（文本级，与 v2/REPRODUCE.md 对齐）:
    head      : SMILES 含 N/n            ; MW 10–1200; comp≤3
    carbonyl  : SMILES 含 =O             ; MW 10–1200; comp≤3
    acid      : 含 =O 且不含 N/n         ; MW 10–1200; comp≤3
    isocyanide: 含 [N+]#[C-] / [C-]#[N+] ; MW 10–800;  comp≤4
通用: 净电荷==0，去卤素(F/Cl/Br/I)，去常见金属离子。

并行 / 断点续跑:
    --ranges N 将 SDF 按字节切 N 段（记录边界对齐），每段一个 worker
    同时写出四个库的 part 文件；part 存在且可读即跳过。

用法:
    python scripts/enamine_db/enamine_stage1.py --sdf <enamine.sdf> --out-dir data/enamine_db --ranges 8 --threads 8
    # 可 --range-idx i 只跑第 i 段（用于单段验证）

环境: python3.10 + RDKit（conda env unimol2）。
"""

import argparse
import gzip
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

COLS = ["cid", "smiles", "inchikey", "total_charge", "mw",
        "component_count", "tpsa", "rotatable_bonds"]

DEFAULT_MW_LO = 10.0

# 四库原子过滤（文本级）: 名字 → 配置
#   toks    : smiles 须含其一（区分大小写）
#   exclude : smiles 含其一则排除（仅 acid：剔除 N/n）
#   iso     : True 表示异腈库（按 [N+]#[C-]/[C-]#[N+] 子串判定）
LIB_CFG = {
    "head":       {"toks": ("N", "n"),                     "mw_hi": 1200.0, "comp": 3},
    "carbonyl":   {"toks": ("=O",),                        "mw_hi": 1200.0, "comp": 3},
    "acid":       {"toks": ("=O",), "exclude": ("N", "n"), "mw_hi": 1200.0, "comp": 3},
    "isocyanide": {"iso": True,                            "mw_hi": 800.0,  "comp": 4},
}

ISO_PATS = ("[N+]#[C-]", "[C-]#[N+]")

# SDF 标签行正则（处理多余空格）
_TAG_RE = re.compile(r">\s*<(.+?)>")

# 卤素检测正则（与 v2/main_component.py 相同）
_HALOGEN_RE = re.compile(
    r"\[(?:F|Cl|Br|I)[+\-0-9]*\]|(?:(?<=^)|(?<=[A-Z()%0-9./\\=+\[\]-]))(?:F|Cl|Br|I)"
)

# 常见金属离子/原子正则（与 v2 相同）
_METAL_RE = re.compile(
    r"\[(?:Li|Be|Na|Mg|Al|K|Ca|Sc|Ti|V|Cr|Mn|Fe|Co|Ni|Cu|Zn|Ga|Ge|"
    r"Rb|Sr|Y|Zr|Nb|Mo|Tc|Ru|Rh|Pd|Ag|Cd|In|Sn|"
    r"Cs|Ba|La|Ce|Pr|Nd|Pm|Sm|Eu|Gd|Tb|Dy|Ho|Er|Tm|Yb|Lu|"
    r"Hf|Ta|W|Re|Os|Ir|Pt|Au|Hg|Tl|Pb|Bi|Po|"
    r"Fr|Ra|Ac|Th|Pa|U|Np|Pu|Am|Cm|Bk|Cf|Es|Fm|Md|No|Lr)[+\-]?[0-9]?\]"
)


def _has_halogen(smi, iso=False):
    """文本级卤素检测；iso 库先移除异腈标记再检测。"""
    if iso:
        smi = smi.replace("[N+]#[C-]", "").replace("[C-]#[N+]", "")
    return bool(_HALOGEN_RE.search(smi))


def _has_metal(smi):
    return bool(_METAL_RE.search(smi))


def _analyze_mol(smiles, iso=False):
    """RDKit 现算 (canonical, inchikey, mw, net_charge, comp, tpsa, rotb)。

    返回 None 表示无法解析（head/carbonyl/acid 会丢弃该记录）。
    iso 库：用 sanitize=False 读入以容忍 `[N+]#[C-]` 超价消毒失败；
    消毒失败时 canonical 仍可输出，mw/tpsa 等为 None（iso Stage2 会兜底）。
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=not iso)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
        canon = Chem.MolToSmiles(mol)
    except Exception:
        try:
            canon = Chem.MolToSmiles(mol)   # 未消毒直接输出（保留电荷写法）
        except Exception:
            canon = smiles
    net_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    comp = len(Chem.GetMolFrags(mol))

    mw = tpsa = rotb = inchikey = None
    try:
        mw = Descriptors.MolWt(mol)
    except Exception:
        mw = None
    try:
        tpsa = Descriptors.TPSA(mol)
    except Exception:
        tpsa = None
    try:
        rotb = Descriptors.NumRotatableBonds(mol)
    except Exception:
        rotb = None
    try:
        inchikey = rdMolDescriptors.MolToInchiKey(mol)
    except Exception:
        inchikey = ""
    return {"canonical": canon, "inchikey": inchikey or "", "mw": mw,
            "net_charge": net_charge, "comp": comp, "tpsa": tpsa, "rotb": rotb}


def _process_range(sdf_path, out_root, start, end, mw_lo, force):
    """处理 [start, end) 字节区间内的记录，同时写四库 part 文件。"""
    t0 = time.time()
    st = {"range": f"{start}-{end}", "records": 0, "missing_smiles": 0,
          "kept": {lib: 0 for lib in LIB_CFG}, "seconds": None}

    # 四个库的输出（part_<start>_<end>.tsv.gz），已存在且可读则跳过该库
    handles = {}
    for lib in LIB_CFG:
        d = os.path.join(out_root, lib, "candidates")
        os.makedirs(d, exist_ok=True)
        out_path = os.path.join(d, f"part_{start}_{end}.tsv.gz")
        st.setdefault("out", {})[lib] = out_path
        if os.path.exists(out_path) and not force:
            try:
                with gzip.open(out_path, "rt") as _chk:
                    _chk.readline()
                handles[lib] = None  # 已存在 → 跳过
                continue
            except Exception:
                os.remove(out_path)
        handles[lib] = (gzip.open(out_path + ".tmp", "wt"), out_path)

    if not any(h is not None for h in handles.values()):
        st["records"], st["seconds"] = -1, 0.0
        return st

    n_rec = n_missing = 0
    try:
        with open(sdf_path, "rb") as fh:
            # —— 对齐到记录边界：跳到 start 后，读到第一个 $$$$ 为止 ——
            fh.seek(start)
            if start > 0:
                for line in fh:
                    if line.rstrip(b"\r\n") == b"$$$$":
                        break
            # 现在 fh 位置 = 下一条记录首行

            rec = {"smiles": "", "catalog": ""}
            cur_tag = None

            def flush():
                nonlocal n_missing
                smi = rec["smiles"]
                if not smi:
                    n_missing += 1
                    return
                cat = rec["catalog"]
                # 文本级过滤（每个库独立判定）
                # info 在首个命中库时计算一次，各库共享
                info = None
                for lib, cfg in LIB_CFG.items():
                    if cfg.get("iso"):
                        hits = any(p in smi for p in ISO_PATS)
                    else:
                        hits = any(t in smi for t in cfg["toks"])
                        if cfg.get("exclude") and any(t in smi for t in cfg["exclude"]):
                            hits = False
                    if not hits:
                        continue
                    if _has_halogen(smi, iso=cfg.get("iso", False)):
                        continue
                    if _has_metal(smi):
                        continue
                    # RDKit 属性（首次命中时计算一次，各库共享）
                    if info is None:
                        info = _analyze_mol(smi, iso=cfg.get("iso", False))
                        if info is None:
                            info = {}  # 不可解析 → head/carbonyl/acid 丢弃；iso 保留
                    if not info:
                        continue
                    if info.get("net_charge", 0) != 0:
                        continue
                    if info.get("comp", 1) > cfg["comp"]:
                        continue
                    mw = info.get("mw")
                    if mw is not None and not (mw_lo <= mw <= cfg["mw_hi"]):
                        continue
                    h = handles[lib]
                    if h is None:
                        continue  # 该库 part 已存在（断点续跑）
                    out, _ = h
                    row = [
                        cat, smi, info.get("inchikey", ""), str(info.get("net_charge", 0)),
                        "" if mw is None else format(mw, ".2f"),
                        str(info.get("comp", 1)),
                        "" if info.get("tpsa") is None else format(info["tpsa"], ".1f"),
                        "" if info.get("rotb") is None else str(info["rotb"]),
                    ]
                    out.write("\t".join(row) + "\n")
                    st["kept"][lib] += 1

            while True:
                if fh.tell() >= end:
                    break
                line = fh.readline()
                if not line:
                    break
                s = line.rstrip(b"\r\n")
                if s == b"$$$$":
                    flush()
                    n_rec += 1
                    rec = {"smiles": "", "catalog": ""}
                    cur_tag = None
                    continue
                m = _TAG_RE.match(line.decode("ascii", "replace").strip())
                if m:
                    tag = m.group(1).strip()
                    cur_tag = tag if tag in ("smiles", "Catalog_ID") else None
                    continue
                if cur_tag is not None:
                    val = s.decode("ascii", "replace").strip()
                    if val:
                        if cur_tag == "smiles":
                            rec["smiles"] = val
                        else:
                            rec["catalog"] = val
                    cur_tag = None
                    continue
            # 末尾未闭合记录（文件被截断）也刷出
            flush()
            st.update({"records": n_rec, "missing_smiles": n_missing})
    finally:
        for lib, h in handles.items():
            if h is None:
                continue
            out, final = h
            out.close()
            os.replace(final + ".tmp", final)
    st["seconds"] = round(time.time() - t0, 1)
    return st


def _worker(args):
    return _process_range(*args)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdf", default="/home/zx/df/disk1/resources/enamine_build_block/"
                                     "Enamine_Building_Blocks_2827624cmpd_20260817.sdf")
    ap.add_argument("--out-dir", default="data/enamine_db")
    ap.add_argument("--ranges", type=int, default=8, help="SDF 按字节切 N 段并行")
    ap.add_argument("--range-idx", type=int, default=None, help="只跑第 N 段（0 起）")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--min-mw", type=float, default=DEFAULT_MW_LO)
    ap.add_argument("--force", action="store_true", help="忽略已存在输出强制重跑")
    args = ap.parse_args()

    sdf_path = Path(args.sdf)
    if not sdf_path.exists():
        print(f"[error] SDF 不存在: {sdf_path}", file=sys.stderr); sys.exit(1)
    size = sdf_path.stat().st_size
    out_root = os.path.abspath(args.out_dir)

    ranges = []
    for i in range(args.ranges):
        start = i * size // args.ranges
        end = (i + 1) * size // args.ranges if i + 1 < args.ranges else size
        ranges.append((str(sdf_path), out_root, start, end, args.min_mw, args.force))
    if args.range_idx is not None:
        if not (0 <= args.range_idx < len(ranges)):
            print(f"[error] --range-idx 越界 0..{len(ranges)-1}", file=sys.stderr); sys.exit(1)
        ranges = [ranges[args.range_idx]]

    print(f"[info] SDF {size/1e9:.2f} GB，切 {args.ranges} 段，线程 {args.threads}", flush=True)
    print(f"[info] 输出 {out_root}", flush=True)

    summary = []
    with ProcessPoolExecutor(max_workers=min(args.threads, len(ranges))) as ex:
        futs = {ex.submit(_worker, r): r for r in ranges}
        for fut in as_completed(futs):
            try:
                r = fut.result()
            except Exception as e:
                print(f"[error] range: {e!r}", file=sys.stderr); continue
            summary.append(r)
            print(f"  [ok] range {r['range']}: rec={r['records']:,} miss={r['missing_smiles']:,} "
                  f"kept={r['kept']} {r['seconds']}s", flush=True)

    agg = {lib: 0 for lib in LIB_CFG}
    tot_rec = tot_miss = 0
    for r in summary:
        for lib in LIB_CFG:
            agg[lib] += r["kept"][lib]
        tot_rec += r["records"]
        tot_miss += r["missing_smiles"]
    print(json.dumps({"ranges_done": len(summary), "records_total": tot_rec,
                      "missing_smiles": tot_miss, "kept": agg}, indent=2))


if __name__ == "__main__":
    main()
