#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iso_stage1.py — Isocyanide（异腈）Ugi-4CR 组分 Stage1：文本级候选池提取
======================================================================
真异腈在 PubChem SMILES 中表示为 `[N+]#[C-]` 或 `[C-]#[N+]`（带显式电荷，
故不能复用 main_component.py 的 `[+/[-` 排除逻辑，需专用脚本）。

规则（与 head/carbonyl/acid 三库对齐的“宁多勿漏”候选池）：
    * SMILES 含 `[N+]#[C-]` 或 `[C-]#[N+]`
    * PUBCHEM_TOTAL_CHARGE == 0（整分子净电荷0；内嵌 [+]/[-] 允许，Stage2去盐）
    * component_count <= 4（盐/配合物留待 Stage2 去金属）
    * 10 <= MW <= 800（宽窗口）
输出：data/isocyanide_v0.1/candidates/*.tsv.gz
    列：cid, smiles, inchikey, total_charge, mw, component_count, tpsa, rotatable_bonds

用法：
    python scripts/iso_stage1.py --out-dir data/isocyanide_v0.1/candidates -t 12
    # 支持 --filelist / 断点续跑（输出存在即跳过）

环境：python3.10，仅标准库。
"""

import argparse
import gzip
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

COLS = ["cid", "smiles", "inchikey", "total_charge", "mw",
        "component_count", "tpsa", "rotatable_bonds"]

ISO_PATS = ("[N+]#[C-]", "[C-]#[N+]")  # 真异腈 R-N≡C


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


def _parse_file(sdf_path, out_dir, mw_lo, mw_hi, max_comp):
    st = {"path": sdf_path, "records": 0, "kept": 0, "seconds": None}
    out_path = os.path.join(out_dir, os.path.basename(sdf_path).replace(".sdf.gz", "") + ".tsv.gz")
    if os.path.exists(out_path):
        try:
            with gzip.open(out_path, "rt") as _f:
                _f.readline()
            st.update({"records": -1, "kept": -1, "seconds": 0.0, "skipped": True})
            return st
        except Exception:
            os.remove(out_path)

    t0 = time.time()
    tmp = out_path + ".tmp"
    n_rec, n_kept = 0, 0
    with gzip.open(sdf_path, "rt", errors="replace") as fh, gzip.open(tmp, "wt") as out:
        out.write("\t".join(COLS) + "\n")
        rec, cur_field, expect_value = {}, None, False
        for line in fh:
            s = line.rstrip("\n").strip()
            if s == "$$$$":
                smi = rec.get("PUBCHEM_SMILES", "").strip()
                ch = _pi(rec.get("PUBCHEM_TOTAL_CHARGE", "x"))
                mw = _pf(rec.get("PUBCHEM_MOLECULAR_WEIGHT", "x"))
                comp = _pi(rec.get("PUBCHEM_COMPONENT_COUNT", "x"))
                if (smi and ch == 0 and mw is not None and mw_lo <= mw <= mw_hi
                        and (comp or 1) <= max_comp
                        and any(p in smi for p in ISO_PATS)):
                    out.write("\t".join([
                        rec.get("PUBCHEM_COMPOUND_CID", "").strip(), smi,
                        rec.get("PUBCHEM_IUPAC_INCHIKEY", "").strip(), str(ch),
                        f"{mw:.2f}", str(comp or ""),
                        rec.get("PUBCHEM_CACTVS_TPSA", "").strip(),
                        rec.get("PUBCHEM_CACTVS_ROTATABLE_BOND", "").strip(),
                    ]) + "\n")
                    n_kept += 1
                n_rec += 1
                rec, cur_field, expect_value = {}, None, False
                continue
            if s.startswith("> <") and s.endswith(">"):
                tag = s[3:-1].strip().split(">")[0].strip()
                cur_field = tag if tag in ("PUBCHEM_SMILES", "PUBCHEM_TOTAL_CHARGE",
                                           "PUBCHEM_MOLECULAR_WEIGHT", "PUBCHEM_COMPONENT_COUNT",
                                           "PUBCHEM_COMPOUND_CID", "PUBCHEM_IUPAC_INCHIKEY",
                                           "PUBCHEM_CACTVS_TPSA", "PUBCHEM_CACTVS_ROTATABLE_BOND") else None
                expect_value = True
                continue
            if expect_value:
                expect_value = False
                if cur_field is not None:
                    rec[cur_field] = s
    os.replace(tmp, out_path)
    st.update({"records": n_rec, "kept": n_kept, "seconds": round(time.time() - t0, 1)})
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pubchem-dir", default="/home/zx/df/disk1/resources/pubchem")
    ap.add_argument("--out-dir", default="data/isocyanide_v0.1/candidates")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--filelist", default=None)
    ap.add_argument("--min-mw", type=float, default=10.0)
    ap.add_argument("--max-mw", type=float, default=800.0)
    ap.add_argument("--max-components", type=int, default=4)
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.filelist:
        files = [ln.strip() for ln in open(args.filelist) if ln.strip()]
        files = [f for f in files if os.path.exists(f)]
    else:
        files = sorted(str(p) for p in Path(args.pubchem_dir).glob("Compound_*.sdf.gz"))
    if not files:
        print("[error] 无候选文件", file=sys.stderr); sys.exit(1)

    print(f"[info] 待扫 {len(files)} 分片，线程 {args.threads}", flush=True)
    summary = []
    with ProcessPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(_parse_file, f, out_dir, args.min_mw, args.max_mw, args.max_components): f
                for f in files}
        for fut in as_completed(futs):
            r = fut.result()
            summary.append(r)
            print(f"  [ok] {os.path.basename(r['path'])}: kept={r['kept']:,} {r['seconds']}s", flush=True)

    print(f"[done] 文件 {len(summary)}，候选合计 "
          f"{sum(r['kept'] for r in summary if r['kept'] > 0):,}")


if __name__ == "__main__":
    main()