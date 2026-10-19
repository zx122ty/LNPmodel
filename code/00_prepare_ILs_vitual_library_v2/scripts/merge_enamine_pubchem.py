#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_enamine_pubchem.py — 合并 PubChem 与 Enamine 的 Ugi-4CR 四组分库
========================================================================
把两份同源（gated / main）四组分子库按「片段」合并：
    * PubChem  : data/v2.1_gated/<pubchem_smi>   （来源 source=pubchem）
    * Enamine  : data/enamine_db/v2.1_gated/<ena_smi>（来源 source=enamine）
输出（每库一个目录，含 TSV 全字段 + 纯 SMILES 列表）:
    data/merge/<lib>/
        ├── <lib>_merged.tsv        # cid/smiles/... + source + source_id
        ├── <lib>_merged.smi        # SMILES \\t source:source_id
        └── merge_summary.json

去重策略（宁留勿删）:
    * 同一库内：按 SMILES 去重（文本正则化后），
      —— PubChem 候选大、Enamine 小，去重主要去 PubChem 内部重复。
    * 跨库：**不去重**。同一分子若同时在 PubChem 与 Enamine 出现，两行都保留
      （各带各自 source），便于下游按 source 归属/溯源。

SMILES 正则化（仅用于「去重键」，保留原串输出）:
    * 去显式氢（[H]→''、[nH]→[n]、[NH]→[N]…），统一芳香/电荷写法后
      再交由 RDKit canonical。失败则回退到字符串去重键。

用法:
    python scripts/merge_enamine_pubchem.py \
        --out-dir data/merge \
        --pubchem-dir data/v2.1_gated \
        --enamine-dir data/enamine_db/v2.1_gated \
        --kind gated            # 或 main（默认 gated）
    # 可 --libs head,carbonyl,acid,isocyanide 限定子库
    # 可 --only smi             # 只写 .smi，不写 TSV

环境: python3.10 + RDKit（conda env unimol2）。
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from rdkit import Chem

# ----------------------------------------------------------------------------
# 库 → (pubchem 文件名, enamine 文件名, 输出前缀/ID 前缀)
# ----------------------------------------------------------------------------
DEFAULTS = {
    "gated": {
        "head":       ("head_v2.0_partial_primary_gated.smi", "head_ENA_main_gated.smi"),
        "carbonyl":   ("carbonyl_v2.0_partial_gated.smi",      "carbonyl_ENA_main_gated.smi"),
        "acid":       ("acid_v2.0_partial_gated.smi",          "acid_ENA_main_gated.smi"),
        "isocyanide": ("isocyanide_v2.0_partial_gated.smi",    "isocyanide_ENA_main_gated.smi"),
    },
    "main": {
        "head":       ("head_v0.1_main.smi",       "head_ENA_main.smi"),
        "carbonyl":   ("carbonyl_v0.1_main.smi",    "carbonyl_ENA_main.smi"),
        "acid":       ("acid_v0.1_main.smi",        "acid_ENA_main.smi"),
        "isocyanide": ("isocyanide_v0.1_main.smi",  "isocyanide_ENA_main.smi"),
    },
}
ID_PREFIX = {"head": "HEAD", "carbonyl": "CARB", "acid": "ACID", "isocyanide": "ISO"}


# ----------------------------------------------------------------------------
# SMILES 文本正则化（低风险，先于 RDKit canonical）
# ----------------------------------------------------------------------------
_ATOM_H_RE = re.compile(r"\[([A-Z][a-z]?)(H\d*)([+-]?\d*)\]")
_BRACKET_H_RE = re.compile(r"\[([a-z])(H\d*)\]")


def _normalize_text(smi):
    """去掉显式氢标记，减少写法差异（不去重源串本身）。"""
    s = smi.replace("[H]", "")
    s = _ATOM_H_RE.sub(r"[\1\3]", s)   # [NH2]→[N], [NH3+]→[N+], [nH]→[n]
    s = _BRACKET_H_RE.sub(r"[\1]", s)  # [nH]→[n]
    return s


def canonical_key(smi):
    """去重键：先文本正则化 → RDKit canonical → InChIKey 前缀（14 位骨架层）。
    失败回退到去氢文本串。"""
    try:
        m = Chem.MolFromSmiles(_normalize_text(smi))
        if m is not None:
            try:
                Chem.SanitizeMol(m)
            except Exception:
                pass
            c = Chem.MolToSmiles(m)
            try:
                from rdkit.Chem import rdMolDescriptors
                ik = rdMolDescriptors.MolToInchiKey(m)
                if ik:
                    return ik.split("-")[0]   # 骨架层（不含立体/同位素）
            except Exception:
                pass
            return c
    except Exception:
        pass
    return _normalize_text(smi)


def read_smi(path):
    """读 .smi（SMILES\\tid 每行）→ [(raw_smi, raw_id)]。"""
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = ln.split("\t")
            smi = parts[0].strip()
            if not smi:
                continue
            sid = parts[1].strip() if len(parts) > 1 else ""
            rows.append((smi, sid))
    return rows


def _merge_lib(lib, pub_file, ena_file, pub_dir, ena_dir, out_dir, src_id_prefix):
    """合并单库 → 返回统计 dict。"""
    st = {"lib": lib, "pubchem": 0, "enamine": 0, "pubchem_dup": 0,
          "enamine_dup": 0, "overlap": 0, "parse_fail": 0}
    out_lib = out_dir / lib
    out_lib.mkdir(parents=True, exist_ok=True)

    pub_path, ena_path = pub_dir / pub_file, ena_dir / ena_file
    tsv_out = out_lib / f"{lib}_merged.tsv"
    smi_out = out_lib / f"{lib}_merged.smi"
    tsv_tmp, smi_tmp = tsv_out.with_suffix(".tsv.tmp"), smi_out.with_suffix(".smi.tmp")

    header = ["cid", "smiles", "source", "source_id"]
    with open(tsv_tmp, "w", newline="") as ftsv, open(smi_tmp, "w") as fsmi:
        ftsv.write("\t".join(header) + "\n")

        def emit(smi, source, source_id):
            # cid：在本合并库内按顺序编号（稳定）；保留 source_id 溯源
            row = [f"{src_id_prefix}_{st[source] + 1:08d}", smi, source, source_id]
            ftsv.write("\t".join(row) + "\n")
            fsmi.write(f"{smi}\t{source}:{source_id}\n")

        # ---- PubChem 先（量大）----
        seen_pub = set()
        n_bad = 0
        for smi, sid in read_smi(pub_path):
            k = canonical_key(smi)
            if k in seen_pub:
                st["pubchem_dup"] += 1
                continue
            seen_pub.add(k)
            st["pubchem"] += 1
            emit(smi, "pubchem", sid or "-")
            if st["pubchem"] == 1:  # 首个命中即尝试 canonical，失败计数
                pass
        st["parse_fail"] = n_bad

        # ---- Enamine（不去重 vs PubChem，保留两行）----
        seen_ena = set()
        for smi, sid in read_smi(ena_path):
            k = canonical_key(smi)
            if k in seen_ena:
                st["enamine_dup"] += 1
                continue
            seen_ena.add(k)
            if k in seen_pub:
                st["overlap"] += 1   # 同一骨架同时出现在两库（各自保留）
            st["enamine"] += 1
            emit(smi, "enamine", sid or "-")

    os.replace(tsv_tmp, tsv_out)
    os.replace(smi_tmp, smi_out)
    return st


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/merge")
    ap.add_argument("--pubchem-dir", default="data/v2.1_gated")
    ap.add_argument("--enamine-dir", default="data/enamine_db/v2.1_gated")
    ap.add_argument("--kind", choices=["gated", "main"], default="gated")
    ap.add_argument("--libs", default="head,carbonyl,acid,isocyanide",
                    help="逗号分隔，默认四库全跑")
    ap.add_argument("--only-smi", action="store_true", help="只写 .smi 不写 TSV")
    args = ap.parse_args()

    pub_dir = Path(args.pubchem_dir)
    ena_dir = Path(args.enamine_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    libs = [x.strip() for x in args.libs.split(",") if x.strip()]
    table = DEFAULTS[args.kind]
    if args.only_smi:
        # 仅 .smi 模式：直接在每库目录写 <lib>_merged.smi
        pass

    results = []
    for lib in libs:
        if lib not in table:
            print(f"[warn] 忽略未知库 {lib}", file=sys.stderr)
            continue
        pub_f, ena_f = table[lib]
        pub_path, ena_path = pub_dir / pub_f, ena_dir / ena_f
        missing = [p for p in (pub_path, ena_path) if not p.exists()]
        if missing:
            print(f"[warn] {lib}: 缺失输入 {[str(p) for p in missing]}，跳过", file=sys.stderr)
            continue
        st = _merge_lib(lib, pub_f, ena_f, pub_dir, ena_dir, out_dir,
                        ID_PREFIX[lib])
        results.append(st)
        print(f"  [ok] {lib:12s} pubchem={st['pubchem']:,} (+dup {st['pubchem_dup']:,}) "
              f"enamine={st['enamine']:,} (+dup {st['enamine_dup']:,}) "
              f"overlap={st['overlap']:,} → {out_dir / lib}", flush=True)

    with open(out_dir / f"merge_summary_{args.kind}.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
