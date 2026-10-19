#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_lipid_gates.py — v2.1 脂质结构门控（两亲性：小亲水头 + 长疏水尾）
========================================================================
对 v2 已生成的 partial 子库 .smi 应用结构门控，输出 gated 子库。不重跑 Stage1/2。

每库门控规则：
  Head(gate)      : MW≤300 · 残余碱性≥1(叔胺/吡啶/咪唑) · TPSA≥35 · 芳环≤1
                    · 最长脂肪链<4（亲水小块）· 必含反应性伯/仲胺
  Carbonyl(gate)  : 含醛/酮(修后SMARTS) · ≥8连续sp3脂肪碳链 · 芳环≤1
                    · 杂原子≤4（排除 醚氧过多/稠环杂芳）
  Acid(gate)      : 羧酸 ≥8连续脂肪链 或 ≥2酯键 · 芳环≤1
  Isocyanide(gate): 链长≤6 优先脂肪/苄基（保留少量；库本已小）

用法：
    python scripts/v2/apply_lipid_gates.py \
        --in "data/head_v2.0/output/head_v2.0_partial_primary.smi:head,..." \
        --out-dir data/v2.1_gated

每个 --in 项 = <smi文件>:<库类型>
输出：<out-dir>/<名称>_gated.smi
"""

import argparse
import os
import re
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

# ----------------------------------------------------------------------------
# SMARTS
# ----------------------------------------------------------------------------
_ALD = Chem.MolFromSmarts("[CX3H1,H2](=O)")
_KET = Chem.MolFromSmarts("[#6][CX3](=O)[#6;!$([CX3](=O)[OX2,NX3])]")
_ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
_CARBOXYLATE = Chem.MolFromSmarts("[CX3](=O)[OX1-]")
_PHOSPHORUS = Chem.MolFromSmarts("[#15]")
_PRIM = Chem.MolFromSmarts("[NX3;H2;!$(NC=O);!$(N=C);!$([N+])]")
_SEC = Chem.MolFromSmarts("[NX3;H1;!$(NC=O);!$(N=C);!$([N+])]")
_TERT = Chem.MolFromSmarts("[NX3;H0;!$(NC=O);!$(N=[!#6]);!$(N-[OX2]);!$(N-[NX3]);!$([N+]);!$(N-[c]);$(N([#6])([#6])[#6])]")
_AROMN = Chem.MolFromSmarts("[n;+0;!$(n[H]);!$([n+]([O-]));!$(n-[#8])]")
_ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2][#6]")
_ISO = ("[N+]#[C-]", "[C-]#[N+]")

# 连续 sp3 脂肪碳链（≥6/8）
def _tail_pattern(n):
    unit = "[CX4;!$(C=O);!$(C-N);!$(C-O);!$(C-S);!$(C-[F,Cl,Br,I])]"
    return Chem.MolFromSmarts("~".join([unit] * n))


_TAIL6 = _tail_pattern(6)
_TAIL8 = _tail_pattern(8)


def _has_p(m):
    return bool(m.HasSubstructMatch(_PHOSPHORUS)) if _PHOSPHORUS else False


def _n(m, pat):
    return len(m.GetSubstructMatches(pat)) if pat is not None else 0


def _tail8(m):
    return m.HasSubstructMatch(_TAIL8)


def _tail6(m):
    return m.HasSubstructMatch(_TAIL6)


def _aromatic_rings(m):
    return len(Chem.GetSymmSSSR(m)) if m is not None else 0


# ----------------------------------------------------------------------------
# Gates
# ----------------------------------------------------------------------------
def gate_head(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False, "parse"
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return False, "sanitize"
    mw = Descriptors.MolWt(m)
    if mw is None or mw > 280:
        return False, "mw>280"
    n_react = _n(m, _PRIM) + _n(m, _SEC)
    if n_react < 1:
        return False, "no_react_amine"
    n_basic = _n(m, _TERT) + _n(m, _AROMN)
    if n_basic < 1:
        return False, "no_basic_center"
    tpsa = Descriptors.TPSA(m)
    if tpsa < 25:
        return False, "tpsa<25"
    if _aromatic_rings(m) > 1:
        return False, "ar_rings>1"
    if _aromatic_rings(m) == 1 and not (m.HasSubstructMatch(_AROMN)):
        return False, "ar_ring_non_basic"   # 芳环必须是吡啶/咪唑等碱性杂环
    if _tail6(m):
        return False, "tail>=6"      # 亲水头不应带长链
    return True, "ok"


def gate_carbonyl(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False, "parse"
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return False, "sanitize"
    n_ald = _n(m, _ALD)
    n_ket = _n(m, _KET)
    if n_ald + n_ket == 0:
        return False, "no_ald_ket"
    if n_ald + n_ket > 1:
        return False, "multi_carb"
    # 两亲性要求尾端是醛（端基亲核）→ 仅保留醛；酮(在链中)排除
    if n_ket > 0 or n_ald == 0:
        return False, "not_aldehyde_tail"
    if not _tail8(m):
        return False, "tail<8"
    if _aromatic_rings(m) != 0:
        return False, "has_aromatic"     # 尾巴须为纯脂肪链（无芳环）
    # 杂原子（非C/H）≤4：允许 1 个羰基 + 0-3 杂（醚/酯少量）
    het = sum(1 for a in m.GetAtoms() if a.GetAtomicNum() not in (1, 6))
    if het > 4:
        return False, "het>4"
    logp = Crippen.MolLogP(m)
    if logp < 2.5:
        return False, "logp<2.5"
    return True, "ok"


_PHOSPHORUS = Chem.MolFromSmarts("[#15]")


def gate_acid(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False, "parse"
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return False, "sanitize"
    if _has_p(m):
        return False, "has_P"
    n_acid = _n(m, _ACID) + _n(m, _CARBOXYLATE)
    if n_acid == 0:
        return False, "no_acid"
    if n_acid > 1:
        return False, "multi_acid"
    n_est = _n(m, _ESTER)
    chain_ok = _tail8(m) or n_est >= 2
    if not chain_ok:
        return False, "no_tail_nor_bi_ester"
    if _aromatic_rings(m) != 0:
        return False, "has_aromatic"     # 纯脂肪族酸/酯酸
    if Crippen.MolLogP(m) < 1.0:
        return False, "logp<1"
    return True, "ok"


def gate_iso(smi):
    # 异腈：保留短/中等（<7 原子到 N 端）；芳基保留(库小)；排除长链+大杂环
    m = Chem.MolFromSmiles(smi, sanitize=False)
    if m is None:
        return False, "parse"
    canonical = Chem.MolToSmiles(m)
    if canonical.count(".") > 0:
        return False, "multi"
    if not any(p in canonical for p in _ISO):
        return False, "no_iso"
    # 链长限制：取 canonical 长度 <60 且重原子≤25 且无长烷链
    if len(canonical) > 60:
        return False, "long"
    # ISBN 库额外：排除 N-N 键 / 额外胺 / 酰胺（产品若带会引入杂氮/不稳定）
    if Chem.MolFromSmiles(smi) is None:
        return False, "parse2"
    mm = Chem.MolFromSmiles(smi, sanitize=False)
    for a in ["N-N"]:
        if a == "N-N" and any(
            b.GetBeginAtom().GetSymbol() == "N" and b.GetEndAtom().GetSymbol() == "N"
            for b in mm.GetBonds()):
            return False, "N_N_bond"
    # 带额外 N（除异腈 [N+] 外）→ 排除
    extra_n = 0
    for a in mm.GetAtoms():
        if a.GetSymbol() == "N" and a.GetFormalCharge() != 1:
            extra_n += 1
            break
    if extra_n:
        return False, "extra_N"
    return True, "ok"


_GATES = {"head": gate_head, "carbonyl": gate_carbonyl,
          "acid": gate_acid, "isocyanide": gate_iso}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", action="append", dest="inputs", default=[],
                    help="<smi文件>:<库类型>，可多次")
    ap.add_argument("--out-dir", default="data/v2.1_gated")
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for item in args.inputs:
        smi_file, kind = item.split(":")
        gate = _GATES[kind]
        rows = []
        with open(smi_file) as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split("\t")
                smi = parts[0]
                sid = parts[1] if len(parts) > 1 else ""
                ok, why = gate(smi)
                if ok:
                    rows.append((smi, sid))
        base = Path(smi_file).stem
        out = outdir / f"{base}_gated.smi"
        with open(out, "w") as f:
            for i, (s, sid) in enumerate(rows, 1):
                f.write(f"{s}\t{sid or f'{kind.upper()}G_{i:07d}'}\n")
        print(f"{kind:12s} {Path(smi_file).name:48s} -> {len(rows):,} 行  →  {out.name}")


if __name__ == "__main__":
    main()