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
# 通用酰胺（其羰基连接任何 N）——head 严格档否决
_AMIDE_CARN = Chem.MolFromSmarts("[CX3](=O)[NX3]")
_ISO = ("[N+]#[C-]", "[C-]#[N+]")
# --- 加强严格档 SMARTS ---
_ACYL = Chem.MolFromSmarts("[CX3](=O)[#6]")        # 酰基（酸/酯/酮共有的羰基-C）
_TER_ALIPH = Chem.MolFromSmarts(                    # 脂肪叔胺（非酰胺、非芳香、非羧酸成环、3 个脂肪 C 取代）
    "[NX3;H0;!$(N-C(=O));!$(N=[!#6]);!$(N-[OX2]);!$(N-[NX3]);!$([N+]);"
    "!$(N-n);!$(N-c);$(N([CX4])([CX4])[CX4])]")
_ALCOHOL_OH = Chem.MolFromSmarts("[OX2H;!$(O-C(=O))]")
_PHENOL = Chem.MolFromSmarts("[OX2H]-[c]")
_GEM_DIOL = Chem.MolFromSmarts("[CX4]([OX2;!$([OX2]-C(=O))])([OX2;!$([OX2]-C(=O))])")
_HYDRAZIDE = Chem.MolFromSmarts("[NX3][NX3;$(N-C(=O))]")
_BORON = Chem.MolFromSmarts("[#5]")
_ALPHA_BRANCH_KET = Chem.MolFromSmarts(             # α 支链酮 R2CH-C(=O)（醛 gate 用）
    "[CX3](=O)[CX4!$([CX4H1]([#6])[CX4])]")

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


def _aro_ring_count(m):
    return len(Chem.GetSymmSSSR(m)) if m is not None else 0


def _aromatic_rings(m):
    """兼容旧名：芳环数（SSSR）。"""
    return _aro_ring_count(m)


def _arom(nn):
    """芳香原子数（严格档/NaN 防护）。"""
    if nn is None:
        return 0
    return sum(1 for a in nn.GetAtoms() if a.GetIsAromatic())


def _chain_best(m):
    """最长脂肪烃链长（sp3 碳 BFS，启发式）。"""
    n = m.GetNumAtoms()
    adj = {}
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 6:
            adj.setdefault(a.GetIdx(), [])
    for b in m.GetBonds():
        c, d = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if c in adj and d in adj:
            adj[c].append(d); adj[d].append(c)
    if not adj:
        return 0
    best = 0
    for s in adj:
        dist = {s: 0}; q = [s]
        maxd = 0
        while q:
            u = q.pop()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    maxd = max(maxd, dist[v])
                    q.append(v)
        best = max(best, maxd)
    return best + 1


from collections import Counter as _Counter


# ----------------------------------------------------------------------------
# 严格档（profile=strict）额外门控
# ----------------------------------------------------------------------------
def gate_head_strict(m, why):
    """Head 严格档：恰好 1 反应胺 + 强制脂肪叔胺 + 否决酰胺/芳杂。"""
    n_prim = _n(m, _PRIM)
    n_sec = _n(m, _SEC)
    if n_prim + n_sec != 1:
        return False, "strict_not_exactly1_react_amine"
    if n_sec == 1:                        # 仲胺头套严格档一律不要（避免颈干扰 pKa 判位）
        return False, "strict_secondary_head"
    if not m.HasSubstructMatch(_TER_ALIPH):
        return False, "strict_no_aliphatic_tertiary_amine"
    if m.HasSubstructMatch(_AMIDE_CARN) or m.HasSubstructMatch(_ACYL):
        return False, "strict_has_amide_or_acyl"
    if _arom(m) != 0:
        # 严格档仅保留纯脂肪叔胺头（无芳环 → 小、纯烷基、pKa 高）
        return False, "strict_has_aromatic_ring"
    return True, "ok"


def gate_acid_strict(m, why):
    """Acid 严格档：必含酰基 + 否决额外OH/手性/多烯 + 链长上限。"""
    if not m.HasSubstructMatch(_ACYL):
        return False, "strict_no_acyl"
    n_oh = _n(m, _ALCOHOL_OH)
    if n_oh > 1:                          # 只有 -COOH 的 OH 算 1
        return False, "strict_extra_OH"
    if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in m.GetAtoms()):
        return False, "strict_stereocenter"
    n_cc = sum(1 for b in m.GetBonds() if b.GetBondType() == Chem.BondType.DOUBLE)
    if n_cc > 2:
        return False, "strict_polyene>2"
    if _chain_best(m) > 30:
        return False, "strict_chain>30"
    return True, "ok"


def gate_iso_strict(m, why):
    """Isocyanide 严格档：无芳、无 B、无偕二醇/N-N/多余OH。"""
    if _arom(m) != 0:
        return False, "strict_has_aromatic"
    if m.HasSubstructMatch(_BORON):
        return False, "strict_has_B"
    if m.HasSubstructMatch(_GEM_DIOL) or m.HasSubstructMatch(_HYDRAZIDE) or \
       m.HasSubstructMatch(_ALCOHOL_OH):
        return False, "strict_OH_or_gemdiol_or_hydrazide"
    return True, "ok"


def gate_carbonyl_strict(m, why):
    """Carbonyl 严格档：否决 α-支链醛，避免产物 α-C 季碳。"""
    if m.HasSubstructMatch(_ALPHA_BRANCH_KET):
        return False, "strict_alpha_branched_ket"
    return True, "ok"


_STRICT_EXTRA = {
    "head": gate_head_strict,
    "acid": gate_acid_strict,
    "isocyanide": gate_iso_strict,
    "carbonyl": gate_carbonyl_strict,
}


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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", action="append", dest="inputs", default=[],
                    help="<smi文件>:<库类型>，可多次")
    ap.add_argument("--out-dir", default="data/v2.1_gated")
    ap.add_argument("--profile", default="default", choices=("default", "strict"),
                    help="default=现有门控；strict=在 default 之上叠加严格档（P1）")
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    strict = args.profile == "strict"

    for item in args.inputs:
        smi_file, kind = item.split(":")
        gate = _GATES[kind]
        gate_extra = _STRICT_EXTRA.get(kind)
        n_pass = 0
        strict_why = Counter()
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
                if ok and strict and gate_extra is not None:
                    m = Chem.MolFromSmiles(smi)
                    if m is None:
                        ok = False; why = "strict_parse"
                    else:
                        try:
                            Chem.SanitizeMol(m)
                        except Exception:
                            ok = False; why = "strict_sanitize"
                    if ok:
                        ok, why = gate_extra(m, why)
                        if not ok:
                            strict_why[why] += 1
                if ok:
                    rows.append((smi, sid)); n_pass += 1
        suffix = "strict" if strict else "gated"
        base = Path(smi_file).stem
        out = outdir / f"{base}_{suffix}.smi"
        with open(out, "w") as f:
            for i, (s, sid) in enumerate(rows, 1):
                f.write(f"{s}\t{sid or f'{kind.upper()}{suffix.upper()}_{i:07d}'}\n")
        print(f"{kind:12s} {Path(smi_file).name:48s} -> {n_pass:,} 行  →  {out.name}")
        if strict and strict_why:
            print(f"    strict 否决原因: {dict(strict_why)}")


# 门控注册表：default 档（真实 gate）——供 CLI 与 import 复用
_GATES = {
    "head": gate_head,
    "carbonyl": gate_carbonyl,
    "acid": gate_acid,
    "isocyanide": gate_iso,
}


# ----------------------------------------------------------------------------
# 执行入口
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    from collections import Counter
    main()