#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ugi_template_validate.py — Ugi-4CR 反应模板小规模枚举验证 + 产物机构图（SDF/PNG）
==================================================================================
验证目标（03.md §6.2）：
  1. 核心骨架 Ugi-4CR core SMILES 与 A/B/C/D 端基替换 → rdMolEnumerator 能否产出
     结构正确的 α-酰氨基酰胺产物
  2. 用 4→2 个真实已知组分对照文献 Ugi 产物，验证 SMART 替换不引入价态异常
  3. 各子库随机抽 20×4，做小规模枚举 → 产物 MW/价态/去重 统计
  4. 输出：产品 SMILES/SDF + 2D 结构图 PNG（含核心原子映射）

Ugi-4CR 产物核：
     R4-C(=O)-N(R1)-CH(R3)-C(=O)-N(R2)
     (羧酸羰基→酰胺N; 醛CH→CHR3; 异腈C→酰胺C=O)
core_smiles = "C(=O)NCC(=O)N"        # 7 重原子：骨架
   A 位(2)  = 胺 N（H 被覆盖）     → R1-NH2 的 N
   B 位(6)  = 末端酰胺 N           → 异腈提供的 N
   C 位(3)  = 次甲基 CH            → 醛的 CH
   D 位(0)  = 酰胺 C=O             → 羧酸羰基

本验证脚本复用 MOLEA 的 star-numbering 组合法（同 ugi_reaction_virtual.py 思路）。

用法：
    python scripts/ugi_template_validate.py --out-dir data/ugi_template_v0.1/output \
                                            --libs data/head_v0.1/output,data/carbonyl_v0.1/output,data/acid_v0.1/output,data/isocyanide_v0.1/output \
                                            --n 20 --draw

产物结构图输出：png/
    product_00001.png ...（含原子坐标，用 rdMolDraw2D 可调）
    products.sdf   （含 source 组分ID）

环境：python3.10 + RDKit 2026.03。

======enamine test==========
 python scripts/ugi_template_validate.py --out-dir data/enamine_db/ugi_template/output \
    --libs data/enamine_db/head/output/head_ENA_main.smi,data/enamine_db/carbonyl/output/carbonyl_ENA_main.smi,data/enamine_db/acid/output/acid_ENA_main.smi,data/enamine_db/isocyanide/output/isocyanide_ENA_main.smi \
    --n 20 --draw


"""

import argparse
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolEnumerator, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

# ----------------------------------------------------------------------------
# Ugi-4CR 核心骨架
# ----------------------------------------------------------------------------
# Ugi-4CR：R1-NH2 + R2-NC + R3-CHO + R4-COOH → R4-C(=O)-N(R1)-CH(R3)-C(=O)-N(R2)
# 骨架 core = C(=O)NCC(=O)N (7 重原子)。连接点（idx = CORE_SMILES 按从左到右 0 起编号）：
#   A位(idx2) = 与 head 的 N 成键的酰胺 N（H 被 R1 覆盖）→ 接 head（伯/仲胺 R1-NH(NR)）
#   D位(idx0) = 首酰胺羰基 C(=O)（原羧酸羰基 C）→ 接 acid（羧酸 R4）
#   C位(idx3) = 中间 CH2（原醛羰基 C，接 R3）→ 接 aldehyde（醛 R3）
#   B位(idx6) = 末端酰胺 N（异腈当年贡献的 N，原连接异腈 C）→ 接 isocyanide（异腈 R2）
#
# ★映射对照 2026-08-29（本脚本与官方已验证的 curate_library/ugi_reaction_virtual.py
#   完全一致）：head→A位(2) · iso→B位(6) · aldehyde→C位(3) · acid→D位(0)。
#   ⚠ 片段拼接顺序（与官方脚本同为 head, iso, ald, acid）与连接点一一对应为：
#     head(star_A) → 2 · iso(star_B) → 6 · ald(star_C) → 3 · acid(star_D) → 0
#   [旧版曾错位一环：head/iso 对调 + 醛/酸错位，产物拓扑错乱（异腈上酰基碳、
#    醛上酰胺N、酸上次甲基），已废弃]
CORE_SMILES = "C(=O)NCC(=O)N"
CORE_POS = {"A": 2, "B": 6, "C": 3, "D": 0}

# 片段拼接顺序与连接点(dst Core idx) 严格对齐官方脚本：
#   head→2, iso→6, aldehyde→3, acid→0
_CORE_DST_ORDER = [("head", 2), ("iso", 6), ("ald", 3), ("acid", 0)]

import re


def fragmentize(smiles: str, kind: str) -> str:
    """把组分 SMILES 转为核心可连的开端片断（去反应团并替换为 *）。

    kind: 'head' / 'aldehyde' / 'acid' / 'isocyanide'
    head: 伯胺 [NH2] 或仲胺 [NH1]（Ugi 胺亲核点均可用）；醛/酸同字面。
    异腈因显式电荷 `[N+]#[C-]` 无法 sanitize → 用字符串去前缀特判。
    """
    if kind == "isocyanide":
        m = re.match(r"\[N\+\]#\[C\-\](.*)|\[C\-\]#\[N\+\](.*)", smiles)
        if not m:
            return None
        r = (m.group(1) or m.group(2) or "").strip()
        return ("*" + r) if r else "[*]"
    if kind == "head":
        # Ugi 主验证限定伯胺（NH2）——仲胺会因 ReplaceSubstructs 断键（N-H 与 N-R 不分）
        # 仲胺（secondary）单独进入高级验证流程
        pat = Chem.MolFromSmarts("[NX3;H2;!$(NC=O);!$(N=C)]")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not mol.HasSubstructMatch(pat):
            return None
        frag = Chem.ReplaceSubstructs(mol, pat, Chem.MolFromSmiles("*"))[0]
        return Chem.MolToSmiles(frag)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if kind == "aldehyde":
        # 本模板验证只支持醛（R-CHO → R-*）；酮(Ugi 低活性)后续单独模板
        if not mol.HasSubstructMatch(Chem.MolFromSmarts("[CX3H1](=O)")):
            return None
        frag = Chem.ReplaceSubstructs(mol, Chem.MolFromSmarts("[CX3H1](=O)"),
                                      Chem.MolFromSmiles("*"), replaceAll=False)[0]
        fs = Chem.MolToSmiles(frag)
        if fs.count("*") != 1 or fs.count(".") > 0:
            return None
        return fs
    p = {
        "acid":       Chem.MolFromSmarts("[CX3](=O)[OX2H1]"),
    }[kind]
    hits = mol.GetSubstructMatches(p)
    if not hits:
        return None
    frag = Chem.ReplaceSubstructs(mol, p, Chem.MolFromSmiles("*"), replaceAll=True)
    if len(frag) == 0:
        return None
    return Chem.MolToSmiles(frag[0])


import re as _re
# 拦截：多片断(.)、硼(\[B，放行 [Br])、残留带电碳 [C-]/[C+]（双异腈/前缀未除净）、
#       [数字…（同位素/原子映射）、O-F 键、非芳环写法的 n1X
# 旧写法 [\[][BC] 会把 [C@H]/[C@@H] 手性碳一并误杀，已改为精确匹配带电碳
_BAD_FRAG = _re.compile(r"\.|\[B(?!r)|\[C[-+]\]|[\[][0-9]|OF|n1[A-Za-z]")
def _frag_ok(f):
    # 开放端片断必须恰有 1 个 *（连接点）；多于 1 个 → 残留反应团（如双异腈剥出的第二个 *
    return (f is not None and f.count("*") == 1
            and not _BAD_FRAG.search(f))


def build_product(head_frag, ald_frag, acid_frag, iso_frag,
                  head_smi, ald_smi, acid_smi, iso_smi):
    """用核心骨架 + 4 开放端片断 → rdMolEnumerator 唯一产物 Mol。

    片段拼接与官方已验证的 ugi_reaction_virtual.py 完全一致：
      组合顺序  head, iso, ald, acid
      各接至    A位(2), B位(6), C位(3), D位(0)
    （atoms 即片段在组合中的相对顺序，与 |m:...| 一一对应）
    """
    if not all(_frag_ok(f) for f in (head_frag, ald_frag, acid_frag, iso_frag)):
        return None
    core = Chem.MolFromSmiles(CORE_SMILES)
    n_core = core.GetNumAtoms()
    # 片段按 head, iso, ald, acid 顺序组合（与官方 A,B,C,D 对齐）
    frags = [Chem.MolFromSmiles(head_frag), Chem.MolFromSmiles(iso_frag),
             Chem.MolFromSmiles(ald_frag), Chem.MolFromSmiles(acid_frag)]
    if any(f is None for f in frags):
        return None
    # 组合顺序上的原子偏移：core 后依次是 head, iso, ald, acid
    offsets = []
    acc = n_core
    for f in frags:
        offsets.append(acc)
        acc += f.GetNumAtoms()
    # 连接点（dst core idx）：head→2, iso→6, ald→3, acid→0
    dst = [CORE_POS["A"], CORE_POS["B"], CORE_POS["C"], CORE_POS["D"]]
    map_part = ",".join(f"{offsets[i]}:{dst[i]}" for i in range(4))
    smi = (f"{CORE_SMILES}.{head_frag}.{iso_frag}.{ald_frag}.{acid_frag} |m:{map_part}|")
    comb = Chem.MolFromSmiles(smi)
    if comb is None:
        return None
    en = rdMolEnumerator.Enumerate(comb)
    if not en or len(en) != 1:
        return None
    prod = en[0]
    try:
        Chem.SanitizeMol(prod)
        prod = Chem.RemoveHs(prod)
    except Exception:
        try:
            prod = Chem.RemoveHs(prod, sanitize=False)
        except Exception:
            pass
    return prod


def random_sample(rows, n, seed):
    random.seed(seed)
    if len(rows) <= n:
        return list(rows)
    return random.sample(list(rows), n)


def load_lib(smi_path):
    """读 .smi（SMILES \t id 或仅 SMILES），返回 [(smiles, id)]。"""
    out = []
    with open(smi_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                out.append((parts[0], parts[1]))
            else:
                out.append((line, ""))
    return out


# ----------------------------------------------------------------------------
# 产物结构自检（映射校验）与产物级规则过滤
# ----------------------------------------------------------------------------
_TER_AMINE = Chem.MolFromSmarts(
    "[NX3;H0;!$(NC=O);!$(N=[!#6]);!$(N-[OX2]);!$(N-[NX3]);!$([N+]);"
    "!$(N-[c]);$(N([#6])([#6])[#6])]")
_AMIDE_N = Chem.MolFromSmarts("[NX3]C(=O)")
_AROMN = Chem.MolFromSmarts("[n;+0;!$(n[H]);!$([n+]([O-]));!$(n-[#8])]")
_ALCOHOL_OH = Chem.MolFromSmarts("[OX2H;!$(O-C(=O))]")


def _nbasic(m):
    """可电离碱性中心计数：脂肪叔胺优先，其次吡啶/咪唑等碱性芳杂N。"""
    return (m.HasSubstructMatch(_TER_AMINE) or m.HasSubstructMatch(_AROMN))


def is_classic_ugi_core(mol):
    """自检核心 α-酰氨基酰胺骨架是否存在：R1-N(R')-C(=O)-CH(R3)-C(=O)-N(R2)。
    即验证两侧酰胺（N-C(=O)）+ 中间 CH-C(=O)-N 顺序。
    """
    if mol is None:
        return False
    patC_N_amide = Chem.MolFromSmarts("C(=O)-[NX3]")          # 末端酰胺 N-C(=O)
    pat_NC_am = Chem.MolFromSmarts("[NX3] [CX4][CX3](=O)[NX3]")  # 头侧 N-C-C(=O)-N(异腈侧)
    # 更强的 Ugi 记号：O=C-N(R1)-CH(R3)-C(=O)-N(R2)
    ugi = Chem.MolFromSmarts("O=C[!C:1][CX4]([CH2,CX4])[CX3](=O)[NX3]" if False else
                             "[CX3](=O)[NX3][CX4][CX3](=O)[NX3]")
    return mol.HasSubstructMatch(ugi)


def make_ug_colored(mol, label=False):
    """复制分子并给四组分连接位打上原子标签，用于排障 PNG。"""
    m = Chem.Mol(mol)
    # A位(head连接的N, idx2), B位(iso连接的N,idx6), C位(醛连接CH,idx3), D位(酸羰基C,idx0)
    marks = {2: "HEAD-N", 6: "ISO-N", 3: "ALD-CH", 0: "ACID-C=O"}
    for idx, note in marks.items():
        if idx < m.GetNumAtoms():
            m.GetAtomWithIdx(idx).SetProp("atomNote", note)
    return m


def cmd_verify_structure(pairs, out_dir):
    """把第 1 个成功产物与四条组分片段关系输出为 atom-coloured PNG。"""
    from rdkit.Chem.Draw import rdMolDraw2D
    if not pairs:
        print("[verify] 无产物可绘制")
        return 0
    prod, src = pairs[0]
    m = make_ug_colored(prod, label=True)
    d = rdMolDraw2D.MolDraw2DCairo(900, 700)
    rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    with open(out_dir / "structure_verify.png", "wb") as f:
        f.write(d.GetDrawingText())
    print(f"[verify] 第一产物 Ugi 拓扑图 → {out_dir}/structure_verify.png")
    print(f"        四组分: head={src[1]} ald={src[2]} acid={src[3]} iso={src[4]}")
    return 1


def _count_hydrophobic_tails(mol, min_len):
    """近似疏水尾条数：数满足“长度≥min_len 的独立脂肪链”。

    只沿非芳香 C-C 键（sp3/sp2）构建骨架邻接；从每个末端甲基出发做
    带 visited 集合的 BFS，深度上限 = min_len，避免环系（螺环/桥环）导致
    指数级路径枚举。同一分支点下的多个甲基共享同一脂肪链，只计一次。
    """
    n = mol.GetNumAtoms()
    adj = [[] for _ in range(n)]
    for b in mol.GetBonds():
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetAtomicNum() == 6 and a2.GetAtomicNum() == 6 and not b.GetIsAromatic():
            h1 = a1.GetHybridization(); h2 = a2.GetHybridization()
            if (h1 in (Chem.HybridizationType.SP3, Chem.HybridizationType.SP2)
                    and h2 in (Chem.HybridizationType.SP3, Chem.HybridizationType.SP2)):
                adj[a1.GetIdx()].append(a2.GetIdx())
                adj[a2.GetIdx()].append(a1.GetIdx())

    count = 0
    seen_branch = set()
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 6 and a.GetTotalNumHs() == 3 and len(adj[a.GetIdx()]) == 1:
            leaf = a.GetIdx()
            nb = adj[leaf][0]
            if nb in seen_branch:          # gem-二甲基等共享分支点只计 1 尾
                continue
            seen_branch.add(nb)
            # BFS 深度上限 min_len：visited 集合防环重复
            seen = {leaf}
            frontier = [(leaf, 0)]
            reached = 0
            while frontier:
                u, d = frontier.pop()
                if d > reached:
                    reached = d
                if d >= min_len - 1:
                    continue
                for v in adj[u]:
                    if v not in seen:
                        seen.add(v)
                        frontier.append((v, d + 1))
            if reached >= min_len - 1:
                count += 1
    return count


def cmd_curate(products, opt):
    """产物级规则过滤。opt: argparse Namespace（curate mw/tpsa/hbd/tail 参数）。

    返回 (passed, stats)，stats 统计每个被淘汰产物的首个不满足条件。
    """
    passed, stats = [], Counter()
    for p in products:
        mw = Descriptors.MolWt(p)
        if not (opt.curate_mw_min <= mw <= opt.curate_mw_max):
            stats["mw_out"] += 1; continue

        tpsa = Descriptors.TPSA(p)
        if tpsa is not None and tpsa > opt.curate_tpsa_max:
            stats["tpsa>max"] += 1; continue

        hbd = Descriptors.NumHDonors(p)
        if hbd > opt.curate_hbd_max:
            stats["hbd>max"] += 1; continue

        if not _nbasic(p):
            stats["no_basic_center"] += 1; continue

        tails = _count_hydrophobic_tails(p, opt.curate_tail_len)
        if tails < opt.curate_tail_n:
            stats["tail_lt_n"] += 1; continue

        passed.append(p)
    return passed, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--libs", default=(
        "data/head_v0.1/output/head_v0.1_filtered.smi,"
        "data/carbonyl_v0.1/output/carbonyl_v0.1_filtered.smi,"
        "data/acid_v0.1/output/acid_v0.1_filtered.smi,"
        "data/isocyanide_v0.1/output/isocyanide_v0.1_filtered.smi"))
    ap.add_argument("--n", type=int, default=20, help="每子库随机抽取数")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="data/ugi_template_v0.1/output")
    ap.add_argument("--draw", action="store_true", help="绘制 2D 结构图 PNG")
    # 结构自检 / 产物规则过滤（v2 新增）
    ap.add_argument("--verify-structure", action="store_true",
                    help="输出边分子图 PNG：atom-labeled + 四组分着色，校验 Ugi 拓扑映射")
    ap.add_argument("--curate", action="store_true",
                    help="产物级过滤：可电离叔胺≥1 · TPSA≤X · HBD≤Y · MW 区间 · 疏水尾≥2")
    ap.add_argument("--curate-tpsa-max", type=float, default=120.0)
    ap.add_argument("--curate-hbd-max", type=int, default=4)
    ap.add_argument("--curate-mw-min", type=float, default=500.0)
    ap.add_argument("--curate-mw-max", type=float, default=1200.0)
    ap.add_argument("--curate-tail-n", type=int, default=2, help="疏水尾数量下限")
    ap.add_argument("--curate-tail-len", type=int, default=8, help="每条疏水尾最低碳数")
    args = ap.parse_args()

    libs = args.libs.split(",")
    if len(libs) != 4:
        print("[error] 需 4 个库路径（head,carbonyl,acid,isocy）", file=sys.stderr)
        sys.exit(1)
    out = Path(args.out_dir)
    (out / "png").mkdir(parents=True, exist_ok=True)

    rows_h = load_lib(Path(libs[0]))
    rows_c = load_lib(Path(libs[1]))
    rows_a = load_lib(Path(libs[2]))
    rows_i = load_lib(Path(libs[3]))

    th = random_sample(rows_h, args.n, args.seed)
    tc = random_sample(rows_c, args.n, args.seed + 1)
    ta = random_sample(rows_a, args.n, args.seed + 2)
    ti = random_sample(rows_i, args.n, args.seed + 3)
    print(f"[info] 采样 Head={len(th)} Carbonyl={len(tc)} Acid={len(ta)} Iso={len(ti)}")

    # 直接笛卡尔积（n=20 → 16e4 组合，可控）
    total = len(th) * len(tc) * len(ta) * len(ti)
    ok = 0
    fail_reason = Counter()
    products = []
    first_src = None
    t0 = time.time()

    for hsmi, hid in th:
        for csmi, cid in tc:
            for asmi, aid in ta:
                for ismi, iid in ti:
                    hf = fragmentize(hsmi, "head")
                    cf = fragmentize(csmi, "aldehyde")
                    af = fragmentize(asmi, "acid")
                    inf = fragmentize(ismi, "isocyanide")
                    if None in (hf, cf, af, inf):
                        fail_reason["no_fragment"] += 1
                        continue
                    # * 端通常在 SMARTS 替换后规范化为 "[*]" 或首字符 *
                    prod = build_product(hf, cf, af, inf, hsmi, csmi, asmi, ismi)
                    if prod is None:
                        fail_reason["combine_fail"] += 1
                        continue
                    try:
                        Chem.SanitizeMol(prod)
                        Chem.GetSymmSSSR(prod)
                    except Exception as e:
                        fail_reason["sanitize"] += 1
                        continue
                    prod.SetProp("_Name", f"U4_{hid}_{cid}_{aid}_{iid}")
                    prod.SetProp("head", hid); prod.SetProp("carbonyl", cid)
                    prod.SetProp("acid", aid);  prod.SetProp("isocyanide", iid)
                    products.append(prod)
                    if len(products) == 1:      # 记录首个成功产物的组分供结构自检
                        first_src = (hid, csmi, asmi, ismi, hsmi, cf, af, inf)
                    ok += 1
                    if ok % 2000 == 0:
                        print(f"  [{ok}/{total}] 已生成", flush=True)

    print(f"\n=== 结果 ===")
    print(f"总组合数: {total:,}")
    print(f"成功产物: {ok:,}  ({100*ok/max(total,1):.1f}%)")
    print(f"失败原因: {dict(fail_reason)}")
    print(f"耗时: {time.time()-t0:.1f}s")

    if not products:
        print("[error] 无产物，终止", file=sys.stderr); sys.exit(1)

    # 去重（canonical）
    seen = set(); uniq = []
    for p in products:
        c = Chem.MolToSmiles(p)
        if c in seen:
            continue
        seen.add(c); uniq.append(p)
    print(f"唯一产物: {len(uniq):,}")

    # 结构自检：把首个产物的四组分连接位着色输出 → 核对 Ugi 拓扑映射
    if args.verify_structure and first_src is not None:
        cmd_verify_structure([(uniq[0], first_src)], out)

    # 产物级规则过滤（默认关闭，--curate 开启）
    curated = uniq
    if args.curate:
        curated, cstats = cmd_curate(uniq, args)
        print(f"\n=== 产物规则过滤 (--curate) ===")
        print(f"过滤前: {len(uniq):,}  过滤后: {len(curated):,}")
        if len(uniq):
            print(f"保留率: {100*len(curated)/len(uniq):.1f}%")
        for reason, cnt in cstats.most_common():
            print(f"  淘汰[{reason:14s}]: {cnt:,}")

    # 写入 SDF + 分子量分布
    writer = Chem.SDWriter(str(out / "products.sdf"))
    mws = []
    for p in curated:
        writer.write(p)
        mws.append(Descriptors.MolWt(p))
    writer.close()
    print(f"MW 分布: min={min(mws):.1f} med={sorted(mws)[len(mws)//2]:.1f} max={max(mws):.1f}")

    # 绘制前 20 个（curated）
    if args.draw and curated:
        for i, p in enumerate(curated[:20]):
            try:
                d = rdMolDraw2D.MolDraw2DCairo(500, 400)
                rdMolDraw2D.PrepareAndDrawMolecule(d, p)
                d.FinishDrawing()
                with open(out / "png" / f"product_{i+1:05d}.png", "wb") as f:
                    f.write(d.GetDrawingText())
            except Exception as e:
                print(f"  draw fail {i}: {e}")
        print(f"[ok] 已绘制 20 张 PNG → {out}/png/")

    print(f"\n产物示例 (canonical):")
    for p in list(curated)[:10]:
        print("  ", Chem.MolToSmiles(p))


if __name__ == "__main__":
    main()