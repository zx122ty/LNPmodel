#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
head_build.py — Ugi-4CR Head（氨基头部）子库 Stage2/3：RDKit 精筛 + 合并去重 + QC
================================================================================
输入：Stage1 候选池 data/head_v0.1/candidates/*.tsv.gz
输出（按 03.md §5.1.4/5.1.5/5.1.6）：
    head_v0.1_main.smi            (SMILES \t head_id，全量主库)
    head_v0.1_main.csv            (全字段)
    head_v0.1_flagged.smi         (近合格：多反应位点 / 残余碱性为 0)
    head_v0.1_excluded.smi        (硬排除：SMILES \t reason)
    head_v0.1_summary.json
    head_v0.1_qc_report.md

Head 定义（Ugi-4CR 胺组分，03.md §5.1.1）：
    * 恰好 1 个反应性伯胺/仲胺（Ugi 消耗 → 酰基酰胺）
    * ≥1 个残余碱性中心（叔胺/碱性芳杂氮）——保证产物仍是可电离脂质
    * MW 100–550（用户确认）
    * 无醛/酮/酸/异腈/异氰酸酯/腈等其它 Ugi 组分
    * 无保护胺、无季铵、无高反应性/毒性片段、无金属/无机盐
    * 无卤素（F, Cl, Br, I）——防止破坏头部 pKa 及引入代谢毒性
    * 去盐、强制中性、按 InChIKey 骨架去重

过滤顺序（务必先于中性化检测化学标签）：
    orig_mol: MRF（硝基/叠氮/重氮/N-N/N-O/过氧/酰卤/酸酐/NHS酯/磺酰卤/环氧化物/迈克尔受体/季铵/金属/保护基/卤素）
    neutral:  活性官能团（醛/酮/酸/异腈/异氰酸酯/腈/亚胺/胍/苯胺型 N/磺酸/膦酸）与胺计数 + PAINS

用法：
    python scripts/head_build.py build  --candidates-dir data/head_v0.1/candidates \
                                        --slices-dir data/head_v0.1/slices -t 24
    python scripts/head_build.py merge  --slices-dir data/head_v0.1/slices \
                                        --out-dir  data/head_v0.1/output
    python scripts/head_build.py qc     --out-dir data/head_v0.1/output

环境：python3.10 + RDKit 2026.03。
"""

import argparse
import csv
import gzip
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
# 兼容不同 RDKit 版本：Uncharger 可能在 rdkit.Chem.MolStandardize.rdMolStandardize 下
try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
    Uncharger = rdMolStandardize.Uncharger
except (ImportError, AttributeError):
    from rdkit.Chem import rdMolStandardize
    Uncharger = rdMolStandardize.Uncharger

# ----------------------------------------------------------------------------
# SMARTS 定义
# ----------------------------------------------------------------------------
_SMARTS = {
    # --- 反应性胺（可被 Ugi 消耗）---
    "prim_amine": "[NX3;H2;!$(NC=O);!$(N=C);!$(N-[OX2]);!$(N-[NX3]);!$([N+])]",
    "sec_amine":  "[NX3;H1;!$(NC=O);!$(N=C);!$(N-[OX2]);!$(N-[NX3]);!$([N+])]",
    # --- 残余碱性中心 ---
    "tertiary_amine": ("[NX3;H0;!$(NC=O);!$(N=[!#6]);!$(N-[OX2]);!$(N-[NX3]);"
                       "!$([N+]);!$(N-[c]);$(N([#6])([#6])[#6])]"),
    # 碱性芳杂氮：吡啶/嘧啶/咪唑/喹啉类（中性 n、非吡咯[nH]、非 N-氧化物）
    "aromatic_N": "[n;+0;!$(n[H]);!$([n+]([O-]));!$(n-[#8])]",
    # --- 排除：其它 Ugi 组分 / 干扰官能团 ---
    "aldehyde":   "[CX3H1,H2](=O)", # 修复：包含甲醛
    "ketone":     "[#6][CX3](=O)[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "carboxylate":     "[CX3](=O)[OX1-]",
    "isocyanide":  "[N+]#[C-]",
    "isocyanate":  "N=C=O",         # 修复：修正 X1 连通度 Bug
    "isothiocyanate": "N=C=S",      # 修复：修正 X1 连通度 Bug
    "nitrile":     "C#N",           # 修复：修正 X1 连通度 Bug
    "imine":       "[CX3]=[NX2;!$([NX2]-[OX1])]",
    "oxime":       "[CX3]=[NX2]-[OX1]",
    "guanidine":   "[NX3][CX3](=[NX2])[NX3]",
    "aniline_type": "[N;+0;$(N-[c])]",
    "primary_aniline": "[N;H2;$(N-[c])]",
    "sulfonic_acid": "[S;X4](=O)(=O)[OX2H1]", # 新增：排除磺酸
    "phosphonic_acid": "[P;X4](=O)([OX2H1])[OX2H1]", # 新增：排除膦酸
    # --- 排除：不稳定 / 高反应性 / 保护基 / 季铵 / 卤素（在原子上直接判定）---
    "azide":    "[N-]=[N+]=[N]",
    "nitro":    "[NX3+](=O)[O-]",
    "diazonium": "[NX2+]#[NX1]",
    "nitroso":   "[NX2]=[OX1]",
    "azo":      "[#6][NX2]=[NX2][#6]",
    "N_N_bond": "[NX3;!$(N(C=O)C=O)]-[NX3;!$(N(C=O)C=O)]",  # 非酰肼的 N-N
    "N_O_bond": "[NX3;!$(NC=O)]-[OX2;!$(O-C(=O))]",
    "peroxide": "[OX2;!$(O-C(=O))][OX2;!$(O-C(=O))]",
    "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
    "anhydride": "[CX3](=O)-[OX2]-[CX3](=O)",
    "NHS_ester": "[CX3](=O)[OX2]N1C(=O)CCC1=O",
    "sulfonyl_halide": "[SX4](=O)(=O)[F,Cl,Br,I]",
    "epoxide":  "[OX2]1[#6][#6]1",
    "aziridine": "[NX3]1[#6][#6]1",
    "michael_acceptor": "[CX3]=[CX3][CX3](=O)[#6,#7,#8,#16]",
    "aldehyde_deriv": "[CX3](=[OX1])[S,N]",
    "quaternary_N": "[N+X4]",
    "hydroxylamine": "[NX3]-[OX2]",
    "hydrazine":  "[NX3]-[NX3]",
    "halogen": "[#9,#17,#35,#53]", # 新增：硬排除卤素（F, Cl, Br, I），保护头部 pKa
    # --- 保护基 ---
    "tert_butyl_carbamate": "C(=O)OC(C)(C)C",
    "benzyl_carbamate":     "[CX3](=O)[OX2][CH2]c1ccccc1",
    "fmoc": "C(=O)[OX2]C1c2ccccc2c2ccccc21",
    "alloc": "C(=O)[OX2]CC=C",
    "trityl": "C(c1ccccc1)(c1ccccc1)c1ccccc1",
    "tosyl":  "[SX4](=O)(=O)c1ccc(C)cc1",
    "mesyl":  "[SX4](=O)(=O)C",
    "phthalimide": "O=C1c2ccccc2C(=O)N1",
}

# 金属 / 无机元素（原子序数集）
_METALS = {3, 4, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
           31, 32, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
           55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
           73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91,
           92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103}

_PATTERNS = {name: Chem.MolFromSmarts(sma) for name, sma in _SMARTS.items()}
if any(p is None for p in _PATTERNS.values()):
    raise RuntimeError("SMARTS 解析失败！")

# 亲电 / 毒性快查目录（PAINS）
_PAINS_CAT = None
def _get_pains():
    global _PAINS_CAT
    if _PAINS_CAT is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CAT = FilterCatalog(params)
    return _PAINS_CAT


# ----------------------------------------------------------------------------
# 原子级工具
# ----------------------------------------------------------------------------
def _has_match(mol, key):
    return _PATTERNS[key] is not None and mol.HasSubstructMatch(_PATTERNS[key])


def _count(mol, key):
    return len(mol.GetSubstructMatches(_PATTERNS[key])) if _PATTERNS[key] is not None else 0


def strip_salts_neutralize(rdmol):
    """去盐（优先保留含反应性胺和残余碱基的片段）+ 强制中性（Uncharger）。

    返回 (frag_mol, info) 或 (None, info)。
    """
    info = {"n_frags": 0, "inorganic": False}
    if rdmol is None:
        return None, info
    try:
        Chem.SanitizeMol(rdmol)
    except Exception:
        return None, info

    frags = list(Chem.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False))
    info["n_frags"] = len(frags)

    def heavy(mol):
        return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)

    # 过滤掉含金属的片段
    valid_frags = []
    has_metal_glob = False
    for f in frags:
        has_metal = any(a.GetAtomicNum() in _METALS for a in f.GetAtoms())
        if has_metal:
            has_metal_glob = True
        else:
            valid_frags.append(f)

    if has_metal_glob:
        info["inorganic"] = True

    # 优先选含N的，其次按反应性胺、残余碱基、重原子数排序（胺优先策略）
    def sort_key(mol):
        has_n = any(a.GetAtomicNum() == 7 for a in mol.GetAtoms())
        n_reactive = _count(mol, "prim_amine") + _count(mol, "sec_amine")
        n_residual = _count(mol, "tertiary_amine") + _count(mol, "aromatic_N")
        return (has_n, n_reactive, n_residual, heavy(mol))
        
    valid_frags.sort(key=sort_key, reverse=True)
    
    best = None
    for f in valid_frags:
        if any(a.GetAtomicNum() == 7 for a in f.GetAtoms()):
            best = Chem.Mol(f)
            break
            
    if best is None:
        return None, info

    # 强制中性：使用 RDKit 官方的 Uncharger
    uncharger = Uncharger()
    try:
        best = uncharger.uncharge(best)
        Chem.SanitizeMol(best)
        best.UpdatePropertyCache(strict=True)
    except Exception:
        return None, info
    return best, info


def _is_stereo(mol):
    return (any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())
            or any(b.GetBondType() == Chem.BondType.DOUBLE and
                   b.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
                   for b in mol.GetBonds()))


# ----------------------------------------------------------------------------
# 单分子过滤
# ----------------------------------------------------------------------------
def analyze_one(smiles: str):
    """候选 SMILES → 判决 dict（不回传 mol，只回传标量，省内存/序列化）。"""
    r = {"smiles": smiles, "exclude": None, "mrf_orig": [], "mrf_neutral": [],
         "n_prim": 0, "n_sec": 0, "n_tert": 0, "n_aromN": 0,
         "n_reactive": 0, "n_residual": 0, "mw": None, "logp": None, "tpsa": None,
         "hbd": None, "hba": None, "stereo": True, "head_class": "unknown",
         "site": "primary/unknown"}

    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        r["exclude"] = "parse_fail"
        return r

    orig = rdmol
    # 1) MRF：对原始分子检测化学标签（硝基/叠氮/… 在去电荷前判定）
    mrf_orig = [k for k in ("azide", "nitro", "diazonium", "nitroso", "azo", "N_N_bond",
                            "N_O_bond", "peroxide", "acyl_halide", "anhydride", "NHS_ester",
                            "sulfonyl_halide", "epoxide", "aziridine", "michael_acceptor",
                            "quaternary_N", "hydroxylamine", "hydrazine", "halogen",
                            "tert_butyl_carbamate", "benzyl_carbamate", "fmoc", "alloc",
                            "trityl", "tosyl", "mesyl", "phthalimide")
                if _has_match(orig, k)]
    r["mrf_orig"] = mrf_orig
    if mrf_orig:
        # 硬排除：过弱酰基/极不稳定的基团必须剔除（保护基/季铵/硝基/卤素…以及 N-N、N-O 键）
        if any(k in mrf_orig for k in ("tert_butyl_carbamate", "benzyl_carbamate", "fmoc",
                                       "alloc", "trityl", "quaternary_N", "azide", "nitro",
                                       "diazonium", "peroxide", "acyl_halide", "anhydride",
                                       "NHS_ester", "sulfonyl_halide", "epoxide", "aziridine",
                                       "michael_acceptor", "hydroxylamine", "hydrazine", "halogen",
                                       "N_N_bond", "N_O_bond")):
            r["exclude"] = "+".join(k for k in mrf_orig)
            return r
        # 仅标记：亚硝基/偶氮等，仍走后续胺计数
        r["mrf_orig"] = mrf_orig

    # 2) 去盐 + 中性化
    neutral, sinfo = strip_salts_neutralize(orig)
    if neutral is None:
        r["exclude"] = "inorganic_or_salt_only" if sinfo["inorganic"] else "no_organic_frag"
        return r
    if sinfo["inorganic"]:
        r["exclude"] = "contains_metal"
        return r

    # 2b) 残片 / 同位素 / 残留盐校验
    has_isotope = any(a.GetIsotope() != 0 for a in neutral.GetAtoms())
    if has_isotope:
        r["exclude"] = "isotope_atom"
        return r
    has_charge = any(a.GetFormalCharge() != 0 for a in neutral.GetAtoms())
    if has_charge:
        r["exclude"] = "residual_charge"
        return r

    # 3) 中性化后的干扰官能团
    mrf_neutral = [k for k in ("aldehyde", "ketone", "carboxylic_acid", "carboxylate",
                               "isocyanide", "isocyanate", "isothiocyanate", "nitrile",
                               "imine", "oxime", "guanidine", "primary_aniline",
                               "sulfonic_acid", "phosphonic_acid")
                   if _has_match(neutral, k)]
    r["mrf_neutral"] = mrf_neutral
    if mrf_neutral:
        r["exclude"] = "+".join(mrf_neutral)
        return r

    # 3b) PAINS 过滤
    pains_cat = _get_pains()
    if pains_cat.HasMatch(neutral):
        r["exclude"] = "PAINS"
        return r

    # 4) 胺计数（中性分子上）
    n_prim = _count(neutral, "prim_amine")
    n_sec = _count(neutral, "sec_amine")
    n_tert = _count(neutral, "tertiary_amine")
    n_aromN = _count(neutral, "aromatic_N")
    r.update({"n_prim": n_prim, "n_sec": n_sec, "n_tert": n_tert, "n_aromN": n_aromN,
              "n_reactive": n_prim + n_sec, "n_residual": n_tert + n_aromN})
    n_resid = r["n_residual"]

    # 5) 其它属性
    r["mw"] = Descriptors.MolWt(neutral)
    r["logp"] = Crippen.MolLogP(neutral)
    r["tpsa"] = Descriptors.TPSA(neutral)
    r["hbd"] = Descriptors.NumHDonors(neutral)
    r["hba"] = Descriptors.NumHAcceptors(neutral)
    r["stereo"] = _is_stereo(neutral)
    r["site"] = "primary" if n_prim == 1 else ("secondary" if n_sec == 1 else "mixed")

    # 6) 头部骨架分类
    if n_tert >= 1 and n_aromN == 0:
        r["head_class"] = "tertiary_aliphatic"
    elif n_aromN >= 1 and n_tert == 0:
        r["head_class"] = "heteroaromatic"
    elif n_tert >= 1 and n_aromN >= 1:
        r["head_class"] = "mixed_aliphatic_aromatic"
    else:
        r["head_class"] = "other"

    # 7) 入选/挂起/排除
    if n_prim + n_sec == 0:
        r["exclude"] = "no_reactive_amine"
    elif n_prim + n_sec > 1:
        r["exclude"] = "multi_reactive"      # → flagged（近合格）
    elif n_resid == 0:
        r["exclude"] = "residual_basic_zero"  # → flagged（近合格）
    elif not (100.0 <= r["mw"] <= 550.0):
        r["exclude"] = "mw_out_of_range"
    else:
        # 非主库的“低危 MRF 标签”（亚硝基/偶氮等）在此搁置到 flagged
        if r["mrf_orig"]:
            r["exclude"] = "mrf_soft:" + "+".join(r["mrf_orig"])
        else:
            r["exclude"] = None   # → main
    return r


# ----------------------------------------------------------------------------
# Stage2：分片并行精筛
# ----------------------------------------------------------------------------
def _build_worker(arg):
    candidate_path, slice_dir, min_mw, max_mw = arg
    base = os.path.basename(candidate_path).replace(".tsv.gz", "")
    out_m = os.path.join(slice_dir, base + ".main.tsv.gz")
    out_f = os.path.join(slice_dir, base + ".flagged.tsv.gz")
    out_e = os.path.join(slice_dir, base + ".excluded.tsv.gz")
    t0 = time.time()
    counts = Counter()
    with gzip.open(candidate_path, "rt") as fin, \
         gzip.open(out_m + ".tmp", "wt") as fm, \
         gzip.open(out_f + ".tmp", "wt") as ff, \
         gzip.open(out_e + ".tmp", "wt") as fe:

        header = "\t".join(["cid", "smiles", "inchikey", "mw", "logp", "tpsa",
                            "hbd", "hba", "n_primary_amine", "n_secondary_amine",
                            "n_reactive_amine_total", "n_tertiary_amine",
                            "n_hetero_aromatic_n", "n_residual_basic_center",
                            "head_class", "reaction_site_type", "stereo_flag",
                            "exclude_reason"])
        for path, fh in ((out_m, fm), (out_f, ff), (out_e, fe)):
            _ = path
        fm.write(header + "\n"); ff.write(header + "\n"); fe.write(header + "\n")
        # 注意：excluded 文件表头与主库相同，exclude_reason 记录原因

        lines = fin  # 流式
        next(lines)  # 跳过表头
        for ln in lines:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3:
                counts["bad_row"] += 1
                continue
            cid, smi, ich = parts[0], parts[1], parts[2]
            r = analyze_one(smi)
            mw = r["mw"]
            out_row = "\t".join([
                cid, r["smiles"], ich,
                "" if mw is None else f"{mw:.2f}",
                "" if r["logp"] is None else f"{r['logp']:.2f}",
                "" if r["tpsa"] is None else f"{r['tpsa']:.1f}",
                str(r["hbd"]), str(r["hba"]),
                str(r["n_prim"]), str(r["n_sec"]), str(r["n_reactive"]),
                str(r["n_tert"]), str(r["n_aromN"]), str(r["n_residual"]),
                r["head_class"], r["site"], "Y" if r["stereo"] else "N",
                r["exclude"] or ""])
            if r["exclude"] is None:
                fm.write(out_row + "\n"); counts["main"] += 1
            elif r["exclude"].startswith("multi_reactive") or \
                 r["exclude"].startswith("residual_basic_zero") or \
                 r["exclude"].startswith("mrf_soft"):
                ff.write(out_row + "\n"); counts["flagged"] += 1
            else:
                fe.write(out_row + "\n"); counts["excluded"] += 1

    for p in (out_m, out_f, out_e):
        os.replace(p + ".tmp", p)
    return {"file": base, **dict(counts), "seconds": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------
# Stage2 主入口
# ----------------------------------------------------------------------------
def _cmd_build(args):
    cand = Path(args.candidates_dir)
    # ---- 子集过滤：优先读入 filelist（分区并行用），再 fallback 到通配 -----
    input_files = []
    for fl in args.filelist or []:
        with open(fl) as f:
            input_files += [ln.strip() for ln in f if ln.strip()]
    if input_files:
        files = [Path(p) for p in input_files if p.startswith(str(cand.resolve()) + os.sep)
                 or not p.startswith("/")]
        files = [f if f.is_absolute() else cand / f for f in files]
        files = [f for f in files if f.exists()]
    else:
        files = sorted(cand.glob("*.tsv.gz"))
    if not files:
        print("[error] 候选文件为空（filelist 或通配均无）", file=sys.stderr); sys.exit(1)
    os.makedirs(args.slices_dir, exist_ok=True)

    print(f"[info] 候选文件 {len(files)}，线程 {args.threads}", flush=True)
    summary = []
    with ProcessPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(_build_worker, (str(f), args.slices_dir,
                                           args.min_mw, args.max_mw)): f for f in files}
        for fut in as_completed(futs):
            f = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                print(f"[error] {f}: {e!r}", file=sys.stderr); continue
            summary.append(r)
            print(f"  [ok] {r['file']}: main={r['main']:,} flagged={r['flagged']:,} "
                  f"excluded={r['excluded']:,} {r['seconds']}s", flush=True)

    agg = Counter()
    for r in summary:
        for k in ("main", "flagged", "excluded"):
            agg[k] += r[k]
    print(json.dumps({"files": len(summary), "main": agg["main"],
                      "flagged": agg["flagged"], "excluded": agg["excluded"]}, indent=2))


# ----------------------------------------------------------------------------
# Stage3：合并 + 去重 + 最终化
# ----------------------------------------------------------------------------


def _cmd_merge(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    slices_dir = Path(args.slices_dir)
    if not list(slices_dir.glob("*.main.tsv.gz")):
        print("[error] 无分片结果", file=sys.stderr); sys.exit(1)

    # 分 3 趟读入（main/flagged/excluded）→ 去重 → 写最终文件
    def load_dedup(suffix):
        rows = []
        seen = set()
        dup = 0
        for s in sorted(slices_dir.glob(f"*.{suffix}.tsv.gz")):
            with gzip.open(s, "rt") as f:
                next(f)  # header
                for ln in f:
                    cols = ln.rstrip("\n").split("\t")
                    if len(cols) < 3:
                        continue
                    # 优先使用 InChIKey，如果没有则用 SMILES
                    key = cols[2] if cols[2] else cols[1]
                    if key in seen:
                        dup += 1
                        continue
                    seen.add(key)
                    rows.append(cols)
        if suffix == "main":
            rows.sort(key=lambda r: float(r[3]) if r[3] else 1e9)  # 主库按 MW 排序
        return rows, dup

    main_rows, main_dup = load_dedup("main")
    flag_rows, flag_dup = load_dedup("flagged")
    excl_rows, excl_dup = load_dedup("excluded")

    header = ["cid", "smiles", "inchikey", "mw", "logp", "tpsa", "hbd", "hba",
              "n_primary_amine", "n_secondary_amine", "n_reactive_amine_total",
              "n_tertiary_amine", "n_hetero_aromatic_n", "n_residual_basic_center",
              "head_class", "reaction_site_type", "stereo_flag", "exclude_reason"]

    # 主库：分配 head_id
    def assign(rows):
        for i, r in enumerate(rows, 1):
            yield [f"HEAD_{i:07d}"] + r

    # 写 CSV（全部字段）
    with open(out / "head_v0.1_main.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head_id"] + header)
        w.writerows(assign(main_rows))
    with open(out / "head_v0.1_flagged.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["head_id"] + header)
        w.writerows(assign(flag_rows))
    with open(out / "head_v0.1_excluded.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(excl_rows)

    # 写 .smi
    with open(out / "head_v0.1_main.smi", "w") as f:
        for i, r in enumerate(main_rows, 1):
            f.write(f"{r[1]}\tHEAD_{i:07d}\n")
    with open(out / "head_v0.1_flagged.smi", "w") as f:
        f.writelines(f"{r[1]}\t{r[17] or ''}\n" for r in flag_rows)
    with open(out / "head_v0.1_excluded.smi", "w") as f:
        f.writelines(f"{r[1]}\t{r[17] or ''}\n" for r in excl_rows)

    # summary.json
    reason_counter = Counter()
    for r in excl_rows:
        reason_counter[r[17] or "unknown"] += 1
    summary = {
        "version": "head_v0.1",
        "n_main_after_dedup": len(main_rows),
        "n_main_duplicates": main_dup,
        "n_flagged_after_dedup": len(flag_rows),
        "n_excluded_after_dedup": len(excl_rows),
        "main_exclude_reason_top": reason_counter.most_common(20),
        "head_class_dist": Counter(r[14] for r in main_rows).most_common(),
        "mw_range": [min(float(r[3]) for r in main_rows), max(float(r[3]) for r in main_rows)],
    }
    with open(out / "head_v0.1_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ----------------------------------------------------------------------------
# QC 报告
# ----------------------------------------------------------------------------
def _cmd_qc(args):
    out = Path(args.out_dir)
    import pandas as pd
    df = pd.read_csv(out / "head_v0.1_main.csv")
    n = len(df)
    with open(out / "head_v0.1_qc_report.md", "w") as f:
        f.write("# Head v0.1 QC 报告\n\n")
        f.write(f"- 主库分子数：**{n:,}**\n\n")
        if n:
            f.write("## 代表性分子（按 MW 排序取中位 20 例）\n\n")
            mid = df.sort_values("mw")
            f.write("| head_id | SMILES | MW | LogP | TPSA | head_class |\n|---|---|---|---|---|---|\n")
            for _, r in mid.iloc[max(0, n//2-10): n//2+10].iterrows():
                f.write(f"| {r.head_id} | `{r.smiles}` | {r.mw:.1f} | {r.logp:.1f} | {r.tpsa:.1f} | {r.head_class} |\n")
        f.write("\n## MW 分布\n\n")
        if n:
            f.write(f"min={df.mw.min():.1f}, Q1={df.mw.quantile(.25):.1f}, "
                    f"median={df.mw.median():.1f}, Q3={df.mw.quantile(.75):.1f}, "
                    f"max={df.mw.max():.1f}\n\n")
            f.write("## LogP 分布\n\n")
            f.write(f"min={df.logp.min():.1f}, Q1={df.logp.quantile(.25):.1f}, "
                    f"median={df.logp.median():.1f}, Q3={df.logp.quantile(.75):.1f}, "
                    f"max={df.logp.max():.1f}\n\n")
    print(f"[ok] QC 报告已写入 {out}/head_v0.1_qc_report.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_b = sub.add_parser("build")
    p_b.add_argument("--candidates-dir", default="data/head_v0.1/candidates")
    p_b.add_argument("--slices-dir", default="data/head_v0.1/slices")
    p_b.add_argument("--threads", type=int, default=24)
    p_b.add_argument("--min-mw", type=float, default=100.0)
    p_b.add_argument("--max-mw", type=float, default=550.0)
    p_b.add_argument("--filelist", action="append", default=[],
                     help="可选：只处理列出的候选文件（可多次指定，每行一个路径），用于分区并行")
    p_b.set_defaults(fn=_cmd_build)

    p_m = sub.add_parser("merge")
    p_m.add_argument("--slices-dir", default="data/head_v0.1/slices")
    p_m.add_argument("--out-dir", default="data/head_v0.1/output")
    p_m.set_defaults(fn=_cmd_merge)

    p_q = sub.add_parser("qc")
    p_q.add_argument("--out-dir", default="data/head_v0.1/output")
    p_q.set_defaults(fn=_cmd_qc)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()