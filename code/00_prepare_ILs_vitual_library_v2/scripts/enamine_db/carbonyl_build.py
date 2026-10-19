#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
carbonyl_build.py — Ugi-4CR Carbonyl/Tail（醛/酮）子库 Stage2/3：RDKit 精筛 + 合并去重 + QC
=============================================================================================
输入：Stage1 候选池 data/carbonyl_ENA/candidates/*.tsv.gz
输出（03.md §5.2.4 / 5.2.2 / 5.2.3）：
    carbonyl_ENA_main.csv / .smi         （主库）
    carbonyl_ENA_flagged.csv / .smi      （近合格：多羰基/低风险标签）
    carbonyl_ENA_excluded.csv / .smi     （排除，含原因）
    carbonyl_ENA_summary.json
    carbonyl_ENA_qc_report.md

Carbonyl/Tail 定义（Ugi-4CR 醛/酮组分，03.md §5.2）：
    * 含 1 个可用于 Ugi 的醛（[CX3H1,H2](=O)）或酮（[#6][CX3](=O)[#6]）
    * 主要提供疏水尾/骨架连接（C6-C20 长链优先，但可放宽）
    * 无自由伯/仲胺、无羧酸、无异腈、无其他强反应组分
    * 无 α,β-不饱和醛/酮（迈克尔受体）、缩醛/缩酮（除非明确去保护）、多羰基
    * 无 β-羟基羰基（Aldol 产物）、无高活性/毒性片段、无金属/无机盐
    * 排除：卤素（F, Cl, Br, I）、磺酸/膦酸（强酸）、PAINS 毒性结构
    * 去盐、强制中性，按 InChIKey 去重

用法：
    python scripts/carbonyl_build.py build --candidates-dir data/carbonyl_ENA/candidates \
                                           --slices-dir data/carbonyl_ENA/slices -t 24 \
                                           --filelist data/carbonyl_ENA/.todo.txt
    python scripts/carbonyl_build.py merge  --slices-dir data/carbonyl_ENA/slices --out-dir data/carbonyl_ENA/output
    python scripts/carbonyl_build.py qc     --out-dir data/carbonyl_ENA/output

环境：python3.10 + RDKit 2026.03。
"""

import argparse
import csv
import gzip
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

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
# SMARTS
# ----------------------------------------------------------------------------
_SMARTS = {
    # --- Ugi 反应性 → 醛 / 酮 ---
    "aldehyde": "[CX3H1,H2](=O)", # 修复：包含甲醛
    # 酮：羰基两个邻接均为碳，且排除酯/酰胺/内酯（邻接带 O/N 的情形）
    "ketone":   "[#6][CX3](=O)[#6;!$([CX3](=O)[OX2,NX3])]",
    # --- 排除：其它 Ugi 组分 / 干扰官能团 ---
    "primary_amine": "[NX3;H2;!$(NC=O);!$(N=C);!$(N-[OX2]);!$(N-[NX3]);!$([N+])]",
    "secondary_amine": "[NX3;H1;!$(NC=O);!$(N=C);!$(N-[OX2]);!$(N-[NX3]);!$([N+])]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "carboxylate":     "[CX3](=O)[OX1-]",
    "isocyanide":  "[N+]#[C-]",
    "isocyanate":  "N=C=O",         # 修复：连通度 Bug
    "isothiocyanate": "N=C=S",      # 修复：连通度 Bug
    "nitrile":     "C#N",           # 修复：连通度 Bug
    "imine":       "[CX3]=[NX2;!$([NX2]-[OX1])]",
    "oxime":       "[CX3]=[NX2]-[OX1]",
    # --- 排除：强酸 / 卤素 / 活泼氢 ---
    "sulfonic_acid": "[S;X4](=O)(=O)[OX2H1]",
    "phosphonic_acid": "[P;X4](=O)([OX2H1])[OX2H1]",
    "halogen": "[#9,#17,#35,#53]", # F, Cl, Br, I
    # --- 多重羰基 / 被保护羰基 / 干扰结构 ---
    "acyclic_ester":  "[CX3](=O)[OX2][#6]",   # 酰氧-烷基（酯）；酸酐也命中但已另行排除
    "amide":   "C(=O)[NX3]",
    "acetal":    "[CX4]([OX2])([OX2])[#6]",
    "thioacetal": "[CX4]([SX2])([SX2])[#6]",
    "aldol_OH": "[CX4]([OX2H1])-[CX4]-[CX3]=[OX1]", # β-羟基羰基
    # --- 迈克尔受体 / 高反应 ---
    "michael_acceptor": "[CX3]=[CX3][CX3](=O)[#6,#7,#8,#16]",
    "epoxide":  "[OX2]1[#6][#6]1",
    "aziridine": "[NX3]1[#6][#6]1",
    # --- 保护/不稳定/毒性 ---
    "nitro":    "[NX3+](=O)[O-]",
    "azide":    "[N-]=[N+]=[N]",
    "peroxide": "[OX2;!$(O-C(=O))][OX2;!$(O-C(=O))]",
    "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
    "anhydride": "[CX3](=O)-[OX2]-[CX3](=O)",
    "sulfonyl_halide": "[SX4](=O)(=O)[F,Cl,Br,I]",
    "quaternary_N": "[N+X4]",
    "tert_butyl_carbamate": "C(=O)OC(C)(C)C",
    "benzyl_carbamate":     "[CX3](=O)[OX2][CH2]c1ccccc1",
    "fmoc": "C(=O)[OX2]C1c2ccccc2c2ccccc21",
    "tosyl":  "[SX4](=O)(=O)c1ccc(C)cc1",
}

_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in _SMARTS.items()}
if any(p is None for p in _PATTERNS.values()):
    raise RuntimeError("SMARTS 解析失败")

_METALS = {3, 4, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
           31, 32, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
           55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
           73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91,
           92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103}

# PAINS 过滤器
_PAINS_CAT = None
def _get_pains():
    global _PAINS_CAT
    if _PAINS_CAT is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CAT = FilterCatalog(params)
    return _PAINS_CAT


def _has(mol, key):
    return _PATTERNS[key] is not None and mol.HasSubstructMatch(_PATTERNS[key])


def _cnt(mol, key):
    return len(mol.GetSubstructMatches(_PATTERNS[key])) if _PATTERNS[key] is not None else 0


def _strip_salts_neutralize(rdmol):
    """去盐（优先保留含羰基片段）+ 强制中性（Uncharger）。"""
    info = {"inorganic": False}
    if rdmol is None:
        return None, info
    try:
        Chem.SanitizeMol(rdmol)
    except Exception:
        return None, info
    
    frags = list(Chem.GetMolFrags(rdmol, asMols=True, sanitizeFrags=False))
    
    def heavy(m):
        return sum(1 for a in m.GetAtoms() if a.GetAtomicNum() > 1)

    valid_frags = []
    has_metal_glob = False
    for f in frags:
        if any(a.GetAtomicNum() in _METALS for a in f.GetAtoms()):
            has_metal_glob = True
        else:
            valid_frags.append(f)

    if has_metal_glob:
        info["inorganic"] = True

    # 羰基优先：优先保留含醛/酮的片段，其次看含氧，最后看重原子数
    def sort_key(mol):
        n_carb = _cnt(mol, "aldehyde") + _cnt(mol, "ketone")
        has_o = any(a.GetAtomicNum() == 8 for a in mol.GetAtoms())
        return (n_carb, has_o, heavy(mol))
        
    valid_frags.sort(key=sort_key, reverse=True)
    
    best = None
    for f in valid_frags:
        if any(a.GetAtomicNum() == 8 for a in f.GetAtoms()):
            best = Chem.Mol(f)
            break
            
    if best is None:
        return None, info

    # 安全中性化
    uncharger = Uncharger()
    try:
        best = uncharger.uncharge(best)
        Chem.SanitizeMol(best)
        best.UpdatePropertyCache(strict=True)
    except Exception:
        return None, info
    return best, info


def _longest_carbon_chain(mol):
    """最大脂肪烃链长（C-C 骨架直径，BFS 两次启发式，多项式时间）。

    最长简单路径在一般图上是 NP-hard；这里用“从任意碳出发 BFS 找最远，
    再从最远点 BFS 找直径”的启发式：在树上精确，在稠环图上给出良好下界，
    完全满足“估碳链长度”的用途，且对类固醇等多环分子不会指数爆炸。
    """
    n = mol.GetNumAtoms()
    adj = [[] for _ in range(n)]
    carbons = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 6:
            carbons.append(a.GetIdx())
    if not carbons:
        return 0
    for b in mol.GetBonds():
        c, d = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if mol.GetAtomWithIdx(c).GetAtomicNum() == 6 and mol.GetAtomWithIdx(d).GetAtomicNum() == 6:
            adj[c].append(d)
            adj[d].append(c)

    def bfs(src):
        dist = [-1] * n
        dist[src] = 0
        q = [src]
        far = src
        for u in q:
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
                    if dist[v] > dist[far]:
                        far = v
        return far, dist[far]

    a, _ = bfs(carbons[0])
    b_far, d = bfs(a)
    return d + 1  # 节点数 = 边数 + 1


def _is_stereo(mol):
    return (any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())
            or any(b.GetBondType() == Chem.BondType.DOUBLE and
                   b.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
                   for b in mol.GetBonds()))


# ----------------------------------------------------------------------------
# 单分子过滤
# ----------------------------------------------------------------------------
def analyze_one(smiles):
    r = {"smiles": smiles, "exclude": None, "mrf_orig": [], "mrf_neutral": [],
         "carbonyl_type": "unknown", "n_reactive_carbonyl": 0,
         "mw": None, "logp": None, "tpsa": None,
         "carbon_chain_length": None, "n_ester": 0, "n_ether": 0, "n_double": 0,
         "stereo": True}

    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        r["exclude"] = "parse_fail"
        return r

    orig = rdmol
    # 1) MRF 硬排除
    mrf_orig = [k for k in ("nitro", "azide", "peroxide", "acyl_halide", "anhydride",
                            "sulfonyl_halide", "quaternary_N", "epoxide", "aziridine",
                            "tert_butyl_carbamate", "benzyl_carbamate", "fmoc", "tosyl",
                            "halogen", "aldol_OH")
                if _has(orig, k)]
    r["mrf_orig"] = mrf_orig
    if mrf_orig:
        r["exclude"] = "+".join(mrf_orig)
        return r

    # 2) 去盐 + 中性化
    neutral, sinfo = _strip_salts_neutralize(orig)
    if neutral is None:
        r["exclude"] = "inorganic_or_salt_only" if sinfo["inorganic"] else "no_organic_frag"
        return r
    if sinfo["inorganic"]:
        r["exclude"] = "contains_metal"
        return r

    # 2b) 残留校验
    if any(a.GetIsotope() != 0 for a in neutral.GetAtoms()):
        r["exclude"] = "isotope_atom"
        return r
    if any(a.GetFormalCharge() != 0 for a in neutral.GetAtoms()):
        r["exclude"] = "residual_charge"
        return r

    # 3) 中性化后干扰基团
    mrf_neutral = [k for k in ("primary_amine", "secondary_amine", "carboxylic_acid",
                               "isocyanide", "isocyanate", "isothiocyanate",
                               "nitrile", "imine", "oxime",
                               "sulfonic_acid", "phosphonic_acid")
                   if _has(neutral, k)]
    r["mrf_neutral"] = mrf_neutral
    if mrf_neutral:
        r["exclude"] = "+".join(mrf_neutral)
        return r

    # 3b) PAINS 过滤
    if _get_pains().HasMatch(neutral):
        r["exclude"] = "PAINS"
        return r

    # 3c) 迈克尔受体（α,β-不饱和醛/酮/酯等）→ 归 flagged（可后续氢化还原，仍保留）
    if _has(neutral, "michael_acceptor"):
        r["exclude"] = "michael_acceptor"
        return r

    # 3d) 缩醛/缩酮（被保护羰基）→ 归 flagged（除非明确去保护）
    if _has(neutral, "acetal") or _has(neutral, "thioacetal"):
        r["exclude"] = "acetal_protected"
        return r

    # 4) 羰基计数与类型判定
    n_carb = _cnt(neutral, "aldehyde") + _cnt(neutral, "ketone")
    r["n_reactive_carbonyl"] = n_carb
    if n_carb == 0:
        r["exclude"] = "no_carbonyl"
        return r
    # 多个反应性羰基 → flagged（可后续逐位点拆分）
    if n_carb > 1:
        r["exclude"] = "multi_carbonyl"
        return r

    is_ald = _cnt(neutral, "aldehyde") == 1
    is_ket = _cnt(neutral, "ketone") == 1
    if is_ald and not is_ket:
        r["carbonyl_type"] = "aldehyde"
    elif is_ket and not is_ald:
        r["carbonyl_type"] = "ketone"
    else:
        r["exclude"] = "ambiguous_carbonyl"
        return r

    # 5) 属性计算
    r["mw"] = Descriptors.MolWt(neutral)
    r["logp"] = Crippen.MolLogP(neutral)
    r["tpsa"] = Descriptors.TPSA(neutral)
    r["carbon_chain_length"] = _longest_carbon_chain(neutral)
    r["n_ester"] = _cnt(neutral, "acyclic_ester")
    r["n_double"] = sum(1 for b in neutral.GetBonds()
                        if b.GetBondType() == Chem.BondType.DOUBLE)
    r["n_ether"] = sum(1 for a in neutral.GetAtoms()
                       if a.GetAtomicNum() == 8 and a.GetTotalNumHs() == 0
                       and a.GetDegree() == 2 and a.GetHybridization() == Chem.HybridizationType.SP3)
    r["stereo"] = _is_stereo(neutral)

    # 6) 主库窗口（醛/酮 tail 需形成疏水尾；决策：chain_length ≥ 8）
    if not (60.0 <= r["mw"] <= 600.0):
        r["exclude"] = "mw_out_of_window"
        return r
    if r["carbon_chain_length"] is None or r["carbon_chain_length"] < 8:
        r["exclude"] = "chain_length_lt_8"
        return r
    return r


# ----------------------------------------------------------------------------
# Stage2 并行精筛
# ----------------------------------------------------------------------------
def _build_worker(arg):
    candidate_path, slice_dir = arg
    # 候选可能仍在由 stage1 原子写（.tmp→rename）；读取前校验存在与可读，避免竞态 FileNotFoundError
    if not os.path.exists(candidate_path):
        return {"file": os.path.basename(candidate_path), "main": -1, "flagged": -1,
                "excluded": -1, "seconds": 0.0, "skipped": "missing_candidate"}
    base = os.path.basename(candidate_path).replace(".tsv.gz", "")
    outs = {k: os.path.join(slice_dir, f"{base}.{k}.tsv.gz") for k in ("main", "flagged", "excluded")}
    t0 = time.time()
    counts = Counter()
    header = "\t".join(["cid", "smiles", "inchikey", "carbonyl_type", "mw", "logp", "tpsa",
                        "carbon_chain_length", "n_ester", "n_ether", "n_double",
                        "n_reactive_carbonyl", "stereo_flag", "exclude_reason"])
    with gzip.open(candidate_path, "rt") as fin, \
         gzip.open(outs["main"] + ".tmp", "wt") as fm, \
         gzip.open(outs["flagged"] + ".tmp", "wt") as ff, \
         gzip.open(outs["excluded"] + ".tmp", "wt") as fe:
        for fh in (fm, ff, fe):
            fh.write(header + "\n")
        next(fin)  # header of candidate
        for ln in fin:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            cid, smi, ich = parts[0], parts[1], parts[2]
            r = analyze_one(smi)
            row = "\t".join([
                cid, r["smiles"], ich, r["carbonyl_type"],
                "" if r["mw"] is None else f"{r['mw']:.2f}",
                "" if r["logp"] is None else f"{r['logp']:.2f}",
                "" if r["tpsa"] is None else f"{r['tpsa']:.1f}",
                "" if r["carbon_chain_length"] is None else str(r["carbon_chain_length"]),
                str(r["n_ester"]), str(r["n_ether"]), str(r["n_double"]),
                str(r["n_reactive_carbonyl"]), "Y" if r["stereo"] else "N",
                r["exclude"] or ""])
            if r["exclude"] is None:
                fm.write(row + "\n"); counts["main"] += 1
            elif r["exclude"] in ("multi_carbonyl", "acetal_protected", "michael_acceptor"):
                ff.write(row + "\n"); counts["flagged"] += 1
            else:
                fe.write(row + "\n"); counts["excluded"] += 1
    for p in outs.values():
        os.replace(p + ".tmp", p)
    return {"file": base, **dict(counts), "seconds": round(time.time() - t0, 1)}


def _cmd_build(args):
    cand = Path(args.candidates_dir)
    input_files = []
    for fl in args.filelist or []:
        with open(fl) as f:
            input_files += [ln.strip() for ln in f if ln.strip()]
    if input_files:
        files = []
        for p in input_files:
            full = Path(p) if p.startswith("/") else (cand / p)
            # 兼容相对路径里已含 data/... 前缀的情况
            if full.exists():
                files.append(full)
            else:
                print(f"[warn] filelist 中缺失: {p}", file=sys.stderr)
        # 若全是相对且都找不到，退回解析绝对
        if not files and not any(p.startswith("/") for p in input_files):
            files = [Path(p) for p in input_files if Path(p).exists()]
    else:
        files = sorted(cand.glob("*.tsv.gz"))
    if not files:
        print("[error] 候选文件为空", file=sys.stderr); sys.exit(1)
    os.makedirs(args.slices_dir, exist_ok=True)
    print(f"[info] 候选文件 {len(files)}，线程 {args.threads}", flush=True)
    summary = []
    with ProcessPoolExecutor(max_workers=args.threads) as ex:
        futs = {ex.submit(_build_worker, (str(f), args.slices_dir)): f for f in files}
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
# Stage3 合并
# ----------------------------------------------------------------------------
def _cmd_merge(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    slices_dir = Path(args.slices_dir)
    if not list(slices_dir.glob("*.main.tsv.gz")):
        print("[error] 无分片结果", file=sys.stderr); sys.exit(1)

    def load_dedup(suffix):
        rows, seen, dup = [], set(), 0
        for s in sorted(slices_dir.glob(f"*.{suffix}.tsv.gz")):
            with gzip.open(s, "rt") as f:
                next(f)
                for ln in f:
                    cols = ln.rstrip("\n").split("\t")
                    if len(cols) < 3:
                        continue
                    # 优化：统一使用 InChIKey 去重
                    key = cols[2] if cols[2] else cols[1]
                    if key in seen:
                        dup += 1; continue
                    seen.add(key); rows.append(cols)
        if suffix == "main":
            rows.sort(key=lambda r: float(r[4]) if r[4] else 1e9)
        return rows, dup

    main_rows, main_dup = load_dedup("main")
    flag_rows, flag_dup = load_dedup("flagged")
    excl_rows, excl_dup = load_dedup("excluded")

    header = ["cid", "smiles", "inchikey", "carbonyl_type", "mw", "logp", "tpsa",
              "carbon_chain_length", "n_ester", "n_ether", "n_double",
              "n_reactive_carbonyl", "stereo_flag", "exclude_reason"]

    def assign(rows):
        for i, r in enumerate(rows, 1):
            yield [f"CARB_{i:07d}"] + r

    with open(out / "carbonyl_ENA_main.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["carbonyl_id"] + header); w.writerows(assign(main_rows))
    with open(out / "carbonyl_ENA_flagged.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["carbonyl_id"] + header); w.writerows(assign(flag_rows))
    with open(out / "carbonyl_ENA_excluded.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(excl_rows)

    with open(out / "carbonyl_ENA_main.smi", "w") as f:
        for i, r in enumerate(main_rows, 1):
            f.write(f"{r[1]}\tCARB_{i:07d}\n")
    with open(out / "carbonyl_ENA_flagged.smi", "w") as f:
        f.writelines(f"{r[1]}\t{r[3]}\n" for r in flag_rows)
    with open(out / "carbonyl_ENA_excluded.smi", "w") as f:
        f.writelines(f"{r[1]}\t{r[13] or ''}\n" for r in excl_rows)

    reason_counter = Counter()
    for r in excl_rows:
        reason_counter[r[13] or "unknown"] += 1
    summ = {
        "version": "carbonyl_ENA",
        "n_main_after_dedup": len(main_rows),
        "n_flagged_after_dedup": len(flag_rows),
        "n_excluded_after_dedup": len(excl_rows),
        "carbonyl_type_dist": Counter(r[3] for r in main_rows).most_common(),
        "exclude_reason_top": reason_counter.most_common(20),
        "mw_range": [round(min(float(r[4]) for r in main_rows), 2),
                     round(max(float(r[4]) for r in main_rows), 2)],
        "n_stereo": sum(1 for r in main_rows if r[12] == "Y"),
    }
    json.dump(summ, open(out / "carbonyl_ENA_summary.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(summ, ensure_ascii=False, indent=2))


# ----------------------------------------------------------------------------
# QC
# ----------------------------------------------------------------------------
def _cmd_qc(args):
    out = Path(args.out_dir)
    import pandas as pd
    df = pd.read_csv(out / "carbonyl_ENA_main.csv")
    n = len(df)
    with open(out / "carbonyl_ENA_qc_report.md", "w") as f:
        f.write("# Carbonyl/Tail v0.1 QC 报告\n\n")
        f.write(f"- 主库分子数：**{n:,}**\n\n")
        if n:
            f.write("## 代表性分子（含酯键/不饱和/不同链长抽样）\n\n")
            sub = df.sample(min(20, n), random_state=7)
            f.write("| carbonyl_id | SMILES | MW | LogP | chain | type |\n|---|---|---|---|---|---|\n")
            for _, r in sub.iterrows():
                f.write(f"| {r.carbonyl_id} | `{r.smiles[:58]}` | {r.mw:.1f} | {r.logp:.1f} | "
                        f"{r.carbon_chain_length} | {r.carbonyl_type} |\n")
            f.write("\n## 碳链长度分布（主库）\n\n")
            f.write(f"min={df.carbon_chain_length.min()}, median={df.carbon_chain_length.median():.0f}, "
                    f"max={df.carbon_chain_length.max()}\n\n")
            f.write("## 羰基类型\n\n")
            for t, c in df.carbonyl_type.value_counts().items():
                f.write(f"- {t}: {c:,}\n")
    print(f"[ok] QC 报告已写入 {out}/carbonyl_ENA_qc_report.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_b = sub.add_parser("build")
    p_b.add_argument("--candidates-dir", default="data/carbonyl_ENA/candidates")
    p_b.add_argument("--slices-dir", default="data/carbonyl_ENA/slices")
    p_b.add_argument("--threads", type=int, default=24)
    p_b.add_argument("--filelist", action="append", default=[])
    p_b.set_defaults(fn=_cmd_build)
    p_m = sub.add_parser("merge")
    p_m.add_argument("--slices-dir", default="data/carbonyl_ENA/slices")
    p_m.add_argument("--out-dir", default="data/carbonyl_ENA/output")
    p_m.set_defaults(fn=_cmd_merge)
    p_q = sub.add_parser("qc")
    p_q.add_argument("--out-dir", default="data/carbonyl_ENA/output")
    p_q.set_defaults(fn=_cmd_qc)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()