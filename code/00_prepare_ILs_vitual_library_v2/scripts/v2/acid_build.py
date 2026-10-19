#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acid_build.py — Ugi-4CR Acid/Linker（羧酸）子库 Stage2/3：RDKit 精筛 + 合并去重 + QC
====================================================================================
输入：Stage1 候选池 data/acid_v0.1/candidates/*.tsv.gz（=O 且无 N/n 的分子）
输出（03.md §5.3.4 / 5.3.2 / 5.3.3）：
    acid_v0.1_main.csv / .smi         （主库）
    acid_v0.1_flagged.csv / .smi      （近合格：多酸/特殊酸）
    acid_v0.1_excluded.csv / .smi     （排除，含原因）
    acid_v0.1_summary.json
    acid_v0.1_qc_report.md

Acid/Linker 定义（Ugi-4CR 羧酸组分，03.md §5.3）：
    * 含 1 个游离羧酸 -COOH / 羧酸根 -COO⁻（中性化后判定）
    * 主要用于连接基/可降解位点/短尾
    * 排除：其它 Ugi 组分（胺/醛酮/异腈）、多酸、酸酐/酰氯/活性酯、迈克尔受体、
           高反应性/毒性片段、金属/无机盐、α-碳活泼（丙二酸二酯类等易副反应）
    * 排除：卤素（F, Cl, Br, I）、磺酸/膦酸（强酸）、PAINS 毒性结构
    * 去盐、强制中性、物理性质计算、按 InChIKey 去重

用法：
    python scripts/acid_build.py build --candidates-dir data/acid_v0.1/candidates \
                                       --slices-dir data/acid_v0.1/slices -t 24 \
                                       --filelist data/acid_v0.1/.acid_fl.txt
    python scripts/acid_build.py merge --slices-dir data/acid_v0.1/slices --out-dir data/acid_v0.1/output
    python scripts/acid_build.py qc    --out-dir data/acid_v0.1/output

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
    # --- 反应性：羧酸 (游离酸 + 羧酸根) ---
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "carboxylate":     "[CX3](=O)[OX1-]",
    # --- 排除：其它 Ugi 组分 ---
    "primary_amine":   "[NX3;H2;!$(NC=O);!$(N=C);!$(N-[OX2]);!$(N-[NX3]);!$([N+])]",
    "secondary_amine": "[NX3;H1;!$(NC=O);!$(N=C);!$(N-[OX2]);!$(N-[NX3]);!$([N+])]",
    "aldehyde":  "[CX3H1,H2](=O)", # 修复：包含甲醛
    "ketone":    "[#6][CX3](=O)[#6]",
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
    "malonate_like": "[CX3](=O)[CX4H2][CX3](=O)", # 丙二酸/丙二酸酯类（α-活泼氢）
    # --- 反应性 / 干扰 ---
    "anhydride": "[CX3](=O)-[OX2]-[CX3](=O)",
    "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
    "NHS_ester": "[CX3](=O)[OX2]N1C(=O)CCC1=O",
    "activated_ester_sulfur": "[CX3](=O)[SX2]",
    "acyclic_ester": "[CX3](=O)[OX2][#6]",
    "michael_acceptor": "[CX3]=[CX3][CX3](=O)[#6,#7,#8,#16]",
    "thioacid": "[CX3](=O)[SX2H1]",
    "peracid":  "[CX3](=O)[OX2][OX2H1]",
    "epoxide":  "[OX2]1[#6][#6]1",
    "aziridine": "[NX3]1[#6][#6]1",
    "nitro":    "[NX3+](=O)[O-]",
    "azide":    "[N-]=[N+]=[N]",
    "peroxide": "[OX2;!$(O-C(=O))][OX2;!$(O-C(=O))]",
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
    """去盐（优先保留含羧酸片段）+ 强制中性（Uncharger）。"""
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

    # 羧酸优先：优先保留含羧酸/羧酸根的片段，其次看含氧，最后看重原子数
    def sort_key(mol):
        n_acid = _cnt(mol, "carboxylic_acid") + _cnt(mol, "carboxylate")
        has_o = any(a.GetAtomicNum() == 8 for a in mol.GetAtoms())
        return (n_acid, has_o, heavy(mol))
        
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
         "n_carboxylic_acid": 0, "n_ester": 0, "n_ether": 0, "n_disulfide": 0,
         "linker_type": "unknown", "mw": None, "logp": None, "tpsa": None, "stereo": True}

    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        r["exclude"] = "parse_fail"
        return r
    orig = rdmol

    # 1) MRF 硬排除
    mrf_orig = [k for k in ("nitro", "azide", "peroxide", "acyl_halide", "anhydride",
                            "sulfonyl_halide", "quaternary_N", "epoxide", "aziridine",
                            "tert_butyl_carbamate", "benzyl_carbamate", "fmoc", "tosyl",
                            "activated_ester_sulfur", "NHS_ester", "thioacid", "peracid",
                            "halogen", "malonate_like")
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
    mrf_neutral = [k for k in ("primary_amine", "secondary_amine", "aldehyde", "ketone",
                               "isocyanide", "isocyanate", "isothiocyanate", "nitrile",
                               "imine", "oxime", "sulfonic_acid", "phosphonic_acid")
                   if _has(neutral, k)]
    r["mrf_neutral"] = mrf_neutral
    if mrf_neutral:
        r["exclude"] = "+".join(mrf_neutral)
        return r

    # 3b) PAINS 过滤
    if _get_pains().HasMatch(neutral):
        r["exclude"] = "PAINS"
        return r

    # 3c) 迈克尔受体
    if _has(neutral, "michael_acceptor"):
        r["exclude"] = "michael_acceptor"
        return r

    # 4) 羧酸计数（Uncharger 处理后主要计数 carboxylic_acid）
    n_acid = _cnt(neutral, "carboxylic_acid") + _cnt(neutral, "carboxylate")
    r["n_carboxylic_acid"] = n_acid
    if n_acid == 0:
        r["exclude"] = "no_carboxylic_acid"
        return r
    if n_acid > 1:
        r["exclude"] = "multi_acid"     # → flagged
        return r

    # 5) 属性计算
    r["mw"] = Descriptors.MolWt(neutral)
    r["logp"] = Crippen.MolLogP(neutral)
    r["tpsa"] = Descriptors.TPSA(neutral)
    r["n_ester"] = _cnt(neutral, "acyclic_ester")
    r["n_ether"] = sum(1 for a in neutral.GetAtoms()
                       if a.GetAtomicNum() == 8 and a.GetTotalNumHs() == 0
                       and a.GetDegree() == 2 and a.GetHybridization() == Chem.HybridizationType.SP3)
    r["n_disulfide"] = len(neutral.GetSubstructMatches(Chem.MolFromSmarts("[SX2][SX2]")))
    r["stereo"] = _is_stereo(neutral)

    # linker 类型
    if r["n_ester"] >= 1:
        r["linker_type"] = "ester_linker"
    elif r["n_ether"] >= 1:
        r["linker_type"] = "ether_linker"
    else:
        r["linker_type"] = "alkyl/hydrocarbon"

    if not (60.0 <= r["mw"] <= 600.0):
        r["exclude"] = "mw_out_of_window"
        return r
    return r


# ----------------------------------------------------------------------------
# Stage2 并行
# ----------------------------------------------------------------------------
def _build_worker(arg):
    candidate_path, slice_dir = arg
    if not os.path.exists(candidate_path):
        return {"file": os.path.basename(candidate_path), "main": -1, "flagged": -1,
                "excluded": -1, "seconds": 0.0, "skipped": "missing_candidate"}
    base = os.path.basename(candidate_path).replace(".tsv.gz", "")
    outs = {k: os.path.join(slice_dir, f"{base}.{k}.tsv.gz") for k in ("main", "flagged", "excluded")}
    t0 = time.time()
    counts = Counter()
    header = "\t".join(["cid", "smiles", "inchikey", "n_carboxylic_acid", "mw", "logp", "tpsa",
                        "n_ester", "n_ether", "n_disulfide", "linker_type",
                        "stereo_flag", "exclude_reason"])
    with gzip.open(candidate_path, "rt") as fin, \
         gzip.open(outs["main"] + ".tmp", "wt") as fm, \
         gzip.open(outs["flagged"] + ".tmp", "wt") as ff, \
         gzip.open(outs["excluded"] + ".tmp", "wt") as fe:
        for fh in (fm, ff, fe):
            fh.write(header + "\n")
        next(fin)
        for ln in fin:
            parts = ln.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            cid, smi, ich = parts[0], parts[1], parts[2]
            r = analyze_one(smi)
            row = "\t".join([
                cid, r["smiles"], ich, str(r["n_carboxylic_acid"]),
                "" if r["mw"] is None else f"{r['mw']:.2f}",
                "" if r["logp"] is None else f"{r['logp']:.2f}",
                "" if r["tpsa"] is None else f"{r['tpsa']:.1f}",
                str(r["n_ester"]), str(r["n_ether"]), str(r["n_disulfide"]),
                r["linker_type"], "Y" if r["stereo"] else "N",
                r["exclude"] or ""])
            if r["exclude"] is None:
                fm.write(row + "\n"); counts["main"] += 1
            elif r["exclude"] in ("multi_acid",):
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
            if full.exists():
                files.append(full)
            else:
                head = cand / p
                if head.exists():
                    files.append(head)
                else:
                    print(f"[warn] filelist 中缺失: {p}", file=sys.stderr)
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

    header = ["cid", "smiles", "inchikey", "n_carboxylic_acid", "mw", "logp", "tpsa",
              "n_ester", "n_ether", "n_disulfide", "linker_type",
              "stereo_flag", "exclude_reason"]

    def assign(rows):
        for i, r in enumerate(rows, 1):
            yield [f"ACID_{i:07d}"] + r

    with open(out / "acid_v0.1_main.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["acid_id"] + header); w.writerows(assign(main_rows))
    with open(out / "acid_v0.1_flagged.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["acid_id"] + header); w.writerows(assign(flag_rows))
    with open(out / "acid_v0.1_excluded.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(excl_rows)

    with open(out / "acid_v0.1_main.smi", "w") as f:
        for i, r in enumerate(main_rows, 1):
            f.write(f"{r[1]}\tACID_{i:07d}\n")
    with open(out / "acid_v0.1_flagged.smi", "w") as f:
        f.writelines(f"{r[1]}\t{r[12] or ''}\n" for r in flag_rows)
    with open(out / "acid_v0.1_excluded.smi", "w") as f:
        f.writelines(f"{r[1]}\t{r[12] or ''}\n" for r in excl_rows)

    reason_counter = Counter()
    for r in excl_rows:
        reason_counter[r[12] or "unknown"] += 1
    summ = {
        "version": "acid_v0.1",
        "n_main_after_dedup": len(main_rows),
        "n_flagged_after_dedup": len(flag_rows),
        "n_excluded_after_dedup": len(excl_rows),
        "linker_type_dist": Counter(r[10] for r in main_rows).most_common(),
        "exclude_reason_top": reason_counter.most_common(20),
        "mw_range": [round(min(float(r[4]) for r in main_rows), 2),
                     round(max(float(r[4]) for r in main_rows), 2)],
        "n_stereo": sum(1 for r in main_rows if r[11] == "Y"),
    }
    json.dump(summ, open(out / "acid_v0.1_summary.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(summ, ensure_ascii=False, indent=2))


# ----------------------------------------------------------------------------
# QC
# ----------------------------------------------------------------------------
def _cmd_qc(args):
    out = Path(args.out_dir)
    import pandas as pd
    df = pd.read_csv(out / "acid_v0.1_main.csv")
    n = len(df)
    with open(out / "acid_v0.1_qc_report.md", "w") as f:
        f.write("# Acid/Linker v0.1 QC 报告\n\n")
        f.write(f"- 主库分子数：**{n:,}**\n\n")
        if n:
            f.write("## 代表性分子（按 linker 类型/酯/醚抽样）\n\n")
            sub = df.sample(min(20, n), random_state=7)
            f.write("| acid_id | SMILES | MW | LogP | linker_type | n_est |\n|---|---|---|---|---|---|\n")
            for _, r in sub.iterrows():
                f.write(f"| {r.acid_id} | `{r.smiles[:58]}` | {r.mw:.1f} | {r.logp:.1f} | "
                        f"{r.linker_type} | {r.n_ester} |\n")
            f.write("\n## linker 类型分布\n\n")
            for t, c in df.linker_type.value_counts().items():
                f.write(f"- {t}: {c:,}\n")
    print(f"[ok] QC 报告已写入 {out}/acid_v0.1_qc_report.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_b = sub.add_parser("build")
    p_b.add_argument("--candidates-dir", default="data/acid_v0.1/candidates")
    p_b.add_argument("--slices-dir", default="data/acid_v0.1/slices")
    p_b.add_argument("--threads", type=int, default=24)
    p_b.add_argument("--filelist", action="append", default=[])
    p_b.set_defaults(fn=_cmd_build)
    p_m = sub.add_parser("merge")
    p_m.add_argument("--slices-dir", default="data/acid_v0.1/slices")
    p_m.add_argument("--out-dir", default="data/acid_v0.1/output")
    p_m.set_defaults(fn=_cmd_merge)
    p_q = sub.add_parser("qc")
    p_q.add_argument("--out-dir", default="data/acid_v0.1/output")
    p_q.set_defaults(fn=_cmd_qc)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()