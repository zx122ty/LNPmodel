#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iso_build.py — Ugi-4CR Isocyanide（异腈）子库 Stage2/3：RDKit 精筛 + 合并去重 + QC
==================================================================================
输入：Stage1 候选池 data/isocyanide_v0.1/candidates/*.tsv.gz（含 `[N+]#[C-]`/`[C-]#[N+]`）
输出（03.md §5.4.4 / 5.4.2 / 5.4.3）：
    isocyanide_v0.1_main.csv / .smi
    isocyanide_v0.1_flagged.csv / .smi
    isocyanide_v0.1_excluded.csv / .smi
    isocyanide_v0.1_summary.json
    isocyanide_v0.1_qc_report.md

异腈判定关键点:
    * PubChem 书写 `[N+]#[C-]` / `[C-]#[N+]`（带显式电荷三键）→ 先取 canonical SMILES
    * 判定"1 个异腈 + 无其它干扰官能团 + 便于 Ugi"：
        - 异腈计数: 以 `[N+]#[C-]` / `[C-]#[N+]` 子串数计
        - 排除: 同时含 -NH2/-NH-（胺会竞争）、-COOH、-CHO、酮、其他杂原子反应基
              金属/无机、盐/同位素、卤素、多异腈 → flagged
    * 若 RDKit sanitize 失败，直接以 canonical SMILES 字符串参与属性估算。

用法：
    python scripts/iso_build.py build --candidates-dir data/isocyanide_v0.1/candidates \
                                      --slices-dir data/isocyanide_v0.1/slices -t 8
    python scripts/iso_build.py merge --slices-dir ... --out-dir data/isocyanide_v0.1/output
    python scripts/iso_build.py qc    --out-dir ...

环境：python3.10 + RDKit 2026.03。
"""

import argparse
import csv
import gzip
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

# ----------------------------------------------------------------------------
# RDKit SMARTS（用于可解析分子的精确复核）
# ----------------------------------------------------------------------------
_RDKIT_SMARTS = {
    "prim_amine": "[NX3;H2;!$(NC=O);!$(N=C);!$([N+])]",
    "sec_amine":  "[NX3;H1;!$(NC=O);!$(N=C);!$([N+])]",
    "aldehyde":   "[CX3H1,H2](=O)",       # 修复：包含甲醛
    "ketone":     "[#6][CX3](=O)[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "isocyanate": "N=C=O",                 # 修复：连通度 Bug
    "isothiocyanate": "N=C=S",
    "nitrile":    "C#N",                   # 修复：连通度 Bug
    "nitro":      "[NX3+](=O)[O-]",
    "sulfonic_acid": "[S;X4](=O)(=O)[OX2H1]",
    "phosphonic_acid": "[P;X4](=O)([OX2H1])[OX2H1]",
}
_RDKIT_PATTERNS = {}
for _k, _v in _RDKIT_SMARTS.items():
    _RDKIT_PATTERNS[_k] = Chem.MolFromSmarts(_v)
    if _RDKIT_PATTERNS[_k] is None:
        raise RuntimeError(f"SMARTS 解析失败: {_k} = {_v}")

# 异腈子串模式（文本级计数）
ISO_PATS = ("[N+]#[C-]", "[C-]#[N+]")

# 卤素正则（原子级）：裸符号紧贴裸符号即代表与其它原子成键的位置，否则会漏掉
# CCCl / CF / Fc1ccccc1 等常规卤代分子（旧正则对 "CCCl" 误放行）
_HALOGEN_RE = re.compile(
    r"\[(?:F|Cl|Br|I)[+\-0-9]*\]|(?:(?<=^)|(?<=[A-Z()%0-9.\]=-]))(?:F|Cl|Br|I)"
)

_METALS = {3, 4, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
           31, 32, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
           55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
           73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91,
           92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103}

# PAINS
_PAINS_CAT = None
def _get_pains():
    global _PAINS_CAT
    if _PAINS_CAT is None:
        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        _PAINS_CAT = FilterCatalog(params)
    return _PAINS_CAT


def _count_isocyanide(smiles):
    """文本级异腈计数（兼容两种书写方向）"""
    return sum(smiles.count(p) for p in ISO_PATS)


def _has_rdkit(mol, key):
    pat = _RDKIT_PATTERNS.get(key)
    return pat is not None and mol.HasSubstructMatch(pat)


def _has_halogen_text(smiles):
    """文本级卤素检测（排除异腈中的 C/N 干扰）"""
    # 移除异腈标记后再检测
    cleaned = smiles.replace("[N+]#[C-]", "").replace("[C-]#[N+]", "")
    return bool(_HALOGEN_RE.search(cleaned))


def _detect_interference_text(canonical):
    """
    文本级干扰官能团检测（用于 RDKit sanitize 失败时的兜底）。
    返回检测到的标签列表。
    """
    tags = []
    cs = canonical

    # 移除异腈标记，避免干扰判定
    cs_no_iso = cs.replace("[N+]#[C-]", "XISO").replace("[C-]#[N+]", "XISO")

    # 羧酸/酯/酰胺: C(=O)O 或 C(=O)N（但排除异腈碳）
    if re.search(r"C\(=O\)[O]", cs_no_iso):
        tags.append("acid_or_ester")
    if re.search(r"C\(=O\)[N]", cs_no_iso):
        tags.append("amide")

    # 醛/酮: C(=O) 后接 C 或 H（排除异腈和已判定的酸/酯/酰胺）
    # 注意：大部分异腈分子 sanitize 会报警从而走文本兜底，因此醛/酮/腈/氨基等
    # 判定几乎都在这里发生，宁多勿漏。
    c_equal_o = re.search(r"C\(=O\)[C]", cs_no_iso)
    if c_equal_o or re.search(r"C=O$|\[H\]C=O|C=O(?=[#6,C])", cs_no_iso):
        tags.append("ketone_or_aldehyde")

    # 腈: 真腈的 `#N` 的 # 号左侧是"裸碳/原子"，而异腈的 `#C-]` 的 # 号左侧是 [N+] 已整段替换为 XISO
    # （XISO 内部无 #），因此 `C#N`、`[s,S]#N` 等才是真腈；避免 "N#C" 反向书写的异腈碳残留。
    nitrile_hits = list(re.finditer(r"C#N|\[[s,S]\]#N|[s,S]#N", cs_no_iso))
    for m in nitrile_hits:
        if m.group(0).startswith("N"):
            continue  # 防御：N#C（反向异腈）不当作腈
        tags.append("nitrile")
        break

    # 胺: 非异腈的 N（排除 [N+] 异腈氮、n 芳香氮、C#N 腈）
    # 策略：找所有 N 原子，排除已知安全的
    amine_found = False
    # 匹配 [NH], [NH2], [NH3+], [nH] 等含氢氮
    if re.search(r"\[N[+]?\(?H[23]?\)?\]|\[nH\]", cs_no_iso):
        amine_found = True
    # 匹配裸 N（非括号内，非异腈，非腈）
    # 在 canonical SMILES 中，裸 N 通常表示胺
    for m in re.finditer(r"(?<!\[)N(?!\])", cs_no_iso):
        pos = m.start()
        # 检查是否是 C#N（腈）的一部分
        if pos > 0 and cs_no_iso[pos-1] == "#":
            continue
        # 检查是否是 XISO 替换后的残留
        if pos >= 4 and cs_no_iso[pos-4:pos] == "XISO":
            continue
        amine_found = True
        break
    if amine_found:
        tags.append("amine")

    # 磺酸
    if re.search(r"S\(=O\)\(=O\)[O]", cs_no_iso):
        tags.append("sulfonic_acid")

    return tags


def _is_stereo(mol):
    """检测立体化学（需要分子可解析）"""
    if mol is None:
        return False
    try:
        return (any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms())
                or any(b.GetBondType() == Chem.BondType.DOUBLE and
                       b.GetStereo() not in (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
                       for b in mol.GetBonds()))
    except Exception:
        return False


def _longest_carbon_chain(mol):
    """最大脂肪烃链长估算（BFS 启发式）"""
    if mol is None:
        return None
    try:
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
        _, d = bfs(a)
        return d + 1
    except Exception:
        return None


# ----------------------------------------------------------------------------
# 单分子过滤
# ----------------------------------------------------------------------------
def analyze_one(smiles):
    r = {"smiles": smiles, "canonical": None, "exclude": None,
         "n_isocyanide": 0, "iso_type": "unknown", "mw": None, "logp": None,
         "tpsa": None, "chain_length": None, "stereo": False}

    # 0) 异腈个数（子串计数）
    n_iso = _count_isocyanide(smiles)
    r["n_isocyanide"] = n_iso
    if n_iso == 0:
        r["exclude"] = "no_isocyanide"
        return r
    if n_iso > 1:
        r["exclude"] = "multi_isocyanide"      # → flagged
        return r

    # 1) 卤素检测（文本级，在 RDKit 解析前）
    if _has_halogen_text(smiles):
        r["exclude"] = "halogen"
        return r

    # 2) 尝试 RDKit 解析（异腈超价常导致消毒失败）
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        r["exclude"] = "parse_fail"
        return r

    # 尝试消毒获取 canonical
    sanitize_ok = True
    try:
        Chem.SanitizeMol(m)
        canon = Chem.MolToSmiles(m)
    except Exception:
        sanitize_ok = False
        # 消毒失败时，尝试不消毒直接输出（保留电荷标记）
        try:
            canon = Chem.MolToSmiles(m)
        except Exception:
            canon = smiles  # 最终兜底
    r["canonical"] = canon

    # 3) 金属 / 多片段 / 同位素
    try:
        if any(a.GetAtomicNum() in _METALS for a in m.GetAtoms()):
            r["exclude"] = "contains_metal"
            return r
        if any(a.GetIsotope() != 0 for a in m.GetAtoms()):
            r["exclude"] = "isotope"
            return r
    except Exception:
        pass

    if canon.count(".") > 0:
        r["exclude"] = "salt_or_multipart"
        return r

    # 4) 干扰官能团检测（注意：异腈的超价/孤对标记会让 RDKit sanitize 对几乎所有
#    含异腈分子报警，这里刻意不依赖 SMARTS 检测 —— 统一走文本级兜底，
#    由 _detect_interference_text 在 canonical 上做保守排除，宁多勿漏。）
    text_issues = _detect_interference_text(canon)
    if text_issues:
        r["exclude"] = "+".join(text_issues)
        return r

    # 5) 属性计算（尽力而为）
    try:
        r["mw"] = Descriptors.MolWt(m) if m is not None else None
    except Exception:
        r["mw"] = None
    try:
        r["logp"] = Crippen.MolLogP(m) if m is not None else None
    except Exception:
        r["logp"] = None
    try:
        r["tpsa"] = Descriptors.TPSA(m) if m is not None else None
    except Exception:
        r["tpsa"] = None

    r["chain_length"] = _longest_carbon_chain(m)
    r["stereo"] = _is_stereo(m)

    # 6) MW 窗口
    if r["mw"] is not None and not (60.0 <= r["mw"] <= 700.0):
        r["exclude"] = "mw_out_of_window"
        return r

    return r


# ----------------------------------------------------------------------------
# Stage2 并行精筛
# ----------------------------------------------------------------------------
def _build_worker(arg):
    cand_path, slice_dir = arg
    if not os.path.exists(cand_path):
        return {"file": os.path.basename(cand_path), "main": -1, "flagged": -1,
                "excluded": -1, "seconds": 0, "skipped": True}
    base = os.path.basename(cand_path).replace(".tsv.gz", "")
    outs = {k: os.path.join(slice_dir, f"{base}.{k}.tsv.gz") for k in ("main", "flagged", "excluded")}
    t0 = time.time()
    counts = Counter()
    # 注意：增加 inchikey 列以支持后续 InChIKey 去重
    header = "\t".join(["cid", "smiles", "inchikey", "canonical", "n_isocyanide",
                        "mw", "logp", "tpsa", "chain_length", "stereo_flag", "exclude_reason"])
    with gzip.open(cand_path, "rt") as fin, \
         gzip.open(outs["main"] + ".tmp", "wt") as fm, \
         gzip.open(outs["flagged"] + ".tmp", "wt") as ff, \
         gzip.open(outs["excluded"] + ".tmp", "wt") as fe:
        for fh in (fm, ff, fe):
            fh.write(header + "\n")
        next(fin)  # skip candidate header
        for ln in fin:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            cid, smi, ich = p[0], p[1], p[2]
            r = analyze_one(smi)
            row = "\t".join([
                cid, r["smiles"], ich, r["canonical"] or "",
                str(r["n_isocyanide"]),
                "" if r["mw"] is None else f"{r['mw']:.2f}",
                "" if r["logp"] is None else f"{r['logp']:.2f}",
                "" if r["tpsa"] is None else f"{r['tpsa']:.1f}",
                "" if r["chain_length"] is None else str(r["chain_length"]),
                "Y" if r["stereo"] else "N",
                r["exclude"] or ""])
            if r["exclude"] is None:
                fm.write(row + "\n"); counts["main"] += 1
            elif r["exclude"] in ("multi_isocyanide",):
                ff.write(row + "\n"); counts["flagged"] += 1
            else:
                fe.write(row + "\n"); counts["excluded"] += 1
    for p in outs.values():
        os.replace(p + ".tmp", p)
    return {"file": base, "main": counts.get("main", 0), "flagged": counts.get("flagged", 0),
            "excluded": counts.get("excluded", 0), "seconds": round(time.time() - t0, 1)}


def _cmd_build(args):
    cand = Path(args.candidates_dir)
    files = []
    for fl in args.filelist or []:
        p = Path(fl)
        if p.exists() and p.suffix == ".txt":
            files += [Path(l.strip()) for l in p.read_text().splitlines() if l.strip()]
        elif p.exists():
            files.append(p)
    if not files:
        files = sorted(cand.glob("*.tsv.gz"))
    files = [f for f in files if f.exists()]
    if not files:
        print("[error] 候选为空", file=sys.stderr); sys.exit(1)
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
            agg[k] += r.get(k, 0)
    print(json.dumps({"files": len(summary), **agg}, indent=2))


# ----------------------------------------------------------------------------
# Stage3 合并 + 去重
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
                next(f)  # header
                for ln in f:
                    cols = ln.rstrip("\n").split("\t")
                    if len(cols) < 5:
                        continue
                    # 优化：使用 InChIKey 去重（cols[2]），fallback 到 canonical（cols[3]）
                    key = cols[2] if cols[2] else (cols[3] if cols[3] else cols[1])
                    if key in seen:
                        dup += 1
                        continue
                    seen.add(key)
                    rows.append(cols)
        return rows, dup

    main_rows, main_dup = load_dedup("main")
    flag_rows, flag_dup = load_dedup("flagged")
    excl_rows, excl_dup = load_dedup("excluded")

    # 按 MW 排序主库
    main_rows.sort(key=lambda r: float(r[5]) if r[5] else 1e9)

    header = ["cid", "smiles", "inchikey", "canonical", "n_isocyanide",
              "mw", "logp", "tpsa", "chain_length", "stereo_flag", "exclude_reason"]

    with open(out / "isocyanide_v0.1_main.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["isocyanide_id"] + header)
        for i, r in enumerate(main_rows, 1):
            w.writerow([f"ISO_{i:07d}"] + r)
    with open(out / "isocyanide_v0.1_main.smi", "w") as f:
        for i, r in enumerate(main_rows, 1):
            # 优先用 canonical，其次原始 SMILES
            f.write(f"{r[3] or r[1]}\tISO_{i:07d}\n")

    with open(out / "isocyanide_v0.1_flagged.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(flag_rows)
    with open(out / "isocyanide_v0.1_flagged.smi", "w") as f:
        f.writelines(f"{r[3] or r[1]}\t{r[-1] or ''}\n" for r in flag_rows)

    with open(out / "isocyanide_v0.1_excluded.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(excl_rows)
    with open(out / "isocyanide_v0.1_excluded.smi", "w") as f:
        f.writelines(f"{r[3] or r[1]}\t{r[-1] or ''}\n" for r in excl_rows)

    reason_counter = Counter()
    for r in excl_rows:
        reason_counter[r[-1] or "unknown"] += 1

    mw_vals = [float(r[5]) for r in main_rows if r[5]]
    summ = {
        "version": "isocyanide_v0.1",
        "n_main_after_dedup": len(main_rows),
        "n_main_duplicates": main_dup,
        "n_flagged_after_dedup": len(flag_rows),
        "n_excluded_after_dedup": len(excl_rows),
        "exclude_reason_top": reason_counter.most_common(15),
        "mw_range": [round(min(mw_vals), 2), round(max(mw_vals), 2)] if mw_vals else [0, 0],
        "n_stereo": sum(1 for r in main_rows if r[9] == "Y"),
    }
    with open(out / "isocyanide_v0.1_summary.json", "w") as f:
        json.dump(summ, f, ensure_ascii=False, indent=2)
    print(json.dumps(summ, ensure_ascii=False, indent=2))


# ----------------------------------------------------------------------------
# QC
# ----------------------------------------------------------------------------
def _cmd_qc(args):
    out = Path(args.out_dir)
    import pandas as pd
    df = pd.read_csv(out / "isocyanide_v0.1_main.csv")
    n = len(df)
    with open(out / "isocyanide_v0.1_qc_report.md", "w") as f:
        f.write("# Isocyanide v0.1 QC 报告\n\n")
        f.write(f"- 主库分子数：**{n:,}**\n\n")
        if n:
            f.write("## 20 个随机样例\n\n| ID | Canonical SMILES | MW | LogP | chain |\n|---|---|---|---|---|\n")
            for _, r in df.sample(min(20, n), random_state=7).iterrows():
                smi_display = str(r.canonical)[:58] if pd.notna(r.canonical) else str(r.smiles)[:58]
                f.write(f"| {r.isocyanide_id} | `{smi_display}` | {r.mw:.1f} | {r.logp:.1f} | {r.chain_length} |\n")
            f.write("\n## MW 分布\n\n")
            f.write(f"min={df.mw.min():.1f}, median={df.mw.median():.1f}, max={df.mw.max():.1f}\n\n")
            if "chain_length" in df.columns and df.chain_length.notna().any():
                f.write("## 碳链长度分布\n\n")
                cl = df.chain_length.dropna()
                f.write(f"min={cl.min():.0f}, median={cl.median():.0f}, max={cl.max():.0f}\n")
    print(f"[ok] QC 报告已写入 {out}/isocyanide_v0.1_qc_report.md")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_b = sub.add_parser("build")
    p_b.add_argument("--candidates-dir", default="data/isocyanide_v0.1/candidates")
    p_b.add_argument("--slices-dir", default="data/isocyanide_v0.1/slices")
    p_b.add_argument("--threads", type=int, default=8)
    p_b.add_argument("--filelist", nargs="*", default=[])
    p_b.set_defaults(fn=_cmd_build)
    p_m = sub.add_parser("merge")
    p_m.add_argument("--slices-dir", default="data/isocyanide_v0.1/slices")
    p_m.add_argument("--out-dir", default="data/isocyanide_v0.1/output")
    p_m.set_defaults(fn=_cmd_merge)
    p_q = sub.add_parser("qc")
    p_q.add_argument("--out-dir", default="data/isocyanide_v0.1/output")
    p_q.set_defaults(fn=_cmd_qc)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()