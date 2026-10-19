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

异性判定关键点:
    * PubChem 书写 `[N+]#[C-]` / `[C-]#[N+]`（带显式电荷三键）→ 先取 canonical SMILES
    * 判定"1 个异腈 + 无其它干扰官能团 + 便于 Ugi"：
        - 异腈计数: 以 `[N+]#[C-]` 子串数计（RDKit 超价无法消毒，用 canonical SMILES 的文本匹配）
        - 排除: 同时含 -NH2/-NH-（胺会竞争）、-COOH、-CHO、酮、其他杂原子反应基
              金属/无机、盐/同位素、多异腈 → flagged
    * 若 RDKit sanitize 失败，直接以 canonical SMILES 字符串参与。

用法：
    python scripts/iso_build.py build --candidates-dir data/isocyanide_v0.1/candidates \
                                      --slices-dir data/isocyanide_v0.1/slices -t 8
    python scripts/iso_build.py merge --slices-dir ... --out-dir data/isocyanide_v0.1/output
    python scripts/iso_build.py qc    --out-dir ...

环境：python3.10 + RDKit 2026.03。
"""

import argparse
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

# 干扰官能团（以 canonical SMILES 字符串正则做粗判）
_EXCLUDE_SMARTS = {
    "amine_prim_or_sec":  r"[N;H2,H1]",
    "carboxylic_acid":    r"C\(=O\)\[O-\]|C\(=O\)O",
    "aldehyde_or_ketone": r"C\(=O\)",
    "isocyanate":         r"\[N[+]#[C][-]O\]|N=C=O",
    "nitrile_cn":         r"C#N(?!\[C)",   # 酰胺 C#N 不算真正的异腈
    "azide_nnn":          r"\[N-]=[N+]=[N\]",
    "nitro":              r"\[O-]\[N+](=O)|\[N+](=O)\[O-\]",
}
# 避免误伤：上面仅使用字符串正则；随后用 RDKit 若可解析则用 SMARTS 复核。

_RDKIT_SMARTS = {
    "prim_amine": Chem.MolFromSmarts("[NX3;H2;!$(NC=O);!$(N=C);!$([N+])]"),
    "sec_amine":  Chem.MolFromSmarts("[NX3;H1;!$(NC=O);!$(N=C);!$([N+])]"),
    "aldehyde":  "[CX3H1](=O)[#6]",
    "ketone":    "[#6][CX3](=O)[#6]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "isocyanate": "[NX1]=[CX1]=[OX1]",
    "nitrile":   "[NX1]#[CX1]",
    "nitro":     "[NX3+](=O)[O-]",
}
for k, v in list(_RDKIT_SMARTS.items()):
    if isinstance(v, str):
        _RDKIT_SMARTS[k] = Chem.MolFromSmarts(v)

_METALS = {3, 4, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
           31, 32, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
           55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
           73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 91,
           92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103}

ISO_PATS = ("[N+]#[C-]", "[C-]#[N+]")   # 真异腈 R-N≡C（子串计数，比正则可靠）


def _count_isocyanide(smiles):
    return sum(smiles.count(p) for p in ISO_PATS)


def _has_rdkit(mol, key):
    return _RDKIT_SMARTS.get(key) is not None and mol.HasSubstructMatch(_RDKIT_SMARTS[key])


def analyze_one(smiles):
    r = {"smiles": smiles, "canonical": None, "exclude": None,
         "n_isocyanide": 0, "iso_type": "unknown", "mw": None, "logp": None,
         "tpsa": None, "chain_length": None, "stereo": False}

    # 异腈个数（子串计数）
    n_iso = _count_isocyanide(smiles)
    r["n_isocyanide"] = n_iso
    if n_iso == 0:
        r["exclude"] = "no_isocyanide"
        return r
    if n_iso > 1:
        r["exclude"] = "multi_isocyanide"      # → flagged
        return r

    # 1) 尝试 RDKit sanitize 得到 canonical；异腈三键带电荷 RDKit 可能消毒失败 → sanitize=False 读入
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    if m is None:
        r["exclude"] = "parse_fail"
        return r
    try:
        Chem.SanitizeMol(m)
        canon = Chem.MolToSmiles(m)
    except Exception:
        canon = Chem.MolToSmiles(m)  # 保留原样（含显式电荷）
    r["canonical"] = canon

    # 2) 金属 / 多片段 / 同位素
    if any(a.GetAtomicNum() in _METALS for a in m.GetAtoms()):
        r["exclude"] = "contains_metal"
        return r
    if canon.count(".") > 0:
        r["exclude"] = "salt_or_multipart"
        return r
    if any(a.GetIsotope() != 0 for a in m.GetAtoms()):
        r["exclude"] = "isotope"
        return r

    # 3) RDKit 可解析时的 干扰基团复核（sanitize 常因异腈超价失败 → 用 canonical 字符串补强）
    sane = Chem.SanitizeMol(m, catchErrors=True)
    if sane is None:
        issues = [k for k in ("carboxylic_acid", "aldehyde", "ketone", "isocyanate",
                              "nitrile", "nitro")
                  if _has_rdkit(m, k)]
        if issues:
            r["exclude"] = "+".join(issues)
            return r
    # 字符串级补强：游离羧酸 / 醛酮 / 胺（无论 sanitize 是否成功）
    #   注意：异腈 canonical 是 [C-]#[N+] / [N+]#[C-]，其 N 是 [N+]，不会误触 amine
    cs = canon
    acid_mm = bool(re.search(r"C\(=O\)[O,N]", cs))
    cf_mm = bool(re.search(r"C\(=O\)[^#]", cs) and "[C-]#[N+]" not in cs)
    # 原子 N 且非异腈的 [N+]（排除括号内 N）、非酰胺（被 acid_mm 判定）
    #   canonical 里胺常写作 CCN / N；异腈是 [N+]；腈是 C#N；芳香氮是小写 n
    def _bad_n_atom(canon):
        for m in re.finditer(r"N", canon):
            start = m.start()
            if start > 0 and canon[start - 1] == "[":
                continue          # 括号内氢形式 N（[NH2]/[N+]/[NH3+] 等）
            nxt = canon[start + 1:start + 4]
            if nxt.startswith("#"):
                continue          # 腈 C#N
            if nxt.startswith("]"):
                continue          # 原子标签右括号结尾（如 [N+] 情形防漏）
            return True           # 普通 N-C/N=N → 胺
        return False
    amine_mm = bool(re.search(r"\[NH[234]?\]|\[nH\]", cs)) or _bad_n_atom(cs)
    if acid_mm or cf_mm or amine_mm:
        tags = []
        if acid_mm: tags.append("acid_or_amide")
        if cf_mm: tags.append("carbonyl")
        if amine_mm: tags.append("amine")
        r["exclude"] = "+".join(tags)
        return r
    # 补 RingInfo（消毒失败时环信息可能未初始化，TPSA/环类属性会崩）
    try:
        Chem.GetSymmSSSR(m)
    except Exception:
        pass

    r["mw"] = Descriptors.MolWt(m) if m is not None else None
    r["logp"] = Crippen.MolLogP(m) if m is not None else None
    r["tpsa"] = Descriptors.TPSA(m) if m is not None else None
    # 碳链长估算（两次 BFS 求直径；不可解析则为 None）
    if m is not None:
        n = m.GetNumAtoms()
        adj2 = {i: [] for i in range(n)}
        for b in m.GetBonds():
            c, d = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if m.GetAtomWithIdx(c).GetAtomicNum() == 6 and m.GetAtomWithIdx(d).GetAtomicNum() == 6:
                adj2[c].append(d); adj2[d].append(c)
        def bfs2(src):
            dist = [-1]*n; dist[src] = 0; q = [src]; far = src
            for u in q:
                for v in adj2[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u]+1; q.append(v)
                        if dist[v] > dist[far]: far = v
            return far, dist[far]
        if adj2:
            a, _ = bfs2(next(iter(adj2)))
            b_far, d = bfs2(a)
            r["chain_length"] = d + 1

    # 5) MW 窗口
    if r["mw"] is not None and not (60.0 <= r["mw"] <= 700.0):
        r["exclude"] = "mw_out_of_window"
        return r
    return r


# ----------------------------------------------------------------------------
def _build_worker(arg):
    cand_path, slice_dir = arg
    if not os.path.exists(cand_path):
        return {"file": os.path.basename(cand_path), "main": -1, "flagged": -1,
                "excluded": -1, "seconds": 0, "skipped": True}
    base = os.path.basename(cand_path).replace(".tsv.gz", "")
    outs = {k: os.path.join(slice_dir, f"{base}.{k}.tsv.gz") for k in ("main", "flagged", "excluded")}
    t0 = time.time(); counts = Counter()
    header = "\t".join(["cid", "smiles", "canonical", "n_isocyanide", "mw", "logp", "tpsa",
                        "chain_length", "stereo_flag", "exclude_reason"])
    with gzip.open(cand_path, "rt") as fin, \
         gzip.open(outs["main"] + ".tmp", "wt") as fm, \
         gzip.open(outs["flagged"] + ".tmp", "wt") as ff, \
         gzip.open(outs["excluded"] + ".tmp", "wt") as fe:
        for fh in (fm, ff, fe):
            fh.write(header + "\n")
        next(fin)
        for ln in fin:
            p = ln.rstrip("\n").split("\t")
            if len(p) < 3: continue
            cid, smi, ich = p[0], p[1], p[2]
            r = analyze_one(smi)
            row = "\t".join([cid, r["smiles"], r["canonical"] or "", str(r["n_isocyanide"]),
                             "" if r["mw"] is None else f"{r['mw']:.2f}",
                             "" if r["logp"] is None else f"{r['logp']:.2f}",
                             "" if r["tpsa"] is None else f"{r['tpsa']:.1f}",
                             "" if r["chain_length"] is None else str(r["chain_length"]),
                             "Y" if r["stereo"] else "N", r["exclude"] or ""])
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
    # filelist 参数可以是多个文本文件（每行一个候选路径）或直接候选路径
    files = []
    for fl in args.filelist:
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
            r = fut.result()
            summary.append(r)
            print(f"  [ok] {r['file']}: main={r['main']:,} flagged={r['flagged']:,} "
                  f"excluded={r['excluded']:,} {r['seconds']}s", flush=True)
    agg = Counter()
    for r in summary:
        for k in ("main", "flagged", "excluded"):
            agg[k] += r[k]
    print(json.dumps({"files": len(summary), **agg}, indent=2))


def _cmd_merge(args):
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    slices_dir = Path(args.slices_dir)
    import csv

    def load_dedup(suffix):
        rows, seen, dup = [], set(), 0
        for s in sorted(slices_dir.glob(f"*.{suffix}.tsv.gz")):
            with gzip.open(s, "rt") as f:
                next(f)
                for ln in f:
                    cols = ln.rstrip("\n").split("\t")
                    if len(cols) < 4: continue
                    key = (cols[2] or cols[1], cols[0]) if suffix == "main" else (cols[0], cols[1])
                    if key in seen: dup += 1; continue
                    seen.add(key); rows.append(cols)
        return rows, dup

    main_rows, _ = load_dedup("main")
    flag_rows, _ = load_dedup("flagged")
    excl_rows, _ = load_dedup("excluded")
    header = ["cid", "smiles", "canonical", "n_isocyanide", "mw", "logp", "tpsa",
              "chain_length", "stereo_flag", "exclude_reason"]

    with open(out / "isocyanide_v0.1_main.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["isocyanide_id"] + header)
        for i, r in enumerate(main_rows, 1):
            w.writerow([f"ISO_{i:07d}"] + r)
    with open(out / "isocyanide_v0.1_main.smi", "w") as f:
        for i, r in enumerate(main_rows, 1):
            f.write(f"{(r[2] or r[1])}\tISO_{i:07d}\n")
    with open(out / "isocyanide_v0.1_flagged.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(flag_rows)
    with open(out / "isocyanide_v0.1_excluded.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(excl_rows)
    with open(out / "isocyanide_v0.1_flagged.smi", "w") as f:
        f.writelines(f"{r[2] or r[1]}\t{r[-1] or ''}\n" for r in flag_rows)
    with open(out / "isocyanide_v0.1_excluded.smi", "w") as f:
        f.writelines(f"{r[2] or r[1]}\t{r[-1] or ''}\n" for r in excl_rows)

    summ = {
        "version": "isocyanide_v0.1",
        "n_main_after_dedup": len(main_rows),
        "n_flagged_after_dedup": len(flag_rows),
        "n_excluded_after_dedup": len(excl_rows),
        "exclude_reason_top": Counter(r[-1] for r in excl_rows).most_common(15),
        "mw_range": [round(min(float(r[4]) for r in main_rows if r[4]), 2),
                     round(max(float(r[4]) for r in main_rows if r[4]), 2)],
    }
    json.dump(summ, open(out / "isocyanide_v0.1_summary.json", "w"), ensure_ascii=False, indent=2)
    print(json.dumps(summ, ensure_ascii=False, indent=2))


def _cmd_qc(args):
    out = Path(args.out_dir)
    df = __import__("pandas").read_csv(out / "isocyanide_v0.1_main.csv")
    n = len(df)
    with open(out / "isocyanide_v0.1_qc_report.md", "w") as f:
        f.write("# Isocyanide v0.1 QC 报告\n\n")
        f.write(f"- 主库分子数：**{n:,}**\n\n")
        if n:
            f.write("## 20 个随机样例\n\n| ID | Canonical SMILES | MW | LogP |\n|---|---|---|---|\n")
            for _, r in df.sample(min(20, n), random_state=7).iterrows():
                f.write(f"| {r.isocyanide_id} | `{r.canonical[:58]}` | {r.mw:.1f} | {r.logp:.1f} |\n")
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
    args = ap.parse_args(); args.fn(args)


if __name__ == "__main__":
    main()