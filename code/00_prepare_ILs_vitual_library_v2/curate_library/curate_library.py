#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
curate_library.py
=================
Curate the input ionizable-lipid component library and export a clean, de-duplicated
4CR_unique.xlsx with one sheet per Ugi-4CR role:

    "diverse amines (headgroups)"   <- A  (R-NH2)
    "aldehydes (linkers)"           <- C  (R-CHO)
    "carboxylic acids (tail A)"     <- D  (R-COOH)
    "isocyanides (tail B)"          <- B  (R-NC)

Input
-----
By default: `../data/sub-library.xlsx`  (the LUMI-lab curated 251 set
`other_model/LUMI-lab/data/sub-library.xlsx`, copied into this project's data dir).
Layout: columns ``ID, SMILES, Category``; Cell values:
    Amines -> A*,  Isocyanides -> B*,  Lipid Aldehydes -> C*,  Lipid Carboxylic Acids -> D*
Other files are supported via CLI args (see --help).

Curation rules (per the task in 01.md)
--------------------------------------
1.  Load the sheet indicated by `--sheet`.
2.  Read the SMILES column; validate every entry with RDKit.
    - rows whose SMILES cannot be parsed are deleted,
    - rows with an empty SMILES are deleted,
    - rows whose molecule lacks the *required functional group* for their category
      (A: -NH2, B: -NC, C: -CHO, D: -COOH) are deleted,
    - rows that are *also* a different category's reagent (give the wrong product)
      are deleted.
3.  Trivial OCR/typo fixes are applied *before* validation (documented in the
    CHANGES log), e.g.  ``I`` -> ``1`` ring closures, ``=0`` -> ``=O``, ``-X`` -> ``X``,
    ``-O`` (dangling) -> ``=O``.
4.  Duplicate canonical SMILES are removed (first occurrence kept).
5.  A fresh sequential ``id`` is filled in: A1..An, B1..Bn, C1..Cn, D1..Dn.

Output
------
Writes ``<out_prefix>.xlsx`` (default ``4CR_unique.xlsx``) with the four role sheets
plus a ``log`` sheet recording every deleted/edited row.  A CSV of all components is
also written for convenience/debugging.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

log = logging.getLogger("curate")

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

# (category value in input  ->  (role sheet name, required functional group SMARTS, id prefix))
CATEGORY_MAP = {
    "Amines": ("diverse amines (headgroups)", "[NH2]", "A"),
    "Isocyanides": ("isocyanides (tail B)", "N#C", "B"),
    "Lipid Aldehydes": ("aldehydes (linkers)", "[CX3H1](=O)", "C"),
    "Lipid Carboxylic Acids": ("carboxylic acids (tail A)", "[CX3](=O)[OX2H1]", "D"),
}

# A molecule that also matches one of these groups (as its *second* reaction
# handle) would react at the wrong or an additional site -> excluded.
#   e.g. R-NC + R-NH2  (isocyanides also carry an amine -> dropped)
#   `[C-]#[N+]` also matches `C[N+]#[C-]` (the zwitterion form of R-NC).
CROSS_CATEGORY_SMARTS = {
    "Amines": ["[C-]#[N+]"],                     # R-NC / R-CN hidden in the amine sheet
    "Isocyanides": ["[NH1;!R]", "[NH2]"],        # 2°/1° amine (N-H) hidden in the isoCN sheet
}

REQUIRED_GROUP = {cat: Chem.MolFromSmarts(s) for cat, (_, s, _) in CATEGORY_MAP.items()}
CROSS = {cat: [Chem.MolFromSmarts(s) for s in pats] for cat, pats in CROSS_CATEGORY_SMARTS.items()}


# ---------------------------------------------------------------------------
# SMILES auto-repair (only trivial taxonomic fixes, each is logged)
# ---------------------------------------------------------------------------

def _repairs(smi: str) -> list[str]:
    """Return a list of human-readable notes describing applied fixes."""
    notes = []
    fixed = smi

    # 'I' used as ring-closure digit (OCR artifact of '1')  e.g. NCCNICCNCC1
    if "I" in fixed and fixed != fixed.replace("I", "1"):
        fixed = fixed.replace("I", "1")
        notes.append("I->1")

    # '=0' numeric zero used as double-bond oxygen  e.g. CCCCCC=0
    if "=0" in fixed:
        fixed = fixed.replace("=0", "=O")
        notes.append("=0->=O")

    # 'X(' used as a closing ')'   e.g. O=C( ... )X(C1=CC=C(C)C=C1)=O
    if "X(" in fixed:
        fixed = fixed.replace("X(", ")")
        notes.append("X(->)")

    # dangling '-O' at the end (should be '=O')  e.g. OC(CCCCCCCCC)-O
    if fixed.endswith("-O"):
        fixed = fixed[:-2] + "=O"
        notes.append("-O->=O")

    return fixed, notes


def _fix_smiles(smi: object) -> tuple[str | None, list[str]]:
    """Apply trivial repairs; return the (possibly repaired) SMILES that parses,
    or None if it still fails, along with the repair notes."""
    if smi is None or (isinstance(smi, float) and pd.isna(smi)):
        return None, ["empty"]
    s = str(smi).strip()
    if not s:
        return None, ["empty"]
    notes: list[str] = []
    if Chem.MolFromSmiles(s) is None:
        s, notes = _repairs(s)
    if Chem.MolFromSmiles(s) is None:
        return None, notes + ["unparsable"]
    return s, notes


def _required_group_missing(cat: str, mol) -> bool:
    return not mol.HasSubstructMatch(REQUIRED_GROUP[cat])


def _has_cross_group(cat: str, mol) -> bool:
    for pat in CROSS.get(cat, []):
        if mol.HasSubstructMatch(pat):
            return True
    return False


# ---------------------------------------------------------------------------
# core curation
# ---------------------------------------------------------------------------

def curate_sheet(df: pd.DataFrame, cat: str, sheet_name: str) -> pd.DataFrame:
    """Clean one input category frame into a role-sheet frame."""
    sheet, _, prefix = CATEGORY_MAP[cat]

    out = df.copy()
    out.columns = [str(c) if not pd.isna(c) else f"col{i}" for i, c in enumerate(out.columns)]
    if "SMILES" not in out.columns:
        raise KeyError(f"category '{cat}': input lacks a 'SMILES' column (got {list(out.columns)})")

    smi_raw = out["SMILES"].astype(object)
    out["smiles_orig"] = smi_raw.astype(str)
    out["fix_notes"] = ""
    out["drop_reason"] = ""

    parsed: list[Chem.Mol | None] = []
    fixed_smi: list[str] = []
    fix_notes: list[list[str]] = []
    drop: list[str] = []
    for v in smi_raw:
        s, notes = _fix_smiles(v)
        mol = Chem.MolFromSmiles(s) if s else None
        cross = (mol is not None) and _has_cross_group(cat, mol)
        missing = (mol is not None) and _required_group_missing(cat, mol)
        if s is None:
            drop.append("unparsable")
        elif cross:
            drop.append("wrong functional group (is a different category's reagent)")
        elif missing:
            drop.append(f"lacks required group {REQUIRED_GROUP[cat]}")
        else:
            drop.append("")
        parsed.append(mol)
        fixed_smi.append(s if s else "")
        fix_notes.append(notes)

    out["canonical_SMILES"] = [
        Chem.MolToSmiles(m) if m is not None else None for m in parsed
    ]
    out["fix_notes_str"] = [";".join(ns) for ns in fix_notes]

    out["SMILES"] = fixed_smi
    out["drop_reason"] = drop

    # --- split into kept / dropped
    keep = out[out["drop_reason"] == ""].copy()
    dropped = out[out["drop_reason"] != ""].copy()

    # duplicate canonical SMILES (first kept)
    keep = keep.drop_duplicates(subset=["canonical_SMILES"], keep="first")
    keep["SMILES"] = keep["canonical_SMILES"].values  # canonical form
    keep.drop(columns=["canonical_SMILES"], inplace=True)

    # fill the fresh id
    keep.insert(0, "id", [f"{prefix}{i+1}" for i in range(len(keep))])

    # assemble the row-level log
    log_rows: list[dict] = []
    for _, r in dropped.iterrows():
        log_rows.append({"id": r.get("id"), "raw_id": r.get("raw_id"), "SMILES_orig": r["smiles_orig"],
                        "reason": r["drop_reason"], "fix_notes": r["fix_notes_str"]})
    for _, r in keep.iterrows():
        if str(r["fix_notes_str"]):
            log_rows.append({"id": r["id"], "raw_id": r.get("raw_id"), "SMILES_orig": r["smiles_orig"],
                            "reason": "fixed (SMILES repaired before validation)", "fix_notes": r["fix_notes_str"]})
    df_log = pd.DataFrame(log_rows)

    # columns kept in the role sheet
    id_cols = ["id", "SMILES"]
    id_cols += [c for c in ("raw_id", "name", "pmid") if c in keep.columns]
    keep = keep[id_cols].reset_index(drop=True)
    return keep, df_log


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="../data/sub-library.xlsx",
                   help="input xlsx with component library (default ../data/sub-library.xlsx)")
    p.add_argument("--sheet", default="Sheet1", help="sheet within --input holding the component rows")
    p.add_argument("--out", default="4CR_unique.xlsx", help="output xlsx path (default 4CR_unique.xlsx)")
    p.add_argument("--id-col", default="ID", help="column holding (maybe dirty) ids  e.g. A1")
    p.add_argument("--category-col", default="Category", help="column holding the category label")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(module)s: %(message)s")

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        log.error("input file not found: %s", in_path)
        return 2
    wb = load_workbook(in_path, read_only=True, data_only=True)
    if args.sheet not in wb.sheetnames:
        log.error("sheet %r not in %s (have: %s)", args.sheet, in_path.name, wb.sheetnames)
        return 2
    df_raw = pd.read_excel(in_path, sheet_name=args.sheet, dtype=str)
    wb.close()

    cats = df_raw[args.category_col].astype(str).str.strip()
    out_sheets: dict[str, pd.DataFrame] = {}
    all_logs: list[pd.DataFrame] = []
    summary = []
    for cat, g in df_raw.groupby(cats):
        if cat.lower() in ("nan", ""):            continue
        if cat not in CATEGORY_MAP:
            log.warning("unknown category %r -> skipped", cat)
            continue
        keep, lg = curate_sheet(g, cat, str(cat))
        sheet, _, _ = CATEGORY_MAP[cat]
        out_sheets[sheet] = keep
        all_logs.append(lg)
        summary.append((cat, len(g), len(keep)))

    log_sheet = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        for sheet, df in out_sheets.items():
            df.to_excel(xw, sheet_name=sheet, index=False)
        if not log_sheet.empty:
            log_sheet.to_excel(xw, sheet_name="curation_log", index=False)
    log.info("wrote %s", out_path)
    for cat, nin, nout in summary:
        log.info("  %-28s %4d -> %4d", cat, nin, nout)

    # also a flat CSV for convenience
    flat = pd.concat(list(out_sheets.values()), ignore_index=True)
    flat.to_csv(out_path.with_suffix(".csv"), index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())