#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
add_eval_to_library_csv.py — fold the downstream benchmark molecules into the
ionizable-lipid virtual library CSV, saving it as *_v2.csv.

The downstream evaluation set (Sheet2 + Sheet3, 2679 unique molecules) is not in
the 300k balanced library (zero InChIKey overlap, measured). To give the v5
pre-training corpus direct coverage of the downstream domain (Domain-Adaptive
Pre-training), we append those missing molecules to the library CSV. The v2 CSV
keeps the exact same column layout as the v1 file (id, A_id, B_id, C_id, D_id,
SMILES); appended eval rows carry ``id=eval_<sheet>_<row>`` and empty R-group
columns.

Usage:
    conda run -n unimol2 python 4_scripts/add_eval_to_library_csv.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_corpus import canonical_inchikey, load_eval_molecules

DEFAULT_SRC_CSV = Path(
    "/home/data/zhengxiao/disk1/my_project/lnpModel/code/00_prepare_ILs_vitual_library_v2/"
    "curate_library/generated_virtual_library_balanced.csv"
)
DEFAULT_OUT_CSV = DEFAULT_SRC_CSV.with_name("generated_virtual_library_balanced_v2.csv")


def main() -> int:
    src_csv = DEFAULT_SRC_CSV
    out_csv = DEFAULT_OUT_CSV

    lib = pd.read_csv(src_csv, dtype=str)
    lib = lib.fillna("")
    print(f"[lib ] {len(lib):,} rows from {src_csv}")

    # Existing InChIKey set of the library.
    lib_iks: set[str] = set()
    for smi in lib["SMILES"].tolist():
        ik = canonical_inchikey(smi)
        if ik:
            lib_iks.add(ik)
    print(f"[lib ] unique InChIKeys: {len(lib_iks):,}")

    # Eval molecules not already in the library.
    eval_mols = [m for m in load_eval_molecules()
                 if canonical_inchikey(m["smiles"]) not in lib_iks]
    print(f"[eval] {len(eval_mols):,} benchmark molecules missing from library")

    if not eval_mols:
        print("Nothing to add — v2 identical to v1.")
        return 0

    # Preserve v1 column order/names; eval rows use empty R-group columns.
    cols = list(lib.columns)
    rows = []
    for m in eval_mols:
        rec = {c: "" for c in cols}
        rec["id"] = m["mol_id"]
        rec["SMILES"] = m["smiles"]
        rows.append(rec)

    v2 = pd.concat([lib, pd.DataFrame(rows, columns=cols)], ignore_index=True)
    # Recompute uniqueness across the merged frame (belt-and-braces; the append
    # was already IK-filtered, but a NaN/dup row in the source would show here).
    merged_iks = set()
    dup_ik = 0
    for smi in v2["SMILES"].tolist():
        ik = canonical_inchikey(smi)
        if ik and ik in merged_iks:
            dup_ik += 1
        if ik:
            merged_iks.add(ik)
    if dup_ik:
        print(f"  WARNING {dup_ik} duplicate InChIKeys in merged frame")

    v2.to_csv(out_csv, index=False)
    print(f"[out ] wrote {len(v2):,} rows -> {out_csv}")
    print(f"  added: {len(eval_mols):,} eval molecules; total unique IK: {len(merged_iks):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
