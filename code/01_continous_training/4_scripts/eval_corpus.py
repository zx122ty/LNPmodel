#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_corpus.py — shared loader for the downstream benchmark molecules.

Used by both:
  * add_eval_to_library_csv.py  (Step A: fold eval molecules into the IL library CSV)
  * build_v5_eval_incremental.py (Step B: give eval molecules conformers + z-scores)

The downstream benchmark is the LUMI Sheet2 (1920 mols) + our Sheet3 (768 mols)
transfection datasets. The dedup is done on InChIKey (canonical identity), so a
SMILES that merely differs in string representation is counted once.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from rdkit import Chem

BENCHMARK_XLSX = Path(
    "/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx"
)

SHEET_SPECS = {
    "Sheet2": {"smiles_col": "SMILES"},
    "Sheet3": {"smiles_col": "SMILES"},
}


def canonical_inchikey(smiles: str) -> str | None:
    """Return the InChIKey of a SMILES, or None if it cannot be parsed."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToInchiKey(mol)


def load_eval_molecules() -> list[dict]:
    """Read all benchmark SMILES (Sheet2 + Sheet3), dedup by InChIKey.

    Returns a list of dicts with keys: mol_id, smiles, source_type, sheet, row.
    ``source_type`` is "virtual" so downstream conformer budgeting matches the
    ionizable-lipid tier (6 generate -> 4 keep).
    """
    xl = pd.ExcelFile(str(BENCHMARK_XLSX))
    seen: set[str] = set()
    out: list[dict] = []
    for sheet, spec in SHEET_SPECS.items():
        df = xl.parse(sheet)
        col = spec["smiles_col"]
        for row_idx, smi in enumerate(df[col].tolist()):
            if smi is None or (isinstance(smi, float) and pd.isna(smi)):
                continue
            smi = str(smi).strip()
            if not smi:
                continue
            ik = canonical_inchikey(smi)
            if ik is None or ik in seen:
                continue
            seen.add(ik)
            out.append({
                "mol_id": f"eval_{sheet}_{row_idx}",
                "smiles": smi,
                "source_type": "virtual",
                "sheet": sheet,
                "row": row_idx,
            })
    return out
