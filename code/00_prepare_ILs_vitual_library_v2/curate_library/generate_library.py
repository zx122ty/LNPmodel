#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_library.py
===================
Enumerate an in-silico ionizable lipid library from curated Ugi-4CR components.

Strategy
--------
1.  Load the curated component library ``4CR_unique.xlsx`` (four sheets:
    diverse amines (headgroups) / aldehydes (linkers) / carboxylic acids (tail A) /
    isocyanides (tail B)).  Every SMILES is canonical and role-correct.
2.  For each component, strip the reactive handle (-NH2, -NC, -CHO, -COOH) and
    replace it with an open-valence dummy atom ``*`` (the "main component" of the
    reference LUMI-lab notebook ``data_generate.ipynb``).
3.  For every combination (A x B x C x D) build the joined SMILES with the RDKit
    position-variation coupling block ``|m:...|`` onto the central Ugi core
    ``C(=O)NCC(=O)N`` and let ``rdMolEnumerator`` close the ring, giving exactly
    one enumerated product (the Ugi-4CR adduct).  This is faithful to the
    reference notebook; validated on 5000 random combinations.
4.  Output a CSV with the product SMILES + its four component ids, streamed every
    ``--save-every`` rows so arbitrarily large libraries fit in memory.

Products:  A(H/N)  ...  D(=O)  ->   D-C(=O)-N(A)-CH(R_C)-C(=O)-N(B)  (amide form)

Usage
-----
    python generate_library.py --components ../data/4CR_unique.xlsx \
                               --out generated_virtual_library.csv -t 24
    python generate_library.py --smoke 5000     # quick feasibility & size estimate

The default component path is ``4CR_unique.xlsx`` next to this script.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolEnumerator

RDLogger.DisableLog("rdApp.*")

log = logging.getLogger("generate")

CORE_SMILES = "C(=O)NCC(=O)N"
CORE_POS = {"A": 2, "B": 6, "C": 3, "D": 0}     # Ugi core attachment index per role
HANDLE_SMARTS = {
    "A": "[NH2]",
    "B": "N#C",
    "C": "[CX3H1](=O)",
    "D": "[CX3](=O)[OX2H1]",
}
SHEET_BY_ROLE = {
    "A": "diverse amines (headgroups)",
    "B": "isocyanides (tail B)",
    "C": "aldehydes (linkers)",
    "D": "carboxylic acids (tail A)",
}

_FRAGS = {k: Chem.MolFromSmarts(v) for k, v in HANDLE_SMARTS.items()}


def _core_num_atoms() -> int:
    return Chem.MolFromSmiles(CORE_SMILES).GetNumAtoms()


def strip_handle(smi: str, role: str) -> str:
    """Replace the role's reactive handle by an open-valence '*' dummy atom."""
    mol = Chem.MolFromSmiles(smi)
    stripped = Chem.ReplaceSubstructs(mol, _FRAGS[role], Chem.MolFromSmiles("*"))[0]
    return Chem.MolToSmiles(stripped)


def build_job_star_positions(core_n: int, fragments) -> dict[str, int]:
    """Absolute index of each fragment's dummy in the joined-smiles atom ordering."""
    pA = core_n
    pB = pA + fragments["A"].GetNumAtoms()
    pC = pB + fragments["B"].GetNumAtoms()
    pD = pC + fragments["C"].GetNumAtoms()
    return {"A": pA, "B": pB, "C": pC, "D": pD}


def enumerate_one(record: dict) -> str | None:
    """Assemble + enumerate a single combination; return product SMILES."""
    frag_smiles = {}
    frags = {}
    for role in "ABCD":
        frag_smiles[role] = record[f"{role}_frag"]
        frags[role] = Chem.MolFromSmiles(frag_smiles[role])
    core_n = _core_num_atoms()
    star = build_job_star_positions(core_n, frags)
    coupling = ",".join(f"{star[r]}:{CORE_POS[r]}" for r in "ABCD")
    joined = f"{CORE_SMILES}." + ".".join(frag_smiles[r] for r in "ABCD")
    mol = Chem.MolFromSmiles(f"{joined} |m:{coupling}|")
    if mol is None:
        return None
    enumerated = rdMolEnumerator.Enumerate(mol)
    if len(enumerated) != 1:
        return None
    return Chem.MolToSmiles(Chem.RemoveHs(enumerated[0]))


def load_components(xlsx_path: Path) -> dict[str, list[dict]]:
    """Return {role: [ {id, smiles, frag} ... ]} in file order."""
    from openpyxl import load_workbook
    wb = load_workbook(xlsx_path, read_only=True, data_only=True)
    out = {}
    for role, sheet in SHEET_BY_ROLE.items():
        rows = list(wb[sheet].iter_rows(values_only=True))[1:]
        rows = [r for r in rows if r and r[0] is not None]
        frag = []
        for row in rows:
            cid = str(row[0])
            smi = str(row[1]).strip()
            f = strip_handle(smi, role)
            frag.append({"id": cid, "smiles": smi, "frag": f})
        out[role] = frag
    wb.close()
    return out


def iter_jobs(components: dict[str, list[dict]]):
    """Yield per-combination job dicts (id strings + pre-stripped fragment SMILES)."""
    for a in components["A"]:
        for b in components["B"]:
            for c in components["C"]:
                for d in components["D"]:
                    yield {
                        "id": f"{a['id']}:{b['id']}:{c['id']}:{d['id']}",
                        "A_id": a["id"], "A_frag": a["frag"],
                        "B_id": b["id"], "B_frag": b["frag"],
                        "C_id": c["id"], "C_frag": c["frag"],
                        "D_id": d["id"], "D_frag": d["frag"],
                    }


class _Worker:
    """Picklable worker: enumerate_one needs module-level helper only."""

    @staticmethod
    def run(rec: dict) -> dict:
        smi = enumerate_one(rec)
        if smi is None:
            return {"id": rec["id"], "A_id": rec["A_id"], "B_id": rec["B_id"],
                    "C_id": rec["C_id"], "D_id": rec["D_id"], "SMILES": None}
        return {"id": rec["id"], "A_id": rec["A_id"], "B_id": rec["B_id"],
                "C_id": rec["C_id"], "D_id": rec["D_id"], "SMILES": smi}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--components", default="4CR_unique.xlsx",
                   help="path to curated 4CR_unique.xlsx (default 4CR_unique.xlsx)")
    p.add_argument("--out", default="generated_virtual_library.csv")
    p.add_argument("-t", "--threads", type=int, default=24)
    p.add_argument("--save-every", type=int, default=200_000)
    p.add_argument("--smoke", type=int, default=0,
                   help="only enumerate N random combinations, print feasibility + projected size, then exit")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(module)s: %(message)s")
    comp_path = Path(args.components).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    components = load_components(comp_path)
    counts = {r: len(components[r]) for r in "ABCD"}
    total = counts["A"] * counts["B"] * counts["C"] * counts["D"]
    log.info("components: %s -> %s total combinations", counts, f"{total:,}")

    if args.smoke:
        import random
        rng = random.Random(0)
        jobs = []
        for _ in range(args.smoke):
            jobs.append({k: (components[k][rng.randrange(counts[k])]) for k in "ABCD"})
        jobs = [{"id": ":".join(j[k]["id"] for k in "ABCD"),
                 **{f"{r}_id": j[r]["id"] for r in "ABCD"},
                 **{f"{r}_frag": j[r]["frag"] for r in "ABCD"}} for j in jobs]
        t0 = time.time()
        fail = 0
        with ProcessPoolExecutor(max_workers=args.threads) as ex:
            res = list(ex.map(_Worker.run, jobs))
        for r in res:
            if r["SMILES"] is None:
                fail += 1
        dt = time.time() - t0
        ok = len(res) - fail
        log.info("smoke: %d/%d succeeded (%.2f%%) in %.1fs "
                 "(%.0f mols/s -> projected ETA %s at %d workers)",
                 ok, len(res), 100*ok/max(1,len(res)), dt,
                 ok/max(dt, 1e-9), fmt_eta(total/max(ok/max(dt,1e-9),1e-9)), args.threads)
        return 0

    # --- full enumeration (streamed) ---
    t0 = time.time()
    written = 0
    first_row = True
    seen_ids: set[str] = set()
    header = ["id", "A_id", "B_id", "C_id", "D_id", "SMILES"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def flush(rows):
        nonlocal first_row, written
        with out_path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            if first_row:
                w.writeheader()
                first_row = False
            w.writerows(rows)
        written += len(rows)

    buf = []
    t_prog = time.time()
    with ProcessPoolExecutor(max_workers=args.threads) as ex:
        it = ex.map(_Worker.run, iter_jobs(components), chunksize=256)
        for i, r in enumerate(it, 1):
            buf.append(r)
            if len(buf) >= args.save_every:
                flush(buf)
                buf = []
                dt = time.time() - t_prog
                rate = args.save_every / max(dt, 1e-9)
                log.info("progress: %s / %s (%.2f%%)  %.0f mol/s  ETA %s",
                         f"{i:,}", f"{total:,}", 100*i/total, rate,
                         fmt_eta((total - i) / max(rate, 1e-9)))
                t_prog = time.time()
    if buf:
        flush(buf)
        buf = []

    log.info("done: %s product rows -> %s (%.1fs)", f"{written:,}", out_path, time.time()-t0)
    log.info("full library size: %s", f"{written:,}")
    return 0


def fmt_eta(seconds: float) -> str:
    t = float(seconds)
    for unit in ("s", "min", "h", "d"):
        if t < 90:
            return f"{t:.0f}{unit}"
        t /= 60
    return f"{t:.1f}d"
if __name__ == "__main__":
    sys.exit(main())