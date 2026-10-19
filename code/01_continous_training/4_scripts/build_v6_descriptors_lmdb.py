#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_v6_descriptors_lmdb.py — rebuild the pretrain LMDB with v6 MTR targets.

v6 keeps v5's corpus and conformers exactly (no re-conformer generation), but
replaces the MTR target vector:
  * v5: 32 curated RDKit descriptors (CLS-read)
  * v6: full RDKit descriptor set (~200 cols, paper-style) + 3 lipid priors
        (tail LogP, tertiary-amine pKa, ionizable-N count) — see v6_descriptors.py

Pipeline:
  1. stream the v5 SDF (already carries conformers + SMILES),
  2. sample ~20k molecules across the corpus (every 50th record) to select
     stable descriptor columns and estimate z-stats,
  3. parallel-compute the v6 descriptor vector for every molecule,
  4. z-score with the sample stats (same z-space for all molecules),
     drop molecules with non-finite descriptors (v5 gate),
  5. write the epoch-sharded LMDB + manifest (consumed by unimol2_mtr task).

Usage:
    conda run -n unimol2 python 4_scripts/build_v6_descriptors_lmdb.py
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import lmdb
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import v6_descriptors as V

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_V5_DIR = REPO_ROOT / "1_data/processed/ionizable_lipids/unimolv2_3d_v5"
DEFAULT_INPUT_LMDB = OLD_V5_DIR / "lmdb_pretrain"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "1_data/processed/ionizable_lipids/unimolv2_3d_v6"

LMDB_MAP_SIZE = 130 * 1024 ** 3
SAMPLE_INTERVAL = 100  # every 100th molecule -> ~10k samples from ~1M corpus
MAX_SAMPLE = 10_000


def compute_desc_task(args: tuple[str, list[str]]) -> tuple[str, np.ndarray]:
    """Worker: compute v6 descriptor vector for one SMILES."""
    smiles, desc_names = args
    return smiles, V.compute_descriptors(smiles, desc_names)


def compute_all_task(smiles: str) -> tuple[str, np.ndarray]:
    """Worker: compute the FULL descriptor vector (all candidates)."""
    return smiles, V.compute_descriptors(
        smiles, V.all_rdkit_names() + V.LIPID_PRIOR_NAMES)


def run_parallel(fn, items, n_workers, desc: str, total: int):
    """Map fn over items with a ProcessPool; returns results list."""
    out = [None] * len(items)
    if n_workers <= 1:
        for i, it in enumerate(tqdm(items, desc=desc, total=total)):
            out[i] = fn(it)
        return out
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(fn, it) for it in items]
        for fut in tqdm(as_completed(futures), desc=desc, total=total):
            pass
        for i, fut in enumerate(futures):
            out[i] = fut.result()
    return out


def iter_lmdb_records(input_lmdb_dir: Path, limit: int | None = None):
    """Yield (smi, atoms, coordinates) records from the v5 LMDB.

    Reading the v5 LMDB is ~30x faster than re-parsing the 25 GB SDF
    (SDMolSupplier is CPU-bound on this box: ~1 GB/min vs ~1 min for LMDB),
    and the records already carry exactly what v6 needs (smi/atoms/coords).
    """
    import lmdb as _lmdb
    import pickle as _pickle
    for part in sorted(input_lmdb_dir.glob("train_part_*.lmdb")):
        env = _lmdb.open(str(part), readonly=True, lock=False, subdir=False)
        with env.begin() as txn:
            cur = txn.cursor()
            for _k, val in cur:
                rec = _pickle.loads(val)
                yield rec["smi"], rec["atoms"], rec["coordinates"]
        env.close()


def collect_molecules(input_lmdb_dir: Path, limit: int | None = None):
    """Read molecules from the v5 LMDB into an OrderedDict (mol_id by SMILES)."""
    molecules: "OrderedDict[str, dict]" = OrderedDict()
    for smi, atoms, coords in iter_lmdb_records(input_lmdb_dir, limit):
        if smi in molecules:
            continue
        molecules[smi] = {"smi": smi, "atoms": atoms, "coordinates": coords}
        if limit is not None and len(molecules) >= limit:
            break
    return molecules


def compute_desc_task(args: tuple[str, list[str]]) -> tuple[str, np.ndarray]:
    """Worker: compute v6 descriptor vector for one SMILES."""
    smiles, desc_names = args
    return smiles, V.compute_descriptors(smiles, desc_names)


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild pretrain LMDB with v6 MTR targets.")
    ap.add_argument("--input-lmdb", type=Path, default=DEFAULT_INPUT_LMDB,
                    help="v5 LMDB dir to read smi/atoms/coordinates from (fast)")
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--valid-ratio", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=20260805)
    ap.add_argument("--max-epoch", type=int, default=10)
    ap.add_argument("--n-workers", type=int, default=max(1, (__import__("os").cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, default=None, help="cap molecules (smoke mode)")
    args = ap.parse_args()

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[lmdb] reading molecules from {args.input_lmdb} …", flush=True)
    molecules = collect_molecules(args.input_lmdb, limit=args.limit)
    mol_ids = list(molecules.keys())
    n_mol = len(mol_ids)
    n_conf = sum(len(m["coordinates"]) for m in molecules.values())
    print(f"[lmdb] loaded {n_mol:,} molecules / {n_conf:,} conformers", flush=True)

    # 1. sample for column selection + z-stats (parallel)
    sample_ids = mol_ids[::SAMPLE_INTERVAL][:MAX_SAMPLE]
    sample_smiles = [molecules[k]["smi"] for k in sample_ids]
    print(f"[desc] selecting columns over {len(sample_smiles):,} samples …",
          flush=True)
    sample_desc = run_parallel(
        compute_all_task,
        sample_smiles, args.n_workers, desc="sample-desc",
        total=len(sample_smiles))
    sample_mat = np.vstack([d for _, d in sample_desc])
    desc_names = V.select_desc_names_from_matrix(sample_mat)
    n_rdkit = len(desc_names) - len(V.LIPID_PRIOR_NAMES)
    print(f"[desc] {len(desc_names)} columns kept ({n_rdkit} RDKit + "
          f"{len(V.LIPID_PRIOR_NAMES)} lipid priors)", flush=True)
    lip_kept = [n for n in desc_names if n in V.LIPID_PRIOR_NAMES]
    print(f"[desc] lipid priors kept: {lip_kept}", flush=True)

    print(f"[desc] estimating z-stats …", flush=True)
    cand = V.all_rdkit_names() + V.LIPID_PRIOR_NAMES
    idxs = [cand.index(n) for n in desc_names]
    big = sample_mat[:, idxs].astype(np.float64)
    # nanmean/nanstd ignore NaN but NOT inf — replace non-finite with NaN so a
    # single float32-overflowed value cannot poison the whole column's z-stats
    # (this was the 100%-drop bug: mean became inf -> every z non-finite).
    big[~np.isfinite(big)] = np.nan
    desc_mean = np.nanmean(big, axis=0).astype(np.float32)
    desc_std = np.nanstd(big, axis=0).astype(np.float32)
    desc_std[~np.isfinite(desc_std)] = 1.0
    desc_std[desc_std < 1e-8] = 1.0
    print(f"[desc] z-stats done ({len(desc_mean)} means)", flush=True)

    # 2. parallel compute descriptors for all molecules
    print(f"[desc] computing descriptors for {n_mol:,} molecules "
          f"({args.n_workers} workers) …", flush=True)
    tasks = [(molecules[k]["smi"], desc_names) for k in mol_ids]
    results = run_parallel(compute_desc_task, tasks, args.n_workers,
                           desc="desc", total=n_mol)
    desc_map = {smi: d for smi, d in results}

    # 3. assemble records (drop non-finite descriptor molecules)
    records: list[dict] = []
    dropped = {"nonfinite": 0, "parse": 0}
    for k in mol_ids:
        rec = molecules[k]
        d = desc_map.get(rec["smi"])
        if d is None or not np.isfinite(d).all():
            dropped["nonfinite"] += 1
            continue
        z = (d - desc_mean) / desc_std
        if not np.isfinite(z).all():
            dropped["nonfinite"] += 1
            continue
        records.append({
            "smi": rec["smi"],
            "atoms": rec["atoms"],
            "coordinates": rec["coordinates"],
            "descriptors": z.astype(np.float32),
        })
    print(f"[lmdb] records kept {len(records):,} / {n_mol:,} "
          f"(dropped {dropped})")

    # 4. train/valid split + write LMDB (same protocol as v5 builder:
    #    shuffle molecule order, first ~1% -> valid)
    rng = random.Random(args.seed)
    rng.shuffle(records)
    n_valid = max(1, int(round(len(records) * args.valid_ratio)))
    valid_records = records[:n_valid]
    train_records = records[n_valid:]
    print(f"[lmdb] train {len(train_records):,} / valid {len(valid_records):,}")

    lmdb_dir = out_dir / "lmdb_pretrain"
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    for part, recs in (("train_part_0", train_records), ("valid_part_0", valid_records)):
        path = lmdb_dir / f"{part}.lmdb"
        if path.exists():
            path.unlink()
        env = lmdb.open(str(path), subdir=False, map_size=LMDB_MAP_SIZE,
                        readonly=False, meminit=False)
        with env.begin(write=True) as txn:
            for idx, rec in enumerate(recs):
                txn.put(idx.to_bytes(4, byteorder="big"), pickle.dumps(rec))
        env.sync()
        env.close()
    print(f"[lmdb] wrote {len(train_records):,} train / {len(valid_records):,} "
          f"valid -> {lmdb_dir}")

    # epoch symlinks: Uni-Mol2's ConformerSampleDataset draws a DIFFERENT part
    # per epoch (train_part_{epoch}.lmdb) — without these the run crashes at
    # epoch 2 with "FileNotFoundError: 0 not found ... max-epoch" (v6 lesson).
    for split in ("train", "valid"):
        for epoch_idx in range(1, max(args.max_epoch, 1)):
            link = lmdb_dir / f"{split}_part_{epoch_idx}.lmdb"
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(f"{split}_part_0.lmdb")

    manifest = {
        "input_lmdb": str(args.input_lmdb),
        "output_dir": str(lmdb_dir),
        "seed": args.seed,
        "valid_ratio": args.valid_ratio,
        "max_epoch": args.max_epoch,
        "total_molecules": len(records),
        "total_conformers": sum(len(r["coordinates"]) for r in records),
        "train_molecules": len(train_records),
        "valid_molecules": len(valid_records),
        "record_keys": ["smi", "atoms", "coordinates", "descriptors"],
        "descriptor_names": desc_names,
        "descriptor_n": len(desc_names),
        "descriptor_mean": [round(float(x), 6) for x in desc_mean],
        "descriptor_std": [round(float(x), 6) for x in desc_std],
        "n_rdkit": n_rdkit,
        "n_lipid_priors": len(V.LIPID_PRIOR_NAMES),
        "note": (
            "v6 MTR targets: full RDKit descriptors + lipid priors "
            "(lip_tail_logp, lip_tert_amine_pka, lip_ionizable_n), z-scored. "
            "Corpus/conformers identical to v5."
        ),
    }
    (lmdb_dir / "lmdb_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[lmdb] manifest -> {lmdb_dir / 'lmdb_manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
