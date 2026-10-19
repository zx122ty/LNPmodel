#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_v5_descriptors_lmdb.py — convert the v5 SDF into the epoch-sharded LMDB
that the v5 continued-pretraining task (``unimol2_mtr``) consumes.

Identical to build_lipids_pretrain_lmdb.py except:
  * every molecule record additionally stores ``"descriptors"``: a float32
    numpy array of the MTR target vector (Z-scored RDKit descriptors),
  * the LMDB manifest carries the descriptor names, mean and std — the v5
    training script reads these to construct the MTR head's output layer and
    to (optionally) unstandardise MTR predictions at logging time.
  * the v5 task signals its own split (flogged via ``Z_DESCRIPTORS`` property
    on each record, computed by build_v5_dataset.py).

Usage:
    conda activate unimol2
    python build_v5_descriptors_lmdb.py --input-sdf .../lipids_unimolv2_3d.sdf
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import OrderedDict
from pathlib import Path

import lmdb
import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_SDF = (
    REPO_ROOT / "1_data/processed/ionizable_lipids/unimolv2_3d_v5/lipids_unimolv2_3d.sdf"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "1_data/processed/ionizable_lipids/unimolv2_3d_v5/lmdb_pretrain"
)
LMDB_MAP_SIZE = 120 * 1024 ** 3  # ~1M mols * (3 confs + desc) LLM-bound; generous

DESC_PROP = "Z_DESCRIPTORS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the v5 SDF into LMDB with MTR descriptor targets."
    )
    parser.add_argument("--input-sdf", type=Path, default=DEFAULT_INPUT_SDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--valid-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--max-epoch", type=int, default=10)
    return parser.parse_args()


def iter_sdf_records(input_sdf: Path):
    """Yield (mol_id, smiles, atom_symbols, coords, desc_str) per conformer record."""
    supplier = Chem.SDMolSupplier(str(input_sdf), removeHs=False, sanitize=False)
    for mol in supplier:
        if mol is None:
            continue
        if mol.GetNumConformers() == 0:
            continue
        # required properties
        if not (mol.HasProp("MOL_ID") and mol.HasProp("SMILES")
                and mol.HasProp("SOURCE_TYPE") and mol.HasProp(DESC_PROP)):
            continue
        mol_id = mol.GetProp("MOL_ID").strip()
        smiles = mol.GetProp("SMILES").strip()
        if not mol_id or not smiles:
            continue
        atom_symbols = np.array([a.GetSymbol() for a in mol.GetAtoms()])
        coords = mol.GetConformer().GetPositions().astype(np.float32)
        if len(atom_symbols) != coords.shape[0]:
            continue
        desc_str = mol.GetProp(DESC_PROP).strip()
        yield mol_id, smiles, atom_symbols, coords, desc_str


def group_molecules(input_sdf: Path):
    """Group conformer records by molecule id; parse descriptors once per molecule."""
    molecules: "OrderedDict[str, dict]" = OrderedDict()
    skipped_mismatch = 0
    skipped_desc = 0
    for mol_id, smiles, atom_symbols, coords, desc_str in tqdm(
        iter_sdf_records(input_sdf), desc="reading conformers", unit="conf"
    ):
        rec = molecules.get(mol_id)
        if rec is None:
            # first conformer carries the descriptor vector
            desc_values = [float(x) for x in desc_str.split(";")] if desc_str else []
            if not desc_values:
                skipped_desc += 1
                continue
            desc_arr = np.array(desc_values, dtype=np.float32)
            if not np.isfinite(desc_arr).all():
                # Non-finite (NaN or ±inf) descriptors would poison valid-set
                # mtr_loss/best_loss at training time; drop the molecule.
                skipped_desc += 1
                continue
            molecules[mol_id] = {
                "smi": smiles,
                "atoms": atom_symbols,
                "coordinates": [coords],
                "descriptors": desc_arr,
            }
            continue
        if len(atom_symbols) != len(rec["atoms"]) or not np.array_equal(
            atom_symbols, rec["atoms"]
        ):
            skipped_mismatch += 1
            continue
        rec["coordinates"].append(coords)
    return molecules, skipped_mismatch, skipped_desc


def write_lmdb(path: Path, records: list[dict]) -> int:
    if path.exists():
        path.unlink()
    lock = path.with_suffix(path.suffix + "-lock")
    if lock.exists():
        lock.unlink()
    env = lmdb.open(str(path), subdir=False, map_size=LMDB_MAP_SIZE,
                    readonly=False, meminit=False, lock=True)
    with env.begin(write=True) as txn:
        for idx, rec in enumerate(records):
            txn.put(idx.to_bytes(4, byteorder="big"), pickle.dumps(rec))
    env.sync()
    env.close()
    if lock.exists():
        lock.unlink()
    return len(records)


def make_epoch_symlinks(output_dir: Path, split: str, max_epoch: int) -> list[str]:
    base = f"{split}_part_0.lmdb"
    links = []
    for epoch_idx in range(1, max(max_epoch, 1)):
        link = output_dir / f"{split}_part_{epoch_idx}.lmdb"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(base)
        links.append(link.name)
    return links


def main() -> None:
    args = parse_args()
    input_sdf = args.input_sdf.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_sdf.exists():
        raise FileNotFoundError(f"Input SDF not found: {input_sdf}")

    molecules, skipped_mismatch, skipped_desc = group_molecules(input_sdf)
    mol_ids = list(molecules.keys())
    total_conformers = sum(len(m["coordinates"]) for m in molecules.values())
    print(f"Grouped {len(mol_ids):,} molecules / {total_conformers:,} conformers "
          f"(skipped_mismatch={skipped_mismatch}, skipped_desc={skipped_desc})")

    if not mol_ids:
        raise SystemExit("No usable molecules — check the SDF Z_DESCRIPTORS property.")

    # fixed train/valid split (molecules, not conformers)
    rng = random.Random(args.seed)
    rng.shuffle(mol_ids)
    n_valid = max(1, int(round(len(mol_ids) * args.valid_ratio))) if args.valid_ratio > 0 else 0
    valid_ids = set(mol_ids[:n_valid])
    train_records = [molecules[mid] for mid in mol_ids if mid not in valid_ids]
    valid_records = [molecules[mid] for mid in mol_ids if mid in valid_ids]

    train_count = write_lmdb(output_dir / "train_part_0.lmdb", train_records)
    valid_count = (write_lmdb(output_dir / "valid_part_0.lmdb", valid_records)
                   if valid_records else 0)

    train_links = make_epoch_symlinks(output_dir, "train", args.max_epoch)
    valid_links = (make_epoch_symlinks(output_dir, "valid", args.max_epoch)
                   if valid_count else [])

    # descriptor metadata shared with the training script
    desc_names = []
    desc_mean = []
    desc_std = []
    card_path = input_sdf.parent / "dataset_cards.json"
    if card_path.exists():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        desc_names = card.get("descriptor_names", [])
        desc_mean = card.get("descriptor_mean", [])
        desc_std = card.get("descriptor_std", [])
    d0 = train_records[0]["descriptors"] if train_records else None
    n_desc = int(d0.shape[0]) if d0 is not None else len(desc_names)

    manifest = {
        "input_sdf": str(input_sdf),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "valid_ratio": args.valid_ratio,
        "max_epoch": args.max_epoch,
        "total_molecules": len(mol_ids),
        "total_conformers": total_conformers,
        "skipped_conformer_mismatch": skipped_mismatch,
        "skipped_desc": skipped_desc,
        "train_molecules": train_count,
        "valid_molecules": valid_count,
        "train_shard": "train_part_0.lmdb",
        "valid_shard": "valid_part_0.lmdb" if valid_count else None,
        "train_epoch_symlinks": train_links,
        "valid_epoch_symlinks": valid_links,
        "record_keys": ["smi", "atoms", "coordinates", "descriptors"],
        "descriptor_names": desc_names,
        "descriptor_n": n_desc,
        "descriptor_mean": desc_mean,
        "descriptor_std": desc_std,
        "source_ratio_general_lipid": 0.7,
        "note": (
            "'descriptors' is the Z-scored MTR target vector for the [CLS] MTR head. "
            "atoms/coordinates include hydrogens; unimol2 task removes H and rebuilds "
            "graph features from 'smi'."
        ),
    }
    (output_dir / "lmdb_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {train_count:,} train / {valid_count:,} valid records -> {output_dir}")
    print(f"  descriptors: n={n_desc} (from dataset_cards.json)")


if __name__ == "__main__":
    main()