"""Convert a prepared DSSTox UniMol2 SDF into the epoch-sharded LMDB format that
the official Uni-Mol2 pretraining task (``unimol2``) expects.

The official pipeline (``4_scripts/Uni-Mol/unimol2/unimol2``) reads training data
through ``data/lmdb_dataset.py``::LMDBDataset, which opens single-file LMDBs named
``{split}_part_{epoch-1}.lmdb`` (epoch is 1-indexed, file index 0-indexed). Each
record is a ``pickle``-d dict keyed by ``idx.to_bytes(4, "big")`` containing::

    {
        "smi":         original SMILES string (parseable by RDKit),
        "atoms":       np.ndarray[str]  of every atom symbol incl. H,
        "coordinates": list[np.ndarray(N, 3) float32]  one entry per conformer,
    }

Alignment contract (important): at training time the graph features are rebuilt
from ``smi`` via ``MolFromSmiles(smi) -> AddHs -> RemoveAllHs``, while the atom
tokens / 3D positions come from ``atoms`` / ``coordinates`` after the pipeline
strips hydrogens. For the heavy-atom ordering of all three to match, ``smi`` MUST
be the same SMILES the molblock was generated from. The prep script builds each
conformer with ``AddHs(MolFromSmiles(SMILES))`` and writes the original ``SMILES``
property, so we store that property verbatim as ``smi``.

We group all conformers sharing a ``DTXSID`` into one molecule record (the task's
``ConformerSampleDataset`` samples one conformer per epoch), split molecules into
train/valid, write ``train_part_0.lmdb`` / ``valid_part_0.lmdb`` physically, and
symlink ``train_part_1.lmdb ... train_part_{max_epoch-1}.lmdb`` to the part-0 file
so a multi-epoch run never needs the data duplicated on disk.
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
    REPO_ROOT / "1_data/processed/ionizable_lipids/unimolv2_3d/lipids_unimolv2_3d.sdf"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "1_data/processed/ionizable_lipids/unimolv2_3d/lmdb_pretrain"
)
DEFAULT_VALID_RATIO = 0.01
# Generous LMDB map size (single file grows up to this); 50 GiB is plenty for ~1M mols.
LMDB_MAP_SIZE = 50 * 1024 ** 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a UniMol2 SDF into epoch-sharded LMDB for continued pretraining."
    )
    parser.add_argument("--input-sdf", type=Path, default=DEFAULT_INPUT_SDF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--valid-ratio", type=float, default=DEFAULT_VALID_RATIO)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--max-epoch",
        type=int,
        default=10,
        help="Number of epoch shards to expose via symlinks (train_part_0..N-1).",
    )
    parser.add_argument("--id-prop", type=str, default="MOL_ID")
    parser.add_argument("--smiles-prop", type=str, default="SMILES")
    return parser.parse_args()


def iter_sdf_conformers(input_sdf: Path, id_prop: str, smiles_prop: str):
    """Yield (mol_id, smiles, atom_symbols, coords) per SDF conformer record.

    Uses ``sanitize=False`` so permissive molecules still load; we only need atom
    symbols, 3D coordinates and the text properties, none of which require a full
    RDKit sanitization. Atom order is the molblock order, which equals the
    ``MolFromSmiles(SMILES)`` heavy-atom order (plus appended H) by construction.
    """
    supplier = Chem.SDMolSupplier(
        str(input_sdf), removeHs=False, sanitize=False
    )
    for mol in supplier:
        if mol is None:
            continue
        if mol.GetNumConformers() == 0:
            continue
        if not mol.HasProp(id_prop) or not mol.HasProp(smiles_prop):
            continue
        mol_id = mol.GetProp(id_prop).strip()
        smiles = mol.GetProp(smiles_prop).strip()
        if not mol_id or not smiles:
            continue
        atom_symbols = np.array([a.GetSymbol() for a in mol.GetAtoms()])
        coords = mol.GetConformer().GetPositions().astype(np.float32)
        if len(atom_symbols) != coords.shape[0]:
            continue
        yield mol_id, smiles, atom_symbols, coords


def group_molecules(input_sdf: Path, id_prop: str, smiles_prop: str):
    """Group conformer records by molecule id into LMDB-ready records.

    Returns an OrderedDict mol_id -> {"smi", "atoms", "coordinates": [array, ...]}.
    Conformers whose atom count/order disagrees with the first seen conformer of
    the same molecule are skipped (defensive; should not happen for one molecule).
    """
    molecules: "OrderedDict[str, dict]" = OrderedDict()
    skipped_mismatch = 0
    for mol_id, smiles, atom_symbols, coords in tqdm(
        iter_sdf_conformers(input_sdf, id_prop, smiles_prop),
        desc="reading conformers",
        unit="conf",
    ):
        rec = molecules.get(mol_id)
        if rec is None:
            molecules[mol_id] = {
                "smi": smiles,
                "atoms": atom_symbols,
                "coordinates": [coords],
            }
            continue
        if len(atom_symbols) != len(rec["atoms"]) or not np.array_equal(
            atom_symbols, rec["atoms"]
        ):
            skipped_mismatch += 1
            continue
        rec["coordinates"].append(coords)
    return molecules, skipped_mismatch


def write_lmdb(path: Path, records: list[dict]) -> int:
    """Write records to a single-file LMDB keyed by 4-byte big-endian indices."""
    if path.exists():
        path.unlink()
    lock = path.with_suffix(path.suffix + "-lock")
    if lock.exists():
        lock.unlink()
    env = lmdb.open(
        str(path),
        subdir=False,
        map_size=LMDB_MAP_SIZE,
        readonly=False,
        meminit=False,
        lock=True,
    )
    with env.begin(write=True) as txn:
        for idx, rec in enumerate(records):
            txn.put(idx.to_bytes(4, byteorder="big"), pickle.dumps(rec))
    env.sync()
    env.close()
    # The official reader opens with lock=False; drop the stray write-lock file.
    if lock.exists():
        lock.unlink()
    return len(records)


def make_epoch_symlinks(output_dir: Path, split: str, max_epoch: int) -> list[str]:
    """Symlink {split}_part_1..max_epoch-1.lmdb to the physical part_0 file."""
    base = f"{split}_part_0.lmdb"
    links = []
    for epoch_idx in range(1, max(max_epoch, 1)):
        link = output_dir / f"{split}_part_{epoch_idx}.lmdb"
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(base)  # relative link within the same dir
        links.append(link.name)
    return links


def main() -> None:
    args = parse_args()
    input_sdf = args.input_sdf.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_sdf.exists():
        raise FileNotFoundError(f"Input SDF not found: {input_sdf}")

    molecules, skipped_mismatch = group_molecules(
        input_sdf, args.id_prop, args.smiles_prop
    )
    mol_ids = list(molecules.keys())
    total_conformers = sum(len(m["coordinates"]) for m in molecules.values())
    print(
        f"Grouped {len(mol_ids)} molecules / {total_conformers} conformers "
        f"({skipped_mismatch} conformer records skipped on atom mismatch)."
    )

    # Split molecules (not conformers) into train/valid for a clean leakage-free split.
    rng = random.Random(args.seed)
    rng.shuffle(mol_ids)
    n_valid = max(1, int(round(len(mol_ids) * args.valid_ratio))) if args.valid_ratio > 0 else 0
    valid_ids = set(mol_ids[:n_valid])
    train_records = [molecules[mid] for mid in mol_ids if mid not in valid_ids]
    valid_records = [molecules[mid] for mid in mol_ids if mid in valid_ids]

    train_count = write_lmdb(output_dir / "train_part_0.lmdb", train_records)
    valid_count = (
        write_lmdb(output_dir / "valid_part_0.lmdb", valid_records)
        if valid_records
        else 0
    )

    train_links = make_epoch_symlinks(output_dir, "train", args.max_epoch)
    valid_links = (
        make_epoch_symlinks(output_dir, "valid", args.max_epoch) if valid_count else []
    )

    manifest = {
        "input_sdf": str(input_sdf),
        "output_dir": str(output_dir),
        "id_prop": args.id_prop,
        "smiles_prop": args.smiles_prop,
        "seed": args.seed,
        "valid_ratio": args.valid_ratio,
        "max_epoch": args.max_epoch,
        "total_molecules": len(mol_ids),
        "total_conformers": total_conformers,
        "skipped_conformer_mismatch": skipped_mismatch,
        "train_molecules": train_count,
        "valid_molecules": valid_count,
        "train_shard": "train_part_0.lmdb",
        "valid_shard": "valid_part_0.lmdb" if valid_count else None,
        "train_epoch_symlinks": train_links,
        "valid_epoch_symlinks": valid_links,
        "record_keys": ["smi", "atoms", "coordinates"],
        "note": (
            "atoms/coordinates include hydrogens; the unimol2 task removes H and "
            "rebuilds graph features from 'smi' at training time."
        ),
    }
    (output_dir / "lmdb_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Wrote {train_count} train / {valid_count} valid molecule records to {output_dir}\n"
        f"Train shards: train_part_0.lmdb (+{len(train_links)} epoch symlinks)"
    )


if __name__ == "__main__":
    main()
