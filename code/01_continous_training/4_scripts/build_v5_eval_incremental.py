#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_v5_eval_incremental.py — incrementally add the downstream benchmark
molecules to the v5 pre-training corpus and rebuild the LMDB.

v5 already has conformers for 300k ionizable lipids (v5_il.sdf) and 700k ChEMBL
(v5_chembl.sdf) — both generated deterministically. Rather than re-running the
full ~14h build, we:
  1. give the 2679 benchmark molecules conformers + Z_DESCRIPTORS, reusing
     build_v5_dataset.prepare_v5_molecule with the EXISTING v5 z-stats
     (dataset_cards.json) so eval molecules live in the same z-space as the
     corpus (a clean ablation: the only variable is the +2679 molecule set),
  2. byte-concatenate v5_il.sdf + v5_eval.sdf + v5_chembl.sdf into a new
     lipids_unimolv2_3d.sdf (new directory unimolv2_3d_v5e/),
  3. rebuild the LMDB from the new SDF.

The old v5 artifacts are untouched — v5e lives in its own directory so the
clean-v5 baseline remains re-runnable.

Usage:
    conda run -n unimol2 python 4_scripts/build_v5_eval_incremental.py
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v5_dataset as B5
import eval_corpus
from build_v5_descriptors_lmdb import DESC_PROP

REPO_ROOT = Path(__file__).resolve().parents[1]

OLD_V5_DIR = REPO_ROOT / "1_data/processed/ionizable_lipids/unimolv2_3d_v5"
NEW_V5E_DIR = REPO_ROOT / "1_data/processed/ionizable_lipids/unimolv2_3d_v5e"

IL_SDF = OLD_V5_DIR / "v5_il.sdf"
CHEMBL_SDF = OLD_V5_DIR / "v5_chembl.sdf"
EVAL_SDF = NEW_V5E_DIR / "v5_eval.sdf"
FINAL_SDF = NEW_V5E_DIR / "lipids_unimolv2_3d.sdf"
FAILURE_CSV = NEW_V5E_DIR / "v5_eval_failures.csv"

LMDB_BUILDER = REPO_ROOT / "4_scripts/build_v5_descriptors_lmdb.py"

FAILURE_FIELDS = B5.P.FAILURE_FIELDS + ["desc_nan"]


def build_config_from_cards(cards: dict) -> dict:
    """Recreate the exact config main() uses, but with the OLD z-stats."""
    return {
        "seed": B5.CONF_SEED,
        "max_atoms": 256,
        "random_coords": True,
        "max_iterations": B5.CONF_MMFF_MAXITER,
        "embed_max_iterations": 200,
        "desc_names": cards["descriptor_names"],
        "desc_mean": np.asarray(cards["descriptor_mean"], dtype=np.float32),
        "desc_std": np.asarray(cards["descriptor_std"], dtype=np.float32),
    }


def generate_eval_sdf(eval_mols: list[dict], config: dict, num_workers: int) -> None:
    """Run prepare_v5_molecule over eval molecules, writing successful ones."""
    stats = {"success": 0, "failed": 0, "reasons": {}}
    tasks = ((i, {"mol_id": m["mol_id"], "smiles": m["smiles"],
                  "source_type": m["source_type"]}, config)
             for i, m in enumerate(eval_mols))
    with EVAL_SDF.open("w", encoding="utf-8") as fh, \
         FAILURE_CSV.open("w", newline="", encoding="utf-8") as ff:
        writer = csv.DictWriter(ff, fieldnames=FAILURE_FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = {ex.submit(B5.prepare_v5_molecule, t): t for t in tasks}
            for fut in as_completed(futures):
                res = fut.result()
                if res.status == "success":
                    stats["success"] += 1
                    B5.append_v5_success(fh, res)
                else:
                    stats["failed"] += 1
                    stats["reasons"][res.failure_reason or "unknown"] = \
                        stats["reasons"].get(res.failure_reason or "unknown", 0) + 1
                    writer.writerow({
                        "row_index": res.row_index, "mol_id": res.mol_id,
                        "smiles": res.smiles, "source_type": res.source_type,
                        "reason": res.failure_reason or "unknown",
                        "detail": res.failure_detail or "",
                        "desc_nan": "yes" if res.failure_reason == "descriptor_nan" else "",
                    })
        fh.flush()
    print(f"[eval] success={stats['success']} failed={stats['failed']} "
          f"reasons={stats['reasons']}")


def cat_streams(srcs: list[Path], dst: Path) -> None:
    """Byte-concatenate SDF files into dst (streaming, no full-RAM copy)."""
    with dst.open("wb") as out:
        for s in srcs:
            with s.open("rb") as f:
                shutil.copyfileobj(f, out, length=64 * 1024 * 1024)
    print(f"[cat ] {len(srcs)} sources -> {dst.name} "
          f"({dst.stat().st_size / 1e9:.2f} GB)")


def main() -> int:
    num_workers = max(1, (__import__("os").cpu_count() or 2) - 2)

    # 1. Load old v5 z-stats (the z-space we must share).
    cards_path = OLD_V5_DIR / "dataset_cards.json"
    cards = json.loads(cards_path.read_text(encoding="utf-8"))
    config = build_config_from_cards(cards)
    print(f"[mtr ] reusing v5 z-stats: {len(config['desc_names'])} descriptors from {cards_path.name}")

    # 2. Eval molecules (InChIKey-dedup, Sheet2+Sheet3).
    eval_mols = eval_corpus.load_eval_molecules()
    print(f"[eval] {len(eval_mols)} benchmark molecules")

    # 3. Conformer generation (reuse v5 worker exactly).
    NEW_V5E_DIR.mkdir(parents=True, exist_ok=True)
    generate_eval_sdf(eval_mols, config, num_workers)

    # 4. Concatenate into the new final SDF (old v5 untouched).
    cat_streams([IL_SDF, EVAL_SDF, CHEMBL_SDF], FINAL_SDF)

    # 5. Write new dataset_cards.json (same z-stats, updated counts).
    new_cards = dict(cards)
    new_cards["il_n"] = 300_000
    new_cards["eval_n"] = len(eval_mols)
    new_cards["note"] = (
        "v5e = v5 corpus + downstream benchmark molecules (Domain-Adaptive "
        "Pre-training). z-stats carried over from v5; only variable is the "
        "+eval molecule set."
    )
    (NEW_V5E_DIR / "dataset_cards.json").write_text(
        json.dumps(new_cards, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[cards] wrote {NEW_V5E_DIR / 'dataset_cards.json'}")

    # 6. Rebuild LMDB from the new SDF (fresh train/valid split).
    lmdb_dir = NEW_V5E_DIR / "lmdb_pretrain"
    cmd = [
        sys.executable, str(LMDB_BUILDER),
        "--input-sdf", str(FINAL_SDF),
        "--output-dir", str(lmdb_dir),
        "--valid-ratio", "0.01",
        "--seed", str(B5.CONF_SEED),
        "--max-epoch", "10",
    ]
    print("[lmdb] building:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Sanity: report LMDB manifest counts.
    manifest = lmdb_dir / "lmdb_manifest.json"
    if manifest.exists():
        m = json.loads(manifest.read_text(encoding="utf-8"))
        print(f"[lmdb] train={m['train_molecules']:,} valid={m['valid_molecules']:,} "
              f"total={m['total_molecules']:,} conformers={m['total_conformers']:,} "
              f"desc_n={m['descriptor_n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
