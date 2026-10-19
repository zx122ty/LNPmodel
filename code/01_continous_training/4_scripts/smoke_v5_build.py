#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
smoke_v5_build.py — minimal v5 data-build smoke test.

Purposely small:
  * 4 ionizable lipids + 4 ChEMBL molecules (full conformer budget),
  * asserts SDF shape (per-mol conformer counts, Z_DESCRIPTORS present),
  * reuses build_v5_dataset functions directly (no multiprocessing).

Run:  conda activate unimol2 && python smoke_v5_build.py
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_v5_dataset import (  # noqa: E402
    CONF_PLAN,
    _DEFAULT_DESC_NAMES,
    compute_descriptors,
    prepare_v5_molecule,
    read_il_corpus,
    sample_chembl,
    DEFAULT_IL_CSV,
    DEFAULT_CHEMBL_TXT,
)

OUT = Path(tempfile.mkdtemp(prefix="v5_smoke_"))
IL_CSV = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/00_prepare_ILs_vitual_library_v2/"
              "curate_library/generated_virtual_library_balanced.csv")
CHEMBL_TXT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training/"
                  "1_data/raw/CHEMBLdb/chembl_37_chemreps.txt.gz")


def main() -> int:
    random.seed(0)
    il = read_il_corpus(IL_CSV, max_rows=4)
    cb = sample_chembl(CHEMBL_TXT, 4, seed=1)  # deterministic reservoir sample
    assert len(il) == 4 and len(cb) == 4, f"got il={len(il)} cb={len(cb)}"

    # descriptor sanity
    all_desc = []
    for r in il + cb:
        v = compute_descriptors(r["smiles"], _DEFAULT_DESC_NAMES)
        assert all(np.isfinite(v)), f"NaN descriptors for {r['smiles'][:40]}"
        all_desc.append(v)
    big = np.vstack(all_desc)
    print(f"descriptors: n={len(_DEFAULT_DESC_NAMES)} "
          f"min/max over sample = {big.min():.2f} / {big.max():.2f}")

    mean = big.mean(axis=0)
    std = big.std(axis=0)
    std[std < 1e-8] = 1.0

    config = {
        "seed": 20260805, "max_atoms": 256, "random_coords": True,
        "max_iterations": 2000, "embed_max_iterations": 200,
        "desc_names": _DEFAULT_DESC_NAMES, "desc_mean": mean.astype(np.float32),
        "desc_std": std.astype(np.float32),
    }

    sdf = OUT / "smoke.sdf"
    import csv
    failures = OUT / "failures.csv"
    with sdf.open("w", encoding="utf-8") as fh, \
         failures.open("w", newline="", encoding="utf-8") as ff:
        writer = csv.DictWriter(ff, fieldnames=["row_index", "mol_id", "smiles",
                                                "source_type", "reason", "detail"])
        writer.writeheader()
        from build_v5_dataset import append_v5_success, prepare_v5_molecule
        for i, r in enumerate(il + cb):
            result = prepare_v5_molecule((i, r, config))
            if result.status == "success":
                n = append_v5_success(fh, result)
                print(f"  OK {r['source_type']:8s} {r['smiles'][:52]}… confs={n}")
            else:
                print(f"  FAIL {r['source_type']:8s} {r['smiles'][:52]}… "
                      f"reason={result.failure_reason} detail={result.failure_detail}")
                writer.writerow({"row_index": result.row_index, "mol_id": result.mol_id,
                                 "smiles": result.smiles, "source_type": result.source_type,
                                 "reason": result.failure_reason, "detail": result.failure_detail})

    print(f"conformer blocks: {sum(1 for _ in sdf.open()) }")
    print(f"SDF: {sdf} ({sdf.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())