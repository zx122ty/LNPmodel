#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How much of what the 3D model sees is actually flat? Measure it, don't estimate.

`05-pKa评估结果.md` §2.4 carries a number every neural-arm result inherits: on
pKa, 41.5% of the molecules handed to Uni-Mol2 are **2D coordinates** -- the
conformer generator fell back to the flat depiction. That is a protocol fact
about the whole neural column, so it has to be reproducible by a reviewer, and
until now it was not: the number existed only as prose, with no script and no
output file behind it (found by grep, 2026-09-18).

The measurement is taken from MolTrain's own input, not from a re-derivation:
`finetune_moltrain` writes `<run_dir>/<dataset>_data.sdf` after its conformer
pass, and that file IS the model's input. Re-generating conformers here could
disagree with it, and then the number would describe a pipeline nobody ran.

The signature of the fallback is the **z-extent**: a molecule that got real 3D
coordinates has depth; a molecule that fell back to a flat depiction has all its
atoms in one plane (z-extent 0). On pKa the two populations are perfectly
separated -- 238 molecules at z < 0.1 A and the rest at a median of 7.3 A -- so
no threshold sensitivity to argue about.

Note what this is NOT: evidence that fixing it would help. The project measured
that (CLAUDE.md 已证伪表: real 3D instead of the fallback scored -0.040). It is
a statement about the conditions the published numbers were obtained under.

Usage:
    python 4_scripts/downstream/conformer_fallback_probe.py
    python 4_scripts/downstream/conformer_fallback_probe.py --sdf <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
sys.path.insert(0, str(ROOT / "4_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ds_registry  # noqa: E402

RUNS = ROOT / "5_outputs/finetune_moltrain"
OUT = ROOT / "5_outputs/downstream_results/conformer_fallback.json"

# A molecule is "flat" when its atoms span no depth. 0.1 A is 1/73 of the median
# z-extent of the non-flat population (7.27 A), so the cut sits in an empty gap
# rather than on a slope -- the histogram is bimodal, not continuous.
FLAT_Z_EXTENT = 0.1


def probe(sdf: Path) -> dict:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    zs = []
    n_unreadable = 0
    for mol in Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=False):
        if mol is None:
            n_unreadable += 1
            continue
        if mol.GetNumConformers() == 0:
            zs.append(0.0)
            continue
        pos = mol.GetConformer().GetPositions()
        zs.append(float(pos[:, 2].max() - pos[:, 2].min()))
    arr = np.asarray(zs, float)
    flat = arr < FLAT_Z_EXTENT
    return {
        "sdf": str(sdf),
        "n_molecules": int(len(arr)),
        "n_unreadable": n_unreadable,
        "n_flat_2d": int(flat.sum()),
        "frac_flat_2d": round(float(flat.mean()), 4),
        "z_extent_median": round(float(np.median(arr)), 3),
        "z_extent_mean": round(float(arr.mean()), 3),
        "z_extent_p10": round(float(np.percentile(arr, 10)), 3),
        "z_extent_max": round(float(arr.max()), 3),
        # Reported so a reader can see the two populations are separated rather
        # than trusting the cutoff: the flat group is exactly the p10 == 0 group.
        "z_extent_median_nonflat": round(float(np.median(arr[~flat])), 3) if (~flat).any() else None,
    }


def find_default_sdf() -> Path:
    """MolTrain's input SDF from a completed pKa run (any split; the conformer
    pass is split-independent -- it is a property of the molecule set)."""
    cands = sorted(RUNS.glob("pKa_exp_*butina_seed42*/pKa_data.sdf"))
    cands = [c for c in cands if ".BUGGY_" not in str(c)]
    if not cands:
        raise SystemExit("no completed pKa run dir with pKa_data.sdf found; pass --sdf")
    return cands[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdf", type=Path, default=None,
                    help="MolTrain input SDF; defaults to the newest completed pKa run")
    ap.add_argument("--json-out", type=Path, default=OUT)
    a = ap.parse_args()

    sdf = a.sdf or find_default_sdf()
    rec = probe(sdf)

    _df, target_cols = ds_registry.load("pKa", verbose=False)
    rec["dataset"] = "pKa"
    rec["n_dataset"] = len(_df)
    rec["target"] = target_cols[0]

    print(f"{sdf}")
    print(f"  molecules        {rec['n_molecules']}  (unreadable {rec['n_unreadable']})")
    print(f"  z-extent         median={rec['z_extent_median']} A  mean={rec['z_extent_mean']}  "
          f"p10={rec['z_extent_p10']}  max={rec['z_extent_max']}")
    print(f"  flat (z < {FLAT_Z_EXTENT} A)  {rec['n_flat_2d']}  "
          f"= {rec['frac_flat_2d']*100:.1f}% of the molecules the 3D model sees")
    if rec["n_molecules"] != rec["n_dataset"]:
        print(f"  ⚠️ SDF has {rec['n_molecules']} molecules but the dataset has "
              f"{rec['n_dataset']} -- the probe would describe the wrong molecule set")

    a.json_out.parent.mkdir(parents=True, exist_ok=True)
    a.json_out.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"\nwrote {a.json_out}")


if __name__ == "__main__":
    main()
