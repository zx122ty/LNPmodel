"""run_all.py — one-command reproducible run of the whole benchmark.

Runs the full pipeline from a clean state and regenerates every deliverable:

  Stage 0  EDA                           scripts/explore_dataset.py
  Stage 2  baseline models               scripts/train.py
  Stage 3  uncertainty (GPR/QRF)         scripts/uncertainty.py
  Stage 4  structural robustness/OOD     scripts/structural_robustness.py
  Stage 5  final model + calibrate       scripts/train_final.py
  Stage 5  score defaults                scripts/predict.py
  (a)      Excel workbook                scripts/export_excel.py
  (c)      calibration diagnostics       scripts/calibration_diagnostics.py
  (b)      optional DL/MLP exploration   scripts/deep_ensemble_explore.py  [--skip-explore to omit]

Run from benchmark/code/:
    python run_all.py                      # full incl. optional MLP exploration
    python run_all.py --skip-explore       # skip the optional DL comparison
    python run_all.py --skip-uncertainty   # skip Stage 3 (faster smoke)
    python run_all.py --seed 123           # override base seed (default 42)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCRIPTS = REPO / "scripts"


def _run(name: str, args: list[str], timeout: int = 1800):
    print(f"\n{'='*70}\n▶ {name}\n{'='*70}")
    t0 = time.time()
    proc = subprocess.run([sys.executable, str(SCRIPTS / f"{name}.py"), *args],
                          cwd=REPO)
    if proc.returncode != 0:
        print(f"✗ {name} FAILED (exit {proc.returncode})")
        sys.exit(proc.returncode)
    print(f"✓ {name} done in {(time.time()-t0)/60:.1f} min")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-explore", action="store_true")
    parser.add_argument("--skip-uncertainty", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = [str(args.seed)]

    _run("explore_dataset", [])
    _run("train", ["--seeds", *seed])
    if not args.skip_uncertainty:
        _run("uncertainty", ["--seeds", "42", "123", "456", "2026"])
    _run("structural_robustness", ["--seeds", "42", "123"])
    _run("train_final", ["--seed", str(args.seed)])
    # score the default workbook rows (predict.py default uses load_dataset)
    _run("predict", [])
    _run("export_excel", [])
    _run("calibration_diagnostics", [])
    if not args.skip_explore:
        _run("deep_ensemble_explore", ["--members", "10"])

    print(f"\n{'='*70}\n✅ All stages complete. Outputs → {REPO/'outputs'}")
    print(f"Excel deliverable → {REPO/'outputs'/'lnp_benchmark_results.xlsx'}")


if __name__ == "__main__":
    main()