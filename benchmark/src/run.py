#!/usr/bin/env python3
"""
run.py — Entry point for the LNP Benchmark Pipeline.

Usage:
    python run.py                              # full benchmark
    python run.py --viz-only ./results         # standalone visualisation
    python run.py --config path/to.yaml        # custom config
    python run.py --n-jobs 8                   # override parallel workers
    python run.py --models XGBoost LightGBM    # only specific models
    python run.py --splits random scaffold     # specific split methods
    python run.py --no-viz                     # skip plot generation
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lnp_benchmark.cli.run_benchmark import BenchmarkPipeline
from lnp_benchmark.eval.visualize import generate_all_plots_standalone


def main():
    parser = argparse.ArgumentParser(description="LNP Benchmark Pipeline")
    parser.add_argument(
        "--config", type=str,
        default="lnp_benchmark/configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=None,
        help="Parallel workers (-1 = all CPUs)",
    )
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        help="Models to run (e.g. XGBoost LightGBM)",
    )
    parser.add_argument(
        "--splits", type=str, nargs="+", default=None,
        help="Split methods (e.g. random scaffold cluster)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Random seeds (e.g. 42 123 456)",
    )
    parser.add_argument(
        "--viz-only", type=str, default=None,
        help="Standalone mode: path to results dir, generates plots only",
    )
    parser.add_argument(
        "--no-viz", action="store_true",
        help="Skip plot generation after run",
    )
    args = parser.parse_args()

    # ── Standalone visualisation mode ────────────────────────────────────
    if args.viz_only:
        results_dir = args.viz_only
        print(f"Standalone visualisation mode — reading from {results_dir}")
        files = generate_all_plots_standalone(results_dir)
        print(f"Generated {len(files)} plots:")
        for f in files:
            print(f"  {f}")
        return

    # ── Full benchmark mode ──────────────────────────────────────────────
    pipeline = BenchmarkPipeline(args.config)

    # CLI overrides
    if args.n_jobs is not None:
        pipeline.cfg["execution"]["n_jobs"] = args.n_jobs
    if args.models:
        pipeline.cfg["models"]["tier1"]["models"] = args.models
    if args.splits:
        pipeline.cfg["splits"]["methods"] = args.splits
    if args.seeds:
        pipeline.cfg["splits"]["seeds"] = args.seeds
    if args.no_viz:
        pipeline.cfg["eval"]["visualize"]["enabled"] = False

    pipeline.run()


if __name__ == "__main__":
    main()