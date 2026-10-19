"""Ionizable-lipid LNP foundation-model pipeline entry point.

Usage
-----
Run all steps:
    python main.py

Run a single step:
    python main.py --step prepare
    python main.py --step pretrain

Use a custom config:
    python main.py --config 3_configs/pipeline.yaml --step pretrain

Smoke test (small sample, no training):
    python main.py --step prepare --set prepare.sample_size=200
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "3_configs/pipeline.yaml"

STEPS = ("prepare", "pretrain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ionizable-lipid pipeline runner.")
    parser.add_argument("--step", choices=STEPS, default=None,
                        help="Run a single step. Omit to run all steps in order.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG,
                        help="Pipeline YAML config (default: 3_configs/pipeline.yaml).")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="Override config values, e.g. --set prepare.sample_size=200")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def _flag(cmd: list[str], name: str, value) -> None:
    """Append --name value unless value is None."""
    if value is not None:
        cmd += [name, str(value)]


def run_prepare(cfg) -> None:
    c = cfg.prepare
    cmd = [
        sys.executable,
        str(REPO_ROOT / "4_scripts/prepare_lipids_unimol_dataset.py"),
        "--input", str(REPO_ROOT / c.input),
        "--output-dir", str(REPO_ROOT / c.output_dir),
        "--smiles-col", c.smiles_col,
        "--id-col", c.id_col,
        "--source-col", c.source_col,
        "--num-conformers-real", str(c.num_conformers_real),
        "--keep-conformers-real", str(c.keep_conformers_real),
        "--num-conformers-virtual", str(c.num_conformers_virtual),
        "--keep-conformers-virtual", str(c.keep_conformers_virtual),
        "--max-iterations", str(c.max_iterations),
        "--max-atoms", str(c.max_atoms),
        "--seed", str(c.seed),
        "--sample-seed", str(c.sample_seed),
    ]
    _flag(cmd, "--num-workers", c.num_workers)
    _flag(cmd, "--max-molecules", c.max_molecules)
    _flag(cmd, "--sample-size", c.sample_size)
    if not c.random_coords:
        cmd += ["--no-random-coords"]
    if c.resume:
        cmd += ["--resume"]
    if c.get("chembl_input") is not None:
        cmd += ["--chembl-input", str(REPO_ROOT / c.chembl_input),
                "--chembl-ratio", str(c.chembl_ratio)]
        _flag(cmd, "--chembl-max", c.get("chembl_max"))
        _flag(cmd, "--chembl-seed", c.get("chembl_seed"))
    _run(cmd)


def run_pretrain(cfg) -> None:
    c = cfg.pretrain
    runner = getattr(c, "runner", "run_lipids_continue_pretrain")
    script = REPO_ROOT / "4_scripts/run_v5_continue_pretrain.py" if runner == "v5" else \
        REPO_ROOT / "4_scripts/run_lipids_continue_pretrain.py"
    cmd = [
        sys.executable,
        str(script),
        "--train-sdf", str(REPO_ROOT / c.train_sdf),
        "--output-dir", str(REPO_ROOT / c.output_dir),
        "--pretrained-checkpoint", str(REPO_ROOT / c.pretrained_checkpoint),
        "--model-size", c.model_size,
        "--valid-ratio", str(c.valid_ratio),
        "--max-epoch", str(c.max_epoch),
        "--batch-size", str(c.batch_size),
        "--update-freq", str(c.update_freq),
        "--learning-rate", str(c.learning_rate),
        "--weight-decay", str(c.weight_decay),
        "--warmup-steps", str(c.warmup_steps),
        "--total-steps", str(c.total_steps),
        "--save-every-n-steps", str(c.save_every_n_steps),
        "--validate-every-n-steps", str(c.validate_every_n_steps),
        "--log-every-n-steps", str(c.log_every_n_steps),
        "--keep-interval-updates", str(c.keep_interval_updates),
        "--num-workers", str(c.num_workers),
        "--masked-token-loss", str(c.masked_token_loss),
        "--masked-coord-loss", str(c.masked_coord_loss),
        "--masked-dist-loss", str(c.masked_dist_loss),
        "--mask-token-prob", str(c.mask_token_prob),
        "--mask-prob", str(c.mask_prob),
        "--drop-feat-prob", str(c.drop_feat_prob),
        "--noise", str(c.noise),
        "--noise-type", c.noise_type,
        "--max-atoms", str(c.max_atoms),
        "--seed", str(c.seed),
    ]
    # v5-only / MTR options
    if runner == "v5":
        cmd += ["--mtr-loss-weight", str(c.get("mtr_loss_weight", 1.0))]
        if c.get("descriptors_json") is not None:
            cmd += ["--descriptors-json", str(REPO_ROOT / c.descriptors_json)]
        # v6: paper-style MTR head options (defaults = v5 behaviour)
        cmd += ["--mtr-pooling", str(c.get("mtr_pooling", "cls"))]
        cmd += ["--mtr-hidden-dim", str(c.get("mtr_hidden_dim", 256))]
        cmd += ["--mtr-activation-fn", str(c.get("mtr_activation_fn", "tanh"))]
        cmd += ["--mtr-dropout", str(c.get("mtr_dropout", 0.0))]
    if c.lmdb_dir is not None:
        cmd += ["--lmdb-dir", str(REPO_ROOT / c.lmdb_dir)]
    if c.prepare_valid_split:
        cmd += ["--prepare-valid-split"]
    if not c.fp16:
        cmd += ["--no-fp16"]
    if c.dry_run:
        cmd += ["--dry-run"]
    _run(cmd)


STEP_FNS = {"prepare": run_prepare, "pretrain": run_pretrain}


def main() -> None:
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    if args.set:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.set))

    # prepare.max_atoms > pretrain.max_atoms would let CroppingDataset silently
    # discard atoms from molecules that passed conformer generation.
    if cfg.prepare.max_atoms > cfg.pretrain.max_atoms:
        sys.exit(
            f"Config error: prepare.max_atoms ({cfg.prepare.max_atoms}) exceeds "
            f"pretrain.max_atoms ({cfg.pretrain.max_atoms}). Uni-Mol2 would "
            f"randomly drop atoms from oversized molecules at training time."
        )

    for step in ([args.step] if args.step else list(STEPS)):
        print(f"\n{'=' * 60}\n  Step: {step}\n{'=' * 60}")
        STEP_FNS[step](cfg)


if __name__ == "__main__":
    main()
