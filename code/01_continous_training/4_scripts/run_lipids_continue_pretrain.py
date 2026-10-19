"""Run UniMol2 continued pretraining (domain adaptation) on the prepared
ionizable-lipid dataset, using the official Uni-Mol2 code + Uni-Core's
``unicore-train`` CLI.

Why this script exists
----------------------
The pretrained checkpoints under ``2_models/pretrained`` are *genuine UniMol2*
two-track models. The weight-complete path is the official UniMol2 implementation
driven by Uni-Core:

  * model/task/loss registered as ``unimol2`` in
    ``4_scripts/Uni-Mol/unimol2/unimol2`` (passed via ``--user-dir``),
  * continued pretraining = ``--finetune-from-model <ckpt>`` (Uni-Core loads the
    weights ``strict=False`` and resets optimizer / lr-scheduler / epoch),
  * enabling ``--masked-token-loss > 0`` builds the model's ``lm_head`` so the
    full checkpoint loads with **zero** missing/unexpected keys.

This script (1) optionally builds the epoch-sharded LMDB the task expects from a
prepared SDF, (2) assembles the ``unicore-train`` argv, and (3) execs it.

Lipid-specific defaults (differ from the source toxicity project):
  * ``--max-atoms 256`` — Uni-Mol2's CroppingDataset randomly *drops atoms* above
    this budget rather than erroring. The lipid corpus reaches 208 heavy atoms,
    so 128 would silently corrupt ~2.5% of molecules.
  * ``--learning-rate 5e-5`` — half the source value; the lipid corpus is ~10x
    smaller, so a gentler rate reduces catastrophic-forgetting risk.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UNIMOL2_USER_DIR = Path(os.environ.get(
    "LNP_UNIMOL2_USER_DIR",
    REPO_ROOT / "4_scripts/Uni-Mol/unimol2/unimol2",
))
BUILD_LMDB_SCRIPT = REPO_ROOT / "4_scripts/build_lipids_pretrain_lmdb.py"

DEFAULT_TRAIN_SDF = (
    REPO_ROOT / "1_data/processed/ionizable_lipids/unimolv2_3d/lipids_unimolv2_3d.sdf"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "2_models/continued_pretrain/unimol2_lipids_84m"
DEFAULT_PRETRAINED_CKPT = REPO_ROOT / "2_models/pretrained/checkpoint_84M.pt"
DEFAULT_VALID_RATIO = 0.01

# --model-size -> registered Uni-Core architecture name.
ARCH_BY_SIZE = {
    "84m": "unimol2_84M",
    "164m": "unimol2_164M",
    "310m": "unimol2_310M",
    "570m": "unimol2_570M",
    "1.1b": "unimol2_1100M",
}
# Embedding dim per arch, used to sanity-check --model-size against the checkpoint.
EMBED_DIM_BY_SIZE = {
    "84m": 768,
    "164m": 768,
    "310m": 1024,
    "570m": 1536,
    "1.1b": 1536,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UniMol2 continued pretraining on a prepared DSSTox dataset."
    )
    parser.add_argument("--train-sdf", type=Path, default=DEFAULT_TRAIN_SDF)
    parser.add_argument(
        "--lmdb-dir",
        type=Path,
        default=None,
        help="Directory with {train,valid}_part_*.lmdb shards. If omitted, it is "
        "derived from --train-sdf (a sibling 'lmdb_pretrain' dir) and built on demand.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAINED_CKPT)
    parser.add_argument("--model-size", choices=sorted(ARCH_BY_SIZE.keys()), default="84m")

    # Optimization / schedule.
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--update-freq", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--total-steps", type=int, default=10000)
    parser.add_argument("--max-epoch", type=int, default=10,
                        help="Number of epoch shards to expose when building LMDB.")
    parser.add_argument("--save-every-n-steps", type=int, default=1000)
    parser.add_argument("--validate-every-n-steps", type=int, default=1000)
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    parser.add_argument("--keep-interval-updates", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)

    # Masking / denoising (UniMol2 full-denoising pretraining objective, 1/1/1).
    parser.add_argument("--masked-token-loss", type=float, default=1.0)
    parser.add_argument("--masked-coord-loss", type=float, default=1.0)
    parser.add_argument("--masked-dist-loss", type=float, default=1.0)
    parser.add_argument("--mask-token-prob", type=float, default=0.15)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--drop-feat-prob", type=float, default=0.5)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--noise-type", choices=["trunc_normal", "uniform", "normal", "none"], default="uniform")
    parser.add_argument("--max-atoms", type=int, default=256)

    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")

    # Data prep.
    parser.add_argument("--valid-ratio", type=float, default=DEFAULT_VALID_RATIO)
    parser.add_argument("--prepare-valid-split", action="store_true",
                        help="(Re)build the LMDB from --train-sdf before training.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the unicore-train command and metadata, then exit.")
    return parser.parse_args()


def detect_embed_dim(checkpoint_path: Path) -> int:
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    return int(state["embed_tokens.weight"].shape[1])


def ensure_lmdb(args: argparse.Namespace, lmdb_dir: Path) -> None:
    """Build the epoch-sharded LMDB from the SDF if missing or requested."""
    train_shard = lmdb_dir / "train_part_0.lmdb"
    need_build = args.prepare_valid_split or not train_shard.exists()
    if not need_build:
        print(f"Reusing existing LMDB at {lmdb_dir}")
        return
    if not args.train_sdf.exists():
        raise FileNotFoundError(f"Training SDF not found: {args.train_sdf}")
    cmd = [
        sys.executable,
        str(BUILD_LMDB_SCRIPT),
        "--input-sdf", str(args.train_sdf),
        "--output-dir", str(lmdb_dir),
        "--valid-ratio", str(args.valid_ratio),
        "--seed", str(args.seed),
        "--max-epoch", str(args.max_epoch),
    ]
    print("Building LMDB:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def build_train_command(args: argparse.Namespace, lmdb_dir: Path, ckpt_dir: Path) -> list[str]:
    arch = ARCH_BY_SIZE[args.model_size]
    has_valid = (lmdb_dir / "valid_part_0.lmdb").exists()
    cmd = [
        "unicore-train",
        str(lmdb_dir),
        "--user-dir", str(UNIMOL2_USER_DIR),
        "--task", "unimol2",
        "--loss", "unimol2",
        "--arch", arch,
        "--mode", "train",
        # Single-process / single-GPU: world-size 1 takes Uni-Core's clean
        # main(args) path (no torchrun / spawn needed).
        "--distributed-world-size", "1",
        "--num-workers", str(args.num_workers),
        "--ddp-backend", "c10d",
        # Continued pretraining: load checkpoint weights, reset optim/sched/epoch.
        "--finetune-from-model", str(args.pretrained_checkpoint),
        # Pretraining objective (full denoising, weights 1/1/1).
        "--masked-token-loss", str(args.masked_token_loss),
        "--masked-coord-loss", str(args.masked_coord_loss),
        "--masked-dist-loss", str(args.masked_dist_loss),
        "--mask-token-prob", str(args.mask_token_prob),
        "--mask-prob", str(args.mask_prob),
        "--drop-feat-prob", str(args.drop_feat_prob),
        "--noise", str(args.noise),
        "--noise-type", args.noise_type,
        "--max-atoms", str(args.max_atoms),
        # Optimizer + polynomial decay schedule.
        "--optimizer", "adam",
        "--adam-betas", "(0.9, 0.99)",
        "--adam-eps", "1e-6",
        "--weight-decay", str(args.weight_decay),
        "--clip-norm", "1.0",
        "--lr-scheduler", "polynomial_decay",
        "--lr", str(args.learning_rate),
        "--warmup-updates", str(args.warmup_steps),
        "--total-num-update", str(args.total_steps),
        "--max-update", str(args.total_steps),
        "--batch-size", str(args.batch_size),
        "--update-freq", str(args.update_freq),
        "--seed", str(args.seed),
        # Checkpointing / logging.
        "--save-dir", str(ckpt_dir),
        "--tmp-save-dir", str(ckpt_dir / "tmp"),
        "--save-interval-updates", str(args.save_every_n_steps),
        "--keep-interval-updates", str(args.keep_interval_updates),
        "--log-interval", str(args.log_every_n_steps),
        "--log-format", "simple",
        "--train-subset", "train",
    ]
    if args.fp16:
        cmd += ["--fp16", "--fp16-init-scale", "4", "--fp16-scale-window", "256"]
    if has_valid:
        cmd += ["--valid-subset", "valid",
                "--validate-interval-updates", str(args.validate_every_n_steps)]
    else:
        # No valid shard: Uni-Core still loads --valid-subset once at startup, so
        # point it at 'train' and disable periodic validation to avoid a crash.
        cmd += ["--valid-subset", "train", "--disable-validation"]
    return cmd


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.train_sdf = args.train_sdf.resolve()
    args.pretrained_checkpoint = args.pretrained_checkpoint.resolve()

    if not args.pretrained_checkpoint.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_checkpoint}")

    # Validate --model-size against the checkpoint embedding dim.
    embed_dim = detect_embed_dim(args.pretrained_checkpoint)
    expected = EMBED_DIM_BY_SIZE[args.model_size]
    if embed_dim != expected:
        raise ValueError(
            f"Checkpoint embed dim {embed_dim} does not match --model-size "
            f"'{args.model_size}' (expects {expected}). Pick the matching size."
        )

    lmdb_dir = (args.lmdb_dir.resolve() if args.lmdb_dir
                else (args.train_sdf.parent / "lmdb_pretrain"))
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    lmdb_tmp = args.output_dir / "lmdb_tmp"
    lmdb_tmp.mkdir(parents=True, exist_ok=True)

    if not args.dry_run:
        ensure_lmdb(args, lmdb_dir)

    cmd = build_train_command(args, lmdb_dir, ckpt_dir)

    metadata = {
        "train_sdf": str(args.train_sdf),
        "lmdb_dir": str(lmdb_dir),
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
        "model_size": args.model_size,
        "arch": ARCH_BY_SIZE[args.model_size],
        "output_dir": str(args.output_dir),
        "checkpoints_dir": str(ckpt_dir),
        "objective": {
            "masked_token_loss": args.masked_token_loss,
            "masked_coord_loss": args.masked_coord_loss,
            "masked_dist_loss": args.masked_dist_loss,
        },
        "total_steps": args.total_steps,
        "warmup_steps": args.warmup_steps,
        "batch_size": args.batch_size,
        "update_freq": args.update_freq,
        "learning_rate": args.learning_rate,
        "fp16": args.fp16,
        "unicore_train_cmd": cmd,
    }
    (args.output_dir / "continued_pretrain_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if args.dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        print("\nDRY RUN — command:\n  " + " ".join(cmd))
        return

    # Redirect the unimol2 LMDBDataset's per-epoch copy dir to a writable path
    # (upstream hardcodes /temp/, which is the non-writable filesystem root here).
    env = dict(os.environ)
    env["UNIMOL2_LMDB_TMP_DIR"] = str(lmdb_tmp)
    # Make the unimol2 package importable as a user-dir module.
    env["PYTHONPATH"] = str(UNIMOL2_USER_DIR.parent) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    print("Launching continued pretraining:\n  " + " ".join(cmd))
    # Run from the output dir so any relative artifacts land there.
    proc = subprocess.run(cmd, env=env, cwd=str(args.output_dir))
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
