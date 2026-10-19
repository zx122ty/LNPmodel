#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_v5_continue_pretrain.py — launch LipidFM v5 continued pre-training with the
MTR objective (masked-token/coord/dist + descriptor regression on [CLS]).

Mirrors run_lipids_continue_pretrain.py but:
  * uses the v5 LMDB (built by build_v5_descriptors_lmdb.py from the v5 SDF),
  * uses the project-unique ``--user-dir unimol2_mtr`` package (task/loss/model),
  * passes --task unimol2_mtr --loss unimol2_mtr --arch unimol2_mtr_84M,
  * wires --mtr-loss-weight and --descriptors-json (the LMDB manifest's
    descriptor metadata double as the descriptors JSON).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MTR_USER_DIR = REPO_ROOT / "4_scripts/Uni-Mol/unimol2/unimol2_mtr"
BASE_USER_DIR = REPO_ROOT / "4_scripts/Uni-Mol/unimol2/unimol2"
NEST_ROOT = BASE_USER_DIR.parent  # dir containing unimol2, unimol2_mtr, unimol2_v5
# unicore's import_user_module imports <user-dir basename>. Pointing --user-dir at
# the v5 shim directory makes unicore import "unimol2_v5", whose side effect
# registers BOTH base + MTR. PYTHONPATH adds NEST_ROOT so the shim can resolve
# `import unimol2` / `import unimol2_mtr`.
SHIM_USER_DIR = REPO_ROOT / "4_scripts/Uni-Mol/unimol2/unimol2_v5"
BUILD_LMDB_SCRIPT = REPO_ROOT / "4_scripts/build_v5_descriptors_lmdb.py"

DEFAULT_TRAIN_SDF = (
    REPO_ROOT / "1_data/processed/ionizable_lipids/unimolv2_3d_v5/lipids_unimolv2_3d.sdf"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "2_models/continued_pretrain/unimol2_lipids_84m_v5"
DEFAULT_PRETRAINED_CKPT = REPO_ROOT / "2_models/pretrained/checkpoint_84M.pt"

ARCH_BY_SIZE = {
    "84m": "unimol2_mtr_84M",
    "164m": "unimol2_mtr_164M",
    "310m": "unimol2_mtr_310M",
    "570m": "unimol2_mtr_570M",
    "1.1b": "unimol2_mtr_1100M",
}
EMBED_DIM_BY_SIZE = {"84m": 768, "164m": 768, "310m": 1024, "570m": 1536, "1.1b": 1536}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LipidFM v5 continued pretraining (MTR).")
    parser.add_argument("--train-sdf", type=Path, default=DEFAULT_TRAIN_SDF)
    parser.add_argument("--lmdb-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pretrained-checkpoint", type=Path, default=DEFAULT_PRETRAINED_CKPT)
    parser.add_argument("--model-size", choices=sorted(ARCH_BY_SIZE.keys()), default="84m")

    # optimisation / schedule
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--update-freq", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--total-steps", type=int, default=30000)
    parser.add_argument("--max-epoch", type=int, default=10)
    parser.add_argument("--save-every-n-steps", type=int, default=1000)
    parser.add_argument("--validate-every-n-steps", type=int, default=1000)
    parser.add_argument("--log-every-n-steps", type=int, default=20)
    parser.add_argument("--keep-interval-updates", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=8)

    # UniMol2 denoising objective
    parser.add_argument("--masked-token-loss", type=float, default=1.0)
    parser.add_argument("--masked-coord-loss", type=float, default=1.0)
    parser.add_argument("--masked-dist-loss", type=float, default=1.0)
    parser.add_argument("--mask-token-prob", type=float, default=0.15)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--drop-feat-prob", type=float, default=0.5)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--noise-type",
                        choices=["trunc_normal", "uniform", "normal", "none"], default="uniform")
    parser.add_argument("--max-atoms", type=int, default=256)

    # ── MTR ────────────────────────────────────────────────────────────────
    parser.add_argument("--mtr-loss-weight", type=float, default=1.0,
                        help="weight of the descriptor-regression loss (0 disables MTR).")
    parser.add_argument("--descriptors-json", type=Path, default=None,
                        help="descriptor metadata (names/mean/std). Defaults to the "
                             "LMDB manifest in --lmdb-dir.")
    # v6: paper-style MTR head options (defaults keep the v5 behaviour)
    parser.add_argument("--mtr-pooling", type=str, default="cls",
                        choices=["cls", "mean"],
                        help="MTR readout: [CLS] or mean pooling (paper DA uses mean).")
    parser.add_argument("--mtr-hidden-dim", type=int, default=256,
                        help="MTR head hidden dim (paper uses 2-layer MLP).")
    parser.add_argument("--mtr-activation-fn", type=str, default="tanh",
                        help="MTR head activation (paper uses relu).")
    parser.add_argument("--mtr-dropout", type=float, default=0.0,
                        help="MTR head dropout (paper uses 0.1).")

    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--valid-ratio", type=float, default=0.01)
    parser.add_argument("--prepare-valid-split", action="store_true",
                        help="(Re)build the LMDB from --train-sdf before training.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def detect_embed_dim(checkpoint_path: Path) -> int:
    import torch
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    return int(state["embed_tokens.weight"].shape[1])


def ensure_lmdb(args: argparse.Namespace, lmdb_dir: Path) -> Path | None:
    """Build the v5 LMDB from the SDF if missing/requested; return descriptors-json."""
    train_shard = lmdb_dir / "train_part_0.lmdb"
    need_build = args.prepare_valid_split or not train_shard.exists()
    if not need_build:
        print(f"Reusing existing LMDB at {lmdb_dir}")
    else:
        if not args.train_sdf.exists():
            raise FileNotFoundError(f"Training SDF not found: {args.train_sdf}")
        cmd = [
            sys.executable, str(BUILD_LMDB_SCRIPT),
            "--input-sdf", str(args.train_sdf),
            "--output-dir", str(lmdb_dir),
            "--valid-ratio", str(args.valid_ratio),
            "--seed", str(args.seed),
            "--max-epoch", str(args.max_epoch),
        ]
        print("Building LMDB:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    manifest = lmdb_dir / "lmdb_manifest.json"
    if args.descriptors_json is not None:
        return args.descriptors_json
    if manifest.exists():
        return manifest
    return None


def build_train_command(args: argparse.Namespace, lmdb_dir: Path, ckpt_dir: Path,
                        desc_json: Path | None) -> list[str]:
    arch = ARCH_BY_SIZE[args.model_size]
    has_valid = (lmdb_dir / "valid_part_0.lmdb").exists()
    cmd = [
        "unicore-train", str(lmdb_dir),
        # unicore imports <user-dir basename>; the shim (unimol2_v5) registers
        # ALL of base + MTR. See SHIM_USER_DIR comment.
        "--user-dir", str(SHIM_USER_DIR),
        "--task", "unimol2_mtr",
        "--loss", "unimol2_mtr",
        "--arch", arch,
        "--mode", "train",
        "--distributed-world-size", "1",
        "--num-workers", str(args.num_workers),
        "--ddp-backend", "c10d",
        "--finetune-from-model", str(args.pretrained_checkpoint),
        "--masked-token-loss", str(args.masked_token_loss),
        "--masked-coord-loss", str(args.masked_coord_loss),
        "--masked-dist-loss", str(args.masked_dist_loss),
        "--mask-token-prob", str(args.mask_token_prob),
        "--mask-prob", str(args.mask_prob),
        "--drop-feat-prob", str(args.drop_feat_prob),
        "--noise", str(args.noise),
        "--noise-type", args.noise_type,
        "--max-atoms", str(args.max_atoms),
        # MTR
        "--mtr-loss-weight", str(args.mtr_loss_weight),
        "--descriptors-json", str(desc_json) if desc_json else "NONE",
        "--mtr-pooling", str(args.mtr_pooling),
        "--mtr-hidden-dim", str(args.mtr_hidden_dim),
        "--mtr-activation-fn", str(args.mtr_activation_fn),
        "--mtr-dropout", str(args.mtr_dropout),
        # optimiser + schedule
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
        cmd += ["--valid-subset", "train", "--disable-validation"]
    # LMDB temp dir must be writable (upstream hardcodes /temp/). Wrap in a
    # simple kv so the training process receives the env var (set below via env).
    cmd += ["--max-epoch", str(args.max_epoch)]
    return cmd


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.train_sdf = args.train_sdf.resolve()
    args.pretrained_checkpoint = args.pretrained_checkpoint.resolve()

    if not args.pretrained_checkpoint.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {args.pretrained_checkpoint}")
    embed_dim = detect_embed_dim(args.pretrained_checkpoint)
    expected = EMBED_DIM_BY_SIZE[args.model_size]
    if embed_dim != expected:
        raise ValueError(f"Checkpoint embed dim {embed_dim} != --model-size {args.model_size}")

    lmdb_dir = (args.lmdb_dir.resolve() if args.lmdb_dir
                else (args.train_sdf.parent / "lmdb_pretrain"))
    lmdb_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    lmdb_tmp = args.output_dir / "lmdb_tmp"
    lmdb_tmp.mkdir(parents=True, exist_ok=True)

    desc_json = None
    if args.mtr_loss_weight > 0:
        desc_json = ensure_lmdb(args, lmdb_dir)

    cmd = build_train_command(args, lmdb_dir, ckpt_dir, desc_json)

    metadata = {
        "train_sdf": str(args.train_sdf),
        "lmdb_dir": str(lmdb_dir),
        "pretrained_checkpoint": str(args.pretrained_checkpoint),
        "model_size": args.model_size,
        "arch": ARCH_BY_SIZE[args.model_size],
        "objective": {
            "masked_token_loss": args.masked_token_loss,
            "masked_coord_loss": args.masked_coord_loss,
            "masked_dist_loss": args.masked_dist_loss,
            "mtr_loss_weight": args.mtr_loss_weight,
            "descriptors_json": str(desc_json) if desc_json else None,
        },
        "total_steps": args.total_steps,
        "learning_rate": args.learning_rate,
        "fp16": args.fp16,
        "unicore_train_cmd": cmd,
    }
    (args.output_dir / "continued_pretrain_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        print("\nDRY RUN — command:\n  " + " ".join(cmd))
        return

    # LMDB temp dir + PYTHONPATH so both the unimol2 and unimol2_mtr packages are
    # importable (NEST_ROOT), matching the --user-dir above.
    env = dict(os.environ)
    env["UNIMOL2_LMDB_TMP_DIR"] = str(lmdb_tmp)
    env["PYTHONPATH"] = str(NEST_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    print("Launching v5 continued pretraining:\n  " + " ".join(cmd))
    proc = subprocess.run(cmd, env=env, cwd=str(args.output_dir))
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()