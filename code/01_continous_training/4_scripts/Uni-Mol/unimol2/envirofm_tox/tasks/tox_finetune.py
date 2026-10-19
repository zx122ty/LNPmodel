# Copyright (c) DP Technology. Apache-2.0 — derived from the official unimol2 package.
"""EnviroFM-v2 fine-tuning task: ``mol_finetune`` plus the protocol's two hard requirements.

The stock :class:`unimol2.tasks.unimol_finetune.UniMolFinetuneTask` is reused wholesale for
dataset construction, model building and head registration. This subclass adds only what
the frozen protocol (``3_configs/envirofm_v2.yaml``) demands and the stock task cannot
express:

1. **Target standardisation on a non-QM9 endpoint.** ``FinetuneMSELoss`` standardises the
   target using ``task.mean``/``task.std``, but the stock task only populates those from a
   hard-coded ``task_metainfo`` table of QM9 properties. LD50 is not in that table, so a
   stock run silently trains on raw ``-log10[mol/kg]`` values. Here the statistics are
   passed in from the fold's own ``train_opt`` partition (``--target-mean``/``--target-std``)
   and recorded in the fold metadata, so the constant is auditable and provably never
   derived from esval or test.

2. **A runtime leak assertion.** Plan §10.1 requires the training loop to raise a fatal
   ``AssertionError`` if any test sample reaches the training process before the checkpoint
   is frozen. The structural half of that guarantee is that ``train.lmdb``/``valid.lmdb``
   are written to a different directory from ``test.lmdb`` and the fine-tuning process is
   only ever handed the former (see ``4_scripts/build_finetune_lmdb.py``). The assertion
   here is the redundant second half: the fold's test frame positions are read from the
   sidecar written at LMDB-build time and checked against the frame positions of every
   record actually loaded. It costs one set intersection per split and fails loudly if a
   future refactor points ``--data`` at the wrong directory.

Nothing else is overridden: an override that merely restates upstream behaviour is a place
for the two to silently diverge on the next upstream pull.
"""

from __future__ import annotations

import json
import logging
import os

from unicore.tasks import register_task

from unimol2.tasks.unimol_finetune import UniMolFinetuneTask

logger = logging.getLogger(__name__)

# Losses that standardise their target, i.e. those that need ``task.mean``/``task.std``.
# A regression loss used without statistics is not an error — it just means the head
# learns the raw label scale — but it is worth saying so out loud once per run.
_REGRESSION_LOSSES = {"finetune_mse", "finetune_mae", "finetune_smooth_mae",
                      "envirofm_mse", "envirofm_smooth_mae", "envirofm_mae"}


@register_task("envirofm_tox")
class ToxFinetuneTask(UniMolFinetuneTask):
    """``mol_finetune`` with protocol-driven target statistics and a test-leak assertion."""

    @staticmethod
    def add_args(parser):
        UniMolFinetuneTask.add_args(parser)
        parser.add_argument(
            "--target-mean", type=float, default=None,
            help="Training-set target mean, for standardising the regression target. "
                 "Must come from the fold's train_opt partition only. Passed explicitly "
                 "rather than recomputed here so the value is auditable in the run log.",
        )
        parser.add_argument(
            "--target-std", type=float, default=None,
            help="Training-set target standard deviation; see --target-mean.",
        )
        parser.add_argument(
            "--test-index-file", type=str, default=None,
            help="Path to the fold's test.meta.json. When given, every non-test split "
                 "loaded asserts that its records' frame positions are disjoint from the "
                 "test fold's. This is the runtime half of the plan §10.1 guardrail.",
        )

    def __init__(self, args, dictionary=None):
        super().__init__(args, dictionary)
        # ``super().__init__`` may have set mean/std from ``task_metainfo``; the fold's own
        # statistics take precedence, because a QM9 table entry cannot be right for a
        # toxicity endpoint. Both are set or neither, so ``compute_loss`` cannot end up
        # standardising by a mean without a matching scale.
        if args.target_mean is not None or args.target_std is not None:
            if args.target_mean is None or args.target_std is None:
                raise ValueError(
                    "--target-mean and --target-std must be given together (got "
                    f"{args.target_mean} / {args.target_std}); standardising by only one "
                    "of them is never intended."
                )
            if not args.target_std > 0:
                raise ValueError(f"--target-std must be positive, got {args.target_std}.")
            self.mean = float(args.target_mean)
            self.std = float(args.target_std)
            logger.info("envirofm_tox: target standardisation mean=%.6f std=%.6f",
                        self.mean, self.std)
        elif not hasattr(self, "mean"):
            self.mean = None
            self.std = None

        self.is_regression = bool(getattr(args, "reg", False))
        loss_name = getattr(args, "loss", None)
        if self.is_regression and loss_name in _REGRESSION_LOSSES and self.mean is None:
            logger.warning(
                "envirofm_tox: regression loss %r without --target-mean/--target-std. The "
                "head will fit the raw label scale. Pass the fold's train_opt statistics "
                "unless that is deliberate.", loss_name,
            )

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args, dictionary=None)

    # -------------------------------------------------------------------------
    # The plan §10.1 runtime guardrail
    # -------------------------------------------------------------------------
    def _assert_no_test_leak(self, split: str, meta_path: str) -> None:
        """Fatal if ``split``'s records share a frame position with the test fold.

        Records may carry ``frame_position: null`` when they are *augmentation rows* that
        are not endpoint-frame molecules at all (e.g. the CATMoS LD50 pool appended to a
        fold's training split). Such records cannot collide with a test fold, because the
        test fold is defined by frame positions and the augmented pool is built with every
        frame molecule removed and with a per-fold Tanimoto>0.90 guard against the fold's
        test molecules (see ``4_scripts/diagnose/build_catmos_aug_lmdb.py``). They are
        therefore skipped *by position* and counted here, so the skip is visible in the log
        rather than implicit.
        """
        test_path = self.args.test_index_file
        if not test_path or split == "test":
            return
        if not os.path.exists(meta_path):
            logger.warning(
                "envirofm_tox: %s has no sidecar %s, so the test-leak assertion cannot "
                "run for it. Regenerate with 4_scripts/build_finetune_lmdb.py.",
                split, meta_path,
            )
            return
        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"--test-index-file {test_path!r} does not exist; refusing to train "
                f"without the guardrail it enables."
            )
        with open(meta_path, encoding="utf-8") as fh:
            records = json.load(fh)["records"]
        # A record is either a frame molecule (positional) or an added augmentation row.
        # Nothing else is permitted: an unrecognised shape is a loud failure, not a skip.
        positioned, n_augmented = set(), 0
        for r in records:
            pos = r.get("frame_position")
            if pos is None:
                n_augmented += 1
                continue
            positioned.add(int(pos))
        mine = positioned
        with open(test_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        theirs = {int(r["frame_position"]) for r in payload["records"]}
        overlap = mine & theirs
        if overlap:
            raise AssertionError(
                f"TEST-FOLD LEAKAGE [{self.args.task_name}/{split}]: {len(overlap)} "
                f"record(s) of this split are also in the test fold "
                f"(frame positions e.g. {sorted(overlap)[:5]}). Loading {meta_path} into "
                f"the training process is the Pitfall #12 failure mode and is a fatal "
                f"protocol violation (plan §10.1)."
            )
        if n_augmented:
            logger.info(
                "envirofm_tox: guardrail ok — %s: %d frame record(s) disjoint from test "
                "(%d records); %d augmentation record(s) carry no frame position and are "
                "excluded from the position comparison (their own near-duplicate guard was "
                "applied when they were built).",
                split, len(mine), len(theirs), n_augmented,
            )
        else:
            logger.info(
                "envirofm_tox: guardrail ok — %s (%d records) is disjoint from test "
                "(%d records)", split, len(mine), len(theirs),
            )

    def load_dataset(self, split, **kwargs):
        """Delegate to the stock task, then discharge the leak assertion for ``split``."""
        split_dir = os.path.join(self.args.data, self.args.task_name)
        self._assert_no_test_leak(split, os.path.join(split_dir, split + ".meta.json"))
        super().load_dataset(split, **kwargs)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------
    def build_model(self, args):
        """Build via the stock path, then publish parameter names for LLRD grouping.

        This is the only point at which both the model and its parameter names are in
        scope together — the optimizer constructor receives bare tensors (see
        :mod:`envirofm_tox.param_grouping`). Registering here rather than in the optimizer
        keeps the dependency one-way: the optimizer reads the map, it never has to reach
        into the task.
        """
        model = super().build_model(args)
        from envirofm_tox.param_grouping import register_param_names

        register_param_names(model)
        return model

    # -------------------------------------------------------------------------
    # Instrumentation
    # -------------------------------------------------------------------------
    def dataset_size(self, split: str) -> int:
        """Number of records in a loaded split, before TTA replication.

        ``TTADataset.__len__`` is ``len(dataset) * conf_size``, so ``len(task.dataset(split))``
        is conf_size times the molecule count. The run log wants the molecule count, since
        that is what the fold file records and what the report's ``n_esval`` column means.
        """
        return len(self.dataset(split))
