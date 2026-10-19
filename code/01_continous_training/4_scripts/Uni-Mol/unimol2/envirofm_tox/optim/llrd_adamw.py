# Copyright (c) DP Technology. Apache-2.0 — derived from the official unimol2 package.
"""AdamW with layer-wise learning rate decay (plan §7.1).

Why this is an optimizer and not an argument
--------------------------------------------
Uni-Core's optimiser plumbing applies **one** learning rate to every parameter group:
``UnicoreOptimizer.set_lr`` writes a single scalar into each group's ``"lr"``, the cosine
scheduler calls it once per update, and ``FP16Optimizer``/``UnicoreAdam`` both build from
``optimizer_config`` with a single ``lr``. There is no CLI surface for per-layer rates, so
LLRD has to be expressed as an optimizer that owns the mapping from the schedule's scalar
to each group's rate.

How the decay is anchored
-------------------------
The schedule's scalar (``--lr``) is by definition the **head** rate, and
``--llrd-backbone-ratio`` is the top backbone block's rate as a fraction of it. That makes
the two numbers the plan fixes — head ``2e-4``, top backbone block ``2e-5`` — exactly
``--lr 2e-4 --llrd-backbone-ratio 0.1``, with ``--llrd-factor 0.85`` decaying geometrically
from the top block down to the stem. :mod:`envirofm_tox.param_grouping` does the name →
depth classification and documents why the names are published by the task.

Why this optimizer requires ``--fp16`` to be **off**
----------------------------------------------------
``FP16Optimizer.build_optimizer`` flattens every parameter in a group into one contiguous
buffer *before* the optimizer class is constructed
(``get_fp16_params`` → ``flatten_parameters_fp32`` → ``param_dict["params"] = [fp32_params]``).
By the time ``__init__`` runs, a group holds a single flattened tensor, so parameter
identity — and therefore layer depth — is unrecoverable. ``__init__`` therefore refuses an
fp16/bf16 run loudly instead of silently decaying every parameter as if it were the stem.

The plan lists FP16 as an OOM mitigation (plan §12.2), but that mitigation is aimed at
``N_heavy <= 128`` molecules and neither Phase 1 endpoint has any: the largest Ames and
LD50 molecules are 56 and 90 heavy atoms. With ``--max-atoms 128`` there is no cropping and
no OOM, so giving up FP16 costs wall-clock, not feasibility.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Collection

import torch
from unicore.optim import UnicoreOptimizer, register_optimizer
from unicore.optim.adam import Adam, UnicoreAdam
from unicore.optim.fused_adam import get_fused_adam_class

from envirofm_tox.param_grouping import lr_scale, name_of, n_layers

logger = logging.getLogger(__name__)


@register_optimizer("envirofm_llrd_adamw")
class EnviroFMLLRDAdam(UnicoreAdam):
    """AdamW whose per-group LR is the schedule's LR times a depth-dependent scale."""

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        UnicoreAdam.add_args(parser)
        parser.add_argument(
            "--llrd-factor", type=float, default=1.0,
            help="Geometric decay factor per transformer block, applied downwards from "
                 "the top block (plan §7.1). 1.0 disables decay.",
        )
        parser.add_argument(
            "--llrd-backbone-ratio", type=float, default=1.0,
            help="Top backbone block LR as a fraction of the head LR (--lr). 0.1 gives "
                 "the plan's 2e-4 head / 2e-5 backbone pair.",
        )

    def __init__(self, args, params):
        if getattr(args, "fp16", False) or getattr(args, "bf16", False):
            raise ValueError(
                "envirofm_llrd_adamw cannot be used with --fp16/--bf16: FP16Optimizer "
                "flattens each parameter group into one buffer before the optimizer is "
                "constructed, so parameter identity (and hence layer depth) is lost. Run "
                "in fp32 — the Phase 1 endpoints top out at 90 heavy atoms, so the OOM "
                "risk FP16 would mitigate does not arise."
            )
        UnicoreOptimizer.__init__(self, args)

        self.factor = float(args.llrd_factor)
        self.backbone_ratio = float(args.llrd_backbone_ratio)
        if not 0.0 < self.factor <= 1.0:
            raise ValueError(f"--llrd-factor must be in (0, 1], got {self.factor}.")
        if not 0.0 < self.backbone_ratio <= 1.0:
            raise ValueError(
                f"--llrd-backbone-ratio must be in (0, 1], got {self.backbone_ratio}.")

        base_lr = args.lr[0] if isinstance(args.lr, Collection) else float(args.lr)
        self._base_lr = float(base_lr)

        groups = self._build_groups(params)
        cfg = dict(self.optimizer_config)      # betas / eps / weight_decay / lr
        cfg.pop("lr", None)                    # every group carries its own
        fused_adam_cls = get_fused_adam_class()
        use_fused_adam = (
            not getattr(args, "use_old_adam", False)
            and fused_adam_cls is not None
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 7
        )
        self._optimizer = (fused_adam_cls(groups, **cfg) if use_fused_adam
                           else Adam(groups, **cfg))

    # -------------------------------------------------------------------------
    # Group construction
    # -------------------------------------------------------------------------
    def _build_groups(self, params) -> list[dict]:
        """Split the incoming groups by LLRD scale, keeping each group's own options.

        ``params`` arrives as a list of ``{"params": [...], ...}`` dicts — from
        ``separate_decay_params`` when a weight decay is configured, otherwise a single
        undifferentiated group. Splitting preserves each source group's options (notably a
        ``weight_decay`` of 0.0 for biases and norms), so adding LLRD never silently
        changes which parameters are decayed.
        """
        names = [name_of(p) for p in
                 (p for g in params for p in g["params"])]
        n_blocks = n_layers([n for n in names if n])

        out: list[dict] = []
        scales_seen: "OrderedDict[float, int]" = OrderedDict()
        unregistered: list[str] = []
        for group in params:
            buckets: "OrderedDict[float, list]" = OrderedDict()
            for p in group["params"]:
                name = name_of(p)
                if name is None:
                    unregistered.append(type(p).__name__)
                scale = lr_scale(name, n_layer_blocks=n_blocks, factor=self.factor,
                                 backbone_ratio=self.backbone_ratio)
                buckets.setdefault(scale, []).append(p)
            for scale, ps in buckets.items():
                sub = {k: v for k, v in group.items() if k != "params"}
                sub["params"] = ps
                sub["llrd_scale"] = scale
                sub["lr"] = self._base_lr * scale
                out.append(sub)
                scales_seen[scale] = scales_seen.get(scale, 0) + len(ps)

        if unregistered:
            logger.warning(
                "envirofm_llrd_adamw: %d parameter(s) had no registered name and were "
                "decayed at the stem rate. The model was probably not built by "
                "ToxFinetuneTask.build_model.", len(unregistered),
            )
        logger.info(
            "envirofm_llrd_adamw: %d blocks, base lr=%.3g, factor=%g, backbone_ratio=%g "
            "-> %d group(s); lr range [%.3g, %.3g]",
            n_blocks, self._base_lr, self.factor, self.backbone_ratio, len(out),
            min(scales_seen) * self._base_lr, max(scales_seen) * self._base_lr,
        )
        for scale, count in scales_seen.items():
            logger.info("  lr=%.4g (x%.4f) over %d parameter(s)",
                        self._base_lr * scale, scale, count)
        return out

    # -------------------------------------------------------------------------
    # The scheduler interface
    # -------------------------------------------------------------------------
    def set_lr(self, lr):
        """Set the head LR; every group follows at its own fixed ratio.

        The cosine scheduler calls this once per update with the scheduled scalar, so this
        is where the single-rate schedule is turned back into a per-depth one.
        """
        self._base_lr = float(lr)
        for group in self.param_groups:
            group["lr"] = self._base_lr * group.get("llrd_scale", 1.0)

    def get_lr(self):
        """The head LR — the schedule's own value, not any single group's rate."""
        return self._base_lr

    @property
    def optimizer_config(self):
        """Config used to *construct* the torch optimizer; per-group LR is added after."""
        cfg = dict(super().optimizer_config)
        cfg.pop("lr", None)
        return cfg
