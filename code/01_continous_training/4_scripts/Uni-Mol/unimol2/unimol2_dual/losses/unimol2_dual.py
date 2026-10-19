# Copyright (c) DP Technology. Apache-2.0 — derived from the official unimol2 loss.
# EnviroFM-v2 Option A pretraining loss (plan §6.2):
#
#   L = λ_token·L_token + λ_coord·L_coord + λ_dist·L_dist
#       + λ_contrast·L_NT-Xent(z_3d, z_2d) + λ_mtr·L_MTR
#
# The denoising terms are the stock Uni-Mol2 losses verbatim. Two additions:
#
#   * **Multi-view NT-Xent** between the [CLS] of the 3D view and the [CLS] of
#     the Kabsch-aligned 2D view of the same molecule (the model runs a second
#     encoder pass for the latter). Negatives are gathered across DDP ranks so
#     the effective negative count is world_size × batch. Temperature τ=0.1.
#   * **MTR descriptor regression** on [CLS] in z-space, NaN-masked (the same
#     convention as unimol2_mtr).

import logging

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss

logger = logging.getLogger(__name__)


def _gather_cross_rank(z: torch.Tensor) -> torch.Tensor:
    """All-gather ``z`` across DDP ranks; local rows keep gradients."""
    if not (dist.is_available() and dist.is_initialized()):
        return z
    world = dist.get_world_size()
    if world == 1:
        return z
    gathered = [torch.zeros_like(z) for _ in range(world)]
    dist.all_gather(gathered, z.detach())
    rank = dist.get_rank()
    gathered[rank] = z
    return torch.cat(gathered, dim=0)


@register_loss("unimol2_dual")
class UniMol2DualLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.pad_idx
        self.seed = task.seed
        self.desc_n = getattr(task, "desc_n", 0)
        self._warned_no_2d = False

    @staticmethod
    def _nt_xent(z_a: torch.Tensor, z_b: torch.Tensor, tau: float) -> torch.Tensor:
        """Symmetric NT-Xent over the paired batch. ``z_a``/``z_b`` are (B, d)."""
        z_a = F.normalize(z_a.float(), dim=-1)
        z_b = F.normalize(z_b.float(), dim=-1)
        z_a = _gather_cross_rank(z_a)
        z_b = _gather_cross_rank(z_b)
        n_local = z_a.size(0)  # rows with gradients (this rank's slice)
        sim_ab = z_a @ z_b.t() / tau          # (B', B')
        sim_ba = z_b @ z_a.t() / tau
        labels = torch.arange(sim_ab.size(0), device=z_a.device)
        loss = 0.5 * (
            F.cross_entropy(sim_ab, labels) + F.cross_entropy(sim_ba, labels)
        )
        # Only this rank's rows contribute to the gradient; the gathered
        # negatives are stop-gradient context.
        return loss

    def forward(self, model, sample, reduce=True):
        input_key = "batched_data"
        masked_tokens = sample[input_key]["target_token"].ne(self.padding_idx)
        sample_size = masked_tokens.long().sum()

        contrast_weight = getattr(self.args, "contrast_loss_weight", 0.0)
        contrast_prob = getattr(self.args, "contrast_prob", 1.0)
        # Per-batch Bernoulli gate (default p=1: always on). Each DDP rank gates
        # independently; a skipped batch simply contributes no contrastive term.
        contrast_on = (
            contrast_weight > 0
            and (contrast_prob >= 1.0 or np.random.rand() < contrast_prob)
        )
        (
            logits_encoder,
            encoder_distance,
            encoder_coord,
            mtr_pred,
            z_3d,
            z_2d,
        ) = model(
            **sample,
            encoder_masked_tokens=masked_tokens,
            contrastive_2d=contrast_on,
        )

        target = sample[input_key]["target_token"]
        if masked_tokens is not None:
            target = target[masked_tokens]

        device = sample[input_key]["src_token"].device
        loss = torch.zeros((), dtype=torch.float32, device=device)

        # ── masked token (atom) loss ────────────────────────────────────────
        if self.args.masked_token_loss > 0:
            masked_token_loss = F.nll_loss(
                F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
                target, ignore_index=self.padding_idx, reduction="mean",
            )
            loss = loss + masked_token_loss * self.args.masked_token_loss
        else:
            masked_token_loss = torch.zeros((), device=device)

        logging_output = {
            "sample_size": 1,
            "bsz": sample[input_key]["src_token"].size(0),
            "seq_len": sample[input_key]["src_token"].size(1)
                       * sample[input_key]["src_token"].size(0),
            "masked_token_loss": masked_token_loss.data,
        }

        # ── coordinate denoising (smooth-L1 per the plan; stock uses L1 — we
        # keep the stock L1 so A0/A1 comparisons stay on identical objectives) ──
        if self.args.masked_coord_loss > 0:
            batch_size = sample[input_key]["atom_mask"].shape[0]
            atom_mask = sample[input_key]["atom_mask"].float()
            pos_mask = atom_mask.unsqueeze(-1)
            pos_target = sample[input_key]["target_pos"].float() * pos_mask

            def get_pos_loss(pos_pred):
                pos_pred = pos_pred.float() * pos_mask
                pos_loss = torch.nn.L1Loss(reduction="none")(pos_pred, pos_target)
                pos_loss = pos_loss.sum(dim=(-1, -2))
                pos_cnt = pos_mask.squeeze(-1).sum(dim=-1) + 1e-10
                return (pos_loss / pos_cnt).sum()

            pos_loss = get_pos_loss(encoder_coord) / batch_size
            loss = loss + pos_loss * self.args.masked_coord_loss
            logging_output["masked_coord_loss"] = pos_loss.data

        # ── pairwise distance reconstruction ────────────────────────────────
        if self.args.masked_dist_loss > 0:
            batch_size = sample[input_key]["atom_mask"].shape[0]
            atom_mask = sample[input_key]["atom_mask"].float()
            pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2).float()
            pos_target = sample[input_key]["target_pos"].float() * atom_mask.unsqueeze(-1)
            dist_target = (pos_target.unsqueeze(-2) - pos_target.unsqueeze(-3)).norm(dim=-1)
            dist_target = dist_target * pair_mask
            dist_cnt = pair_mask.sum(dim=(-1, -2)) + 1e-10

            def get_dist_loss(dist_pred):
                dist_pred = dist_pred.float() * pair_mask
                dist_loss = torch.nn.L1Loss(reduction="none")(dist_pred, dist_target)
                return (dist_loss.sum(dim=(-1, -2)) / dist_cnt).sum()

            dist_loss = get_dist_loss(encoder_distance) / batch_size
            loss = loss + dist_loss * self.args.masked_dist_loss
            logging_output["masked_dist_loss"] = dist_loss.data

        # ── multi-view NT-Xent (plan §6.2, τ=0.1) ───────────────────────────
        contrast_loss = None
        if contrast_on:
            if z_2d is None or z_3d is None:
                if not self._warned_no_2d:
                    logger.warning(
                        "unimol2_dual loss: contrast_loss_weight>0 but no 2D view "
                        "available (ablation != hybrid or missing pos_2d_aligned); "
                        "skipping the contrastive term.")
                    self._warned_no_2d = True
            else:
                tau = getattr(self.args, "contrast_temperature", 0.1)
                contrast_loss = self._nt_xent(z_3d, z_2d, tau)
                loss = loss + contrast_loss * self.args.contrast_loss_weight
                logging_output["contrast_loss"] = contrast_loss.data

        # ── MTR descriptor regression on [CLS] (z-space, NaN-masked) ────────
        mtr_loss = None
        if getattr(self.args, "mtr_loss_weight", 0.0) > 0 and mtr_pred is not None:
            mtr_targets = sample[input_key].get("descriptors", None)
            if mtr_targets is None:
                mtr_targets = sample.get("descriptors", None)
            if mtr_targets is not None and mtr_targets.shape[-1] == mtr_pred.shape[-1]:
                tgt = mtr_targets.float()
                valid_mask = ~torch.isnan(tgt)
                tgt_safe = torch.where(valid_mask, tgt, torch.zeros_like(tgt))
                mse = F.mse_loss(mtr_pred, tgt_safe, reduction="none")
                mse = torch.where(valid_mask, mse, torch.zeros_like(mse))
                n_valid = valid_mask.sum().clamp(min=1).float()
                mtr_loss = mse.sum() / n_valid
                loss = loss + mtr_loss * self.args.mtr_loss_weight
                logging_output["mtr_loss"] = mtr_loss.data

        logging_output["loss"] = loss.data
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / max(sample_size, 1), sample_size, round=5)
        metrics.log_scalar("seq_len", seq_len / max(bsz, 1), 1, round=3)

        masked_loss = sum(log.get("masked_token_loss", 0) for log in logging_outputs)
        metrics.log_scalar("masked_token_loss",
                           masked_loss / max(sample_size, 1), sample_size, round=5)

        for key in ("masked_coord_loss", "masked_dist_loss", "contrast_loss", "mtr_loss"):
            total = sum(log.get(key, 0) for log in logging_outputs)
            if total > 0:
                metrics.log_scalar(key, total / max(sample_size, 1), sample_size, round=5)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        return True
