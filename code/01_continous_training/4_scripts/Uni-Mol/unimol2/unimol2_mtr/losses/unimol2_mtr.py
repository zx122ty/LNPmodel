# Copyright (c) DP Technology. Apache-2.0 — derived from the official unimol2 loss.
# v5: adds a masked (per-sample) descriptor-regression loss on the [CLS] token,
# driven by the MTR targets stored in each LMDB record.

import logging

import torch
import torch.nn.functional as F
from unicore import metrics
from unicore.losses import UnicoreLoss, register_loss

logger = logging.getLogger(__name__)

NAN_REPLACEMENT = 0.0  # target replaced when a descriptor is NaN (should not happen)


@register_loss("unimol2_mtr")
class UniMol2MTRLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.pad_idx
        self.seed = task.seed
        self.desc_n = task.desc_n
        self.desc_mean = torch.tensor(task.desc_mean or [], dtype=torch.float32)
        self.desc_std = torch.tensor(task.desc_std or [], dtype=torch.float32)
        self.has_stats = self.desc_mean.numel() == self.desc_n and \
            self.desc_std.numel() == self.desc_n
        # mtr prediction on/off is controlled by the loss weight (args), set by
        # the run script; model.mtr_head is always built (spec'd via --desc-n).
        self.mtr_on = getattr(self.args, "mtr_loss_weight", 0.0) > 0
        if self.mtr_on:
            logger.info(
                "unimol2_mtr loss: %d descriptor targets, has_stats=%s",
                self.desc_n, self.has_stats,
            )

    def forward(self, model, sample, reduce=True):
        input_key = "batched_data"
        masked_tokens = sample[input_key]["target_token"].ne(self.padding_idx)
        sample_size = masked_tokens.long().sum()

        (logits_encoder, encoder_distance, encoder_coord, mtr_pred) = model(
            **sample, encoder_masked_tokens=masked_tokens,
        )
        target = sample[input_key]["target_token"]
        if masked_tokens is not None:
            target = target[masked_tokens]

        # masked token (atom) loss — skipped entirely in MTR-only mode
        # (masked_token_loss=0: model returns logits=None, so we must not touch
        # logits_encoder here; the MTR-only DA mode relies on this branch).
        masked_token_loss = torch.zeros((), dtype=torch.float32,
                                        device=sample[input_key]["src_token"].device)
        masked_pred = torch.zeros((), dtype=torch.long, device=masked_token_loss.device)
        masked_hit = torch.zeros((), dtype=torch.long, device=masked_token_loss.device)
        masked_hit_at6 = torch.zeros((), dtype=torch.long, device=masked_token_loss.device)
        target_hit_at6 = torch.zeros((), dtype=torch.long, device=masked_token_loss.device)
        masked_cnt = torch.zeros((), dtype=torch.long, device=masked_token_loss.device)
        loss = torch.zeros((), dtype=torch.float32, device=masked_token_loss.device)
        if self.args.masked_token_loss > 0:
            masked_token_loss = F.nll_loss(
                F.log_softmax(logits_encoder, dim=-1, dtype=torch.float32),
                target, ignore_index=self.padding_idx, reduction="mean",
            )
            masked_pred = logits_encoder.argmax(dim=-1)
            masked_hit = (masked_pred == target).long().sum()
            masked_hit_at6 = (masked_pred == 6).long().sum()
            target_hit_at6 = (target == 6).long().sum()
            masked_cnt = sample_size
            loss = masked_token_loss * self.args.masked_token_loss

        logging_output = {
            "sample_size": 1,
            "bsz": sample[input_key]["src_token"].size(0),
            "seq_len": sample[input_key]["src_token"].size(1)
            * sample[input_key]["src_token"].size(0),
            "masked_token_loss": masked_token_loss.data,
            "masked_token_hit": masked_hit.data,
            "masked_hit_at6": masked_hit_at6.data,
            "target_hit_at6": target_hit_at6.data,
            "masked_token_cnt": masked_cnt,
        }

        if self.args.masked_coord_loss > 0:
            batch_size = sample[input_key]["atom_mask"].shape[0]
            atom_mask = sample[input_key]["atom_mask"].float()
            pos_mask = atom_mask.unsqueeze(-1)
            pos_target = sample[input_key]["target_pos"].float() * pos_mask

            def get_pos_loss(pos_pred):
                pos_pred = pos_pred.float() * pos_mask
                pos_loss = torch.nn.L1Loss(reduction="none")(
                    pos_pred, pos_target).sum(dim=(-1, -2))
                pos_cnt = pos_mask.squeeze(-1).sum(dim=-1) + 1e-10
                return (pos_loss / pos_cnt).sum()

            pos_loss = get_pos_loss(encoder_coord) / batch_size
            loss = loss + pos_loss * self.args.masked_coord_loss
            logging_output["masked_coord_loss"] = pos_loss.data

        if self.args.masked_dist_loss > 0:
            batch_size = sample[input_key]["atom_mask"].shape[0]
            atom_mask = sample[input_key]["atom_mask"].float()
            pair_mask = atom_mask.unsqueeze(-1) * atom_mask.unsqueeze(-2).float()
            pos_target = sample[input_key]["target_pos"].float() * atom_mask.unsqueeze(-1)
            dist_target = (pos_target.unsqueeze(-2) - pos_target.unsqueeze(-3)).norm(dim=-1)
            dist_target = dist_target * pair_mask
            dist_cnt = pair_mask.sum(dim=(-1, -2)) + 1e-10

            def get_dist_loss(dist_pred, return_sum=True):
                dist_pred = dist_pred.float() * pair_mask
                dist_loss = torch.nn.L1Loss(reduction="none")(
                    dist_pred, dist_target).sum(dim=(-1, -2))
                return (dist_loss / dist_cnt).sum() if return_sum else dist_loss / dist_cnt

            dist_loss = get_dist_loss(encoder_distance) / batch_size
            loss = loss + dist_loss * self.args.masked_dist_loss
            logging_output["masked_dist_loss"] = dist_loss.data

        # ── MTR: descriptor regression on [CLS] ─────────────────────────────
        if self.args.mtr_loss_weight > 0 and mtr_pred is not None:
            mtr_targets = sample[input_key].get("descriptors", None)
            if mtr_targets is None:
                mtr_targets = sample.get("descriptors", None)

            if mtr_targets is not None:
                # infer desc_n from the batch if task-level default was not overridden
                batch_desc_n = mtr_targets.shape[-1]
                tgt = mtr_targets.float()
                valid_mask = ~torch.isnan(tgt)
                tgt_safe = torch.where(valid_mask, tgt, torch.zeros_like(tgt))

                mse = F.mse_loss(mtr_pred, tgt_safe, reduction="none")
                mse = torch.where(valid_mask, mse, torch.zeros_like(mse))
                n_valid = valid_mask.sum().clamp(min=1).float()
                mtr_loss = mse.sum() / n_valid

                # mtr accuracy proxy: mean absolute error (z-space)
                diff = (mtr_pred - tgt_safe).abs()
                mtr_mae = torch.where(valid_mask, diff, torch.zeros_like(diff)
                                      ).sum() / n_valid
                # optionally unstandardise MAE to original units if stats exist
                if self.has_stats and self.desc_n == batch_desc_n:
                    batch_std = self.desc_std.to(mtr_pred.device)
                    mtr_mae_raw = torch.where(
                        valid_mask, diff * batch_std, torch.zeros_like(diff)
                    ).sum() / n_valid
                else:
                    mtr_mae_raw = mtr_mae
                logging_output["mtr_loss"] = mtr_loss.data
                logging_output["mtr_mae"] = mtr_mae.data
                logging_output["mtr_mae_raw"] = mtr_mae_raw.data

                loss = loss + mtr_loss * self.args.mtr_loss_weight

        logging_output["loss"] = loss.data
        return loss, 1, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split="valid") -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=5)
        metrics.log_scalar("seq_len", seq_len / bsz, 1, round=3)

        masked_loss = sum(log.get("masked_token_loss", 0) for log in logging_outputs)
        metrics.log_scalar("masked_token_loss", masked_loss / sample_size,
                           sample_size, round=5)
        masked_cnt_total = sum(log.get("masked_token_cnt", 0)
                               for log in logging_outputs)
        masked_acc = (sum(log.get("masked_token_hit", 0) for log in logging_outputs)
                      / masked_cnt_total) if masked_cnt_total > 0 else 0.0
        metrics.log_scalar("masked_acc", masked_acc, sample_size, round=5)
        masked_hit_at6_total = (sum(log.get("masked_hit_at6", 0)
                                    for log in logging_outputs)
                                / masked_cnt_total) if masked_cnt_total > 0 else 0.0
        metrics.log_scalar("masked_hit_at6", masked_hit_at6_total,
                           sample_size, round=3)
        target_hit_at6_total = (sum(log.get("target_hit_at6", 0)
                                    for log in logging_outputs)
                                / masked_cnt_total) if masked_cnt_total > 0 else 0.0
        metrics.log_scalar("target_hit_at6", target_hit_at6_total,
                           sample_size, round=3)

        masked_coord_loss = sum(log.get("masked_coord_loss", 0) for log in logging_outputs)
        if masked_coord_loss > 0:
            metrics.log_scalar("masked_coord_loss", masked_coord_loss / sample_size,
                               sample_size, round=5)
        masked_dist_loss = sum(log.get("masked_dist_loss", 0) for log in logging_outputs)
        if masked_dist_loss > 0:
            metrics.log_scalar("masked_dist_loss", masked_dist_loss / sample_size,
                               sample_size, round=5)

        mtr_loss = sum(log.get("mtr_loss", 0) for log in logging_outputs)
        if mtr_loss > 0:
            metrics.log_scalar("mtr_loss", mtr_loss / sample_size, sample_size, round=5)
        mtr_mae = sum(log.get("mtr_mae", 0) for log in logging_outputs)
        if mtr_mae > 0:
            metrics.log_scalar("mtr_mae", mtr_mae / sample_size, sample_size, round=5)
        mtr_mae_raw = sum(log.get("mtr_mae_raw", 0) for log in logging_outputs)
        if mtr_mae_raw > 0:
            metrics.log_scalar("mtr_mae_raw", mtr_mae_raw / sample_size,
                               sample_size, round=5)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        return True