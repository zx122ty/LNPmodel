# Copyright (c) DP Technology. Apache-2.0 — derived from the official unimol2 model.
# EnviroFM-v2 Option A: dual-stream hybrid 2D-topology + 3D-conformer backbone.
#
# What is new relative to stock ``unimol2`` (plan §5.2):
#
#   1. **Ring-membership bias** ``b_ring(i,j)`` — a learned per-pair categorical
#      embedding for "shared aromatic ring / shared aliphatic ring / neither",
#      added to the pairwise attention bias in every layer, on top of the stock
#      shortest-path + bond-order encoding (``EdgeFeature``) and the 3D Gaussian
#      RBF (``SE3InvariantKernel``). The pair flag is computed once per molecule
#      by the task (RDKit ring-info) and shipped in the batch as ``ring_flag``.
#
#   2. **Ablation toggles** (plan §5.2 sub-ablations, strict runtime switches):
#        * ``--ablation pure_2d``  — coordinates zeroed, RBF term removed;
#        * ``--ablation pure_3d``  — shortest-path / bond / ring terms removed
#                                    (reverts the bias to the vanilla Uni-Mol2
#                                    3D-only form);
#        * ``--ablation hybrid``   — full dual-stream model (default).
#      The toggles change only what is fed to the encoder; every module is still
#      built, so a checkpoint trained in one mode loads into another without
#      missing keys.
#
#   3. **Multi-view contrastive readout** — ``forward`` returns the [CLS] token
#      (and, when requested, a second forward pass over the 2D view) so the loss
#      can compute the NT-Xent term between the 3D and 2D views of the same
#      molecule. The 2D view is a *separate encoder pass* with zeroed
#      coordinates (``pure_2d`` mode on the same weights), which is what makes
#      the contrastive target conformer-invariant by construction.
#
#   4. **MTR head** — descriptor-regression MLP on [CLS] (reused verbatim from
#      ``unimol2_mtr``), plan §5.2 item 4 / §6.2 L_mtr.
#
# Everything else is copied from the stock model so a checkpoint trained here is
# loadable by the stock fine-tuning path (same parameter names for every shared
# module) and vice versa.

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import LayerNorm, init_bert_params

from unimol2.models.unimol2 import (
    AtomFeature,
    EdgeFeature,
    SE3InvariantKernel,
    MovementPredictionHead,
    MaskLMHead,
)
from unimol2.models.transformer_encoder_with_pair import TransformerEncoderWithPair
from unimol2_mtr.models.unimol2_mtr import MTRHead

logger = logging.getLogger(__name__)

ABLATION_MODES = ("hybrid", "pure_2d", "pure_3d")


@register_model("unimol2_dual")
class UniMol2DualModel(BaseUnicoreModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--encoder-layers", type=int, metavar="L",
                            help="num encoder layers")
        parser.add_argument("--encoder-embed-dim", type=int, metavar="H",
                            help="encoder embedding dimension")
        parser.add_argument("--encoder-ffn-embed-dim", type=int, metavar="F",
                            help="encoder embedding dimension for FFN")
        parser.add_argument("--encoder-attention-heads", type=int, metavar="A",
                            help="num attention heads")
        parser.add_argument("--activation-fn",
                            choices=utils.get_available_activation_fns(),
                            help="activation function to use")
        parser.add_argument("--pooler-activation-fn",
                            choices=utils.get_available_activation_fns(),
                            help="activation function to use for pooler layer")
        parser.add_argument("--emb-dropout", type=float, metavar="D",
                            help="dropout probability for embeddings")
        parser.add_argument("--dropout", type=float, metavar="D",
                            help="dropout probability")
        parser.add_argument("--attention-dropout", type=float, metavar="D",
                            help="dropout probability in attention weights")
        parser.add_argument("--activation-dropout", type=float, metavar="D",
                            help="dropout probability after activation in FFN")
        parser.add_argument("--pooler-dropout", type=float, metavar="D",
                            help="dropout probability in the masked_lm pooler layers")
        parser.add_argument("--max-seq-len", type=int,
                            help="number of positional embeddings to learn")
        parser.add_argument("--post-ln", type=bool,
                            help="use post layernorm or pre layernorm")
        parser.add_argument("--masked-token-loss", type=float, metavar="D",
                            help="mask loss ratio")
        parser.add_argument("--masked-dist-loss", type=float, metavar="D",
                            help="masked distance loss ratio")
        parser.add_argument("--masked-coord-loss", type=float, metavar="D",
                            help="masked coord loss ratio")
        parser.add_argument("--masked-coord-dist-loss", type=float, metavar="D",
                            help="masked coord dist loss ratio")
        parser.add_argument("--pair-embed-dim", type=int, metavar="N",
                            help="pairwise bias embedding dimension")
        parser.add_argument("--pair-hidden-dim", type=int, metavar="N",
                            help="pairwise hidden dimension")
        parser.add_argument("--pair-dropout", type=float, metavar="D",
                            help="dropout probability for pair")
        parser.add_argument("--droppath-prob", type=float,
                            help="stochastic path probability", default=0.0)
        parser.add_argument("--notri", action="store_true", help="disable trimul")
        parser.add_argument("--gaussian-std-width", type=float, default=1.0)
        parser.add_argument("--gaussian-mean-start", type=float, default=0.0)
        parser.add_argument("--gaussian-mean-stop", type=float, default=9.0)
        parser.add_argument("--mode", type=str, default="train",
                            choices=["train", "infer"])

        # ── Option A additions (plan §5.2) ───────────────────────────────────
        parser.add_argument("--ablation", type=str, default="hybrid",
                            choices=ABLATION_MODES,
                            help="dual-stream ablation toggle: hybrid (full), "
                                 "pure_2d (coordinates zeroed, RBF off), "
                                 "pure_3d (topology/bond/ring terms off)")
        parser.add_argument("--ring-embed-dim", type=int, default=None,
                            help="dimension of the ring-membership pair bias; "
                                 "defaults to --pair-embed-dim")

        # MTR head (plan §5.2 item 4)
        parser.add_argument("--desc-n", type=int, default=32,
                            help="descriptor vector length (MTR output dim)")
        parser.add_argument("--mtr-pooling", type=str, default="cls",
                            choices=["cls", "mean"],
                            help="readout for the MTR head")
        parser.add_argument("--mtr-hidden-dim", type=int, default=256,
                            help="MTR head hidden dim")
        parser.add_argument("--mtr-activation-fn", type=str, default="tanh",
                            help="MTR head activation")
        parser.add_argument("--mtr-dropout", type=float, default=0.0,
                            help="MTR head dropout")

    def __init__(self, args, dictionary=None):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.token_num = 128
        self.padding_idx = 0
        self.mask_idx = 127
        self.embed_tokens = nn.Embedding(
            self.token_num, args.encoder_embed_dim, self.padding_idx
        )

        num_atom = 512
        num_degree = 128
        num_edge = 64
        num_pair = 512
        num_spatial = 512

        self.atom_feature = AtomFeature(
            num_atom=num_atom,
            num_degree=num_degree,
            hidden_dim=args.encoder_embed_dim,
        )
        self.edge_feature = EdgeFeature(
            pair_dim=args.pair_embed_dim,
            num_edge=num_edge,
            num_spatial=num_spatial,
        )

        # Option A: ring-membership pair bias. Three categories:
        # 0 = no shared ring (or padding), 1 = shared aromatic ring,
        # 2 = shared aliphatic ring. Index 0 is the zero vector by
        # construction (padding_idx), so a molecule with no rings adds nothing.
        ring_dim = args.ring_embed_dim or args.pair_embed_dim
        self.ring_bias = nn.Embedding(3, ring_dim, padding_idx=0)
        nn.init.zeros_(self.ring_bias.weight)

        self._num_updates = None

        self.encoder = TransformerEncoderWithPair(
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            pair_dim=args.pair_embed_dim,
            pair_hidden_dim=args.pair_hidden_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            activation_fn=args.activation_fn,
            droppath_prob=args.droppath_prob,
        )
        if args.masked_token_loss > 0:
            self.lm_head = MaskLMHead(
                embed_dim=args.encoder_embed_dim,
                output_dim=self.token_num,
                activation_fn=args.activation_fn,
                weight=None,
            )

        K = 128
        self.se3_invariant_kernel = SE3InvariantKernel(
            pair_dim=args.pair_embed_dim,
            num_pair=num_pair,
            num_kernel=K,
            std_width=args.gaussian_std_width,
            start=args.gaussian_mean_start,
            stop=args.gaussian_mean_stop,
        )

        self.movement_pred_head = MovementPredictionHead(
            args.encoder_embed_dim, args.pair_embed_dim, args.encoder_attention_heads
        )

        # MTR head (descriptor regression on [CLS]); always built so a checkpoint
        # with mtr_loss_weight=0 still loads into one without.
        self.mtr_head = MTRHead(
            input_dim=args.encoder_embed_dim,
            output_dim=args.desc_n,
            hidden_dim=args.mtr_hidden_dim,
            activation_fn=args.mtr_activation_fn,
            pooler_dropout=args.mtr_dropout,
        )

        self.classification_heads = nn.ModuleDict()
        self.dtype = torch.float32
        self.apply(init_bert_params)

    def half(self):
        super().half()
        self.se3_invariant_kernel = self.se3_invariant_kernel.float()
        self.atom_feature = self.atom_feature.float()
        self.edge_feature = self.edge_feature.float()
        self.dtype = torch.half
        return self

    def bfloat16(self):
        super().bfloat16()
        self.se3_invariant_kernel = self.se3_invariant_kernel.float()
        self.atom_feature = self.atom_feature.float()
        self.edge_feature = self.edge_feature.float()
        self.dtype = torch.bfloat16
        return self

    def float(self):
        super().float()
        self.dtype = torch.float32
        return self

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        if getattr(task, "desc_n", None) and task.desc_n != getattr(args, "desc_n", None):
            args.desc_n = task.desc_n
            logger.info("unimol2_dual: desc_n synced from task -> %d", args.desc_n)
        return cls(args, task.dictionary)

    # ── the dual-stream forward pass ─────────────────────────────────────────
    def _encode(self, batched_data, pos, use_3d, use_2d):
        """Run one encoder pass.

        ``use_3d`` gates the SE(3)-invariant Gaussian RBF term; ``use_2d`` gates
        the topology / bond-order / ring terms. The ablation toggles are exactly
        these two booleans: hybrid = (T, T), pure_2d = (F, T), pure_3d = (T, F).
        """
        src_token = batched_data["src_token"]
        data_x = batched_data["atom_feat"]
        atom_mask = batched_data["atom_mask"]
        pair_type = batched_data["pair_type"]

        n_mol, n_atom = data_x.shape[:2]
        dtype = self.dtype

        token_feat = self.embed_tokens(src_token)
        x = self.atom_feature(batched_data, token_feat).type(dtype)

        # NOTE: the pairwise bias is built from the *unrepeated* (B, N+1, N+1)
        # mask — same order as the stock model — and only the attention mask gets
        # the head repeat.
        attn_mask = batched_data["attn_bias"].clone()
        attn_bias = torch.zeros_like(attn_mask)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.encoder_attention_heads, 1, 1)
        attn_mask = attn_mask.type(self.dtype)

        atom_mask_cls = torch.cat(
            [torch.ones(n_mol, 1, device=atom_mask.device, dtype=atom_mask.dtype),
             atom_mask], dim=1).type(self.dtype)
        pair_mask = atom_mask_cls.unsqueeze(-1) * atom_mask_cls.unsqueeze(-2)

        # 2D stream: shortest path + bond order (stock EdgeFeature) + ring bias.
        attn_bias = attn_bias.unsqueeze(-1).repeat(1, 1, 1, self.args.pair_embed_dim)
        if use_2d:
            attn_bias = self.edge_feature(batched_data, attn_bias)
            ring_flag = batched_data.get("ring_flag", None)
            if ring_flag is not None:
                # ring_flag is (n_mol, N, N) long over heavy atoms only (the
                # collater pads atom-block matrices without the [CLS] row/col);
                # place it into the atom block exactly like EdgeFeature does.
                rb = self.ring_bias(ring_flag).type(dtype)
                attn_bias[:, 1:, 1:, :] = attn_bias[:, 1:, 1:, :] + rb

        # 3D stream: SE(3)-invariant Gaussian RBF on pair distances.
        if use_3d:
            delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
            dist = delta_pos.norm(dim=-1)
            attn_bias_3d = self.se3_invariant_kernel(dist.detach(), pair_type)
            attn_bias[:, 1:, 1:, :] = attn_bias[:, 1:, 1:, :] + attn_bias_3d

        x, pair = self.encoder(
            x, attn_bias.type(dtype),
            atom_mask=atom_mask_cls, pair_mask=pair_mask, attn_mask=attn_mask,
        )
        return x, pair, attn_mask

    def forward(self, batched_data, encoder_masked_tokens=None,
                classification_head_name=None, features_only=False,
                contrastive_2d=False, **kwargs):
        src_token = batched_data["src_token"]
        data_x = batched_data["atom_feat"]
        atom_mask = batched_data["atom_mask"]
        pos = batched_data["src_pos"]

        ablation = getattr(self.args, "ablation", "hybrid")
        use_3d = ablation != "pure_2d"
        use_2d = ablation != "pure_3d"
        if not use_3d:
            # pure_2d: coordinates are zeroed so the movement head and any
            # residual 3D path see nothing; the RBF term is skipped above.
            pos = torch.zeros_like(pos)

        x, pair, attn_mask = self._encode(batched_data, pos, use_3d=use_3d, use_2d=use_2d)

        delta_pos = (pos.unsqueeze(1) - pos.unsqueeze(2)).detach()
        node_output = self.movement_pred_head(
            x[:, 1:, :], pair[:, 1:, 1:, :], attn_mask[:, :, 1:, 1:], delta_pos,
        )
        pos = pos + node_output

        encoder_distance = None
        encoder_coord = None
        logits = None
        mtr_pred = None

        if not features_only:
            if self.args.masked_token_loss > 0:
                logits = self.lm_head(x[:, 1:, :], encoder_masked_tokens)
            if self.args.masked_coord_loss > 0:
                encoder_coord = pos
            if self.args.masked_dist_loss > 0:
                encoder_distance = (pos.unsqueeze(1) - pos.unsqueeze(2)).norm(dim=-1)

        # MTR readout (always computed; the loss decides whether it is used).
        if self.args.mtr_pooling == "cls":
            mtr_pred = self.mtr_head(x[:, 0, :])
        else:
            mask = batched_data["atom_mask"].unsqueeze(-1).float()
            pooled = (x[:, 1:, :] * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            mtr_pred = self.mtr_head(pooled)

        if classification_head_name is not None:
            logits = self.classification_heads[classification_head_name](x)

        # Multi-view contrastive readout (plan §6.2 L_contrast): a second encoder
        # pass over the 2D view of the SAME molecule with the same weights.
        # Multi-view contrastive readout (plan §6.2 L_contrast). The "3D view"
        # is the main pass above; the "2D view" is a dedicated encoder pass over
        # the molecule's Kabsch-aligned 2D layout with the RBF term off (the
        # pure_2d configuration of these same weights), so the pair (z_3d, z_2d)
        # is two genuinely different geometric views of one molecule. Excluded
        # only in pure_2d, where the main pass already lacks 3D (degenerate
        # pair). pure_3d + contrastive is the plan §11 A2 / A-Lite recipe: the
        # vanilla Uni-Mol2 encoder as the 3D view against the topology-only 2D
        # view -- the contrastive wrapper without the hybrid pair injection.
        # The loss requests the pass only when contrast_loss_weight > 0 and the
        # per-batch Bernoulli gate fires, so the mode alone never triggers it.
        z_3d = x[:, 0, :]
        z_2d = None
        if contrastive_2d and ablation != "pure_2d":
            pos_2d_aligned = batched_data.get("pos_2d_aligned", None)
            if pos_2d_aligned is not None:
                x2, _, _ = self._encode(batched_data, pos_2d_aligned,
                                        use_3d=False, use_2d=True)
                z_2d = x2[:, 0, :]

        if self.args.mode == "infer":
            return x, pair

        return (
            logits,
            encoder_distance,
            encoder_coord,
            mtr_pred,
            z_3d,
            z_2d,
        )

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes,
                        inner_dim, prev_inner_dim))
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        """State from trainer to pass along to model at every update."""
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class ClassificationHead(nn.Module):
    """Sentence-level classification head (kept for fine-tuning parity)."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture("unimol2_dual", "unimol2_dual")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.pair_embed_dim = getattr(args, "pair_embed_dim", 512)
    args.pair_hidden_dim = getattr(args, "pair_hidden_dim", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 48)
    args.dropout = getattr(args, "dropout", 0.1)
    args.emb_dropout = getattr(args, "emb_dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
    args.max_seq_len = getattr(args, "max_seq_len", 512)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.post_ln = getattr(args, "post_ln", False)
    args.masked_token_loss = getattr(args, "masked_token_loss", -1.0)
    args.masked_coord_loss = getattr(args, "masked_coord_loss", -1.0)
    args.masked_dist_loss = getattr(args, "masked_dist_loss", -1.0)
    args.x_norm_loss = getattr(args, "x_norm_loss", -1.0)
    args.delta_pair_repr_norm_loss = getattr(args, "delta_pair_repr_norm_loss", -1.0)
    args.notri = getattr(args, "notri", False)
    args.gaussian_std_width = getattr(args, "gaussian_std_width", 1.0)
    args.gaussian_mean_start = getattr(args, "gaussian_mean_start", 0.0)
    args.gaussian_mean_stop = getattr(args, "gaussian_mean_stop", 9.0)
    # Option A defaults
    args.ablation = getattr(args, "ablation", "hybrid")
    args.ring_embed_dim = getattr(args, "ring_embed_dim", None)
    # MTR head (plan §5.2: 32 descriptors)
    args.desc_n = getattr(args, "desc_n", 32)
    args.mtr_pooling = getattr(args, "mtr_pooling", "cls")
    args.mtr_hidden_dim = getattr(args, "mtr_hidden_dim", 256)
    args.mtr_activation_fn = getattr(args, "mtr_activation_fn", "tanh")
    args.mtr_dropout = getattr(args, "mtr_dropout", 0.0)


@register_model_architecture("unimol2_dual", "unimol2_dual_84M")
def unimol_dual_84m_architecture(args):
    base_architecture(args)


@register_model_architecture("unimol2_dual", "unimol2_dual_164M")
def unimol_dual_164m_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 48)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
    base_architecture(args)


@register_model_architecture("unimol2_dual", "unimol2_dual_310M")
def unimol_dual_310m_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 32)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    base_architecture(args)


@register_model_architecture("unimol2_dual", "unimol2_dual_570M")
def unimol_dual_570m_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 32)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1536)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 96)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1536)
    base_architecture(args)


@register_model_architecture("unimol2_dual", "unimol2_dual_1100M")
def unimol_dual_1100m_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 64)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1536)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 96)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1536)
    base_architecture(args)
