# Copyright (c) DP Technology. Apache-2.0 — derived from the official unimol2 model.
# v5: adds a shared MTR (descriptor-regression) MLP head on the [CLS] token.
# The MTR head is computed inside forward and returned as a 4th output, so the
# loss gets the pred with a clean gradient path into the backbone.

import logging

import torch
import torch.nn as nn
from unicore import utils
from unicore.models import BaseUnicoreModel, register_model, register_model_architecture
from unicore.modules import init_bert_params

from unimol2.models.unimol2 import (
    AtomFeature,
    EdgeFeature,
    SE3InvariantKernel,
    MovementPredictionHead,
    MaskLMHead,
)
from unimol2.models.transformer_encoder_with_pair import TransformerEncoderWithPair

logger = logging.getLogger(__name__)


@register_model("unimol2_mtr")
class UniMol2MTRModel(BaseUnicoreModel):
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
                            help="num encoder attention heads")
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
                            help="dropout probability for attention weights")
        parser.add_argument("--activation-dropout", type=float, metavar="D",
                            help="dropout probability after activation in FFN")
        parser.add_argument("--pooler-dropout", type=float, metavar="D",
                            help="dropout probability in the masked_lm pooler layers")
        parser.add_argument("--max-seq-len", type=int,
                            help="number of positional embeddings to learn")
        parser.add_argument("--post-ln", type=bool, help="use post layernorm or pre")
        parser.add_argument("--masked-token-loss", type=float, metavar="D",
                            help="mask loss ratio")
        parser.add_argument("--masked-dist-loss", type=float, metavar="D",
                            help="masked distance loss ratio")
        parser.add_argument("--masked-coord-loss", type=float, metavar="D",
                            help="masked coord loss ratio")
        parser.add_argument("--masked-coord-dist-loss", type=float, metavar="D",
                            help="masked coord dist loss ratio")
        parser.add_argument("--pair-embed-dim", type=int, metavar="N",
                            help="encoder embedding dimension for FFN")
        parser.add_argument("--pair-hidden-dim", type=int, metavar="N",
                            help="encoder embedding dimension for FFN")
        parser.add_argument("--pair-dropout", type=float, metavar="D",
                            help="dropout probability for pair")
        parser.add_argument("--droppath-prob", type=float, metavar="D",
                            help="stochastic path probability", default=0.0)
        parser.add_argument("--notri", action="store_true", help="disable trimul")
        parser.add_argument("--gaussian-std-width", type=float, default=1.0)
        parser.add_argument("--gaussian-mean-start", type=float, default=0.0)
        parser.add_argument("--gaussian-mean-stop", type=float, default=9.0)
        parser.add_argument("--mode", type=str, default="train",
                            choices=["train", "infer"])

        # MTR head
        parser.add_argument("--desc-n", type=int, default=34,
                            help="descriptor vector length (MTR output dim)")
        parser.add_argument("--mtr-pooling", type=str, default="cls",
                            choices=["cls", "mean"],
                            help="readout for the MTR head: [CLS] token or mean "
                                 "pooling over tokens (paper-style DA uses mean)")
        parser.add_argument("--mtr-hidden-dim", type=int, default=256,
                            help="MTR head hidden dim (paper uses 2-layer MLP)")
        parser.add_argument("--mtr-activation-fn",
                            choices=utils.get_available_activation_fns(),
                            default="tanh",
                            help="MTR head activation (paper uses ReLU)")
        parser.add_argument("--mtr-dropout", type=float, default=0.0,
                            help="MTR head dropout (paper uses 0.1)")

    def __init__(self, args, dictionary=None):
        super().__init__()
        base_architecture(args)
        self.args = args
        self.token_num = 128
        self.padding_idx = 0
        self.mask_idx = 127
        self.embed_tokens = nn.Embedding(self.token_num, args.encoder_embed_dim,
                                         self.padding_idx)

        num_atom = 512
        num_degree = 128
        num_edge = 64
        num_pair = 512
        num_spatial = 512

        self.atom_feature = AtomFeature(
            num_atom=num_atom, num_degree=num_degree, hidden_dim=args.encoder_embed_dim)
        self.edge_feature = EdgeFeature(
            pair_dim=args.pair_embed_dim, num_edge=num_edge, num_spatial=num_spatial)
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
            pair_dim=args.pair_embed_dim, num_pair=num_pair, num_kernel=K,
            std_width=args.gaussian_std_width, start=args.gaussian_mean_start,
            stop=args.gaussian_mean_stop,
        )
        self.movement_pred_head = MovementPredictionHead(
            args.encoder_embed_dim, args.pair_embed_dim, args.encoder_attention_heads)

        # ── MTR head (descriptor regression; readout by --mtr-pooling) ─────
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
        # The task knows the true descriptor dimension (from the LMDB manifest);
        # sync it before construction so the MTR head matches the data.
        if getattr(task, "desc_n", None) and task.desc_n != getattr(args, "desc_n", None):
            args.desc_n = task.desc_n
            logger.info("unimol2_mtr: desc_n synced from task -> %d", args.desc_n)
        return cls(args, task.dictionary)

    def forward(self, batched_data, encoder_masked_tokens=None,
                classification_head_name=None, features_only=False, **kwargs):
        src_token = batched_data["src_token"]
        data_x = batched_data["atom_feat"]
        atom_mask = batched_data["atom_mask"]
        pair_type = batched_data["pair_type"]
        pos = batched_data["src_pos"]

        n_mol, n_atom = data_x.shape[:2]
        token_feat = self.embed_tokens(src_token)
        x = self.atom_feature(batched_data, token_feat)
        dtype = self.dtype
        x = x.type(dtype)

        attn_mask = batched_data["attn_bias"].clone()
        attn_bias = torch.zeros_like(attn_mask)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.encoder_attention_heads, 1, 1)
        attn_bias = attn_bias.unsqueeze(-1).repeat(1, 1, 1, self.args.pair_embed_dim)
        attn_bias = self.edge_feature(batched_data, attn_bias)
        attn_mask = attn_mask.type(self.dtype)
        atom_mask_cls = torch.cat(
            [torch.ones(n_mol, 1, device=atom_mask.device, dtype=atom_mask.dtype),
             atom_mask], dim=1).type(self.dtype)
        pair_mask = atom_mask_cls.unsqueeze(-1) * atom_mask_cls.unsqueeze(-2)

        delta_pos = pos.unsqueeze(1) - pos.unsqueeze(2)
        dist = delta_pos.norm(dim=-1)
        attn_bias_3d = self.se3_invariant_kernel(dist.detach(), pair_type)
        new_attn_bias = attn_bias.clone()
        new_attn_bias[:, 1:, 1:, :] += attn_bias_3d
        new_attn_bias = new_attn_bias.type(dtype)
        x, pair = self.encoder(x, new_attn_bias, atom_mask=atom_mask_cls,
                               pair_mask=pair_mask, attn_mask=attn_mask)
        node_output = self.movement_pred_head(
            x[:, 1:, :], pair[:, 1:, 1:, :], attn_mask[:, :, 1:, 1:], delta_pos.detach())
        pos = pos + node_output

        encoder_distance = None
        encoder_coord = None
        logits = None

        if not features_only:
            if self.args.masked_token_loss > 0:
                logits = self.lm_head(x[:, 1:, :], encoder_masked_tokens)
            if self.args.masked_coord_loss > 0:
                encoder_coord = pos
            if self.args.masked_dist_loss > 0:
                encoder_distance = (pos.unsqueeze(1) - pos.unsqueeze(2)).norm(dim=-1)

        if classification_head_name is not None:
            logits = self.classification_heads[classification_head_name](x)

        # ── MTR head (always built; prediction gated by the loss) ──────────
        mtr_pred = None
        if not features_only:
            if self.args.mtr_pooling == "mean":
                # paper-style: mean pooling over tokens (incl. [CLS])
                mtr_pred = self.mtr_head(x.mean(dim=1))
            else:
                mtr_pred = self.mtr_head(x[:, 0, :])

        if self.args.mode == "infer":
            return x, pair
        else:
            return (logits, encoder_distance, encoder_coord, mtr_pred)

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim))
        self.classification_heads[name] = ClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
        )

    def set_num_updates(self, num_updates):
        self._num_updates = num_updates

    def get_num_updates(self):
        return self._num_updates


class MTRHead(nn.Module):
    """Shared descriptor-regression head on the [CLS] token (MLP)."""

    def __init__(self, input_dim, output_dim, hidden_dim=256, activation_fn="tanh",
                 pooler_dropout=0.0):
        super().__init__()
        self.dense = nn.Linear(input_dim, hidden_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, cls_feat, **kwargs):
        x = self.dropout(cls_feat)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class ClassificationHead(nn.Module):
    """Sentence-level classification head (kept for fine-tuning parity)."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


@register_model_architecture("unimol2_mtr", "unimol2_mtr")
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
    args.desc_n = getattr(args, "desc_n", 34)
    args.mtr_loss_weight = getattr(args, "mtr_loss_weight", -1.0)


@register_model_architecture("unimol2_mtr", "unimol2_mtr_84M")
def unimol_84m_architecture(args):
    base_architecture(args)


@register_model_architecture("unimol2_mtr", "unimol2_mtr_164M")
def unimol_164m_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 48)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
    base_architecture(args)


@register_model_architecture("unimol2_mtr", "unimol2_mtr_310M")
def unimol_310m_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 32)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    base_architecture(args)


@register_model_architecture("unimol2_mtr", "unimol2_mtr_570M")
def unimol_570m_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 32)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1536)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 96)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1536)
    base_architecture(args)


@register_model_architecture("unimol2_mtr", "unimol2_mtr_1100M")
def unimol_1100m_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 64)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1536)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 96)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1536)
    base_architecture(args)