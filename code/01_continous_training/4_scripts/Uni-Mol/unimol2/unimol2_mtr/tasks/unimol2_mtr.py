# Copyright (c) DP Technology. Apache-2.0 — derived from the official unimol2 task.
# v5: adds MTR (descriptor regression on the [CLS] token) to the continued
# pre-training objective. All other behaviour is identical to the stock task.

import logging
import os

import numpy as np
from unicore.data import (
    NestedDictionaryDataset,
    EpochShuffleDataset,
)
from unimol2.data import (
    Add2DConformerDataset,
    ConformerSampleDataset,
    CroppingDataset,
    IndexAtomDataset,
    KeyDataset,
    LMDBDataset,
    NormalizeDataset,
    RemoveHydrogenDataset,
    Unimol2FeatureDataset,
)
from unicore.tasks import UnicoreTask, register_task

from unimol2_mtr.data.mtr_dataset import MTRTargetDataset  # MTR descriptor targets

logger = logging.getLogger(__name__)

DESC_N = 34  # placeholder; overridden from --descriptors-json at runtime


@register_task("unimol2_mtr")
class UniMol2MTRTask(UnicoreTask):
    """Uni-Mol2 task + a descriptor-regression (MTR) head on the [CLS] token."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list",
        )
        parser.add_argument(
            "--mask-prob", default=0.15, type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--drop-feat-prob", default=0.5, type=float,
            help="probability of dropout the atom / bond feat",
        )
        parser.add_argument(
            "--mask-token-prob", default=0.15, type=float,
            help="probability of mask token",
        )
        parser.add_argument(
            "--leave-unmasked-prob", default=0.0, type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob", default=0.0, type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--noise-type", default="uniform",
            choices=["trunc_normal", "uniform", "normal", "none"],
            help="noise type in coordinate noise",
        )
        parser.add_argument("--noise", default=1.0, type=float, help="coordinate noise")
        parser.add_argument(
            "--max-atoms", type=int, default=256,
            help="selected maximum number of atoms in a molecule",
        )

        # ── MTR head ─────────────────────────────────────────────────────────
        parser.add_argument(
            "--descriptors-json", type=str, default=None,
            help="Path to a JSON file with descriptor metadata "
                 "(keys: descriptor_names / descriptor_mean / descriptor_std).",
        )
        parser.add_argument(
            "--mtr-loss-weight", type=float, default=1.0,
            help="weight of the MTR (descriptor regression) loss in the total loss.",
        )

    def __init__(self, args, dictionary=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        self.pad_idx = 0
        self.mask_idx = 127

        # descriptor metadata
        self.desc_n = args.desc_n
        self.desc_names: list[str] = []
        self.desc_mean: list[float] = []
        self.desc_std: list[float] = []
        if getattr(args, "descriptors_json", None):
            import json
            with open(args.descriptors_json, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self.desc_names = meta.get("descriptor_names", [])
            self.desc_mean = meta.get("descriptor_mean", [])
            self.desc_std = meta.get("descriptor_std", [])
            if self.desc_names:
                self.desc_n = len(self.desc_names)
            self.has_descriptor_stats = bool(self.desc_mean and self.desc_std)
            logger.info(
                "unimol2_mtr: %d descriptors (stats: %s) — MTR head on [CLS]",
                self.desc_n, self.has_descriptor_stats,
            )

        from multiprocessing.pool import ThreadPool
        self.lmdb_copy_thread = ThreadPool(processes=1)

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args, dictionary=None)

    def load_dataset(self, split, combine=False, **kwargs):
        epoch = kwargs["epoch"]

        def one_dataset(seed):
            if self.args.mode == "train":
                # raw LMDB record (smi/atoms/coordinates/descriptors) is read once
                # per sample; Add2DConformerDataset DROPS non-graph keys, so the
                # MTR wrapper reads descriptors from this raw_lmdb directly.
                raw_lmdb = LMDBDataset(
                    self.args.data, split, epoch, self.args.max_epoch,
                    self.lmdb_copy_thread,
                )
                raw_dataset = Add2DConformerDataset(
                    raw_lmdb, "smi", "atoms", "coordinates"
                )
            else:
                raw_lmdb = LMDBDataset(self.args.data, split, 1, 1, self.lmdb_copy_thread)
                raw_dataset = raw_lmdb

            smi_dataset = KeyDataset(raw_dataset, "smi")
            dataset = ConformerSampleDataset(
                raw_dataset, seed, "atoms", "coordinates", "coordinates_2d"
            )
            dataset = RemoveHydrogenDataset(
                dataset, "atoms", "coordinates", "coordinates_2d", True
            )
            dataset = CroppingDataset(
                dataset, seed, "atoms", "coordinates", "coordinates_2d",
                self.args.max_atoms,
            )
            dataset = NormalizeDataset(
                dataset, "coordinates", "coordinates_2d", normalize_coord=True
            )

            token_dataset = KeyDataset(dataset, "atoms")
            origin_token_dataset = IndexAtomDataset(smi_dataset, token_dataset)
            coord_dataset = KeyDataset(dataset, "coordinates")
            coordinates_2d = KeyDataset(dataset, "coordinates_2d")

            v2_feat = Unimol2FeatureDataset(
                smi_dataset=smi_dataset,
                token_dataset=origin_token_dataset,
                mask_token_prob=self.args.mask_token_prob,
                pad_idx=self.pad_idx,
                mask_idx=self.mask_idx,
                src_pos_dataset=coord_dataset,
                mask_pos_prob=self.args.mask_prob,
                noise=self.args.noise,
                noise_type=self.args.noise_type,
                drop_feat_prob=self.args.drop_feat_prob,
                seed=seed,
                src_2d_pos_dataset=coordinates_2d,
            )

            # MTR target (descriptor vector read from the raw LMDB record).
            v2_feat = MTRTargetDataset(
                v2_feat,
                raw_lmdb,
                "descriptors",
                pad_idx=self.pad_idx,
            )
            return v2_feat

        dataset = {
            "batched_data": one_dataset(self.args.seed),
        }
        dataset = NestedDictionaryDataset(dataset)
        if split in ["train", "train.small"]:
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)
        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models
        model = models.build_model(args, self)
        return model

    @property
    def desc_len(self) -> int:
        return self.desc_n