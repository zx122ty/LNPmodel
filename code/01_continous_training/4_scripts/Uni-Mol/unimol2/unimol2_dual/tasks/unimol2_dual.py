# Copyright (c) DP Technology. Apache-2.0 — derived from the official unimol2 task.
# EnviroFM-v2 Option A pretraining task: stock Uni-Mol2 denoising pipeline plus
# three batch fields the dual-stream model needs:
#
#   * ``ring_flag``      (n, N, N) long — shared-ring category per atom pair
#                                (0 none / 1 aromatic / 2 aliphatic), plan §5.2 b_ring.
#   * ``descriptors``    (n, desc_n) float32 — z-scored RDKit physchem targets for
#                                the MTR head (plan §5.2 item 4).
#   * ``pos_2d_aligned`` (n, N, 3) float32 — the molecule's 2D layout Kabsch-
#                                aligned onto its sampled 3D conformer, so the
#                                contrastive pair (z_3d, z_2d) compares the same
#                                frame rather than two arbitrary orientations.
#
# Everything else is inherited from the stock task: same masking, same noise,
# same conformer sampling, same crop/normalise order.

from __future__ import annotations

import logging
import threading
from functools import lru_cache

import numpy as np
import torch
from unicore.data import NestedDictionaryDataset, EpochShuffleDataset
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
from unimol2.data.unimol2_dataset import (
    get_optimal_transform,
    pad_1d_feat,
    pad_2d,
)
from unicore.tasks import UnicoreTask, register_task

logger = logging.getLogger(__name__)


# ── per-molecule 2D graph features: ring flags + descriptors ─────────────────
from unimol2_dual.descriptors import compute_descriptors

_RING_CACHE_LOCK = threading.Lock()
_RING_CACHE: dict[str, np.ndarray | None] = {}


def _heavy_ring_flag(smi: str) -> np.ndarray | None:
    """(N, N) long array over heavy atoms: 0 none, 1 shared aromatic ring,
    2 shared aliphatic ring. Computed once per SMILES and cached."""
    with _RING_CACHE_LOCK:
        if smi in _RING_CACHE:
            return _RING_CACHE[smi]

    from rdkit import Chem

    mol = Chem.MolFromSmiles(smi)
    out = None
    if mol is not None:
        mol = Chem.AddHs(mol)
        n_heavy = sum(1 for a in mol.GetAtoms() if a.GetSymbol() != "H")
        # uint8 keeps the per-molecule cache small (~1 KB vs 7 KB at int64);
        # the collater converts to long.
        flag = np.zeros((n_heavy, n_heavy), dtype=np.uint8)
        # RDKit >= 2026 removed RingInfo.NumAromaticRings(); a ring is aromatic
        # when every atom in it carries the aromatic flag (the same criterion
        # RDKit's own NumAromaticRings used).
        ri = mol.GetRingInfo()
        heavy_idx = [i for i, a in enumerate(mol.GetAtoms()) if a.GetSymbol() != "H"]
        pos_of = {global_i: k for k, global_i in enumerate(heavy_idx)}
        atoms_by_idx = list(mol.GetAtoms())
        for ring_atoms in ri.AtomRings():
            is_aromatic = all(atoms_by_idx[i].GetIsAromatic() for i in ring_atoms)
            cat = 1 if is_aromatic else 2
            heavy_ring = [pos_of[i] for i in ring_atoms if i in pos_of]
            for a in heavy_ring:
                for b in heavy_ring:
                    if a != b and flag[a, b] < cat:
                        flag[a, b] = cat
        out = flag

    with _RING_CACHE_LOCK:
        if len(_RING_CACHE) > 200_000:  # bound memory on the full corpus
            _RING_CACHE.clear()
        _RING_CACHE[smi] = out
    return out


class DualFeatureDataset(torch.utils.data.Dataset):
    """Wraps the stock feature pipeline and attaches ring_flag / descriptors /
    pos_2d_aligned to each item, then collates them alongside the parent batch.

    The parent (Unimol2FeatureDataset) returns an already-collated dict for a
    single molecule (batch dim 1); we mirror that shape so the outer
    NestedDictionaryDataset sees one consistent item per index.
    """

    def __init__(self, dataset, smi_dataset, coord_3d_dataset, coord_2d_dataset,
                 desc_names: list[str], desc_mean: np.ndarray, desc_std: np.ndarray,
                 aux_lookup: dict | None = None):
        super().__init__()
        self.aux_lookup = aux_lookup
        self.dataset = dataset
        self.smi_dataset = smi_dataset
        self.coord_3d_dataset = coord_3d_dataset
        self.coord_2d_dataset = coord_2d_dataset
        self.desc_names = list(desc_names)
        self.desc_mean = np.asarray(desc_mean, dtype=np.float32)
        self.desc_std = np.where(
            np.asarray(desc_std, dtype=np.float32) < 1e-8, 1.0,
            np.asarray(desc_std, dtype=np.float32))

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch, **unused):
        self.dataset.set_epoch(epoch, **unused)

    @lru_cache(maxsize=16)
    def _cached(self, epoch: int, idx: int):
        # Shallow copy: the parent's __getitem__ is lru_cached and returns a
        # shared dict; mutating it would leak our extra keys into its cache.
        item = dict(self.dataset[idx])
        smi = self.smi_dataset[idx]

        flag = _heavy_ring_flag(smi)
        if flag is not None:
            item["ring_flag"] = torch.from_numpy(flag.astype(np.int64)).long()

        if self.aux_lookup is not None:
            vec = self.aux_lookup.get(smi)
            if vec is None:
                vec = np.full(len(self.desc_names), np.nan, dtype=np.float32)
            item["descriptors"] = torch.from_numpy(np.asarray(vec, dtype=np.float32))
            desc = None
        else:
            desc = compute_descriptors(smi, self.desc_names)
        if desc is not None and len(self.desc_names):
            z = (desc - self.desc_mean) / self.desc_std
            # NaN-safe: the loss masks invalid targets; keep them as NaN so a
            # missing descriptor is visible rather than silently zero.
            item["descriptors"] = torch.from_numpy(z.astype(np.float32))

        # Kabsch-align the 2D layout onto the sampled 3D conformer (same
        # transform the stock pipeline uses to align noised coords to target).
        pos_3d = np.asarray(self.coord_3d_dataset[idx], dtype=np.float64)
        pos_2d = np.asarray(self.coord_2d_dataset[idx], dtype=np.float64)
        if pos_3d.shape == pos_2d.shape and len(pos_3d):
            # get_optimal_transform(src, tgt) maps src onto tgt (Kabsch), so the
            # 2D layout is the source and the sampled conformer the target.
            R, T = get_optimal_transform(pos_2d, pos_3d)
            aligned = pos_2d @ R + T
            item["pos_2d_aligned"] = torch.from_numpy(aligned.astype(np.float32))

        return item

    def __getitem__(self, idx: int):
        return self._cached(self.dataset.epoch, idx)

    def collater(self, items):
        batched = self.dataset.collater(items)
        # The parent pads every atom matrix to max_node_num; match that exactly
        # (atom_mask is [B, max_node_num], so its dim 1 is the node pad length).
        max_node = batched["atom_mask"].shape[1]
        if "ring_flag" in items[0]:
            batched["ring_flag"] = pad_2d(
                [it["ring_flag"] for it in items], max_node, pad_value=0)
        if "descriptors" in items[0]:
            batched["descriptors"] = torch.stack(
                [it["descriptors"] for it in items])
        if "pos_2d_aligned" in items[0]:
            batched["pos_2d_aligned"] = pad_1d_feat(
                [it["pos_2d_aligned"] for it in items], max_node)
        return batched


@register_task("unimol2_dual")
class UniMol2DualTask(UnicoreTask):
    """Uni-Mol2 denoising task + dual-stream batch fields (plan §5.2/§6.2)."""

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument("--mask-prob", default=0.15, type=float,
                            help="probability of replacing a token with mask")
        parser.add_argument("--drop-feat-prob", default=0.5, type=float,
                            help="probability of dropout the atom / bond feat")
        parser.add_argument("--mask-token-prob", default=0.15, type=float,
                            help="probability to mask token")
        parser.add_argument("--leave-unmasked-prob", default=0.0, type=float,
                            help="probability that a masked token is unmasked")
        parser.add_argument("--random-token-prob", default=0.0, type=float,
                            help="probability of replacing a token with a random token")
        parser.add_argument("--noise-type", default="uniform",
                            choices=["trunc_normal", "uniform", "normal", "none"],
                            help="noise type in coordinate noise")
        parser.add_argument("--noise", default=1.0, type=float,
                            help="coordinate noise for masked atoms")
        parser.add_argument("--max-atoms", type=int, default=256,
                            help="selected maximum number of atoms in a molecule")

        # ── MTR head targets (plan §5.2 item 4) ──────────────────────────────
        parser.add_argument("--descriptors-json", type=str, default=None,
                            help="JSON with descriptor_names / descriptor_mean / "
                                 "descriptor_std for the MTR targets")
        parser.add_argument("--aux-targets-pkl", type=str, default=None,
                            help="pickle {'names': [...], 'targets': {smiles: "
                                 "float32 vector, NaN = unobserved}}; replaces the "
                                 "RDKit descriptors as the MTR targets "
                                 "(supervised multi-task pretraining)")
        # ── loss weights (plan §6.2: L_token/L_coord/L_dist already come from
        #    --masked-*-loss; these two are Option A additions) ──────────────
        parser.add_argument("--mtr-loss-weight", type=float, default=0.5,
                            help="weight of the MTR descriptor-regression loss "
                                 "(plan §6.2 lambda_mtr)")
        parser.add_argument("--contrast-loss-weight", type=float, default=2.0,
                            help="weight of the multi-view NT-Xent contrastive "
                                 "loss (plan §6.2 lambda_contrast)")
        parser.add_argument("--contrast-temperature", type=float, default=0.1,
                            help="NT-Xent temperature tau (plan §6.2)")
        parser.add_argument("--contrast-prob", type=float, default=1.0,
                            help="per-batch probability of running the 2D "
                                 "contrastive view (second encoder pass); a "
                                 "compute-budget lever for the 96 GPU-hour "
                                 "ceiling — plan §6.2 specifies no gating")

    def __init__(self, args, dictionary=None):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        self.pad_idx = 0
        self.mask_idx = 127

        self.desc_names: list[str] = []
        self.desc_mean: np.ndarray | None = None
        self.desc_std: np.ndarray | None = None
        if getattr(args, "descriptors_json", None):
            import json
            with open(args.descriptors_json, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            self.desc_names = list(meta.get("descriptor_names", []))
            if self.desc_names:
                self.desc_mean = np.asarray(meta["descriptor_mean"], dtype=np.float32)
                self.desc_std = np.asarray(meta["descriptor_std"], dtype=np.float32)
                logger.info("unimol2_dual: %d MTR descriptors loaded", len(self.desc_names))

        self.aux_lookup = None
        if getattr(args, "aux_targets_pkl", None):
            import pickle
            with open(args.aux_targets_pkl, "rb") as fh:
                blob = pickle.load(fh)
            self.desc_names = list(blob["names"])
            self.aux_lookup = blob["targets"]
            logger.info("unimol2_dual: %d aux supervised targets, %d molecules",
                        len(self.desc_names), len(self.aux_lookup))

        from multiprocessing.pool import ThreadPool
        self.lmdb_copy_thread = ThreadPool(processes=1)

    @property
    def desc_n(self) -> int:
        return len(self.desc_names)

    @classmethod
    def setup_task(cls, args, **kwargs):
        return cls(args, dictionary=None)

    def load_dataset(self, split, combine=False, **kwargs):
        epoch = kwargs["epoch"]

        def one_dataset(seed):
            if self.args.mode == "train":
                raw_dataset = LMDBDataset(
                    self.args.data, split, epoch, self.args.max_epoch,
                    self.lmdb_copy_thread)
                # 10 3D + 1 2D conformers per record
                raw_dataset = Add2DConformerDataset(
                    raw_dataset, "smi", "atoms", "coordinates")
            else:
                raw_dataset = LMDBDataset(
                    self.args.data, split, 1, 1, self.lmdb_copy_thread)

            smi_dataset = KeyDataset(raw_dataset, "smi")
            dataset = ConformerSampleDataset(
                raw_dataset, seed, "atoms", "coordinates", "coordinates_2d")
            dataset = RemoveHydrogenDataset(
                dataset, "atoms", "coordinates", "coordinates_2d", True)
            dataset = CroppingDataset(
                dataset, seed, "atoms", "coordinates", "coordinates_2d",
                self.args.max_atoms)
            dataset = NormalizeDataset(
                dataset, "coordinates", "coordinates_2d", normalize_coord=True)

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

            # Option A batch fields. The raw LMDB record is read directly (same
            # index space as the pipeline above) so ring/descriptor alignment is
            # exact by construction.
            v2_feat = DualFeatureDataset(
                v2_feat,
                smi_dataset=smi_dataset,
                coord_3d_dataset=coord_dataset,
                coord_2d_dataset=coordinates_2d,
                desc_names=self.desc_names,
                desc_mean=(self.desc_mean if self.desc_mean is not None
                           else np.zeros(len(self.desc_names))),
                desc_std=(self.desc_std if self.desc_std is not None
                          else np.ones(len(self.desc_names))),
                aux_lookup=self.aux_lookup,
            )
            return v2_feat

        dataset = {"batched_data": one_dataset(self.args.seed)}
        dataset = NestedDictionaryDataset(dataset)
        if split in ["train", "train.small"]:
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)
        self.datasets[split] = dataset

    def build_model(self, args):
        from unicore import models
        return models.build_model(args, self)

    def build_loss(self, args):
        from unicore import losses
        return losses.build_loss(args, self)
