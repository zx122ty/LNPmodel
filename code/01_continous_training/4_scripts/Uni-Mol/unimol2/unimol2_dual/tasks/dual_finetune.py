# Copyright (c) DP Technology. Apache-2.0 — derived from the official unimol2 package.
"""EnviroFM-v2 Option A fine-tuning task: ``envirofm_tox`` + ring flags.

The dual-stream model reads a ``ring_flag`` pair matrix in every encoder pass.
During pretraining the stock denoising pipeline never produced one, so this
package's pretraining task computed it on the fly; the fine-tuning pipeline has
the same gap and needs the same fix — but it must not lose anything the
protocol lock already enforces, which is why this subclasses
:mod:`envirofm_tox` (target standardisation from the fold's own train_opt
partition + the runtime test-leak assertion) rather than the stock task.

The only new behaviour here is a wrapper dataset that attaches ``ring_flag`` to
every fine-tune batch, computed with the exact same function as pretraining so
the two phases cannot drift apart. Everything else — head registration, LLRD
parameter publication, leak guardrail, target statistics — is inherited.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import torch
from unicore.data import BaseWrapperDataset
from unicore.tasks import register_task

from envirofm_tox.tasks.tox_finetune import ToxFinetuneTask
from unimol2_dual.tasks.unimol2_dual import _heavy_ring_flag
from unimol2.data.unimol2_dataset import pad_2d

logger = logging.getLogger(__name__)


class RingFlagDataset(BaseWrapperDataset):
    """Attach the shared-ring pair flag to a fine-tune feature batch.

    The wrapped dataset is the stock ``Unimol2FinetuneFeatureDataset`` (or its
    TTA-expanded form), whose items are already-collated single-molecule dicts
    with an ``atom_mask`` of shape ``(1, N)`` — the same convention the
    pretraining wrapper relies on. The flag is computed from the molecule's
    SMILES with the *pretraining* function (same cache, same RDKit aromaticity
    rule), so a pair that was "shared aromatic ring" at step 0 of pretraining
    is exactly that at every fine-tuning step.

    The SMILES comes from an index-aligned dataset (the nested dataset's
    ``smi_name`` leaf), not from the feature item: the stock feature dict does
    not carry one, and the two leaves are the same length by construction —
    the nested dataset asserts equal lengths at build time.
    """

    def __init__(self, dataset, smi_dataset):
        super().__init__(dataset)
        self.dataset = dataset
        self.smi_dataset = smi_dataset

    def set_epoch(self, epoch, **unused):
        self.dataset.set_epoch(epoch, **unused)
        if hasattr(self.smi_dataset, "set_epoch"):
            self.smi_dataset.set_epoch(epoch, **unused)

    @lru_cache(maxsize=16)
    def _flag_for(self, smi: str):
        flag = _heavy_ring_flag(smi)
        return None if flag is None else torch.from_numpy(flag.astype(np.int64)).long()

    def __getitem__(self, index: int):
        item = self.dataset[index]
        if not isinstance(item, dict):
            return item
        smi = self.smi_dataset[index]
        if not isinstance(smi, str) or not smi:
            return item
        flag = self._flag_for(smi)
        if flag is not None:
            # Shallow copy: the parent's getitem is lru_cached and returns a
            # shared dict; mutating it would leak our key into its cache.
            item = dict(item)
            item["ring_flag"] = flag
        return item

    def collater(self, items):
        batched = self.dataset.collater(items)
        if "ring_flag" in items[0]:
            # atom_mask is [B, max_node_num]; dim 1 is the node pad length.
            max_node = batched["atom_mask"].shape[1]
            flags = [it["ring_flag"] for it in items]
            batched["ring_flag"] = pad_2d(flags, max_node, pad_value=0)
        return batched


@register_task("envirofm_dual")
class DualFinetuneTask(ToxFinetuneTask):
    """``envirofm_tox`` with the Option A ring flag attached to every batch."""

    def load_dataset(self, split, **kwargs):
        super().load_dataset(split, **kwargs)
        # Wrap the already-built nested dataset's batched_data leaf. The nested
        # dict is flattened by NestedDictionaryDataset.__init__, so reach the
        # leaf through self.datasets[split].defn — the same object graph the
        # collater walks. For train the stock task wraps the nested dataset in
        # an EpochShuffleDataset (no_shuffle off), which has no .defn; walk
        # through BaseWrapperDataset.dataset until the nested one is reached so
        # every split gets the ring flag identically.
        nest = self.datasets[split]
        while not hasattr(nest, "defn") and hasattr(nest, "dataset"):
            nest = nest.dataset
        defn = getattr(nest, "defn", None)
        if not isinstance(defn, dict):
            raise RuntimeError(
                f"envirofm_dual: could not reach a NestedDictionaryDataset for "
                f"{split!r} (no .defn found through the wrapper chain); the "
                f"stock task layout changed."
            )
        key = "net_input.batched_data"
        smi_key = "smi_name"
        if key in defn and smi_key in defn:
            defn[key] = RingFlagDataset(defn[key], defn[smi_key])
            logger.info("envirofm_dual: ring flag attached to %s", split)
        else:
            raise RuntimeError(
                f"envirofm_dual: nested dataset for {split!r} has no "
                f"{key!r}/{smi_key!r} leaves; the stock task layout changed.")
