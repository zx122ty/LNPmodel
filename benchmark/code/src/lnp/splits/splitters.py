"""Dataset splitters — random, scaffold (Bemis-Murcko), and cluster (Butina).

These produce train/val/test index arrays that are **structure-aware** where
requested, so similar molecules are kept together in one split to avoid
data leakage when reporting generalisation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union

import numpy as np
from rdkit import Chem


@dataclass
class Split:
    """Train / val / test integer index arrays into the full dataset."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    @property
    def sizes(self) -> dict:
        return dict(train=len(self.train), val=len(self.val), test=len(self.test))


class BaseSplitter(ABC):
    def __init__(self, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def split(self, smiles_list) -> Split: ...

    def _claim_groups(self, groups, n):
        """Assign shuffled groups to test, then val, then train by target size."""
        val_target = int(n * self.val_ratio)
        test_target = int(n * self.test_ratio)
        train_idx, val_idx, test_idx = [], [], []
        for group in groups:
            if len(test_idx) < test_target:
                test_idx.extend(group)
            elif len(val_idx) < val_target:
                val_idx.extend(group)
            else:
                train_idx.extend(group)
        return Split(
            train=np.array(train_idx, dtype=int),
            val=np.array(val_idx, dtype=int),
            test=np.array(test_idx, dtype=int),
        )


class RandomSplitter(BaseSplitter):
    @property
    def name(self) -> str:
        return f"random_seed{self.seed}"

    def split(self, smiles_list) -> Split:
        rng = np.random.RandomState(self.seed)
        n = len(smiles_list)
        groups = [[i] for i in rng.permutation(n)]
        return self._claim_groups(groups, n)


class ScaffoldSplitter(BaseSplitter):
    """Group by Bemis-Murcko scaffold; assign scaffold groups to splits."""

    @property
    def name(self) -> str:
        return f"scaffold_seed{self.seed}"

    def split(self, smiles_list) -> Split:
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaffold_to_idx = {}
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(str(smi)) if smi else None
            if mol is None:
                continue
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffold_to_idx.setdefault(scaf, []).append(i)

        groups = list(scaffold_to_idx.values())
        rng = np.random.RandomState(self.seed)
        rng.shuffle(groups)
        return self._claim_groups(groups, len(smiles_list))


class ClusterSplitter(BaseSplitter):
    """Butina clustering on Morgan fingerprints; clusters go to one split."""

    def __init__(self, val_ratio=0.1, test_ratio=0.1, seed=42, cutoff: float = 0.7):
        super().__init__(val_ratio, test_ratio, seed)
        self.cutoff = cutoff

    @property
    def name(self) -> str:
        return f"cluster{self.cutoff}_seed{self.seed}"

    def split(self, smiles_list) -> Split:
        from rdkit import DataStructs
        from rdkit.Chem import AllChem
        from rdkit.ML.Cluster import Butina

        fps, valid_idx = [], []
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(str(smi)) if smi else None
            if mol is not None:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
                valid_idx.append(i)

        num = len(fps)
        if num < 2:
            return self._claim_groups([[i] for i in range(len(smiles_list))], len(smiles_list))

        dists = []
        for i in range(num):
            for j in range(i + 1, num):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                dists.append(1 - sim)

        clusters = [list(c) for c in Butina.ClusterData(dists, num, 1 - self.cutoff, isDistData=True)]
        rng = np.random.RandomState(self.seed)
        rng.shuffle(clusters)

        # Map clustered positions back to original dataset indices.
        groups = [[valid_idx[pos] for pos in cluster] for cluster in clusters]
        return self._claim_groups(groups, len(smiles_list))


def get_splitter(kind: str, seed: int = 42, val_ratio=0.1, test_ratio=0.1, cutoff=0.7) -> BaseSplitter:
    """Factory helper."""
    kind = kind.lower().split("_")[0]
    if kind in ("random", "r"):
        return RandomSplitter(val_ratio, test_ratio, seed)
    if kind in ("scaffold", "scaf"):
        return ScaffoldSplitter(val_ratio, test_ratio, seed)
    if kind in ("cluster", "c", "clust"):
        return ClusterSplitter(val_ratio, test_ratio, seed, cutoff)
    raise ValueError(f"Unknown splitter kind: {kind}")