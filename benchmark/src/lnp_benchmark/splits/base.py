"""
BaseSplitter — Abstract base class for all dataset splitting strategies.

All splitters follow the same contract:
    splitter.split(dataset) → {"train": idx, "val": idx, "test": idx}

where idx is a 1D boolean or integer index array into the dataset.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


IndexArray = Union[np.ndarray, List[int]]


@dataclass
class Split:
    """Container for a single train/val/test split."""
    train: np.ndarray   # boolean mask or integer indices
    val: np.ndarray
    test: np.ndarray

    def to_mask(self, n: int) -> Dict[str, np.ndarray]:
        """Convert indices to boolean masks of length *n*."""
        mask = np.zeros(n, dtype=bool)
        return {
            "train": mask.copy(),
            "val": mask.copy(),
            "test": mask.copy(),
        }

    @property
    def sizes(self) -> Dict[str, int]:
        return {
            "train": len(self.train),
            "val": len(self.val),
            "test": len(self.test),
        }


class BaseSplitter(ABC):
    """Abstract splitter with a unified .split() signature."""

    def __init__(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ):
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for result tracking, e.g. 'random_seed42'."""
        ...

    @abstractmethod
    def split(self, smiles_list: List[str]) -> Split:
        """
        Return train/val/test index arrays.

        Parameters
        ----------
        smiles_list : List[str]
            Canonical SMILES of the full dataset (used by scaffold/cluster
            splitters to enforce OOD splits; random splitter ignores it).

        Returns
        -------
        Split with .train, .val, .test as integer index arrays.
        """
        ...

    def _split_indices(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Utility: split indices 0..n-1 into train/val/test by ratio."""
        indices = np.random.permutation(n)
        val_end = int(n * self.val_ratio)
        test_end = val_end + int(n * self.test_ratio)
        return (
            indices[test_end:],          # train
            indices[:val_end],           # val
            indices[val_end:test_end],   # test
        )


# ── Concrete splitters ──────────────────────────────────────────────────────

class RandomSplitter(BaseSplitter):
    """Simple random split by ratio."""

    @property
    def name(self) -> str:
        return f"random_seed{self.seed}"

    def split(self, smiles_list: List[str]) -> Split:
        rng = np.random.RandomState(self.seed)
        n = len(smiles_list)
        idx = rng.permutation(n)
        val_end = int(n * self.val_ratio)
        test_end = val_end + int(n * self.test_ratio)
        return Split(
            train=idx[test_end:],
            val=idx[:val_end],
            test=idx[val_end:test_end],
        )


class ScaffoldSplitter(BaseSplitter):
    """Bemis-Murcko scaffold-based split.

    Molecules are grouped by scaffold; the groups are randomly assigned to
    train/val/test to guarantee no scaffold overlap between sets.
    """

    @property
    def name(self) -> str:
        return f"scaffold_seed{self.seed}"

    def split(self, smiles_list: List[str]) -> Split:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        scaffold_to_indices: Dict[str, list] = {}
        for i, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
            scaffold_to_indices.setdefault(scaffold, []).append(i)

        scaffold_sets = list(scaffold_to_indices.values())
        rng = np.random.RandomState(self.seed)
        rng.shuffle(scaffold_sets)

        n = len(smiles_list)
        val_target = int(n * self.val_ratio)
        test_target = val_target + int(n * self.test_ratio)

        train_idx, val_idx, test_idx = [], [], []
        for group in scaffold_sets:
            if len(test_idx) < test_target:
                test_idx.extend(group)
            elif len(val_idx) < val_target:
                val_idx.extend(group)
            else:
                train_idx.extend(group)

        return Split(
            train=np.array(train_idx),
            val=np.array(val_idx),
            test=np.array(test_idx),
        )


class ClusterSplitter(BaseSplitter):
    """Tanimoto similarity clustering split.

    Molecules are clustered by Morgan fingerprint similarity (Butina
    clustering), then clusters are assigned to train/val/test.
    """

    def __init__(
        self,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        cutoff: float = 0.7,
    ):
        super().__init__(val_ratio, test_ratio, seed)
        self.cutoff = cutoff

    @property
    def name(self) -> str:
        return f"cluster{self.cutoff}_seed{self.seed}"

    def split(self, smiles_list: List[str]) -> Split:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        from rdkit.ML.Cluster import Butina

        mols = [Chem.MolFromSmiles(s) for s in smiles_list]
        fps = []
        valid_idx = []
        for i, mol in enumerate(mols):
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                fps.append(fp)
                valid_idx.append(i)

        # Distance matrix
        dists = []
        for i in range(len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[i])
            for j in range(i + 1, len(fps)):
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                dists.append(1 - sim)

        clusters = Butina.ClusterData(dists, len(fps), 1 - self.cutoff, isDistData=True)
        # Convert tuple clusters to list for shuffling
        clusters = [list(c) for c in clusters]
        rng = np.random.RandomState(self.seed)
        rng.shuffle(clusters)

        n = len(valid_idx)
        val_target = int(n * self.val_ratio)
        test_target = val_target + int(n * self.test_ratio)

        train_idx, val_idx, test_idx = [], [], []
        for cluster in clusters:
            members = [valid_idx[i] for i in cluster]
            if len(test_idx) < test_target:
                test_idx.extend(members)
            elif len(val_idx) < val_target:
                val_idx.extend(members)
            else:
                train_idx.extend(members)

        return Split(
            train=np.array(train_idx),
            val=np.array(val_idx),
            test=np.array(test_idx),
        )