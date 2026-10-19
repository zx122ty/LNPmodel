"""
BaseFeaturizer — Abstract base class for molecular featurisation.

Every featurizer converts a list of SMILES strings into a fixed-size
feature matrix (n_samples, n_features).
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
import pandas as pd


class BaseFeaturizer(ABC):
    """Abstract featurizer with a standardised .featurize() signature."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the output feature vectors."""
        ...

    @abstractmethod
    def featurize(
        self,
        smiles_list: Union[List[str], np.ndarray, pd.Series],
    ) -> np.ndarray:
        """Convert SMILES to a feature matrix of shape (n, dim).

        Returns a NumPy array of float32.
        """
        ...


class MorganFPFeaturizer(BaseFeaturizer):
    """Morgan (circular) fingerprints — standard 2D-QSAR baseline."""

    def __init__(self, n_bits: int = 2048, radius: int = 2):
        self.n_bits = n_bits
        self.radius = radius

    @property
    def dim(self) -> int:
        return self.n_bits

    def featurize(self, smiles_list: Union[List[str], np.ndarray, pd.Series]) -> np.ndarray:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                fps.append(np.zeros(self.n_bits, dtype=np.float32))
            else:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
                fps.append(np.array(fp, dtype=np.float32))
        return np.array(fps)


class RDKitDescriptorFeaturizer(BaseFeaturizer):
    """2D physicochemical descriptors — 1D-QSAR baseline."""

    DESCRIPTOR_NAMES = [
        "MolWt", "MolLogP", "NumHDonors", "NumHAcceptors", "TPSA",
        "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
        "NumSaturatedRings", "NumHeteroatoms", "NumValenceElectrons",
        "NumRadicalElectrons", "QED",
    ]

    @property
    def dim(self) -> int:
        return len(self.DESCRIPTOR_NAMES)

    def featurize(self, smiles_list: Union[List[str], np.ndarray, pd.Series]) -> np.ndarray:
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        results = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                results.append(np.full(self.dim, np.nan, dtype=np.float32))
            else:
                results.append(np.array([
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.TPSA(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.NumAliphaticRings(mol),
                    Descriptors.NumSaturatedRings(mol),
                    Descriptors.NumHeteroatoms(mol),
                    Descriptors.NumValenceElectrons(mol),
                    Descriptors.NumRadicalElectrons(mol),
                    Descriptors.qed(mol),
                ], dtype=np.float32))
        return np.array(results)


class UniMolReprFeaturizer(BaseFeaturizer):
    """UniMol molecular representation via unimol_tools.

    Supports both UniMol v1 (default) and UniMol v2.
    """

    def __init__(
        self,
        model_name: str = "unimolv1",
        batch_size: int = 32,
        use_cuda: bool = True,
        max_atoms: int = 256,
        pretrained_model_path: Optional[str] = None,
        pretrained_dict_path: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_cuda = use_cuda
        self.max_atoms = max_atoms
        self.pretrained_model_path = pretrained_model_path
        self.pretrained_dict_path = pretrained_dict_path
        self._model = None

    @property
    def dim(self) -> int:
        return 512 if self.model_name == "unimolv1" else 512

    def _lazy_init(self):
        if self._model is not None:
            return
        from unimol_tools import UniMolRepr
        kwargs = dict(
            data_type="molecule",
            batch_size=self.batch_size,
            remove_hs=False,
            model_name=self.model_name,
            use_cuda=self.use_cuda,
            max_atoms=self.max_atoms,
        )
        if self.pretrained_model_path is not None:
            kwargs["pretrained_model_path"] = self.pretrained_model_path
        if self.pretrained_dict_path is not None:
            kwargs["pretrained_dict_path"] = self.pretrained_dict_path
        self._model = UniMolRepr(**kwargs)

    def featurize(self, smiles_list: Union[List[str], np.ndarray, pd.Series]) -> np.ndarray:
        self._lazy_init()
        smiles_list = [str(s) for s in smiles_list]
        reprs = self._model.get_repr(data=smiles_list, return_atomic_reprs=False)
        return np.array(reprs, dtype=np.float32)