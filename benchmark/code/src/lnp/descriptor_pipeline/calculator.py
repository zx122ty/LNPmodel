"""Main descriptor calculator — orchestrates backends, caching, and filtering."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from rdkit import Chem

from .backends import CustomLNPBackend, FingerprintBackend, MordredBackend, RDKit2DBackend
from .backends.base import DescriptorBackend
from .cache import DescriptorCache
from .config import DescriptorPipelineConfig
from .filters import DescriptorFilter

if TYPE_CHECKING:
    from rdkit.Chem import Mol

logger = logging.getLogger(__name__)


class DescriptorCalculator:
    """Orchestrates descriptor computation across multiple backends.

    Typical usage::

        from descriptor_pipeline import DescriptorCalculator, DescriptorPipelineConfig

        config = DescriptorPipelineConfig(descriptor_tier=2)
        calc = DescriptorCalculator(config)

        mols = calc.load_mols_from_dir("path/to/mol_files/")
        df = calc.compute(mols)              # raw descriptors
        df = calc.compute_and_filter(mols)   # with variance/correlation filtering
    """

    def __init__(self, config: DescriptorPipelineConfig | None = None) -> None:
        self._config = config or DescriptorPipelineConfig()
        self._backends = self._build_backends()
        self._filter = DescriptorFilter(self._config.filter_config)

        self._cache: DescriptorCache | None = None
        if self._config.use_cache and self._config.cache_dir is not None:
            self._cache = DescriptorCache(self._config.cache_dir)

        logging.basicConfig(level=getattr(logging, self._config.log_level.upper(), logging.INFO))

    def _build_backends(self) -> list[DescriptorBackend]:
        backends: list[DescriptorBackend] = []
        cfg = self._config

        if cfg.enable_rdkit:
            backends.append(RDKit2DBackend(subset=cfg.effective_rdkit_subset()))

        if cfg.enable_fingerprints:
            backends.append(FingerprintBackend(configs=cfg.fingerprints))

        if cfg.enable_custom_lnp:
            backends.append(CustomLNPBackend(
                smarts_motifs=cfg.smarts_motifs,
                compute_handcrafted=cfg.compute_handcrafted_lnp,
            ))

        if cfg.enable_mordred:
            backends.append(MordredBackend(
                subset=cfg.effective_mordred_subset(),
                use_curated=cfg.mordred_use_curated,
            ))

        logger.info("Active backends: %s", [b.name for b in backends])
        return backends

    @staticmethod
    def load_mols_from_dir(mol_dir: str | Path) -> dict[str, Mol]:
        """Load all .mol files from a directory.

        Returns:
            Dict mapping molecule name (stem of filename) to RDKit Mol.
        """
        mol_dir = Path(mol_dir)
        mols: dict[str, Mol] = {}
        failed: list[str] = []

        for path in sorted(mol_dir.glob("*.mol")):
            name = path.stem
            mol = Chem.MolFromMolFile(str(path), sanitize=True, removeHs=True)
            if mol is None:
                mol = Chem.MolFromMolFile(str(path), sanitize=False, removeHs=True)
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol)
                    except Exception:
                        failed.append(name)
                        continue
                else:
                    failed.append(name)
                    continue
            mols[name] = mol

        logger.info("Loaded %d/%d .mol files", len(mols), len(mols) + len(failed))
        if failed:
            logger.warning("Failed to parse: %s", failed)
        return mols

    @staticmethod
    def load_mols_from_smiles(smiles_dict: dict[str, str]) -> dict[str, Mol]:
        """Create Mol objects from a name→SMILES mapping."""
        mols: dict[str, Mol] = {}
        for name, smi in smiles_dict.items():
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mols[name] = mol
            else:
                logger.warning("Invalid SMILES for %s: %s", name, smi)
        return mols

    @staticmethod
    def mols_to_smiles(mols: dict[str, Mol]) -> dict[str, str]:
        """Convert Mol objects to canonical SMILES."""
        return {name: Chem.MolToSmiles(mol) for name, mol in mols.items()}

    def compute(self, mols: dict[str, Mol]) -> pd.DataFrame:
        """Compute descriptors from all active backends (no filtering).

        Returns:
            Concatenated DataFrame of all backend outputs, indexed by molecule name.
        """
        if not mols:
            raise ValueError("No molecules provided")

        smiles_set = frozenset(Chem.MolToSmiles(m) for m in mols.values())
        frames: list[pd.DataFrame] = []

        for backend in self._backends:
            t0 = time.perf_counter()

            # Check cache
            cached = None
            if self._cache is not None:
                cache_key = {"backend": backend.name, "tier": self._config.descriptor_tier}
                cached = self._cache.get(backend.name, smiles_set, cache_key)

            if cached is not None:
                frames.append(cached)
            else:
                df = backend.compute(mols)
                frames.append(df)
                if self._cache is not None:
                    self._cache.put(backend.name, smiles_set, cache_key, df)

            elapsed = time.perf_counter() - t0
            logger.info("Backend %s: %.2fs", backend.name, elapsed)

        if not frames:
            return pd.DataFrame(index=pd.Index(sorted(mols.keys()), name="name"))

        result = pd.concat(frames, axis=1)
        result = result.loc[sorted(mols.keys())]

        logger.info("Total descriptors: %d features × %d molecules", result.shape[1], result.shape[0])
        return result

    def compute_and_filter(self, mols: dict[str, Mol]) -> pd.DataFrame:
        """Compute descriptors and apply filtering pipeline."""
        raw = self.compute(mols)
        filtered = self._filter.filter(raw)
        return filtered

    def compute_with_report(self, mols: dict[str, Mol]) -> tuple[pd.DataFrame, dict]:
        """Compute, filter, and return both the result and a diagnostic report."""
        raw = self.compute(mols)

        report: dict = {
            "n_molecules": raw.shape[0],
            "n_raw_descriptors": raw.shape[1],
            "backends": [b.name for b in self._backends],
            "nan_per_column": raw.isna().sum().to_dict(),
            "tier": self._config.descriptor_tier,
        }

        filtered = self._filter.filter(raw)
        report["n_filtered_descriptors"] = filtered.shape[1]
        report["n_removed"] = raw.shape[1] - filtered.shape[1]
        report["p_over_n_raw"] = round(raw.shape[1] / max(raw.shape[0], 1), 2)
        report["p_over_n_filtered"] = round(filtered.shape[1] / max(filtered.shape[0], 1), 2)

        if report["p_over_n_filtered"] > 1.0:
            logger.warning(
                "p/n ratio = %.1f after filtering. Risk of overfitting is HIGH "
                "for n=%d samples. Consider using Tier 1 descriptors or additional "
                "feature selection inside CV folds.",
                report["p_over_n_filtered"], filtered.shape[0],
            )

        return filtered, report
