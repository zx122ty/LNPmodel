"""Hash-based descriptor caching.

Cache keys are derived from a hash of (canonical SMILES set + config digest),
so results are invalidated automatically when molecules or settings change.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _smiles_hash(smiles_set: frozenset[str]) -> str:
    """Deterministic hash of the molecule set."""
    ordered = "\n".join(sorted(smiles_set))
    return hashlib.sha256(ordered.encode()).hexdigest()[:16]


def _config_hash(config_dict: dict) -> str:
    """Deterministic hash of the configuration."""
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


class DescriptorCache:
    """File-based cache for computed descriptor DataFrames.

    Cache files are stored as Parquet for speed and type preservation.
    Falls back to CSV if pyarrow is not available.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._use_parquet = self._check_parquet()

    @staticmethod
    def _check_parquet() -> bool:
        try:
            import pyarrow  # noqa: F401
            return True
        except ImportError:
            return False

    def _cache_path(self, backend_name: str, mol_hash: str, cfg_hash: str) -> Path:
        ext = ".parquet" if self._use_parquet else ".csv"
        return self._dir / f"cache_{backend_name}_{mol_hash}_{cfg_hash}{ext}"

    def get(
        self,
        backend_name: str,
        smiles_set: frozenset[str],
        config_dict: dict,
    ) -> pd.DataFrame | None:
        mol_h = _smiles_hash(smiles_set)
        cfg_h = _config_hash(config_dict)
        path = self._cache_path(backend_name, mol_h, cfg_h)

        if not path.exists():
            return None

        try:
            if self._use_parquet:
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, index_col=0)
            logger.info("Cache HIT for %s (%d × %d)", backend_name, df.shape[0], df.shape[1])
            return df
        except Exception:
            logger.warning("Cache read failed for %s, recomputing", backend_name, exc_info=True)
            path.unlink(missing_ok=True)
            return None

    def put(
        self,
        backend_name: str,
        smiles_set: frozenset[str],
        config_dict: dict,
        df: pd.DataFrame,
    ) -> None:
        mol_h = _smiles_hash(smiles_set)
        cfg_h = _config_hash(config_dict)
        path = self._cache_path(backend_name, mol_h, cfg_h)

        try:
            if self._use_parquet:
                df.to_parquet(path)
            else:
                df.to_csv(path)
            logger.debug("Cached %s → %s", backend_name, path)
        except Exception:
            logger.warning("Cache write failed for %s", backend_name, exc_info=True)

    def clear(self) -> int:
        """Remove all cached files. Returns number of files removed."""
        count = 0
        for f in self._dir.glob("cache_*"):
            f.unlink()
            count += 1
        logger.info("Cleared %d cached descriptor files", count)
        return count
