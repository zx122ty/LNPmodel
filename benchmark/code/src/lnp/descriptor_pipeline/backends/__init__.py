"""Backend registry — maps short names to backend classes."""

from .base import DescriptorBackend
from .custom_lnp import CustomLNPBackend
from .fingerprints import FingerprintBackend, FingerprintTransformer
from .mordred_backend import MordredBackend
from .rdkit_2d import RDKit2DBackend

BACKEND_REGISTRY: dict[str, type[DescriptorBackend]] = {
    "rdkit": RDKit2DBackend,
    "fingerprint": FingerprintBackend,
    "lnp": CustomLNPBackend,
    "mordred": MordredBackend,
}

__all__ = [
    "DescriptorBackend",
    "RDKit2DBackend",
    "FingerprintBackend",
    "FingerprintTransformer",
    "CustomLNPBackend",
    "MordredBackend",
    "BACKEND_REGISTRY",
]
