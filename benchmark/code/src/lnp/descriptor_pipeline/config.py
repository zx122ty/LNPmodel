"""Configuration dataclasses for the descriptor pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class FingerprintConfig:
    """Configuration for a single fingerprint type."""

    fp_type: Literal["morgan", "maccs", "atom_pair", "topological"]
    radius: int = 2
    n_bits: int = 1024
    pca_components: int | None = 30

    @property
    def name(self) -> str:
        if self.fp_type == "morgan":
            return f"morgan_r{self.radius}_{self.n_bits}"
        return self.fp_type


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for post-computation descriptor filtering."""

    variance_threshold: float = 1e-10
    near_zero_variance_threshold: float = 0.01
    correlation_threshold: float = 0.95
    nan_threshold: float = 0.1
    nan_strategy: Literal["drop_cols", "drop_rows", "impute_median", "impute_zero"] = "drop_cols"
    inf_handling: Literal["to_nan", "drop"] = "to_nan"


CURATED_RDKIT_SUBSET = [
    "MolWt", "HeavyAtomMolWt", "ExactMolWt",
    "MolLogP", "MolMR",
    "TPSA", "LabuteASA",
    "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds",
    "NumHeteroatoms", "NumValenceElectrons",
    "FractionCSP3",
    "RingCount", "NumAromaticRings", "NumAliphaticRings",
    "HeavyAtomCount",
    "NHOHCount", "NOCount",
    "NumRadicalElectrons",
    "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge",
    "BalabanJ", "BertzCT",
    "Chi0", "Chi1",
    "HallKierAlpha",
    "Kappa1", "Kappa2", "Kappa3",
    "PEOE_VSA1", "PEOE_VSA6", "PEOE_VSA7",
    "SMR_VSA5", "SMR_VSA7",
    "SlogP_VSA2", "SlogP_VSA5",
    "EState_VSA3", "EState_VSA5",
    "fr_ester", "fr_ether", "fr_NH0", "fr_NH1", "fr_NH2",
    "fr_Al_OH", "fr_Ar_OH",
    "fr_alkyl_halide", "fr_amide",
    "fr_C_O", "fr_C_O_noCOO",
]

CURATED_MORDRED_SUBSET = [
    "nAcid", "nBase",
    "SLogP", "SMR",
    "TopoPSA",
    "nHBAcc", "nHBDon",
    "nRot",
    "nHeavyAtom", "nAtom",
    "MW",
    "WPath", "WPol",
    "Zagreb1", "Zagreb2",
    "mZagreb1", "mZagreb2",
    "ABC", "ABCGG",
    "SpAbs_Dt", "SpMax_Dt",
    "BalabanJ",
    "BertzCT",
    "nBonds", "nBondsD", "nBondsS",
    "C1SP2", "C1SP3", "C2SP2", "C2SP3", "C3SP3",
    "AATS1d", "AATS1v", "AATS2d",
    "ATSC1d", "ATSC1v",
    "SsOH", "SdO", "SssO", "SsNH2", "SssNH",
    "Xch-5d", "Xch-6d",
    "ATS1d", "ATS1v",
    "GATS1d", "GATS1v",
    "MATS1d", "MATS1v",
    "ETA_epsilon_1", "ETA_epsilon_3",
]

DEFAULT_SMARTS_MOTIFS: dict[str, str] = {
    "n_ionizable_N": "[NX3;!$(NC=O);!$(NS=O);!$(N=*)]",
    "n_ester": "[CX3](=[OX1])[OX2][#6]",
    "n_thioether": "[#6][SX2][#6]",
    "n_hydroxyl": "[OX2H1][#6]",
    "n_C_double_bond": "[CX3]=[CX3]",
    "has_piperazine": "C1CN(CCN1)",
    "has_piperidine": "[NR1]1[CR1][CR1][CR1][CR1][CR1]1",
    "n_branch_C": "[CX4;!R]([#6])([#6])([#6])[#6]",
}


@dataclass
class DescriptorPipelineConfig:
    """Master configuration for the descriptor computation pipeline.

    Descriptor tier system (designed for small-sample SAR):
      Tier 1: Curated physicochemical + domain features (~20-50)
      Tier 2: Fingerprints with PCA reduction (~30-60)
      Tier 3: Full RDKit 2D descriptors (~200)
      Tier 4: Mordred descriptors (opt-in, subset-filtered)
    """

    # --- Backend selection ---
    enable_rdkit: bool = True
    enable_fingerprints: bool = True
    enable_custom_lnp: bool = True
    enable_mordred: bool = False

    # --- Descriptor tier ---
    descriptor_tier: Literal[1, 2, 3, 4] = 2

    # --- RDKit config ---
    rdkit_subset: list[str] | None = None
    rdkit_use_curated: bool = True

    # --- Mordred config ---
    mordred_subset: list[str] | None = None
    mordred_use_curated: bool = True

    # --- Fingerprint config ---
    fingerprints: list[FingerprintConfig] = field(default_factory=lambda: [
        FingerprintConfig(fp_type="morgan", radius=2, n_bits=1024, pca_components=30),
        FingerprintConfig(fp_type="maccs", radius=0, n_bits=167, pca_components=None),
    ])

    # --- Custom LNP features ---
    smarts_motifs: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SMARTS_MOTIFS))
    compute_handcrafted_lnp: bool = True

    # --- Filtering ---
    filter_config: FilterConfig = field(default_factory=FilterConfig)

    # --- Caching ---
    cache_dir: Path | None = None
    use_cache: bool = True

    # --- Computation ---
    n_jobs: int = 1
    random_seed: int = 42

    # --- Logging ---
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.descriptor_tier == 1:
            self.rdkit_use_curated = True
            self.enable_fingerprints = False
            self.enable_mordred = False
        elif self.descriptor_tier == 2:
            self.rdkit_use_curated = True
            self.enable_fingerprints = True
            self.enable_mordred = False
        elif self.descriptor_tier == 3:
            self.rdkit_use_curated = False
            self.enable_mordred = False
        elif self.descriptor_tier == 4:
            self.rdkit_use_curated = False
            self.enable_mordred = True
            self.mordred_use_curated = True

    def effective_rdkit_subset(self) -> list[str] | None:
        if self.rdkit_subset is not None:
            return self.rdkit_subset
        if self.rdkit_use_curated:
            return list(CURATED_RDKIT_SUBSET)
        return None

    def effective_mordred_subset(self) -> list[str] | None:
        if not self.enable_mordred:
            return None
        if self.mordred_subset is not None:
            return self.mordred_subset
        if self.mordred_use_curated:
            return list(CURATED_MORDRED_SUBSET)
        return None
