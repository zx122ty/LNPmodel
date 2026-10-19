"""Post-computation descriptor filtering.

Designed for HDLSS (high-dimensional low-sample-size) settings where
aggressive filtering is necessary to prevent spurious correlations.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import FilterConfig

logger = logging.getLogger(__name__)


class DescriptorFilter:
    """Filter descriptors by variance, missingness, and pairwise correlation."""

    def __init__(self, config: FilterConfig | None = None) -> None:
        self._cfg = config or FilterConfig()

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply full filtering pipeline and return cleaned DataFrame."""
        original_cols = df.shape[1]

        df = self._handle_inf(df)
        df = self._handle_nan(df)
        df = self._remove_low_variance(df)
        df = self._remove_near_zero_variance(df)
        df = self._remove_high_correlation(df)

        logger.info(
            "Filtering: %d → %d descriptors (removed %d)",
            original_cols, df.shape[1], original_cols - df.shape[1],
        )
        return df

    def _handle_inf(self, df: pd.DataFrame) -> pd.DataFrame:
        n_inf = np.isinf(df.select_dtypes(include=[np.number]).values).sum()
        if n_inf == 0:
            return df

        if self._cfg.inf_handling == "to_nan":
            df = df.replace([np.inf, -np.inf], np.nan)
            logger.info("Replaced %d inf values with NaN", n_inf)
        elif self._cfg.inf_handling == "drop":
            bad_cols = df.columns[df.isin([np.inf, -np.inf]).any()].tolist()
            df = df.drop(columns=bad_cols)
            logger.info("Dropped %d columns containing inf", len(bad_cols))
        return df

    def _handle_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        nan_frac = df.isna().mean()
        high_nan = nan_frac[nan_frac > self._cfg.nan_threshold].index.tolist()

        if high_nan:
            logger.info(
                "Dropping %d columns with >%.0f%% missing values",
                len(high_nan), self._cfg.nan_threshold * 100,
            )
            df = df.drop(columns=high_nan)

        if self._cfg.nan_strategy == "drop_cols":
            remaining_nan = df.columns[df.isna().any()].tolist()
            if remaining_nan:
                df = df.drop(columns=remaining_nan)
                logger.info("Dropped %d additional columns with any NaN", len(remaining_nan))

        elif self._cfg.nan_strategy == "impute_median":
            for col in df.columns[df.isna().any()]:
                df[col] = df[col].fillna(df[col].median())
            logger.info("Imputed remaining NaN with column median")

        elif self._cfg.nan_strategy == "impute_zero":
            df = df.fillna(0.0)
            logger.info("Imputed remaining NaN with zero")

        elif self._cfg.nan_strategy == "drop_rows":
            n_before = df.shape[0]
            df = df.dropna()
            logger.info("Dropped %d rows with NaN", n_before - df.shape[0])

        return df

    def _remove_low_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.select_dtypes(include=[np.number])
        var = numeric.var()
        low = var[var < self._cfg.variance_threshold].index.tolist()
        if low:
            df = df.drop(columns=low)
            logger.info("Removed %d constant-variance descriptors (var < %g)", len(low), self._cfg.variance_threshold)
        return df

    def _remove_near_zero_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove features where >95% of values are the same."""
        threshold = self._cfg.near_zero_variance_threshold
        if threshold <= 0:
            return df

        to_drop = []
        for col in df.select_dtypes(include=[np.number]).columns:
            freq = df[col].value_counts(normalize=True)
            if freq.iloc[0] > (1.0 - threshold):
                to_drop.append(col)

        if to_drop:
            df = df.drop(columns=to_drop)
            logger.info("Removed %d near-zero-variance descriptors", len(to_drop))
        return df

    def _remove_high_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        threshold = self._cfg.correlation_threshold
        if threshold >= 1.0:
            return df

        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return df

        corr = numeric.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))

        to_drop: set[str] = set()
        for col in upper.columns:
            high_corr = upper.index[upper[col] > threshold].tolist()
            for paired in high_corr:
                if col not in to_drop:
                    to_drop.add(paired)

        if to_drop:
            df = df.drop(columns=list(to_drop))
            logger.info(
                "Removed %d highly correlated descriptors (|r| > %.2f)",
                len(to_drop), threshold,
            )
        return df


def compute_correlation_report(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """Return a DataFrame of highly correlated descriptor pairs for manual review."""
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()
    pairs = []

    for i in range(len(corr.columns)):
        for j in range(i + 1, len(corr.columns)):
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                pairs.append({
                    "feature_1": corr.columns[i],
                    "feature_2": corr.columns[j],
                    "correlation": round(r, 4),
                    "abs_correlation": round(abs(r), 4),
                })

    report = pd.DataFrame(pairs)
    if not report.empty:
        report = report.sort_values("abs_correlation", ascending=False)
    return report.reset_index(drop=True)
