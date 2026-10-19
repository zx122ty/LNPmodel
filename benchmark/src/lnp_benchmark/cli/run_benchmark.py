"""
BenchmarkPipeline — Central orchestrator with parallel execution.

Uses joblib.Parallel to run (featurizer, split, model) combinations
concurrently, dramatically speeding up the benchmark.
"""

import copy
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed

from ..data.base import BaseLipidDataset
from ..data.loaders import LNPCSVDataset, LNPExcelDataset
from ..splits.base import RandomSplitter, ScaffoldSplitter, ClusterSplitter, Split
from ..features.base import (
    BaseFeaturizer,
    MorganFPFeaturizer,
    RDKitDescriptorFeaturizer,
    UniMolReprFeaturizer,
)
from ..models.base import BaseModel, ModelResult, compute_metrics
from ..models.tier1_ml import (
    make_xgboost, make_lightgbm, make_random_forest,
    make_mlp, make_linear, make_svr, optuna_optimise_xgboost,
)
from ..eval.metrics import (
    aggregate_seeds, paired_ttest, top_error_molecules,
    sample_by_fraction,
)
from ..eval.visualize import generate_all_plots
from ..utils.seed import seed_everything
from ..utils.logging import setup_logger


logger = setup_logger(__name__)


TIER1_FACTORY = {
    "XGBoost": make_xgboost,
    "LightGBM": make_lightgbm,
    "RandomForest": make_random_forest,
    "MLP": make_mlp,
    "Ridge": make_linear,
    "SVR": make_svr,
}


class BenchmarkPipeline:
    """End-to-end benchmark pipeline with parallel execution."""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self._setup_env()
        self._results: List[ModelResult] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self):
        """Execute the full benchmark pipeline."""
        logger.info("=" * 60)
        logger.info("LNP Benchmark Pipeline — Starting")
        logger.info("=" * 60)

        # 1. Load & standardise data
        dataset = self._load_dataset()
        df = dataset.to_dataframe()
        smiles_list = df["smiles_canonical"].tolist()
        y = df["target"].values
        logger.info(f"Loaded {len(df)} standardised records")

        # 2. Build featurizers
        featurizers = self._build_featurizers()
        logger.info(f"Built {len(featurizers)} featurizers: {list(featurizers.keys())}")

        # 3. Precompute all feature matrices
        feature_cache: Dict[str, np.ndarray] = {}
        for feat_name, featurizer in featurizers.items():
            logger.info(f"Featurizing with {feat_name} ...")
            feature_cache[feat_name] = featurizer.featurize(smiles_list)
            logger.info(f"  Feature matrix shape: {feature_cache[feat_name].shape}")

        # 4. Precompute all splits (they share the same smiles_list)
        split_cache: Dict[str, Dict[int, Split]] = {}
        for split_method in self.cfg["splits"]["methods"]:
            split_cache[split_method] = {}
            for seed in self.cfg["splits"]["seeds"]:
                splitter = self._build_splitter(split_method, seed)
                s = splitter.split(smiles_list)
                split_cache[split_method][seed] = s
                logger.info(
                    f"  Split {splitter.name}: "
                    f"train={s.sizes['train']}, "
                    f"val={s.sizes['val']}, "
                    f"test={s.sizes['test']}"
                )

        # 5. Build the flat task list for parallel execution
        tasks = []
        for feat_name, X in feature_cache.items():
            for split_method in self.cfg["splits"]["methods"]:
                for seed in self.cfg["splits"]["seeds"]:
                    split = split_cache[split_method][seed]
                    split_name = f"{split_method}_seed{seed}"
                    models = self._build_models()
                    for model in models:
                        tasks.append(
                            delayed(self._run_task)(
                                feat_name, model, X, y, split, split_name, seed
                            )
                        )

        logger.info(f"Running {len(tasks)} tasks in parallel ...")

        # 6. Execute in parallel
        n_jobs = self.cfg.get("execution", {}).get("n_jobs", -1)
        batch_size = self.cfg.get("execution", {}).get("batch_size", 1)
        results = Parallel(n_jobs=n_jobs, batch_size=batch_size, verbose=10)(
            tasks
        )

        # 7. Collect results
        for r in results:
            if r is not None:
                self._results.append(r)

        # 8. Aggregate & save
        self._save_results()

        # 9. Visualize
        viz_cfg = self.cfg.get("eval", {}).get("visualize", {})
        if viz_cfg.get("enabled", True):
            logger.info("Generating plots ...")
            raw_csv = str(self._output_dir / "raw_results.csv")
            agg_csv = str(self._output_dir / "aggregated_results.csv")
            _ = generate_all_plots(raw_csv, agg_csv, str(self._output_dir / "figures"))
            logger.info(f"Plots saved to {self._output_dir / 'figures'}")

    # ── Single task (runs in a worker process) ─────────────────────────────

    def _run_task(
        self,
        feat_name: str,
        model: BaseModel,
        X: np.ndarray,
        y: np.ndarray,
        split: Split,
        split_name: str,
        seed: int,
    ) -> Optional[ModelResult]:
        """Train + evaluate one model on one split, called in a worker."""
        try:
            X_train = X[split.train]
            y_train = y[split.train]
            X_val = X[split.val]
            y_val = y[split.val]
            X_test = X[split.test]
            y_test = y[split.test]

            model.fit(X_train, y_train, X_val, y_val)
            y_pred = model.predict(X_test)
            y_val_pred = model.predict(X_val)
            metrics = compute_metrics(y_test, y_pred)

            return ModelResult(
                model_name=f"{feat_name}_{model.name}",
                split_name=split_name,
                seed=seed,
                metrics=metrics,
                y_pred=y_pred,
                y_true=y_test,
                y_val_pred=y_val_pred,
                y_val_true=y_val,
            )
        except Exception as e:
            logger.error(f"Task failed: {feat_name}_{model.name}@{split_name}: {e}")
            return None

    # ── Internal helpers ────────────────────────────────────────────────────

    def _setup_env(self):
        env_cfg = self.cfg.get("env", {})
        if "hf_endpoint" in env_cfg:
            os.environ["HF_ENDPOINT"] = env_cfg["hf_endpoint"]
        seed_everything(env_cfg.get("seed", 42))
        self._output_dir = Path(self.cfg["output"]["base_dir"])
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _load_dataset(self) -> BaseLipidDataset:
        path = self.cfg["data"]["input_path"]
        if path.endswith(".xlsx"):
            return LNPExcelDataset(path)
        else:
            return LNPCSVDataset(
                path=path,
                smiles_col=self.cfg["data"].get("smiles_col", "mol"),
                target_col=self.cfg["data"].get("target_col", "RLU (log2)"),
                dataset_name="LUMI_lab_1920",
            )

    def _build_featurizers(self) -> Dict[str, BaseFeaturizer]:
        feats = {}
        morgan_cfg = self.cfg["features"].get("morgan", {})
        feats["MorganFP"] = MorganFPFeaturizer(
            n_bits=morgan_cfg.get("n_bits", 2048),
            radius=morgan_cfg.get("radius", 2),
        )
        if self.cfg["features"].get("rdkit_descriptors", True):
            feats["RDKitDesc"] = RDKitDescriptorFeaturizer()
        unimol_cfg = self.cfg["features"].get("unimol", {})
        if unimol_cfg.get("enabled", False):
            feats["UniMol"] = UniMolReprFeaturizer(
                model_name=unimol_cfg.get("model_name", "unimolv1"),
                batch_size=unimol_cfg.get("batch_size", 32),
                max_atoms=unimol_cfg.get("max_atoms", 256),
                pretrained_model_path=unimol_cfg.get("pretrained_model_path"),
                pretrained_dict_path=unimol_cfg.get("pretrained_dict_path"),
            )
        return feats

    def _build_splitter(self, method: str, seed: int):
        split_cfg = self.cfg["splits"]
        if method == "random":
            return RandomSplitter(
                val_ratio=split_cfg["val_ratio"],
                test_ratio=split_cfg["test_ratio"],
                seed=seed,
            )
        elif method == "scaffold":
            return ScaffoldSplitter(
                val_ratio=split_cfg["val_ratio"],
                test_ratio=split_cfg["test_ratio"],
                seed=seed,
            )
        elif method == "cluster":
            return ClusterSplitter(
                val_ratio=split_cfg["val_ratio"],
                test_ratio=split_cfg["test_ratio"],
                seed=seed,
                cutoff=split_cfg.get("cluster_cutoff", 0.7),
            )
        else:
            raise ValueError(f"Unknown split method: {method}")

    def _build_models(self) -> List[BaseModel]:
        models = []
        tier1_cfg = self.cfg["models"].get("tier1", {})
        if tier1_cfg.get("enabled", True):
            for name in tier1_cfg.get("models", ["XGBoost"]):
                factory = TIER1_FACTORY.get(name)
                if factory:
                    models.append(factory(seed=42))
        return models

    def _save_results(self):
        save_pred = self.cfg["output"].get("save_predictions", True)
        rows = []
        for r in self._results:
            row = {
                "model_name": r.model_name,
                "split_name": r.split_name,
                "seed": r.seed,
                **r.metrics,
            }
            if save_pred and r.y_pred is not None and r.y_true is not None:
                row["y_true"] = r.y_true.tolist()
                row["y_pred"] = r.y_pred.tolist()
            rows.append(row)
        raw_df = pd.DataFrame(rows)
        raw_path = self._output_dir / "raw_results.csv"
        raw_df.to_csv(raw_path, index=False)
        logger.info(f"Saved raw results to {raw_path}")

        agg_df = aggregate_seeds(rows)
        agg_path = self._output_dir / "aggregated_results.csv"
        agg_df.to_csv(agg_path, index=False)
        logger.info(f"Saved aggregated results to {agg_path}")

        leaderboard_cols = [c for c in agg_df.columns if c.endswith("_mean")]
        if leaderboard_cols:
            sort_col = "R2_mean" if "R2_mean" in leaderboard_cols else leaderboard_cols[0]
            leaderboard = agg_df.sort_values(sort_col, ascending=False)
            leaderboard_path = self._output_dir / "leaderboard.csv"
            leaderboard.to_csv(leaderboard_path, index=False)
            logger.info(f"Leaderboard saved to {leaderboard_path}")
            logger.info("\n" + leaderboard.to_string())

        self._save_significance(rows)
        self._save_top_errors()

    def _save_significance(self, rows: List[Dict]):
        df = pd.DataFrame(rows)
        if "Spearman" not in df.columns:
            return
        models = df["model_name"].unique()
        sig_records = []
        for i, m1 in enumerate(models):
            for m2 in models[i + 1:]:
                s1 = df[df["model_name"] == m1]["Spearman"].values
                s2 = df[df["model_name"] == m2]["Spearman"].values
                if len(s1) == len(s2) and len(s1) >= 2:
                    test = paired_ttest(s1.tolist(), s2.tolist())
                    sig_records.append({
                        "model_a": m1,
                        "model_b": m2,
                        "mean_diff": test["mean_diff"],
                        "p_value": test["p_value"],
                    })
        if sig_records:
            sig_df = pd.DataFrame(sig_records)
            sig_path = self._output_dir / "significance_tests.csv"
            sig_df.to_csv(sig_path, index=False)
            logger.info(f"Significance tests saved to {sig_path}")

    def _save_top_errors(self):
        top_k = self.cfg["eval"].get("error_analysis", {}).get("top_k", 50)
        if not top_k:
            return
        for r in self._results[-1:]:
            errors_df = top_error_molecules(
                r.y_true, r.y_pred,
                [""] * len(r.y_true), top_k=top_k,
            )
            err_path = self._output_dir / "top_errors.csv"
            errors_df.to_csv(err_path, index=False)
            logger.info(f"Top error molecules saved to {err_path}")