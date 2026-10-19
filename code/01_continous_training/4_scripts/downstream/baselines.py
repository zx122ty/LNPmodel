#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classical baselines for the downstream evaluations, as ``fit_predict`` closures.

These plug into ``run_matrix.run_sklearn_model``, which owns the fold contract:
every model receives the SAME frozen train/test indices, and predictions are
persisted per cell before scoring. Nothing here computes a split -- that is the
one rule this package exists to enforce.

Three arms, in the order the plan lists them (26-09-16 §5.2):

* ``ecfp4_rf``    -- Morgan/ECFP4 (r=2, 2048) + RandomForest. The project's
  historical non-neural reference (``frozen_readout.py`` prints the same arm),
  so a new task can be dropped beside the existing Sheet2 numbers.
* ``ecfp4_xgb``   -- the same fingerprint with gradient-boosted trees. On Sheet2
  the tuned XGB was the stronger of the two, and the whole "does the neural model
  actually win" question was settled with it.
* ``mlp_morgan``  -- the LUMI paper's own baseline family: a plain MLP on
  fingerprints. The project had never built one (handoff §6, item 6); the paper
  reports that this arm "dropped sharply on the top 25% subset", which is a claim
  about RANKING, so it needs the screening metrics too.

A note on scaling: the MLP standardises its inputs with statistics from the
TRAINING fold only. Fitting the scaler on all rows would leak test information
into every fold, and with a 2048-bit fingerprint the leak is small enough to look
like noise rather than a bug -- which is exactly why it is spelled out here.

Reproducibility note on ``ecfp4_rf``: the one pre-existing artifact
(``pKa|ecfp4_rf|butina|seed42|k5``) was produced by a driver script that was not
archived, so its exact RF settings are unknown. Re-running the arm with the
settings below reproduces it to within 0.006 R2 per fold (the residual is RF
internal parallelism, not a different configuration). Rather than tune the arm to
match a number whose provenance is lost, the parameters are frozen here and the
run is redone; both numbers are reported in the pKa write-up. Anything larger
than ~0.01 would have meant a genuinely different model and needed a re-run of
the comparison, which is why the check was run at all.
"""
from __future__ import annotations

import numpy as np

# Fingerprint settings are frozen to the project's reference arm
# (``frozen_readout.py``: radius 2, 2048 bits) so downstream numbers are
# comparable with the Sheet2 feature benchmark.
FP_RADIUS = 2
FP_BITS = 2048

# Fixed seeds. These models are deterministic given a seed, and the seed must NOT
# be the partition seed: the point of a 5-seed x 5-fold grid is that the model is
# held constant while the partition varies. Varying both would confound the two.
MODEL_SEED = 0


def morgan_fingerprints(smiles: list[str], radius: int = FP_RADIUS,
                        n_bits: int = FP_BITS) -> np.ndarray:
    """ECFP4 bit vectors as a dense float32 matrix.

    Dense, not sparse: 2048 bits is small, and every downstream consumer
    (sklearn trees, torch) is faster on a contiguous array than on CSR.
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return np.asarray(
        [np.asarray(gen.GetFingerprint(Chem.MolFromSmiles(s)), dtype=np.float32)
         for s in smiles], dtype=np.float32)


def make_ecfp4_rf(X: np.ndarray, n_estimators: int = 300):
    """RandomForest on fingerprints, one regressor per target.

    Per-target rather than a native multi-output forest: it matches how the
    neural arm is built (MolTrain puts an independent head on each of the 16
    MTR columns), and a shared multi-output forest would couple the columns
    through the split criterion -- a modelling choice the neural arm does not
    make. n_jobs=-1: this arm is CPU-only while the GPU runs the neural arms.
    """
    from sklearn.ensemble import RandomForestRegressor

    def fit_predict(tr, te):
        Y = _y_slice(tr)
        out = np.empty((len(te), Y.shape[1]))
        for j in range(Y.shape[1]):
            m = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1,
                                      random_state=MODEL_SEED)
            m.fit(X[tr], Y[:, j])
            out[:, j] = m.predict(X[te])
        return out

    return fit_predict


def make_ecfp4_xgb(X: np.ndarray, **params):
    """XGBoost on fingerprints, one booster per target.

    Defaults are the *tuned* Sheet2 configuration, not library defaults: on
    Sheet2 the default-parameter probe scored 0.649 while the tuned grid reached
    0.6448-0.79 depending on protocol, and using defaults here would re-run the
    mistake that made the traditional baseline look weaker than it is.
    """
    import xgboost as xgb

    cfg = dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
               colsample_bytree=0.5, reg_lambda=1.0)
    cfg.update(params)

    def fit_predict(tr, te):
        Y = _y_slice(tr)
        out = np.empty((len(te), Y.shape[1]))
        for j in range(Y.shape[1]):
            m = xgb.XGBRegressor(tree_method="hist", n_jobs=-1,
                                 random_state=MODEL_SEED, **cfg)
            m.fit(X[tr], Y[:, j])
            out[:, j] = m.predict(X[te])
        return out

    return fit_predict


def make_mlp_morgan(X: np.ndarray, hidden=(512, 256), epochs: int = 300,
                    lr: float = 1e-3, batch_size: int = 64, dropout: float = 0.1,
                    patience: int = 30, val_frac: float = 0.15):
    """2-hidden-layer MLP on fingerprints, in torch.

    Deliberately the *same shape* as the LUMI paper's fingerprint baseline so the
    comparison answers "does the pretrained 3D encoder beat an MLP on the same
    input?", not "does a better-tuned MLP beat it?". Two implementation choices
    that are not free:

    * The validation split for early stopping is carved out of the TRAINING fold.
      MolTrain does not do this (it validates on the test fold -- 坑 #12), so this
      arm is scored on a strictly harder protocol than the neural one. That
      asymmetry is stated in the report rather than papered over; it means a tie
      here is a win for the MLP, not for the neural model.
    * Inputs are standardised with training-fold statistics only (see module docstring).
    * The target is z-scored with training-fold statistics too. A 2048-bit
      fingerprint row has unit-scale entries while apparent pKa spans ~4-11, so
      an unscaled target puts the MSE loss and the Adam step size in a regime
      where the network barely moves -- the failure looks like "MLPs are bad at
      this" rather than "the loss was mis-scaled".
    """
    import torch
    import torch.nn as nn

    def fit_predict(tr, te):
        torch.manual_seed(MODEL_SEED)
        rng = np.random.default_rng(MODEL_SEED)
        y = _y_slice(tr).ravel()

        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd > 1e-8, sd, 1.0)
        Xtr = torch.from_numpy(((X[tr] - mu) / sd).astype(np.float32))
        Xte = torch.from_numpy(((X[te] - mu) / sd).astype(np.float32))
        y_mu, y_sd = float(y.mean()), float(y.std()) or 1.0
        ytr = torch.from_numpy(((y - y_mu) / y_sd).astype(np.float32)).reshape(-1, 1)

        # Internal validation split. A single-row val set would make early
        # stopping meaningless, so fall back to no early stopping on tiny folds.
        n_val = int(len(tr) * val_frac)
        use_val = n_val >= 8
        if use_val:
            perm = rng.permutation(len(tr))
            vi, ti = perm[:n_val], perm[n_val:]
        else:
            ti = np.arange(len(tr))

        layers, d = [], X.shape[1]
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        net = nn.Sequential(*layers)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
        lossf = nn.MSELoss()

        best, best_state, bad = np.inf, None, 0
        for _ in range(epochs):
            net.train()
            order = rng.permutation(len(ti))
            for b in range(0, len(ti), batch_size):
                idx = ti[order[b:b + batch_size]]
                opt.zero_grad()
                lossf(net(Xtr[idx]), ytr[idx]).backward()
                opt.step()
            if not use_val:
                continue
            net.eval()
            with torch.no_grad():
                vl = float(lossf(net(Xtr[vi]), ytr[vi]))
            if vl < best - 1e-5:
                best, bad = vl, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)

        net.eval()
        with torch.no_grad():
            # Undo the target scaling: the caller scores on the ORIGINAL scale, and
            # a z-scored prediction would silently score ~0 R2 against raw pKa.
            return net(Xte).numpy() * y_sd + y_mu

    return fit_predict


# --------------------------------------------------------------------------
# Target plumbing
# --------------------------------------------------------------------------
# ``run_sklearn_model`` hands fit_predict only (train_idx, test_idx), because the
# fold contract is its responsibility and the target is the caller's. Rather than
# pass y through every closure signature, it is bound once per run via
# ``bind_target``. A module global would be simpler and would silently keep the
# previous dataset's targets if two datasets were evaluated in one process.
_Y: np.ndarray | None = None


def bind_target(y: np.ndarray) -> None:
    """Set the target matrix the closures slice. Called once per (dataset, run).

    Always stored 2-D (``n x n_targets``) even for a single target: the
    single-target case is a 1-column matrix, not a vector. Keeping one shape
    everywhere is what lets the same closure serve pKa (1 column) and MTR (16)
    without a branch -- and a branch here is how a (n,) vs (n,1) mix-up would
    reach ``run_matrix``'s shape assertion as a confusing failure instead of a
    clear one.
    """
    global _Y
    _Y = np.asarray(y, float)
    if _Y.ndim == 1:
        _Y = _Y[:, None]


def _y_slice(idx) -> np.ndarray:
    if _Y is None:
        raise RuntimeError("baselines.bind_target(y) must be called before fitting")
    return _Y[np.asarray(idx, int)]


def atompair_fingerprints(smiles: list[str], n_bits: int = 1024) -> np.ndarray:
    """AtomPair bit vectors as a dense float32 matrix."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    gen = AllChem.GetAtomPairGenerator(fpSize=n_bits)
    return np.asarray(
        [np.asarray(gen.GetFingerprint(Chem.MolFromSmiles(s)), dtype=np.float32)
         for s in smiles], dtype=np.float32)


def make_xgb_atompair(X: np.ndarray, **params):
    """XGBoost on Topological AtomPair fingerprints (1024-bit).

    Settings frozen to arm_zoo.py: n_estimators=600, max_depth=6, lr=0.05,
    subsample=0.8, colsample_bytree=0.4, reg_lambda=1.0.
    """
    import xgboost as xgb

    cfg = dict(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8,
               colsample_bytree=0.4, reg_lambda=1.0)
    cfg.update(params)

    def fit_predict(tr, te):
        Y = _y_slice(tr)
        out = np.empty((len(te), Y.shape[1]))
        for j in range(Y.shape[1]):
            m = xgb.XGBRegressor(tree_method="hist", n_jobs=-1,
                                 random_state=MODEL_SEED, **cfg)
            m.fit(X[tr], Y[:, j])
            out[:, j] = m.predict(X[te])
        return out

    return fit_predict


REGISTRY = {
    "ecfp4_rf": make_ecfp4_rf,
    "ecfp4_xgb": make_ecfp4_xgb,
    "mlp_morgan": make_mlp_morgan,
    "xgb_atompair": make_xgb_atompair,
}

