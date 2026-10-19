"""LUMI-lab evaluation protocol: sequential folds + relaxed Spearman.

Both are transcribed from LUMI-lab's own source so that "we ran their protocol"
is literally true rather than approximately true:

* ``relaxed_spearman`` — verbatim port of ``relaxed_spearman_correlation`` in
  ``model/unimol/hg_mterics.py:10-39``. It is LUMI's ``metric_for_best_model``
  (``unimol_huggingface_kfold.py:262``); note their ``compute_metrics`` reports
  pearson/spearman/mse/top25 but **never R²**.

* ``sequential_folds`` — LUMI's default ``--kfold-mode sequential``
  (``unimol_huggingface_kfold.py:24-25``) is ``KFold(n_splits=5, shuffle=False)``
  over the dataset in file order (``:178-193``).

WHY THIS SPLIT IS NOT A SCAFFOLD SPLIT
--------------------------------------
Sheet2 IS LUMI's 4CR-1920 library (verified: 1920/1920 canonical-SMILES overlap
and identical labels). It is a fully enumerated 4-component combinatorial set —
R1(20) x R2(4) x R3(4) x R4(6) = 1920, each molecule a unique Markush code.

Under ``shuffle=False`` every R-group VALUE that appears in a test fold has also
been seen in training (measured: R1 20/20, R2 4/4, R3 1-2/1-2, R4 6/6 for all
five folds). So this protocol measures **combinatorial interpolation** — the
model has seen every building block and is asked about a new assembly — whereas
our scaffold split measures **extrapolation to unseen scaffolds**. They are
different questions, and a domain-pretrained model can legitimately win one
while losing the other. Keep both; do not silently replace one with the other.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def relaxed_spearman(preds, targets, relax_ratio: float = 0.05) -> float:
    """Spearman where rank gaps within ``relax_ratio * n`` are forgiven.

    Verbatim port of LUMI-lab ``hg_mterics.py:10-39`` (including their use of the
    uncorrected 1 - 6*sum(d^2)/(n*(n^2-1)) form on tie-corrected ranks, which is
    why the value can exceed the classical Spearman).
    """
    preds = np.asarray(preds, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if len(preds) != len(targets):
        raise ValueError("preds and targets must have the same length")
    n = len(preds)
    if n < 2:
        return float("nan")

    x_rank = rankdata(preds)
    y_rank = rankdata(targets)
    delta = relax_ratio * n

    d = x_rank - y_rank
    d = np.where(np.abs(d) <= delta, 0, d)
    return float(1 - (6 * np.sum(d**2)) / (n * (n**2 - 1)))


def sequential_folds(n: int, kfold: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """LUMI's default split: contiguous blocks in file order, no shuffling.

    Byte-identical to ``sklearn.model_selection.KFold(n_splits=kfold, shuffle=False)``,
    which is what LUMI actually constructs. sklearn distributes the remainder to the
    FIRST ``n % kfold`` folds (making them one larger), so naive ``i*n//kfold``
    slicing diverges whenever ``n`` is not divisible by ``kfold`` — on Sheet3
    (n=768) that gave [153,154,153,154,154] instead of sklearn's
    [154,154,154,153,153]. Returns ``[(train_idx, test_idx), ...]``.
    """
    idx = np.arange(n)
    sizes = np.full(kfold, n // kfold, dtype=int)
    sizes[: n % kfold] += 1  # sklearn puts the remainder in the leading folds
    folds = []
    start = 0
    for size in sizes:
        end = start + int(size)
        test = idx[start:end]
        train = np.concatenate([idx[:start], idx[end:]])
        folds.append((train, test))
        start = end
    return folds
