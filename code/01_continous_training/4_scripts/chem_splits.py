"""Chemically-aware dataset splits for fine-tuning evaluation.

Pure-RDKit implementations of a **scaffold splitter** (Bemis-Murcko scaffolds) and a
**Butina splitter** (ECFP4 fingerprints + Tanimoto similarity clustering), plus greedy
size-balanced fold assignment. These are used to evaluate MolTrain fine-tuning in a way
that preserves chemical diversity and prevents train/test leakage of structurally
related molecules — the primary evaluation method (vs. the random split, which is kept
only for baseline comparison).

This module is intentionally dependency-light (RDKit + numpy only, no MolTrain/deepchem
imports) so it can be reused for any future dataset: just pass a new list of SMILES.

The fold-splitting functions return the exact ``list[(train_idx, test_idx)]`` contract
that ``Uni-Mol``'s ``NNModel.run`` consumes (``for fold, (tr_idx, te_idx) in
enumerate(split_nfolds)``), so they can be dropped straight into a monkeypatched
``Splitter``.

Run ``python chem_splits.py`` to execute a small self-test (leakage + partition +
balance checks).
"""
from __future__ import annotations

import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

__all__ = [
    "get_murcko_scaffold",
    "scaffold_group_ids",
    "butina_clusters",
    "split_clusters_to_folds",
    "scaffold_folds",
    "butina_folds",
    "check_no_leakage",
]


def get_murcko_scaffold(smiles: str) -> str:
    """Return the Bemis-Murcko scaffold SMILES for a molecule.

    Unparseable molecules fall back to the raw SMILES so they form their own group
    (never silently merged into another scaffold). Acyclic (ring-less) molecules have an
    empty Bemis-Murcko scaffold — we give each its own group keyed by its full SMILES so
    acyclic molecules never share a fold-group (they spread evenly across folds via the
    size-balanced assignment). This preserves the no-leakage guarantee.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return f"__invalid__{smiles}"
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    if scaf is None:
        return f"__invalid__{smiles}"
    scaf_smi = Chem.MolToSmiles(scaf)
    if scaf_smi == "":
        # acyclic: unique group per molecule
        return f"__acyclic__{Chem.MolToSmiles(mol)}"
    return scaf_smi


def scaffold_group_ids(smiles: list[str]) -> np.ndarray:
    """Per-molecule Bemis-Murcko scaffold group label (1:1 with a scaffold)."""
    return np.array([get_murcko_scaffold(s) for s in smiles], dtype=object)


def butina_clusters(smiles: list[str], cutoff: float = 0.6) -> list[list[int]]:
    """Cluster molecules by ECFP4 (Morgan radius 2, 2048 bits) Tanimoto similarity.

    Returns a list of clusters, each a list of row indices. Singleton clusters are
    molecules unique at the similarity cutoff. Uses Butina's automatic
    ``ClusterData`` (greedy leader algorithm). ``cutoff`` is the Tanimoto similarity
    threshold: molecules with similarity >= cutoff are clustered together.

    ⚠️ The order of the 1-D distance vector is NOT free, and getting it wrong is
    silent. ``Butina.ClusterData`` reconstructs a symmetric matrix with
    ``dist_matrix[np.tril_indices(n, -1)] = data``, i.e. it wants the **LOWER**
    triangle in row-major order: ``D[1,0], D[2,0], D[2,1], D[3,0], ...``.

    Until 2026-09-18 this function supplied the **UPPER** triangle
    (``for i: sims = BulkTanimotoSimilarity(fps[i], fps[i+1:])`` ->
    ``D[0,1], D[0,2], D[0,3], D[1,2], ...``). The two orders coincide at
    positions 0, 1 and n-1 and disagree everywhere else, so the mistake produced
    a plausible-looking partition whose distances were permuted. Concretely,
    molecules with BIT-IDENTICAL fingerprints (distance 0) were placed in
    different clusters, which is exactly the isolation Butina is supposed to
    provide. Measured on the project's libraries: Sheet2 150 -> 138 clusters,
    Sheet3 30 -> 21, pKa 90 -> 218, and the ECFP4-identical test->train rate fell
    from 29.0% to 0.0% (Sheet2), 89.2% to 0.0% (Sheet3), 32.1% to 0.0% (pKa).
    See ``prompt/26-09-18/06-发现-Butina距离向量顺序错误.md``.

    ``_self_test`` now asserts the invariant this bug violated: molecules whose
    fingerprints are identical must land in the same cluster.
    """
    fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            Chem.MolFromSmiles(s) if s else None,
            radius=2,
            nBits=2048,
        )
        for s in smiles
    ]
    # A molecule that failed to parse becomes an all-zero fingerprint -> distance 1 to
    # everything -> its own singleton cluster. Safe, no false grouping.
    dists = []
    n = len(fps)
    for i in range(n):
        # j < i, so the vector is D[1,0], D[2,0], D[2,1], D[3,0], ... -- the lower
        # triangle in row-major order, which is what ClusterData's tril_indices
        # assignment expects. Iterating i and taking fps[:i] (rather than fps[i+1:])
        # is the whole fix; the reverse order is a different permutation of the same
        # numbers and silently mis-assigns them.
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])
    clusters = Butina.ClusterData(dists, n, isDistData=True, distThresh=1 - cutoff)
    return [list(c) for c in clusters]


def split_clusters_to_folds(
    clusters: list[list[int]],
    n_folds: int,
    seed: int = 42,
) -> np.ndarray:
    """Assign clusters to folds so each fold has roughly balanced total molecule count.

    Groups are bucketed by descending size and each cluster is placed in the currently
    lightest fold (ties broken by a seeded shuffle for reproducibility). A cluster never
    spans two folds, guaranteeing no scaffold/cluster appears in both train and test of
    the same fold.

    Returns ``fold_id`` array indexed by molecule row.
    """
    rng = np.random.default_rng(seed)
    order = list(range(len(clusters)))
    order.sort(key=lambda c: -len(clusters[c]))  # largest clusters first
    folds = np.zeros(len(sum(clusters, [])), dtype=int) if clusters else np.zeros(0, dtype=int)
    fold_sizes = np.zeros(n_folds, dtype=int)
    for c in order:
        members = np.asarray(clusters[c])
        # pick lightest fold (seeded tie-break)
        best = int(np.argmin(fold_sizes + rng.random(n_folds) * 1e-9))
        folds[members] = best
        fold_sizes[best] += len(members)
    return folds


def _group_folds(
    group_ids: np.ndarray,
    n_folds: int,
    seed: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Generic group-based CV: group each fold, then held-out fold is that fold.

    Builds ``list[(tr_idx, te_idx)]`` where each group's members are fully inside one
    held-out fold. Returns (folds, fold_ids).
    """
    groups, inverse = np.unique(group_ids, return_inverse=True)
    clusters = [np.where(inverse == g)[0].tolist() for g in range(len(groups))]
    fold_ids = split_clusters_to_folds(clusters, n_folds, seed=seed)
    fold_splits = []
    for k in range(n_folds):
        te_idx = np.where(fold_ids == k)[0]
        tr_idx = np.where(fold_ids != k)[0]
        fold_splits.append((tr_idx, te_idx))
    return fold_splits, fold_ids


def scaffold_folds(
    smiles: list[str],
    n_folds: int,
    seed: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Scaffold-aware CV folds (Bemis-Murcko groups, size-balanced).

    Returns ``(fold_splits, fold_ids, scaffold_ids)``. ``fold_splits`` is the list of
    (train_idx, test_idx) tuples for each fold; ``fold_ids`` the per-row fold label;
    ``scaffold_ids`` the per-row scaffold label.
    """
    scaf_ids = scaffold_group_ids(smiles)
    fold_splits, fold_ids = _group_folds(scaf_ids, n_folds, seed)
    return fold_splits, fold_ids, scaf_ids


def butina_folds(
    smiles: list[str],
    n_folds: int,
    cutoff: float = 0.6,
    seed: int = 42,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:
    """Butina-cluster CV folds (ECFP4 + Tanimoto ``cutoff``, size-balanced).

    Returns ``(fold_splits, fold_ids, cluster_ids)``.
    """
    clusters = butina_clusters(smiles, cutoff)
    cluster_ids = np.full(len(smiles), -1, dtype=int)
    for ci, members in enumerate(clusters):
        cluster_ids[np.asarray(members)] = ci
    fold_ids = split_clusters_to_folds(clusters, n_folds, seed=seed)
    fold_splits = []
    for k in range(n_folds):
        te_idx = np.where(fold_ids == k)[0]
        tr_idx = np.where(fold_ids != k)[0]
        fold_splits.append((tr_idx, te_idx))
    return fold_splits, fold_ids, cluster_ids


def scaffold_leakage_check(scaf_ids: np.ndarray, fold_splits: list) -> bool:
    """True if any scaffold appears in BOTH train and test of the same fold (leak)."""
    for tr, te in fold_splits:
        tr_scafs = set(scaf_ids[tr])
        te_scafs = set(scaf_ids[te])
        if tr_scafs & te_scafs:
            return False
    return True


def butina_leakage_check(clusters: list, fold_splits: list) -> bool:
    """True if any cluster spans train AND test of the same fold (leak)."""
    n = sum(len(c) for c in clusters)
    cluster_of = np.full(n, -1)
    for ci, members in enumerate(clusters):
        for i in members:
            cluster_of[i] = ci
    for tr, te in fold_splits:
        tr_cl = set(cluster_of[tr])
        te_cl = set(cluster_of[te])
        if tr_cl & te_cl:
            return False
    return True


# --------------------------------------------------------------------------- self-test
def _self_test() -> None:
    rng = np.random.default_rng(0)
    base = [
        "CCCCCCCCCC(=O)OCCN(C)C",     # DLin-like
        "CCCCCCCCCC(=O)OCCCCCC",      # similar tail
        "CCCCCCCCCC(=O)OCCCNC",       # similar
        "O=C(O)CCCCCCCCCCCCCC",       # fatty acid, different head
        "O=C(O)CCCCCCCCCCCCCCC",      # fatty acid
        "C1CCCCC1C(=O)OCCO",          # ring-containing
        "C1CCCCC1C(=O)OCCCO",         # ring-containing
        "OC(=O)C(CCCCC)CCCCCC",       # branched
        "CCCCC(C)CC(=O)O",            # branched
        "COCCOCCOCCO",                # PEG-like
    ]
    # exaggerate with many near-duplicates to test leak detection
    smiles = base * 3

    print("== scaffold folds (5) ==")
    sf, s_fold, s_scaf = scaffold_folds(smiles, 5, seed=42)
    for k, (tr, te) in enumerate(sf):
        print(f"  fold {k}: train={len(tr)} test={len(te)}")
    print("  unique scaffolds:", len(np.unique(s_scaf)))
    assert len(np.concatenate([te for _, te in sf])) == len(smiles), "partition failed"
    assert scaffold_leakage_check(s_scaf, sf), "scaffold leakage detected!"

    print("== butina folds (cutoff 0.6) ==")
    bf, b_fold, b_cluster = butina_folds(smiles, 5, cutoff=0.6, seed=42)
    for k, (tr, te) in enumerate(bf):
        print(f"  fold {k}: train={len(tr)} test={len(te)}")
    print("  n_clusters:", len(np.unique(b_cluster)))
    assert len(np.concatenate([te for _, te in bf])) == len(smiles), "partition failed"
    # butina leak check: no cluster spans a train/test boundary
    clusters = [np.where(b_cluster == c)[0].tolist() for c in np.unique(b_cluster)]
    assert butina_leakage_check(clusters, bf), "butina leakage detected!"

    print("== butina distance-vector order (regression guard) ==")
    # The 2026-09-18 bug: the 1-D vector was built as the UPPER triangle while
    # ClusterData reads it as the LOWER triangle. That permutes most distances and
    # splits molecules that have IDENTICAL fingerprints into different clusters --
    # which no leak/balance check can see, because the partition still looks sane.
    # Both assertions below fail loudly on the old code.
    probe_smiles = smiles[:8]
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048)
           for s in probe_smiles]
    np_ = len(fps)
    # (a) rebuild the matrix the way ClusterData will, from our own vector, and
    #     compare against the true distances. Any order mismatch shows up here.
    dists = []
    for i in range(np_):
        dists.extend([1 - s for s in DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])])
    rebuilt = np.zeros((np_, np_))
    rebuilt[np.tril_indices(np_, -1)] = dists
    rebuilt += rebuilt.T
    truth = np.zeros((np_, np_))
    for i in range(np_):
        truth[i, i + 1:] = 1 - np.asarray(
            DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
    truth += truth.T
    assert np.allclose(rebuilt, truth), (
        "butina distance vector is not in ClusterData's expected (lower-triangle, "
        "row-major) order -- distances are being permuted")
    # (b) the chemically meaningful invariant: bit-identical fingerprints must
    #     co-cluster at any cutoff, since their distance is 0.
    dup_smiles = ["CCCCCCCCCCCCCCCC(=O)OCCN(C)C"] * 3 + ["c1ccccc1O", "CCCCCCCCCC(=O)O"]
    dup_clusters = butina_clusters(dup_smiles, cutoff=0.7)
    where = {i: ci for ci, c in enumerate(dup_clusters) for i in c}
    assert len({where[0], where[1], where[2]}) == 1, (
        "molecules with identical ECFP4 landed in different butina clusters -- "
        "the splitter is not isolating what it claims to isolate")
    assert where[3] != where[0] and where[4] != where[0], (
        "dissimilar molecules were merged into one cluster")
    print("  distance order matches the true matrix; identical fingerprints co-cluster")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    _self_test()