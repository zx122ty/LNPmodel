#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify the 3D-conformer fine-tune path preserves fold alignment and leaks nothing.

The 3D path swaps MolTrain's data input from a CSV path to a dict carrying
atoms/coordinates. That introduces two alignment hazards worth testing explicitly:

  1. ROW ALIGNMENT — atoms[i] must belong to SMILES[i] and TARGET[i]. If the smiles
     validity filter in MolDataReader reordered or dropped rows independently of the
     coordinate columns, every molecule would be trained against another's label.
  2. FOLD IDENTITY — the injected cached folds are integer indices into the dataset
     order. If the featurised array is ordered differently, a scaffold split silently
     degenerates into a leaky one.

Both are checked by comparing the atom-symbol sequence the MODEL actually sees
against the one we passed in, per row, and by diffing the installed folds against
the cached ones. No training is required.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import GetPeriodicTable

RDLogger.DisableLog("rdApp.*")

REPO = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
sys.path.insert(0, str(REPO / "4_scripts"))
sys.path.insert(0, str(REPO / "4_scripts/Uni-Mol/unimol_tools"))
# Match finetune_moltrain.py's own sys.path: it inserts the VENDORED unimol_tools
# first, so that (0.1.6) is the library the real run uses. Verifying against the
# site-packages copy would exercise different input handling and prove nothing.

import conformer_3d  # noqa: E402
import unimol_tools  # noqa: E402
from unimol_tools.data.datahub import DataHub  # noqa: E402

_TOOLS_ROOT = Path(unimol_tools.__file__).parent.parent
print(f"[0] unimol_tools <- {_TOOLS_ROOT}")

XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
DS = "Sheet2"


def main() -> int:
    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", "RLU (log2)"])
    smiles = d["SMILES"].astype(str).str.strip().tolist()
    target = pd.to_numeric(d["RLU (log2)"]).values.astype(float)

    print(f"[1] generating 3D for {len(smiles)} molecules …")
    # Reuse the same cache file the fine-tune itself reads, so this check exercises
    # the exact geometries that will reach the model (and stays cheap on reruns).
    mols, kept, stats = conformer_3d.generate_3d_conformers(
        smiles, cache_path=conformer_3d.default_cache_path(DS, len(smiles)),
        n_workers=32)
    # Dropping is only meaningful if the caller also re-slices SMILES and TARGET,
    # exactly as finetune_moltrain.py does before installing the remapped folds.
    smiles = [smiles[i] for i in kept]
    target = target[kept]
    assert len(mols) == len(smiles) == len(target)
    print(f"    kept {len(kept)} rows ({stats.rate_3d:.1%} 3D)")

    payload = {"SMILES": smiles, "TARGET": target, "ROMol": mols}

    print("[2] building DataHub (featurisation only, no training) …")
    # model_name is load-bearing: DataHub._init_data only takes the
    # atoms/coordinates branch inside the `model_name == 'unimolv2'` arm
    # (datahub.py:111-124). Omitting it leaves no_h_list unbound. The real
    # fine-tune passes it through MolTrain(config.model_name).
    hub = DataHub(data=payload, is_train=True, save_path="/tmp/verify3d",
                  task="regression", data_type="molecule", model_name="unimolv2",
                  smiles_col="SMILES", target_cols="TARGET",
                  target_normalize="standard", conf_cache_level=0,
                  max_atoms=conformer_3d.MAX_ATOMS, seed=42, split_seed=42,
                  split="random", kfold=5)

    # The featurised inputs are what the model consumes; src_tokens are atomic numbers.
    # DataHub stores them under 'unimol_input' (datahub.py:124).
    inputs = hub.data["unimol_input"]
    print(f"    featurised rows: {len(inputs)}  (expected {len(smiles)})")
    assert len(inputs) == len(smiles), "row count changed during featurisation"

    print("[3] BOND GRAPH: model-visible graph must match the molecule's real graph …")
    # The regression case is a mol with no bonds at all: degree collapses, edge_attr
    # empties and shortest_path saturates at its 511 sentinel. Assert the graph the
    # model sees is the real one, and that the coordinates really are 3D.
    pt = GetPeriodicTable()
    bad_graph, bad_geo, mismatches = [], [], []
    for i, feat in enumerate(inputs):
        seen = [pt.GetElementSymbol(int(z)) for z in feat["src_tokens"]]
        expected = [a.GetSymbol() for a in Chem.RemoveHs(mols[i]).GetAtoms()]
        if seen != expected:
            mismatches.append(i)
        # degree > 2 means real branching is present (a bond-free mol gives all-2).
        if int(np.asarray(feat["degree"]).max()) <= 2:
            bad_graph.append(i)
        if int(np.asarray(feat["shortest_path"]).max()) >= 511:
            bad_graph.append(i)
        c = np.asarray(feat["src_coord"])
        if float(c[:, 2].max() - c[:, 2].min()) < conformer_3d.FLAT_Z_EXTENT:
            bad_geo.append(i)
    if mismatches:
        print(f"    !! {len(mismatches)} rows carry the wrong atoms; first: {mismatches[:3]}")
        return 1
    if bad_graph:
        print(f"    !! {len(bad_graph)} rows lost their bond graph; first: {bad_graph[:3]}")
        return 1
    if bad_geo:
        print(f"    !! {len(bad_geo)} rows are flat; first: {bad_geo[:3]}")
        return 1
    print(f"    OK: all {len(inputs)} rows have the correct atoms, a real bond graph "
          f"(degree>2 and no 511 sentinel), and non-planar coordinates")

    print("[4] FOLD IDENTITY: injecting cached folds and diffing …")
    cache = json.loads((REPO / f"5_outputs/split_cache/{DS}_scaffold_k5_seed1337.json")
                       .read_text())
    folds = [(np.asarray(f["train"], int), np.asarray(f["test"], int))
             for f in cache["folds"]]
    # Same transformation finetune_moltrain.py applies when rows were dropped.
    folds = conformer_3d.remap_folds(folds, kept)

    from unimol_tools.data import split as split_mod
    split_mod.Splitter.split = lambda self, **kw: folds

    hub2 = DataHub(data=payload, is_train=True, save_path="/tmp/verify3d",
                   task="regression", data_type="molecule", model_name="unimolv2",
                   smiles_col="SMILES", target_cols="TARGET",
                   target_normalize="standard", conf_cache_level=0,
                   max_atoms=conformer_3d.MAX_ATOMS, seed=1337, split_seed=1337,
                   split="random", kfold=5)
    installed = hub2.data["split_nfolds"]
    assert len(installed) == len(folds), "fold count changed"
    for k, ((tr_a, te_a), (tr_b, te_b)) in enumerate(zip(installed, folds)):
        assert np.array_equal(np.asarray(tr_a), tr_b), f"fold {k} train differs"
        assert np.array_equal(np.asarray(te_a), te_b), f"fold {k} test differs"
    print(f"    OK: all {len(folds)} folds identical to the cached scaffold folds")
    sizes = [len(te) for _tr, te in folds]
    print(f"    fold test sizes: {sizes}")

    print("[5] LEAKAGE: train/test disjoint, and no scaffold straddles the split …")
    # Reuse the project's own scaffold labelling (chem_splits.scaffold_group_ids) so the
    # leak check tests the SAME definition the splitter used, not a re-derivation that
    # could disagree with it.
    import chem_splits
    scaf = chem_splits.scaffold_group_ids(smiles)
    for k, (tr, te) in enumerate(folds):
        inter = set(tr) & set(te)
        assert not inter, f"fold {k} has {len(inter)} rows in both train and test"
        shared = set(scaf[tr]) & set(scaf[te])
        if shared:
            raise AssertionError(f"fold {k} shares scaffolds {list(shared)[:3]}")
    print(f"    OK: no row and no Murcko scaffold appears in both sides of any fold")

    print("\n=== ALL CHECKS PASSED ===")
    print(f"3D audit: {stats.summary()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
