#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_v5_dataset.py — build the LipidFM v5 pre-training dataset.

v5 corpus (target ~1,000,000 unique molecules):
    *  300,000 ionizable lipids   (source: generated_virtual_library_balanced.csv)
    *  700,000 ChEMBL general     (source: chembl_37_chemreps.txt.gz, reservoir-sampled)

The 7:3 ratio (general : ionizable-lipid) is implemented as *counts*: the LMDB built
from this SDF holds 700k general + 300k lipid molecules, and Uni-Mol2's
ConformerSampleDataset->EpochShuffle resamples them together every epoch, so each
epoch (≈ each batch) sees ~70% general molecules — a strict "data replay"
anti-forgetting signal, exactly as in the reference advice.

For every molecule we also compute a fixed MTR (Multi-Task Regression) target vector
of Z-scored RDKit descriptors. These become the regression labels of the optional
[CLS]-supervised MTR head added during v5 continued pretraining.

Design
------
Two stages, each streamed and resumable:
  * IL stage    : 300k ionizable lipids -> v5_il.sdf
  * ChEMBL stage: 700k general molecules -> v5_chembl.sdf
Then byte-concatenated into lipids_unimolv2_3d.sdf.
On-disk order is irrelevant to training (the LMDB epoch-shuffles), so the two-stage
layout avoids having to sort 1M conformer records in RAM.

Descriptor z-stats (mean/std) are computed on a representative ~20k sample first and
written both into dataset_cards.json and (as Z_DESCRIPTORS) onto every SDF record.

Conformer budgets (generate -> keep):
  * ionizable lipids: 24 -> 20  (these are the domain molecules; dense 3D coverage)
  * ChEMBL general  : 14 -> 10  (generic-chemistry regularisers; cheaper budget)

Resume: each stage skips when its marker file (v5_il.done / v5_chembl.done) and the
final SDF are present. --force rebuilds everything.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.DisableLog("rdApp.*")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_lipids_unimol_dataset as P  # reuses conformer optimization + SDF writer

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── data sources ──────────────────────────────────────────────────────────────
DEFAULT_IL_CSV = Path(
    "/home/data/zhengxiao/disk1/my_project/lnpModel/code/00_prepare_ILs_vitual_library_v2/"
    "curate_library/generated_virtual_library_balanced.csv"
)
DEFAULT_CHEMBL_TXT = REPO_ROOT / "1_data/raw/CHEMBLdb/chembl_37_chemreps.txt.gz"
DEFAULT_OUT_DIR = REPO_ROOT / "1_data/processed/ionizable_lipids/unimolv2_3d_v5"

# ── MTR descriptor set (fixed; Z-normalised by this build) ────────────────────
# Names come from RDKit's global descriptor list (Descriptors._descList). A short
# smoke-critical postfix check at import time replaces any name that is not an
# exact registry member (e.g. NumHeavyAtoms, fr_AlkOH) with nothing, so we never
# silently emit NaN targets.
_DEFAULT_DESC_NAMES = [
    "MolWt", "HeavyAtomMolWt", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "NumSaturatedRings",
    "NumAliphaticRings", "TPSA", "MolLogP", "MolMR", "FractionCSP3", "NumValenceElectrons",
    "MaxPartialCharge", "MinPartialCharge", "MaxAbsPartialCharge", "MinAbsPartialCharge",
    "HallKierAlpha", "Chi0", "Chi1", "Kappa1", "Kappa2",
    "fr_C_O", "fr_C_O_noCOO", "fr_NH0", "fr_NH1", "fr_amide", "fr_Ar_N",
    "fr_alkyl_halide", "EState_VSA1", "EState_VSA2",
]
_DESC_REGISTRY = {name for name, _ in Descriptors._descList}
_DEFAULT_DESC_NAMES = [n for n in _DEFAULT_DESC_NAMES if n in _DESC_REGISTRY]
if len(_DEFAULT_DESC_NAMES) < 30:
    raise RuntimeError(f"descriptor registry damaged: {_DEFAULT_DESC_NAMES}")

# conformer budgets (generate -> keep), per source type.
# v5 has 1M molecules → per-molecule conformer density must stay lean or the
# build takes ~200h. ILs keep a modest 6->4; ChEMBL general 4->3.
# (Uni-Mol's ConformerSampleDataset draws ONE conformer per molecule per epoch,
# so these budgets control 3D augmentation diversity, not sample count.)
CONF_PLAN = {
    "virtual": {"num": 6, "keep": 4},
    "real":    {"num": 6, "keep": 4},
    "chembl":  {"num": 4, "keep": 3},
}
CONF_SEED = 20260805
CONF_MMFF_MAXITER = 2000
# ETKDG embed-budget. v4 hit 30–40s hangs per pathological (bridged-caged-head)
# lipid because maxIterations=200 makes distance-geometry retry forever. For v5 we
# (a) pre-screen every molecule with a 0.2s random-coords embed and exclude the
# failure cases cheaply, and (b) cap the real embed at 20 iterations so a
# borderline molecule fails in seconds, not minutes.
EMBED_PRE_ITERATIONS = 3
EMBED_ITERATIONS = 20

Z_DESC_PROP = "Z_DESCRIPTORS"

_FAILURE_FIELDS = P.FAILURE_FIELDS + ["desc_nan"]  # extend upstream failure CSV


# ── corpus construction ───────────────────────────────────────────────────────

def read_il_corpus(csv_path: Path, max_rows: int | None = None) -> list[dict[str, str]]:
    """Read the balanced ionizable-lipid virtual library (already unique)."""
    df = pd.read_csv(csv_path, dtype=str)
    if max_rows is not None:
        df = df.iloc[:max_rows]
    out = []
    for _, r in df.iterrows():
        smi = (r.get("SMILES") or "").strip()
        mid = (r.get("id") or "").strip()
        if not smi or not mid:
            continue
        out.append({"mol_id": mid, "smiles": smi, "source_type": "virtual"})
    return out


def sample_chembl(
    txt_path: Path, n_samples: int, seed: int, exclude: set[str] | None = None,
    progress: Callable[[int], None] | None = None, max_lines: int | None = None,
) -> list[dict[str, str]]:
    """Deterministic reservoir-sample N unique ChEMBL molecules.

    ``max_lines`` limits how many source lines are scanned (smoke tests only — a
    reservoir sample over a prefix is deterministic but not representative).
    """
    rng = random.Random(seed)
    reservoir: list[dict[str, str]] = []
    seen = 0
    exclude = exclude or set()
    open_fn = gzip.open if str(txt_path).endswith(".gz") else open
    finite = n_samples if max_lines is None else min(n_samples, max(1, max_lines))
    with open_fn(txt_path, "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if max_lines is not None and seen >= max_lines:
                break
            smi = (row.get("canonical_smiles") or "").strip()
            mol_id = (row.get("chembl_id") or "").strip()
            if not mol_id or not smi or smi in exclude:
                continue
            seen += 1
            item = {"mol_id": mol_id, "smiles": smi, "source_type": "chembl"}
            if len(reservoir) < finite:
                reservoir.append(item)
            else:
                swap = rng.randint(0, seen - 1)
                if swap < finite:
                    reservoir[swap] = item
            if progress is not None and seen % 100_000 == 0:
                progress(seen)
    return reservoir


def build_corpus(
    il_csv: Path, chembl_txt: Path, il_n: int, chembl_n: int, chembl_seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    print(f"[corpus] loading ionizable lipids: {il_csv} (n={il_n}) …")
    il = read_il_corpus(il_csv, max_rows=il_n)
    print(f"[corpus] IL rows: {len(il):,}")

    il_smiles = {r["smiles"] for r in il}
    limit = chembl_n
    if limit <= 0:
        print("[corpus] no ChEMBL requested — skipping the 2.9M-line reservoir scan.")
        return il, []
    print(f"[corpus] reservoir-sampling ChEMBL: n={limit:,} seed={chembl_seed} …")
    chembl = sample_chembl(
        chembl_txt, limit, chembl_seed,
        exclude=il_smiles,
        progress=lambda seen: print(f"  … scanned {seen:,} ChEMBL", flush=True),
    )
    print(f"[corpus] ChEMBL rows: {len(chembl):,} (cross-source dedup applied)")
    return il, chembl


# ── descriptors ───────────────────────────────────────────────────────────────

def compute_descriptors(smiles: str, desc_names: list[str]) -> np.ndarray:
    """Descriptors float32 vector; NaN where the molecule fails to parse/calculate."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.full(len(desc_names), float("nan"), dtype=np.float32)
    feats = dict(Descriptors._descList)
    vals = np.empty(len(desc_names), dtype=np.float64)
    for i, name in enumerate(desc_names):
        try:
            v = feats[name](mol)
            vals[i] = v if v is not None else float("nan")
        except Exception:
            vals[i] = float("nan")
    return vals.astype(np.float32)


# ── single-molecule worker ────────────────────────────────────────────────────

def prepare_v5_molecule(task: tuple[int, dict, dict]) -> P.MoleculeResult:
    """Canonicalise + descriptor-validate + build conformers for one molecule."""
    row_index, row, config = task
    mol_id = str(row.get("mol_id") or "").strip()
    smiles = str(row.get("smiles") or "").strip()
    source_type = str(row.get("source_type") or "").strip()

    def fail(reason: str, detail: str, **kw) -> P.MoleculeResult:
        return P.MoleculeResult(status="failed", row_index=row_index, mol_id=mol_id,
                                smiles=smiles, source_type=source_type,
                                failure_reason=reason, failure_detail=detail, **kw)

    if not smiles or not mol_id:
        return fail("missing_field", "empty SMILES or id")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return fail("invalid_smiles", "RDKit failed to parse SMILES")

    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    desc = compute_descriptors(canonical, config["desc_names"])
    if not np.isfinite(desc).all():
        # Reject NaN AND ±inf (inf can appear e.g. MolLogP on a few exotic
        # molecules). inf would otherwise poison valid-set metrics (mtr_loss inf
        # -> best_loss never updated) at training time.
        return fail("descriptor_nan", f"non-finite in {int(~np.isfinite(desc)).sum()} descriptors",
                    canonical_smiles=canonical)

    heavy = mol.GetNumHeavyAtoms()
    if heavy > config["max_atoms"]:
        return fail("too_many_heavy_atoms",
                    f"heavy_atoms={heavy} exceeds max_atoms={config['max_atoms']}",
                    canonical_smiles=canonical, heavy_atom_count=heavy,
                    total_atom_count=mol.GetNumAtoms())

    plan = CONF_PLAN.get(source_type, CONF_PLAN["chembl"])
    num_conf, keep_conf = plan["num"], plan["keep"]

    mol_h = Chem.AddHs(mol)  # embed AND optimize on the same with-H molecule
    # 1 random-coords embed with a tiny iteration budget (~0.2 s) — long flexible
    # chains need random coords to embed at all. ~3.5% of the virtual IL library
    # cannot embed even one (bridged/caged chiral heads); those would otherwise
    # hang the real embed for 30-40 s each (v4 2026-08-19 regression). Fail them
    # fast into the failure CSV instead of pinning a worker.
    pre_params = AllChem.ETKDGv3()
    pre_params.randomSeed = int(config["seed"]) + int(row_index)
    pre_params.enforceChirality = True
    pre_params.useRandomCoords = True
    pre_params.maxIterations = int(config.get("embed_pre_iterations", EMBED_PRE_ITERATIONS))
    pre_params.pruneRmsThresh = -1.0
    pre_params.numThreads = 1
    try:
        pre_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=1, params=pre_params)
    except Exception:
        pre_ids = []
    # A second try without random coords: some molecules with aromatic / charge
    # containing heads embed *better* without random-coordinate initialisation
    # (measured ~5/600 where random-coords embedding hangs but plain succeeds).
    if not pre_ids:
        pre_params2 = AllChem.ETKDGv3()
        pre_params2.randomSeed = int(config["seed"]) + int(row_index)
        pre_params2.enforceChirality = True
        pre_params2.useRandomCoords = False
        pre_params2.maxIterations = int(config.get("embed_pre_iterations",
                                                   EMBED_PRE_ITERATIONS))
        pre_params2.pruneRmsThresh = -1.0
        pre_params2.numThreads = 1
        try:
            pre_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=1, params=pre_params2)
        except Exception:
            pre_ids = []
    if not pre_ids:
        return fail("embed_precheck_failed", "ETKDG pre-screen could not embed",
                    canonical_smiles=canonical, heavy_atom_count=heavy,
                    total_atom_count=mol.GetNumAtoms())

    params = AllChem.ETKDGv3()
    params.randomSeed = int(config["seed"]) + int(row_index)
    params.enforceChirality = True
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.pruneRmsThresh = -1.0
    params.numThreads = 1
    params.useRandomCoords = bool(config["random_coords"])
    # bounded embed budget (module comment): keep per-molecule cost low
    params.maxIterations = int(config.get("embed_iterations", EMBED_ITERATIONS))

    try:
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=num_conf, params=params))
    except Exception as exc:
        return fail("embed_exception", str(exc), canonical_smiles=canonical,
                    heavy_atom_count=heavy, total_atom_count=mol.GetNumAtoms())
    if not conf_ids:
        return fail("no_embedded_conformers", f"requested={num_conf}",
                    canonical_smiles=canonical, heavy_atom_count=heavy,
                    total_atom_count=mol.GetNumAtoms())

    optimized: list[tuple[float, int, str]] = []
    rejected: list[str] = []
    for cid in conf_ids:
        ok, energy, method = P.optimize_conformer(mol_h, cid, int(config["max_iterations"]))
        if ok and math.isfinite(energy):
            optimized.append((energy, cid, method))
        else:
            rejected.append(method)
    if not optimized:
        return fail("no_optimized_conformers",
                    f"embedded={len(conf_ids)} rejected={','.join(rejected[:5])}",
                    canonical_smiles=canonical, heavy_atom_count=heavy,
                    total_atom_count=mol.GetNumAtoms())

    optimized.sort(key=lambda item: item[0])
    selected = optimized[:keep_conf]
    conformers = [
        P.PreparedConformer(Chem.MolToMolBlock(mol_h, confId=cid), rank, energy)
        for rank, (energy, cid, _) in enumerate(selected, start=1)
    ]

    z = (desc - config["desc_mean"]) / config["desc_std"]
    z_str = ";".join(f"{v:.6f}" for v in np.asarray(z, dtype=float))

    # NOTE: failure_detail is repurposed to carry the Z_DESCRIPTORS string on
    # success. append_v5_success reads it back from the result.
    return P.MoleculeResult(
        status="success", row_index=row_index, mol_id=mol_id,
        smiles=smiles, source_type=source_type,
        canonical_smiles=canonical,
        selected_conformers=conformers,
        heavy_atom_count=heavy, total_atom_count=mol.GetNumAtoms(),
        failure_reason=None, failure_detail=z_str,
    )


def append_v5_success(handle, result: P.MoleculeResult) -> int:
    """Write every conformer of a successful molecule, embedding the MTR target."""
    assert result.selected_conformers is not None
    z_str = result.failure_detail or "" if result.status == "success" else ""
    for conf in result.selected_conformers:
        handle.write(P.sdf_record(conf.mol_block, {
            "MOL_ID": result.mol_id,
            "ROW_INDEX": result.row_index,
            "SMILES": result.smiles,
            "CANONICAL_SMILES": result.canonical_smiles or "",
            "SOURCE_TYPE": result.source_type,
            "HEAVY_ATOM_COUNT": result.heavy_atom_count or "",
            "TOTAL_ATOM_COUNT": result.total_atom_count or "",
            "CONFORMER_RANK": conf.conformer_rank,
            "CONFORMER_ENERGY": f"{conf.conformer_energy:.8f}",
            Z_DESC_PROP: z_str,
        }))
    return len(result.selected_conformers)


def iter_results(tasks: Iterable[tuple], num_workers: int) -> Iterator[P.MoleculeResult]:
    if num_workers <= 1:
        for task in tasks:
            yield prepare_v5_molecule(task)
        return
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = {ex.submit(prepare_v5_molecule, t): t for t in tasks}
        for fut in as_completed(futures):
            yield fut.result()


# ── stage plumbing ────────────────────────────────────────────────────────────

def _write_sdf_stage(
    rows: list[dict], config: dict, out_sdf: Path, failure_csv: Path,
    stage_name: str, num_workers: int,
) -> dict:
    stats = {"success": 0, "failed": 0, "written_conformers": 0, "reasons": Counter()}
    with out_sdf.open("w", encoding="utf-8") as fh, \
         failure_csv.open("a", newline="", encoding="utf-8") as ff:
        writer = csv.DictWriter(ff, fieldnames=_FAILURE_FIELDS)
        if failure_csv.stat().st_size == 0:
            writer.writeheader()
        tasks = ((i, {"mol_id": r["mol_id"], "smiles": r["smiles"],
                      "source_type": r["source_type"]}, config)
                 for i, r in enumerate(rows))
        for result in iter_results(tasks, num_workers):
            if result.status == "success":
                stats["written_conformers"] += append_v5_success(fh, result)
                stats["success"] += 1
            else:
                stats["reasons"][result.failure_reason or "unknown"] += 1
                stats["failed"] += 1
                writer.writerow({
                    "row_index": result.row_index,
                    "mol_id": result.mol_id,
                    "smiles": result.smiles,
                    "source_type": result.source_type,
                    "reason": result.failure_reason or "unknown",
                    "detail": result.failure_detail or "",
                    "desc_nan": "yes" if result.failure_reason == "descriptor_nan" else "",
                })
        fh.flush()
    print(f"[{stage_name}] success={stats['success']:,} failed={stats['failed']:,} "
          f"conformers={stats['written_conformers']:,}")
    return dict(stats, reasons=dict(stats["reasons"]))


def _stage_done(marker: Path, sdf: Path) -> bool:
    return marker.exists() and sdf.exists() and sdf.stat().st_size > 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the LipidFM v5 pretraining dataset.")
    ap.add_argument("--il-csv", type=Path, default=DEFAULT_IL_CSV)
    ap.add_argument("--chembl-txt", type=Path, default=DEFAULT_CHEMBL_TXT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--il-n", type=int, default=300_000)
    ap.add_argument("--chembl-n", type=int, default=700_000)
    ap.add_argument("--chembl-seed", type=int, default=20260826)
    ap.add_argument("--n-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--desc-n", type=int, default=20_000,
                    help="Number of molecules used to estimate descriptor mean/std.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Smoke mode: cap total molecules (IL first, then ChEMBL).")
    ap.add_argument("--force", action="store_true", help="Rebuild stages even if done.")
    args = ap.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    il_sdf = out_dir / "v5_il.sdf"
    chembl_sdf = out_dir / "v5_chembl.sdf"
    final_sdf = out_dir / "lipids_unimolv2_3d.sdf"
    failures = out_dir / "lipids_unimolv2_failures.csv"
    il_marker, cb_marker = out_dir / "v5_il.done", out_dir / "v5_chembl.done"

    # effective counts (apply --limit before sampling: cheap and near-identical)
    il_n, chembl_n = args.il_n, args.chembl_n
    if args.limit is not None:
        il_n = min(il_n, args.limit)
        chembl_n = max(0, args.limit - il_n)

    il_rows, chembl_rows = build_corpus(
        args.il_csv, args.chembl_txt, il_n, chembl_n, args.chembl_seed,
    )

    # descriptor z-stats on a representative sample
    desc_names = _DEFAULT_DESC_NAMES
    n_sample = min(args.desc_n, len(il_rows) + len(chembl_rows))
    sample = []
    rng = random.Random(1234)
    n_il_sample = min(n_sample, len(il_rows))
    sample += rng.sample(il_rows, n_il_sample)
    if len(sample) < n_sample and chembl_rows:
        sample += rng.sample(chembl_rows, n_sample - len(sample))
    print(f"[mtr] z-stats over {len(sample):,} sample molecules ({len(desc_names)} descs) …")
    big = np.vstack([compute_descriptors(r["smiles"], desc_names) for r in sample])
    if np.isnan(big).all(axis=0).any():
        bad = [n for n, v in zip(desc_names, np.isnan(big).all(axis=0)) if v]
        print(f"  WARNING descriptor(s) NaN everywhere: {bad}")
    desc_mean = np.nanmean(big, axis=0).astype(np.float32)
    desc_std = np.nanstd(big, axis=0).astype(np.float32)
    desc_std[desc_std < 1e-8] = 1.0
    print("[mtr] z-stats done.")

    config = {
        "seed": CONF_SEED,
        "max_atoms": 256,
        "random_coords": True,
        "max_iterations": CONF_MMFF_MAXITER,
        "embed_max_iterations": 200,
        "desc_names": desc_names,
        "desc_mean": desc_mean,
        "desc_std": desc_std,
    }

    # IL stage
    if args.force or not _stage_done(il_marker, il_sdf):
        if failures.exists():
            failures.unlink()
        print(f"[IL] conformers: {len(il_rows):,} ionizable lipids -> {il_sdf}")
        _write_sdf_stage(il_rows, config, il_sdf, failures, "IL", args.n_workers)
        il_marker.write_text(json.dumps({"rows": len(il_rows)}), encoding="utf-8")
    else:
        print(f"[IL] stage up-to-date: {il_sdf}")

    # ChEMBL stage (append to the same failure CSV)
    if args.force or not _stage_done(cb_marker, chembl_sdf):
        print(f"[ChEMBL] conformers: {len(chembl_rows):,} general molecules -> {chembl_sdf}")
        _write_sdf_stage(chembl_rows, config, chembl_sdf, failures, "ChEMBL", args.n_workers)
        cb_marker.write_text(json.dumps({"rows": len(chembl_rows)}), encoding="utf-8")
    else:
        print(f"[ChEMBL] stage up-to-date: {chembl_sdf}")

    # final: concatenate (streaming)
    if args.force or not final_sdf.exists() or final_sdf.stat().st_size == 0:
        with final_sdf.open("wb") as out:
            for src in (il_sdf, chembl_sdf):
                with src.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        out.write(chunk)
        print(f"[final] concatenated into {final_sdf} ({final_sdf.stat().st_size/1e9:.1f} GB)")
    else:
        print(f"[final] up-to-date: {final_sdf}")

    card = {
        "il_source": str(args.il_csv),
        "chembl_source": str(args.chembl_txt),
        "il_n": len(il_rows),
        "chembl_n": len(chembl_rows),
        "ratio_general_lipid": 0.7,
        "descriptor_names": desc_names,
        "descriptor_n": len(desc_names),
        "descriptor_mean": [round(float(x), 6) for x in desc_mean],
        "descriptor_std": [round(float(x), 6) for x in desc_std],
        "conformer_plan": CONF_PLAN,
        "seed": CONF_SEED,
        "note": "LMDB epoch-shuffle yields ~70% general / 30% ionizable-lipid per epoch.",
    }
    (out_dir / "dataset_cards.json").write_text(
        json.dumps(card, indent=2), encoding="utf-8")
    print(json.dumps(card, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())