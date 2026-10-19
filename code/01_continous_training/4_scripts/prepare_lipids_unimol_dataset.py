"""Prepare 3D conformers for ionizable lipids (Step 1).

Adapted from the EnviroFM DSSTox pipeline. Three changes were required because
ionizable lipids are far more flexible than drug-like molecules (median 39
rotatable bonds vs ~5), and the original ETKDG/MMFF settings silently discarded
most of the corpus:

  1. ``useRandomCoords=True`` — plain ETKDG distance-geometry embedding fails on
     long flexible chains. Measured on 24 molecules of 60-140 heavy atoms:
     8.3% of molecules reached 5 conformers without it, 100% with it.
  2. ``max_iterations`` default 500 -> 2000 — MMFF94s must return status 0 or the
     conformer is dropped. At 500 iterations 0/24 converged; at 2000, 23/24.
  3. ``max_atoms`` default 128 -> 256 — Uni-Mol2's CroppingDataset randomly
     *drops atoms* above this budget instead of erroring, silently corrupting
     molecules. 256 covers the whole corpus (max 208 heavy atoms).

Input is the Stage-1 parquet corpus (not CSV): it already carries standardized
SMILES, InChIKey, and a real/virtual label, so per-source conformer counts can be
applied (real lipids are scarce and get more conformers than virtual ones).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = REPO_ROOT / "1_data/raw/ionizable_lipids/pretrain_corpus.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "1_data/processed/ionizable_lipids/unimolv2_3d"

# ChEMBL replay rows are offset so their indices never collide with corpus rows.
CHEMBL_ROW_INDEX_OFFSET = 10_000_000

FAILURE_FIELDS = ["row_index", "mol_id", "smiles", "source_type", "reason", "detail"]


@dataclass
class PreparedConformer:
    mol_block: str
    conformer_rank: int
    conformer_energy: float


@dataclass
class MoleculeResult:
    status: str
    row_index: int
    mol_id: str
    smiles: str
    source_type: str = ""
    failure_reason: str | None = None
    failure_detail: str | None = None
    canonical_smiles: str | None = None
    selected_conformers: list[PreparedConformer] | None = None
    heavy_atom_count: int | None = None
    total_atom_count: int | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate 3D conformers for ionizable lipids.")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--smiles-col", default="smiles_std")
    p.add_argument("--id-col", default="inchikey")
    p.add_argument("--source-col", default="source_type")
    p.add_argument("--num-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    # Real lipids are scarce (12k) so they get denser conformer coverage than the
    # 100k virtual ones. Uni-Mol2 samples ONE conformer per molecule per epoch,
    # so these counts control 3D augmentation diversity, not sample count.
    p.add_argument("--num-conformers-real", type=int, default=24)
    p.add_argument("--keep-conformers-real", type=int, default=20)
    p.add_argument("--num-conformers-virtual", type=int, default=6)
    p.add_argument("--keep-conformers-virtual", type=int, default=4)
    p.add_argument("--max-iterations", type=int, default=2000)
    p.add_argument("--max-atoms", type=int, default=256)
    p.add_argument("--random-coords", dest="random_coords", action="store_true", default=True)
    p.add_argument("--no-random-coords", dest="random_coords", action="store_false")
    p.add_argument("--seed", type=int, default=20260805)
    p.add_argument("--max-molecules", type=int, default=None)
    p.add_argument("--sample-size", type=int, default=None,
                   help="Randomly subsample N corpus molecules (smoke tests).")
    p.add_argument("--sample-seed", type=int, default=20260805)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--chembl-input", type=Path, default=None)
    p.add_argument("--chembl-ratio", type=float, default=0.20)
    p.add_argument("--chembl-max", type=int, default=None)
    p.add_argument("--chembl-seed", type=int, default=None)
    return p.parse_args()


# ── conformer generation ─────────────────────────────────────────────────────

def optimize_conformer(mol: Chem.Mol, conf_id: int, max_iterations: int) -> tuple[bool, float, str]:
    """MMFF94s with UFF fallback. Returns (ok, energy, method)."""
    try:
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if props is not None:
            status = AllChem.MMFFOptimizeMolecule(
                mol, mmffVariant="MMFF94s", maxIters=max_iterations, confId=conf_id
            )
            if status != 0:
                return False, math.inf, "mmff_not_converged"
            ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
            if ff is None:
                return False, math.inf, "mmff_forcefield_missing"
            return True, float(ff.CalcEnergy()), "MMFF94s"
    except Exception as exc:
        return False, math.inf, f"mmff_exception:{type(exc).__name__}"

    try:
        status = AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=max_iterations)
    except Exception as exc:
        return False, math.inf, f"uff_exception:{type(exc).__name__}"
    if status != 0:
        return False, math.inf, "uff_not_converged"
    ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
    if ff is None:
        return False, math.inf, "uff_forcefield_missing"
    return True, float(ff.CalcEnergy()), "UFF"


def prepare_molecule(task: tuple[int, dict, dict]) -> MoleculeResult:
    """Process one molecule. Never raises — the worker pool must not die."""
    row_index, row, config = task
    mol_id = str(row.get("mol_id") or "").strip()
    smiles = str(row.get("smiles") or "").strip()
    source_type = str(row.get("source_type") or "")

    def fail(reason: str, detail: str, **kw) -> MoleculeResult:
        return MoleculeResult(status="failed", row_index=row_index, mol_id=mol_id,
                              smiles=smiles, source_type=source_type,
                              failure_reason=reason, failure_detail=detail, **kw)

    if not smiles or not mol_id:
        return fail("missing_field", "empty SMILES or id")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return fail("invalid_smiles", "RDKit failed to parse SMILES")

    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    heavy = mol.GetNumHeavyAtoms()
    if heavy > config["max_atoms"]:
        return fail("too_many_heavy_atoms",
                    f"heavy_atoms={heavy} exceeds max_atoms={config['max_atoms']}",
                    canonical_smiles=canonical, heavy_atom_count=heavy,
                    total_atom_count=mol.GetNumAtoms())

    # Real lipids get dense conformer coverage; virtual and ChEMBL replay
    # molecules get the cheaper budget. ChEMBL is a generic-chemistry regulariser,
    # not domain data, so spending 20 conformers each would waste hours.
    is_cheap = source_type in ("virtual", "chembl")
    num_conf = config["num_conformers_virtual"] if is_cheap else config["num_conformers_real"]
    keep_conf = config["keep_conformers_virtual"] if is_cheap else config["keep_conformers_real"]

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(config["seed"]) + int(row_index)
    params.enforceChirality = True
    params.useSmallRingTorsions = True
    params.useMacrocycleTorsions = True
    params.pruneRmsThresh = -1.0
    params.numThreads = 1
    # Essential for flexible lipids — see module docstring.
    params.useRandomCoords = bool(config["random_coords"])
    # ETKDGv3 leaves maxIterations at 0 = retry distance-geometry embedding
    # forever. The v4 corpus introduced bridged/caged heads (quinuclidine
    # C1C[N@]2CC[C@@H]1CC2, adamantane) whose chiral bridgeheads combined with
    # enforceChirality make embedding fail over and over: workers pinned at 100%
    # CPU with zero output and the whole run wedged at 4% (2026-08-19).
    # Cap the retries so an unembeddable molecule fails fast into the failure CSV.
    params.maxIterations = int(config.get("embed_max_iterations", 200))

    try:
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_conf, params=params))
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
        ok, energy, method = optimize_conformer(mol, cid, int(config["max_iterations"]))
        if ok and math.isfinite(energy):
            optimized.append((energy, cid, method))
        else:
            rejected.append(method)

    # Keep whatever converged rather than discarding the molecule outright: for
    # highly flexible lipids a strict "must reach keep_conf" gate throws away
    # usable structures. One good conformer is still a valid training sample.
    if not optimized:
        return fail("no_optimized_conformers",
                    f"embedded={len(conf_ids)} rejected={','.join(rejected[:5])}",
                    canonical_smiles=canonical, heavy_atom_count=heavy,
                    total_atom_count=mol.GetNumAtoms())

    optimized.sort(key=lambda item: item[0])
    selected = optimized[:keep_conf]
    conformers = [
        PreparedConformer(Chem.MolToMolBlock(mol, confId=cid), rank, energy)
        for rank, (energy, cid, _) in enumerate(selected, start=1)
    ]

    return MoleculeResult(status="success", row_index=row_index, mol_id=mol_id,
                          smiles=smiles, source_type=source_type,
                          canonical_smiles=canonical, selected_conformers=conformers,
                          heavy_atom_count=heavy, total_atom_count=mol.GetNumAtoms())


# ── input streams ────────────────────────────────────────────────────────────

def read_corpus_rows(args, skip_rows: int = 0) -> Iterator[tuple[int, dict]]:
    """Stream the parquet corpus as (row_index, row-dict) pairs."""
    df = pd.read_parquet(args.input)
    has_source = args.source_col in df.columns
    cols = [args.id_col, args.smiles_col] + ([args.source_col] if has_source else [])
    df = df[cols]

    if args.sample_size is not None and args.sample_size < len(df):
        df = df.sample(n=args.sample_size, random_state=args.sample_seed)
    if args.max_molecules is not None:
        df = df.head(args.max_molecules)

    for i, rec in enumerate(df.itertuples(index=False)):
        if i < skip_rows:
            continue
        yield i, {
            "mol_id": getattr(rec, args.id_col),
            "smiles": getattr(rec, args.smiles_col),
            # Absent column means the whole corpus is treated as domain data.
            "source_type": getattr(rec, args.source_col) if has_source else "real",
        }


def count_corpus_rows(args) -> int:
    if args.sample_size is not None:
        return args.sample_size
    if args.max_molecules is not None:
        return args.max_molecules
    return len(pd.read_parquet(args.input, columns=[args.id_col]))


def sample_chembl(chembl_input: Path, n_samples: int, seed: int) -> list[dict]:
    """Reservoir-sample ChEMBL molecules for anti-forgetting replay."""
    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen = 0
    open_fn = gzip.open if str(chembl_input).endswith(".gz") else open
    with open_fn(chembl_input, "rt", encoding="utf-8") as fh:
        for row in tqdm(csv.DictReader(fh, delimiter="\t"), desc="Sampling ChEMBL", unit="mol"):
            smi = (row.get("canonical_smiles") or "").strip()
            mol_id = (row.get("chembl_id") or "").strip()
            if not smi or not mol_id:
                continue
            seen += 1
            item = {"mol_id": mol_id, "smiles": smi, "source_type": "chembl"}
            if len(reservoir) < n_samples:
                reservoir.append(item)
            else:
                swap = rng.randint(0, seen - 1)
                if swap < n_samples:
                    reservoir[swap] = item
    return reservoir


def compute_chembl_positions(n_corpus: int, n_chembl: int, seed: int) -> list[int]:
    """Insertion positions for ChEMBL rows within the combined stream."""
    rng = random.Random(seed)
    total = n_corpus + n_chembl
    return sorted(rng.sample(range(total), min(n_chembl, total)))


def iter_interleaved(corpus_rows, chembl_rows, chembl_positions, n_total,
                     resume_output_pos: int = 0) -> Iterator[tuple[int, dict]]:
    """Scatter ChEMBL rows deterministically through the corpus stream."""
    pos_to_idx = {pos: idx for idx, pos in enumerate(chembl_positions)}
    corpus_iter = iter(corpus_rows)
    for output_pos in range(resume_output_pos, n_total):
        if output_pos in pos_to_idx:
            ci = pos_to_idx[output_pos]
            yield CHEMBL_ROW_INDEX_OFFSET + ci, chembl_rows[ci]
        else:
            pair = next(corpus_iter, None)
            if pair is not None:
                yield pair


def iter_results(tasks: Iterable[tuple[int, dict, dict]], num_workers: int) -> Iterator[MoleculeResult]:
    if num_workers <= 1:
        for task in tasks:
            yield prepare_molecule(task)
        return
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(prepare_molecule, t): t for t in tasks}
        from concurrent.futures import as_completed
        for future in as_completed(futures):
            yield future.result()


# ── output ───────────────────────────────────────────────────────────────────

def sdf_record(mol_block: str, props: dict) -> str:
    lines = [mol_block.rstrip("\n")]
    for key, value in props.items():
        lines.append(f">  <{key}>")
        lines.append(str(value))
        lines.append("")
    lines.append("$$$$")
    return "\n".join(lines) + "\n"


def append_success(handle, result: MoleculeResult) -> int:
    assert result.selected_conformers is not None
    for conf in result.selected_conformers:
        handle.write(sdf_record(conf.mol_block, {
            # MOL_ID / SMILES are the keys build_lmdb reads; SMILES must be the
            # exact string the molblock was generated from (alignment contract).
            "MOL_ID": result.mol_id,
            "ROW_INDEX": result.row_index,
            "SMILES": result.smiles,
            "CANONICAL_SMILES": result.canonical_smiles or "",
            "SOURCE_TYPE": result.source_type,
            "HEAVY_ATOM_COUNT": result.heavy_atom_count or "",
            "TOTAL_ATOM_COUNT": result.total_atom_count or "",
            "CONFORMER_RANK": conf.conformer_rank,
            "CONFORMER_ENERGY": f"{conf.conformer_energy:.8f}",
        }))
    return len(result.selected_conformers)


def append_failure(path: Path, result: MoleculeResult) -> None:
    with path.open("a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=FAILURE_FIELDS).writerow({
            "row_index": result.row_index,
            "mol_id": result.mol_id,
            "smiles": result.smiles,
            "source_type": result.source_type,
            "reason": result.failure_reason,
            "detail": result.failure_detail,
        })


def default_checkpoint() -> dict:
    return {"processed_rows": 0, "successful_molecules": 0, "failed_molecules": 0,
            "written_conformers": 0, "chembl_rows_emitted": 0, "chembl_successful": 0,
            "conformers_by_source": {}, "failure_reasons": {}}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sdf_path = args.output_dir / "lipids_unimolv2_3d.sdf"
    failure_path = args.output_dir / "lipids_unimolv2_failures.csv"
    summary_path = args.output_dir / "lipids_unimolv2_summary.json"
    checkpoint_path = args.output_dir / "lipids_unimolv2_checkpoint.json"

    if not args.resume:
        for path in (sdf_path, failure_path, summary_path, checkpoint_path):
            path.unlink(missing_ok=True)

    checkpoint = default_checkpoint()
    if args.resume and checkpoint_path.exists():
        checkpoint.update(json.loads(checkpoint_path.read_text(encoding="utf-8")))
    skip_rows = int(checkpoint.get("processed_rows", 0)) if args.resume else 0

    if not failure_path.exists():
        with failure_path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FAILURE_FIELDS).writeheader()

    config = {
        "num_conformers_real": args.num_conformers_real,
        "keep_conformers_real": args.keep_conformers_real,
        "num_conformers_virtual": args.num_conformers_virtual,
        "keep_conformers_virtual": args.keep_conformers_virtual,
        "max_iterations": args.max_iterations,
        "max_atoms": args.max_atoms,
        "random_coords": args.random_coords,
        "seed": args.seed,
    }

    print(f"Input        : {args.input}")
    print(f"Output       : {sdf_path}")
    print(f"Workers      : {args.num_workers}")
    print(f"max_atoms    : {args.max_atoms} (heavy atoms)")
    print(f"randomCoords : {args.random_coords}   MMFF maxIters: {args.max_iterations}")
    print(f"conformers   : real {args.num_conformers_real}->{args.keep_conformers_real}, "
          f"virtual {args.num_conformers_virtual}->{args.keep_conformers_virtual}")

    # ── ChEMBL anti-forgetting replay ─────────────────────────────────────────
    chembl_rows: list[dict] = []
    chembl_positions: list[int] = []
    n_corpus = 0
    resume_output_pos = skip_rows

    if args.chembl_input is not None:
        chembl_seed = args.chembl_seed if args.chembl_seed is not None else args.seed + 1
        if args.resume and checkpoint.get("n_corpus_effective"):
            n_corpus = checkpoint["n_corpus_effective"]
            n_chembl_target = checkpoint["n_chembl_target"]
            chembl_seed = checkpoint["chembl_seed_used"]
        else:
            n_corpus = count_corpus_rows(args)
            n_chembl_target = int(round(n_corpus * args.chembl_ratio))
            if args.chembl_max is not None:
                n_chembl_target = min(n_chembl_target, args.chembl_max)

        print(f"\nSampling {n_chembl_target:,} ChEMBL molecules "
              f"(ratio={args.chembl_ratio:.0%}, n_corpus={n_corpus:,})…")
        chembl_rows = sample_chembl(args.chembl_input, n_chembl_target, chembl_seed)

        if args.resume and checkpoint.get("chembl_positions"):
            chembl_positions = checkpoint["chembl_positions"]
        else:
            chembl_positions = compute_chembl_positions(n_corpus, len(chembl_rows), chembl_seed + 1)

        checkpoint["n_corpus_effective"] = n_corpus
        checkpoint["n_chembl_target"] = n_chembl_target
        checkpoint["chembl_seed_used"] = chembl_seed
        checkpoint["chembl_positions"] = chembl_positions
        resume_output_pos = skip_rows + int(checkpoint.get("chembl_rows_emitted", 0))
        print(f"Interleaving {len(chembl_rows):,} ChEMBL into {n_corpus:,} lipids "
              f"(resume_pos={resume_output_pos:,})")

    rows = read_corpus_rows(args, skip_rows=skip_rows)
    if chembl_rows:
        n_total = n_corpus + len(chembl_rows)
        combined: Iterator[tuple[int, dict]] = iter_interleaved(
            rows, chembl_rows, chembl_positions, n_total, resume_output_pos)
    else:
        combined = rows

    tasks = ((idx, row, config) for idx, row in combined)
    total_expected = (n_corpus + len(chembl_rows)) if chembl_rows else count_corpus_rows(args)

    mode = "a" if args.resume else "w"
    with sdf_path.open(mode, encoding="utf-8") as sdf_handle:
        desc = "Preparing lipids" + ("+ChEMBL" if chembl_rows else "")
        for result in tqdm(iter_results(tasks, args.num_workers), desc=desc,
                          total=max(total_expected - resume_output_pos, 0), ncols=76):
            is_chembl = result.row_index >= CHEMBL_ROW_INDEX_OFFSET
            key = "chembl_rows_emitted" if is_chembl else "processed_rows"
            checkpoint[key] = int(checkpoint.get(key, 0)) + 1

            if result.status == "success":
                written = append_success(sdf_handle, result)
                ok_key = "chembl_successful" if is_chembl else "successful_molecules"
                checkpoint[ok_key] = int(checkpoint.get(ok_key, 0)) + 1
                checkpoint["written_conformers"] = int(checkpoint.get("written_conformers", 0)) + written
                by_src = checkpoint.setdefault("conformers_by_source", {})
                src = result.source_type or "unknown"
                by_src[src] = int(by_src.get(src, 0)) + written
            else:
                append_failure(failure_path, result)
                if not is_chembl:
                    checkpoint["failed_molecules"] = int(checkpoint.get("failed_molecules", 0)) + 1
                reasons = checkpoint.setdefault("failure_reasons", {})
                reason = result.failure_reason or "unknown"
                reasons[reason] = int(reasons.get(reason, 0)) + 1

            done = int(checkpoint.get("processed_rows", 0)) + int(checkpoint.get("chembl_rows_emitted", 0))
            if done % 100 == 0:
                checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")

    checkpoint_path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")
    summary = {
        "input": str(args.input),
        "output_sdf": str(sdf_path),
        "chembl_input": str(args.chembl_input) if args.chembl_input else None,
        "chembl_ratio": args.chembl_ratio if args.chembl_input else None,
        **{k: checkpoint.get(k) for k in
           ("processed_rows", "successful_molecules", "failed_molecules",
            "chembl_rows_emitted", "chembl_successful", "written_conformers",
            "conformers_by_source", "failure_reasons")},
        "config": config,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
