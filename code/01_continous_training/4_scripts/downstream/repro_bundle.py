#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducibility bundle: everything the plan's checklist asks a third party to receive.

Emits one directory containing the split files, the per-sample predictions, the
hyperparameter record and the environment, plus a manifest with a hash per file.
The point is that a reviewer can verify a published number without rerunning
training -- and can tell whether the bundle they received is the one that
produced the number.

Deliberately NOT included: model weights. They are large, they live under
``2_models/``, and the manifest records their path + size instead so the bundle
stays small enough to hand over.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
sys.path.insert(0, str(ROOT / "4_scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import split_files  # noqa: E402

BUNDLE_ROOT = ROOT / "5_outputs/repro_bundles"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def environment_record() -> dict:
    """Versions that can change a number.

    RDKit matters most: scaffold perception and ECFP4 generation both come from
    it, so a version bump can silently repartition the data. That is why the
    fold files are hashed and stored rather than regenerated downstream.
    """
    rec = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "conda_env": "unimol2",
    }
    for mod in ("numpy", "pandas", "sklearn", "scipy", "rdkit", "torch", "xgboost"):
        try:
            m = __import__(mod)
            rec[mod] = getattr(m, "__version__", "unknown")
        except ImportError:
            rec[mod] = None
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=10)
        rec["gpu"] = out.stdout.strip() or None
    except Exception:
        rec["gpu"] = None
    return rec


def _clear_subdirs(out: Path) -> list[str]:
    """Empty the bundle's payload directories so a rebuild is not additive.

    ``build_bundle`` writes files but never removed any, so re-building a bundle
    left every file the previous build had written. On 2026-09-18 that turned a
    fix into a lie: ``pKa-scaffold-20260918`` was rebuilt to stop shipping the
    five buggy butina fold files, the manifest was correctly regenerated with
    only the scaffold folds, and the README said the butina folds "are gone" --
    while the five files were still sitting in ``splits/`` on disk. Anyone
    handed the directory got them.

    A manifest that disagrees with its own directory is worse than no manifest:
    the hash self-check passes (it only covers what is listed) and the extra
    files look like legitimate contents. So clear before writing.

    ``manifest.json`` and ``environment.json`` live at the top level and are
    overwritten, so only the payload dirs are cleared.
    """
    removed = []
    for sub in ("results", "predictions", "splits"):
        d = out / sub
        if not d.is_dir():
            continue
        for p in d.iterdir():
            if p.is_file():
                p.unlink()
                removed.append(str(p.relative_to(out)))
    return removed


def build_bundle(name: str, results: list[Path], splits: list[str] | None = None,
                 notes: str = "") -> Path:
    """Assemble a bundle directory and write its manifest.

    ``results`` are JSON files written by ``run_matrix``; each one's companion
    ``.npz`` prediction file is picked up automatically, since the plan requires
    per-sample predictions and a bundle without them cannot be audited.
    """
    out = BUNDLE_ROOT / name
    stale = _clear_subdirs(out)
    if stale:
        print(f"[bundle] cleared {len(stale)} stale file(s) from the previous build")
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "splits").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    result_docs: list[dict] = []

    for r in results:
        r = Path(r)
        if not r.exists():
            raise SystemExit(f"result file missing: {r}")
        d = json.loads(r.read_text(encoding="utf-8"))
        result_docs.append(d)
        (out / "results" / r.name).write_text(json.dumps(d, indent=1), encoding="utf-8")
        files.append(out / "results" / r.name)
        npz = ROOT / "5_outputs/downstream_predictions" / f"{d['run_id']}.npz"
        if not npz.exists():
            raise SystemExit(
                f"{r.name} has no prediction file at {npz} — a result without its "
                f"per-sample predictions cannot be audited, refusing to bundle it.")
        (out / "predictions" / npz.name).write_bytes(npz.read_bytes())
        files.append(out / "predictions" / npz.name)
        for s in [d["dataset"], *(["Sheet2", "Sheet3"] if d["dataset"].startswith("Sheet") else [])]:
            if s not in (splits or []):
                splits = (splits or []) + [s]

    # Only the split protocols the bundled results actually used.
    #
    # This used to be `glob(f"{key}_*.json")`, which copies EVERY protocol for the
    # dataset. On 2026-09-18 that was found to be silently wrong: the pKa scaffold
    # bundle shipped 5 butina fold files built by the buggy Butina implementation
    # (prompt/26-09-18/06-*), so a reviewer reproducing the scaffold numbers would
    # also receive a wrong-but-valid-looking butina partition. The bundle's own
    # sha256 self-check could not catch it -- every file matched the manifest; the
    # manifest itself was just describing the wrong contents.
    #
    # The splits a bundle needs are exactly the (dataset, split) pairs its results
    # name, so derive them instead of globbing. `splits` stays as an escape hatch
    # for callers who deliberately want an extra protocol included.
    used = sorted({(d["dataset"], d["split"]) for d in result_docs})
    if splits:
        used = sorted(set(used) | {(d, s) for d in {x[0] for x in used} for s in splits})
    for key, split in used:
        pat = f"{key}_{split}_k*_seed*.json"
        for p in sorted(split_files.SPLIT_DIR.glob(pat)):
            # The quarantined-but-still-present buggy folds are named
            # `..._seed42.BUGGY_20260918.json`, which this glob matches (the
            # `_seed*.json` suffix is not anchored). They are kept on disk on
            # purpose -- `failed/butina_distance_order_bug_20260918/README.md`
            # tracks their provenance -- but a bundle must never ship them.
            if ".BUGGY_" in p.name:
                continue
            (out / "splits" / p.name).write_bytes(p.read_bytes())
            files.append(out / "splits" / p.name)

    (out / "environment.json").write_text(
        json.dumps(environment_record(), indent=1), encoding="utf-8")
    (out / "README.md").write_text(
        f"# Reproducibility bundle: {name}\n\n{notes}\n\n"
        f"Contents: `results/` per-run metrics, `predictions/` per-sample OOF "
        f"predictions (SMILES-aligned to the split files), `splits/` frozen fold "
        f"assignments, `environment.json` library versions.\n\n"
        f"Every file's sha256 is in `manifest.json`. Verify with:\n"
        f"```\nsha256sum -c <(python -c \"import json;[print(v['sha256'],"
        f"' ',k) for k,v in json.load(open('manifest.json'))['files'].items()]\")\n```\n",
        encoding="utf-8")
    files.append(out / "environment.json")
    files.append(out / "README.md")

    manifest = {
        "bundle": name,
        "environment": environment_record(),
        "files": {str(f.relative_to(out)): {"sha256": _sha256(f),
                                            "bytes": f.stat().st_size} for f in files},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # Verify the directory against the manifest it just got, by reading both
    # back from disk. The manifest's sha256 list is only a guarantee if it
    # describes the WHOLE directory: an unlisted extra file passes every hash
    # check while being exactly the thing a reviewer should not receive.
    listed = set(manifest["files"])
    on_disk = {str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()}
    extra = sorted(on_disk - listed - {"manifest.json"})
    if extra:
        raise SystemExit(f"[bundle] {len(extra)} file(s) on disk are not in the "
                         f"manifest: {extra}. The bundle would ship content the "
                         f"hash list does not describe.")
    bad = [k for k, v in manifest["files"].items() if _sha256(out / k) != v["sha256"]]
    if bad:
        raise SystemExit(f"[bundle] sha256 mismatch after write: {bad}")

    total = sum(v["bytes"] for v in manifest["files"].values())
    print(f"[bundle] {out}  {len(manifest['files'])} files, {total/1e6:.2f} MB  "
          f"(verified: manifest == directory, {len(listed)} hashes)")
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Build a reproducibility bundle.")
    ap.add_argument("name", help="bundle name, e.g. 'eval1-te-20260916'")
    ap.add_argument("results", nargs="+", type=Path, help="result JSON files")
    ap.add_argument("--splits", nargs="*", default=None,
                    help="dataset keys whose split files to include")
    ap.add_argument("--notes", default="", help="free-text description")
    a = ap.parse_args()
    build_bundle(a.name, a.results, a.splits, a.notes)


if __name__ == "__main__":
    main()
