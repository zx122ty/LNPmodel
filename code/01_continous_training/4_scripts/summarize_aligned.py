"""Summarise the aligned scaffold-split comparison (stock / v2 / v3).

Reads the per-run entries in ``5_outputs/finetune_moltrain/moltrain_results.json``
(keyed by ``<base>|<split>|seed<N>``) and prints the headline table:
fold-mean R² ± std, averaged over seeds where several were run.

Only runs sharing (split, kfold, lr, epochs) are compared; anything else is
listed separately so a mismatched protocol can never be silently averaged in.

Usage:
    python 4_scripts/summarize_aligned.py [--split scaffold]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AGG = REPO_ROOT / "5_outputs/finetune_moltrain/moltrain_results.json"

# Present the bases in pretraining-scale order so a scaling trend is readable.
ORDER = ["stock-84M", "unimol2_lipids_84m_v2", "unimol2_lipids_84m_v3",
         "unimol2_lipids_84m_v4"]


def base_of(run_key: str) -> str:
    label = run_key.split("|")[0]
    return label.split("/")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="scaffold")
    ap.add_argument("--lr", type=float, default=None,
                    help="Only include runs at this learning rate (an lr sweep puts "
                         "several lrs in the file; averaging across them is meaningless).")
    ap.add_argument("--sweep", action="store_true",
                    help="Group by (base, lr) instead of base, to read an lr sweep.")
    ap.add_argument("--results", type=Path, default=AGG)
    args = ap.parse_args()

    if not args.results.exists():
        raise SystemExit(f"no results file at {args.results}")
    runs = json.loads(args.results.read_text(encoding="utf-8")).get("runs", {})

    # base -> dataset -> list of (seed, fold_mean_r2, fold_std, spearman)
    table: dict = defaultdict(lambda: defaultdict(list))
    protocols: set = set()
    for key, rec in runs.items():
        if not isinstance(rec, dict) or "metrics" not in rec:
            continue  # legacy/pre-refactor entry, not comparable
        if rec.get("split") != args.split:
            continue
        if args.lr is not None and rec.get("lr") != args.lr:
            continue
        protocols.add((rec.get("lr"), rec.get("epochs"), rec.get("kfold")))
        for ds, m in rec["metrics"].items():
            if "r2" not in m:
                continue
            # epochs MUST be in the group key: a 20-epoch and a 40-epoch run at the
            # same lr are different protocols, and averaging them silently reported
            # "n=2" for what were really two incomparable points.
            group = (f"{base_of(key)} @lr{rec.get('lr')}/ep{rec.get('epochs')}"
                     if args.sweep else base_of(key))
            table[group][ds].append(
                (rec.get("seed"), m["r2"], m.get("r2_std"), m.get("spearman")))

    if len(protocols) > 1 and not args.sweep:
        print(f"!! WARNING: mixed protocols present {protocols} — NOT comparable")
        print("   (use --lr to filter, or --sweep to group by lr)\n")

    datasets = sorted({ds for b in table.values() for ds in b})
    bases = [b for b in ORDER if b in table] + \
            [b for b in sorted(table) if b not in ORDER]

    for ds in datasets:
        print(f"\n=== {ds}  (split={args.split}, headline = fold-mean R² ± std) ===")
        print(f"{'base':<28} {'R² (mean over seeds)':<26} {'Spearman':<12} seeds")
        for b in bases:
            entries = table[b].get(ds, [])
            if not entries:
                continue
            r2s = [e[1] for e in entries]
            sps = [e[3] for e in entries if e[3] is not None]
            seeds = [e[0] for e in entries]
            if len(r2s) > 1:
                r2_txt = f"{st.mean(r2s):.4f}  (n={len(r2s)}, sd={st.stdev(r2s):.4f})"
            else:
                sd = entries[0][2]
                r2_txt = f"{r2s[0]:.4f}" + (f" ± {sd:.4f} (folds)" if sd is not None else "")
            sp_txt = f"{st.mean(sps):.4f}" if sps else "—"
            print(f"{b:<28} {r2_txt:<26} {sp_txt:<12} {seeds}")

    print("\nNote: per-fold R² is the headline; pooled R² is diagnostic only.")


if __name__ == "__main__":
    main()
