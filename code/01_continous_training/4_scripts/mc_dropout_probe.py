#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does MC Dropout work on this model? Measured, not assumed.

The proposal is the standard one: at inference, force every torch.nn.Dropout
module into training mode and forward the same molecule N times; the spread of the
N predictions is the epistemic uncertainty. It is a good idea in general and costs
nothing to try, so it is worth trying rather than arguing about.

Two things can make it produce a confidently wrong answer here, and both are
checked directly:

  1. REGRESSION SETS pooler_dropout TO 0. `NNModel._init_model` (nnmodel.py:125-128)
     does `params['pooler_dropout'] = 0` for regression tasks, and the head is
     `LinearHead` -> `nn.Dropout(p=0)` (unimolv2.py:462). A p=0 dropout is the
     identity even in train mode, so it contributes exactly zero variance.
  2. THE ENCODER'S OWN Dropout IS A CUSTOM CLASS, not nn.Dropout.
     `transformersv2.py:45` defines `Dropout(nn.Module)` whose forward is
     `if self.p > 0 and self.training: ... else: return x`. It does not read the
     `training` flag of its CHILDREN (it has none) and it is not an instance of
     `torch.nn.Dropout`, so the usual `isinstance(m, nn.Dropout)` loop skips it.
     Even if reached, it is gated on `self.training`, so calling `model.train()`
     would be required -- which also switches BatchNorm/LayerNorm statistics and
     is not what "just enable dropout" is supposed to do.

So the script measures the actual sigma under the expert's snippet, then repeats
with `model.train()` (the more aggressive variant) and with a non-zero p injected
into every dropout it can find, to separate "the flags were not reached" from
"reaching them does not help".

Sigma is then judged the way P1 judged the ensemble sigma: against the trivial
spread |mu - ybar| and against the label-using oracle |y - ybar|, and against the
13-arm ensemble sigma on the same molecules. A sigma that does not clear the
trivial control is not an uncertainty estimate.

Usage:
    python 4_scripts/mc_dropout_probe.py --seed 42 --fold 0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

ROOT = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/code/01_continous_training")
XLSX = Path("/home/data/zhengxiao/disk1/my_project/lnpModel/benchmark/input_data/dataset.xlsx")
sys.path.insert(0, str(ROOT / "4_scripts"))
sys.path.insert(0, str(ROOT / "4_scripts/Uni-Mol/unimol_tools"))

DS = "Sheet2"
CKPT = ROOT / ("2_models/continued_pretrain/unimol2_lipids_84m_v5/checkpoints/"
               "checkpoint_1_30000.pt")
RUN = "Sheet2_exp_unimol2_lipids_84m_v5_checkpoint_1_30000.pt_scaffold_seed{}_lr5e-05_ep40"
N_MC = 20


def build_model(fold_dir: Path, fold: int, device: str):
    """Rebuild the fine-tuned fold model and load its weights."""
    import finetune_moltrain as fm
    fm.USE_STOCK_84M = False
    fm.CHECKPOINT = CKPT          # must stay a Path: the loader patch calls .exists()
    fm._install_lipidfm_loader()

    from unimol_tools.models.unimolv2 import UniMolV2Model
    torch.manual_seed(0)
    model = UniMolV2Model(output_dim=1, model_size="84m")
    sd = torch.load(fold_dir / f"model_{fold}.pth", map_location="cpu")
    model.load_state_dict(sd["model_state_dict"])
    return model.to(device)


def dropout_inventory(model) -> list:
    """Every module that could inject noise in train mode, with its p."""
    import torch.nn as nn
    from unimol_tools.models.transformersv2 import Dropout as CustomDropout
    out = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Dropout):
            out.append((name, "nn.Dropout", m.p))
        elif isinstance(m, CustomDropout):
            out.append((name, "transformersv2.Dropout", m.p))
        elif type(m).__name__ == "DropPath":
            out.append((name, "DropPath", getattr(m, "drop_prob", None)))
    return out


def set_dropout_train(model, also_encoder: bool) -> int:
    """The expert's snippet, plus an option to also flip the encoder's custom
    Dropout gates. Returns how many modules were touched."""
    import torch.nn as nn
    from unimol_tools.models.transformersv2 import Dropout as CustomDropout
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
            n += 1
    if also_encoder:
        for m in model.modules():
            if isinstance(m, CustomDropout):
                m.train()
                m.p = max(m.p, 0.1)      # p=0 layers are identity regardless of mode
                n += 1
    return n


@torch.no_grad()
def mc_predict(model, X, idx, device, n_mc=N_MC, batch=32):
    """N stochastic forwards over the molecules at `idx`. Returns (n_mc, len(idx)).

    Only `idx` is forwarded. The question here is whether sigma varies AT ALL
    under the proposed snippet, and that is answerable on one fold's test set --
    forwarding all 1920 molecules x 3 variants on CPU takes hours and buys nothing
    extra. Pass idx = arange(n) to disable.

    Batching goes through `model.batch_collate_fn` -- the model's own collate, which
    pads the variable-length atom/token arrays (unimolv2.py:356-412). Hand-rolling a
    `torch.as_tensor(np.stack(...))` over the payload does not work: `src_tokens` is
    a plain Python list whose length differs per molecule, and stacking it either
    raises or silently leaves a list the encoder cannot consume.
    """
    samples = [(X[i], 0.0) for i in idx]
    outs = []
    for t in range(n_mc):
        torch.manual_seed(1000 + t)
        preds = []
        for i in range(0, len(samples), batch):
            batch_in, _ = model.batch_collate_fn(samples[i:i + batch])
            batch_in = {k: (v.to(device) if torch.is_tensor(v) else v)
                        for k, v in batch_in.items()}
            preds.append(model(**batch_in).cpu().numpy())
        outs.append(np.concatenate(preds).ravel())
    return np.stack(outs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-mc", type=int, default=N_MC)
    ap.add_argument("--cpu", action="store_true",
                    help="force CPU (the vendored UniMolV2Model builds on CPU, and a "
                         "full-library MC sweep there takes hours)")
    args = ap.parse_args()

    fold_dir = ROOT / "5_outputs/finetune_moltrain" / RUN.format(args.seed)
    d = pd.read_excel(XLSX, sheet_name=DS).dropna(subset=["SMILES", "RLU (log2)"])
    smis = d["SMILES"].astype(str).str.strip().tolist()
    y = pd.to_numeric(d["RLU (log2)"]).values.astype(float)
    folds = json.loads((ROOT / f"5_outputs/split_cache/{DS}_scaffold_k5_seed{args.seed}"
                        ".json").read_text())["folds"]
    te = np.asarray(folds[args.fold]["test"], int)

    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    model = build_model(fold_dir, args.fold, device)
    inv = dropout_inventory(model)
    nz = [v for v in inv if (v[2] or 0) > 0]
    print(f"dropout inventory ({len(inv)} modules, {len(nz)} with p>0):")
    for name, kind, p in inv:
        if (p or 0) > 0:
            print(f"    {name:55s} {kind:24s} p={p}")

    from unimol_tools.data.datahub import DataHub
    # Build the model input exactly the way fine-tuning does: the SMILES path of
    # UniMolV2Feature (datahub.py:120-123). Passing ROMol would take the
    # transform_mols branch and the atoms/coordinates route loses the bond graph
    # entirely -- see CLAUDE.md 坑 #2.
    hub = DataHub(data={"SMILES": smis, "TARGET": y}, is_train=False,
                  task="regression", smiles_col="SMILES", target_cols="TARGET",
                  model_name="unimolv2", model_size="84m", data_type="molecule",
                  max_atoms=256, remove_hs=False, multi_process=False, seed=42)
    X = hub.data["unimol_input"]      # list of per-molecule dicts
    assert isinstance(X, list), f"expected a list of dicts, got {type(X)}"
    print(f"  feature payload: {len(X)} molecules x {len(X[0])} keys")
    import joblib
    scaler = joblib.load(fold_dir / "target_scaler.ss")

    ybar = y[np.asarray(folds[args.fold]["train"], int)].mean()
    report = {"seed": args.seed, "fold": args.fold, "n_mc": args.n_mc,
              "dropout_modules": len(inv),
              "dropout_with_p_gt_0": [[n, k, p] for n, k, p in nz], "variants": {}}

    variants = [
        ("expert_snippet", False, False),      # only isinstance(nn.Dropout).train()
        ("encoder_train_mode", True, False),   # also flip the custom Dropout gates
        ("encoder_p0.1_injected", True, True), # and give p=0 layers a real rate
    ]
    for tag, also_encoder, inject in variants:
        model.eval()
        if inject:
            from unimol_tools.models.transformersv2 import Dropout as CustomDropout
            for m in model.modules():
                if isinstance(m, CustomDropout):
                    m.p = 0.1
        touched = set_dropout_train(model, also_encoder)
        P = mc_predict(model, X, te, device, args.n_mc)     # (n_mc, n_test) z-scale
        P_raw = scaler.inverse_transform(P.reshape(-1, 1)).reshape(P.shape)
        mu = P_raw.mean(0)                                  # over the test molecules
        sd = P_raw.std(0, ddof=1)
        yt, mut = y[te], mu
        r2 = float(r2_score(yt, mut))
        rho = float(spearmanr(sd, np.abs(yt - mut)).statistic)
        report["variants"][tag] = {
            "modules_touched": touched, "r2_test_fold": r2,
            "sigma_mean": float(sd.mean()), "sigma_max": float(sd.max()),
            "frac_sigma_exactly_0": float((sd == 0).mean()),
            "spearman_test_fold": rho}
        print(f"\n  [{tag}] touched={touched}  R2={r2:.4f}  "
              f"sigma mean={sd.mean():.4f} max={sd.max():.4f}  "
              f"exactly-zero: {(sd == 0).mean():.1%}")
        print(f"      Spearman(sigma,|err|) test-fold={rho:+.4f}")

    # ---- reference: the 13-arm ensemble sigma on the SAME molecules ----
    z = np.load(ROOT / "5_outputs/ensemble_oof.npz")
    panel = z["scaffold__panel"]                 # (n_arms, n_partitions, n)
    ens_sig = panel.mean(1).std(0, ddof=1)       # model-direction sigma, whole library
    ens_mu = panel.mean(0).mean(0)
    ybar_tr = ybar                                # train-fold mean from above
    report["ensemble_reference"] = {
        "spearman_test_fold": float(spearmanr(ens_sig[te],
                                              np.abs(y[te] - ens_mu[te])).statistic),
        "trivial_control_test_fold": float(spearmanr(
            np.abs(ens_mu[te] - y[te].mean()), np.abs(y[te] - ens_mu[te])).statistic),
        "sigma_mean": float(ens_sig[te].mean()), "sigma_max": float(ens_sig[te].max()),
        "frac_sigma_exactly_0": float((ens_sig[te] == 0).mean())}
    print(f"\n  [reference] 13-arm ensemble sigma, same test fold: "
          f"Spearman={report['ensemble_reference']['spearman_test_fold']:+.4f}  "
          f"trivial control="
          f"{report['ensemble_reference']['trivial_control_test_fold']:+.4f}  "
          f"sigma mean={report['ensemble_reference']['sigma_mean']:.4f}")

    out = ROOT / "5_outputs/mc_dropout_probe.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
