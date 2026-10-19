#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPU stability soak test — targets the failure mode this box actually has.

The three recorded failures were all `Xid 79: GPU has fallen off the bus`, i.e. the
device disappearing from the PCIe bus, not an OOM or a compute error. Two of the
three happened under load and one happened while completely IDLE, so "run a matmul
loop" is not by itself the right probe: the previous gpu_full_power_test.py already
does that and would not have predicted the idle failure.

This test therefore exercises the things that stress the LINK and the driver rather
than only the SM:

  1. Host<->device transfers of varying size — PCIe traffic, which is where a
     fallen-off-bus manifests first.
  2. Memory allocation/free churn — forces driver page-table work.
  3. Synchronize() after every op, so a hang is detected as a hang rather than
     silently coasting on async queues.
  4. Idle gaps between rounds — the third failure happened at 0% utilisation, so a
     soak that never idles would miss it.

Every round records wall time and checks the device is still enumerable; any
exception is caught and reported rather than aborting, so one bad round does not
hide the rest of the picture.

Usage:
    conda run -n unimol2 python 4_scripts/gpu_soak_test.py --minutes 30
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time


def _smi(query: str) -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception as exc:  # nvidia-smi itself failing is a signal
        return f"<nvidia-smi failed: {type(exc).__name__}>"


def _visible() -> bool:
    """Cheapest possible 'is the device still there' check."""
    return "Unknown Error" not in _smi("name") and _smi("name") != ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--idle-gap", type=float, default=5.0,
                    help="seconds of idle between rounds (the 3rd failure was at idle)")
    ap.add_argument("--transfer-mb", type=int, default=512)
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("!! CUDA not available at start — device already unhealthy")
        return 2
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"initial nvidia-smi: {_smi('name,memory.used,utilization.gpu,temperature.gpu,power.draw')}",
          flush=True)

    deadline = time.time() + args.minutes * 60
    rnd = 0
    failures = []

    while time.time() < deadline:
        rnd += 1
        t0 = time.time()
        status = "ok"
        try:
            # 1. allocation churn — driver page-table work
            blocks = [torch.empty(64 * 1024 * 1024 // 4, device="cuda")
                      for _ in range(8)]
            for b in blocks:
                b.fill_(rnd)
            del blocks

            # 2. PCIe traffic both directions with sync, so a link stall surfaces
            # Values stay small on purpose: float32 represents integers exactly only
            # up to 2**24, so an arange over a 512MB buffer would round the tail and
            # a "last element" check would report a false corruption.
            n = args.transfer_mb * 1024 * 1024 // 4
            host = torch.arange(n, dtype=torch.float32) % 1024.0
            dev = host.to("cuda")
            torch.cuda.synchronize()
            back = dev.cpu()
            torch.cuda.synchronize()
            assert torch.equal(back, host), "host<->device transfer corrupted"

            # 3. compute, with sync so errors are attributed to this round
            a = torch.randn(2048, 2048, device="cuda")
            c = a @ a
            torch.cuda.synchronize()
            assert torch.isfinite(c).all(), "non-finite matmul result"
            del a, c, host, dev, back

            if not _visible():
                status = "DEVICE GONE (nvidia-smi cannot see it)"
        except Exception as exc:
            status = f"{type(exc).__name__}: {exc}"
            failures.append((rnd, status))

        dt = time.time() - t0
        if rnd % 5 == 0 or status != "ok":
            print(f"  round {rnd:4d}  {dt:6.2f}s  {status}", flush=True)
        if status != "ok":
            print(f"    nvidia-smi: {_smi('name,memory.used,temperature.gpu')}", flush=True)

        # 4. deliberate idle — the third recorded failure happened at 0% util
        time.sleep(args.idle_gap)

    print(f"\n=== soak finished: {rnd} rounds, {len(failures)} failures ===", flush=True)
    print(f"final nvidia-smi: {_smi('name,memory.used,utilization.gpu,temperature.gpu,power.draw')}",
          flush=True)
    if failures:
        print("FAILURES:")
        for r, s in failures[:10]:
            print(f"  round {r}: {s}")
        return 1
    print("VERDICT: no failures — device survived the soak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
