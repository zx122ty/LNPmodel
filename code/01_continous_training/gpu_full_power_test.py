#!/usr/bin/env python
"""GPU full-power stress test with live temperature/power logging.

Pulls the NVIDIA GPU to maximum power with dense matmul compute and
continuously logs temperature, power draw, utilization, and fan speed.

Run in the `unimol2` conda env:

    conda activate unimol2
    python gpu_full_power_test.py --seconds 300 --size 8192
"""
import argparse
import subprocess
import sys
import threading
import time

import torch

torch.cuda.set_device(0)
DEV = "cuda"
print(f"Using GPU: {torch.cuda.get_device_name(0)}", flush=True)

LOG_PATH = "gpu_full_power_log.csv"
START = time.time()
stop = threading.Event()


def query_gpu():
    """Return {temp_c, power_w, util_pct, mem_mib, fan_pct} from nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=temperature.gpu,power.draw,utilization.gpu,"
         "memory.used,fan.speed",
         "--format=csv,noheader,nounits"],
        text=True).strip()
    parts = [p.strip().rstrip("]").lstrip("[") for p in out.split(",")]

    def f(x):
        try:
            return float(x) if x not in ("N/A", "") else -1.0
        except ValueError:
            return -1.0

    return {
        "temp_c": f(parts[0]),
        "power_w": f(parts[1]),
        "util_pct": f(parts[2]),
        "mem_mib": f(parts[3]),
        "fan_pct": f(parts[4]),
    }


def monitor(interval):
    """Log sensors to CSV + a single-line stdout readout until stop set."""
    with open(LOG_PATH, "w") as f:
        header = "elapsed_s,temp_c,power_w,util_pct,mem_mib,fan_pct\n"
        f.write(header)
        while not stop.is_set():
            t0 = time.time()
            try:
                s = query_gpu()
                line = (f"{t0-START:.1f},{s['temp_c']},{s['power_w']},"
                        f"{s['util_pct']},{s['mem_mib']},{s['fan_pct']}\n")
                f.write(line)
                f.flush()
                sys.stdout.write(
                    f"\r[{t0-START:6.0f}s] temp {s['temp_c']:4.1f}C | "
                    f"power {s['power_w']:6.1f}W/{450:.0f}W | "
                    f"util {s['util_pct']:3.0f}% | "
                    f"mem {s['mem_mib']:6.0f}MiB | "
                    f"fan {s['fan_pct']:3.0f}%    ")
                sys.stdout.flush()
            except subprocess.CalledProcessError as e:
                f.write(f"ERROR nvidia-smi: {e}\n")
            stop.wait(interval)


def stress_loop(size):
    """Continuous dense matmul + activation compute to maximize power."""
    a = torch.randn(size, size, device=DEV)
    b = torch.randn(size, size, device=DEV)
    torch.cuda.synchronize()
    start = time.time()
    i = 0
    while not stop.is_set():
        c = a @ b
        force = c.sum()          # materialize so work is not dead-code-eliminated
        # Alternate pure SGEMM with an activation path to keep both the
        # tensor-core and SIMD/FP32 units saturated.
        if i % 4 == 0:
            r = torch.relu(a) @ torch.sigmoid(b) + force
            del r
        del c, force
        i += 1
        if i % 25 == 0:
            torch.cuda.synchronize()
            print(f"\n-- {i} iterations, elapsed {time.time()-start:.1f}s --",
                  flush=True)


def main():
    ap = argparse.ArgumentParser(description="GPU full-power temperature test")
    ap.add_argument("--seconds", type=int, default=300, help="run duration (s)")
    ap.add_argument("--size", type=int, default=8192,
                    help="square matmul matrix size (larger = more bandwidth)")
    ap.add_argument("--interval", type=float, default=1.0,
                    help="sensor sample interval (s)")
    args = ap.parse_args()

    m = threading.Thread(target=monitor, args=(args.interval,), daemon=True)
    m.start()
    l = threading.Thread(target=stress_loop, args=(args.size,), daemon=True)
    l.start()

    time.sleep(args.seconds)
    stop.set()
    m.join(timeout=5)
    l.join(timeout=5)

    # Final summary.
    s = query_gpu()
    print("\n\n=== FINAL READINGS after {:.0f}s ===".format(args.seconds))
    print(f"Temperature : {s['temp_c']:.1f} C")
    print(f"Power draw  : {s['power_w']:.1f} W")
    print(f"GPU util    : {s['util_pct']:.0f} %")
    print(f"Memory used : {s['mem_mib']:.0f} MiB")
    print(f"Fan speed   : {s['fan_pct']:.0f} %")
    print(f"\nFull log written to {LOG_PATH}")


if __name__ == "__main__":
    main()