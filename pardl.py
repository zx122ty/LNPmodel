#!/usr/bin/env python3
"""Parallel ranged downloader.

download.pytorch.org serves the cu130 torch wheels at ~45 KB/s on a single
connection and frequently stalls outright, which makes `pip install torch` from
that index unusable here. The same host serves ranged requests at ~1.3 MB/s per
connection, so we split the file into N chunks and pull them concurrently,
resuming any chunk that dies.

Each worker owns one chunk and retries it until complete -- one flaky chunk
never aborts the others. A chunk is only ever appended to, so a resumed run
picks up where the previous one left off. Assembly refuses to run unless every
part is exactly the expected size.

Usage: pardl.py URL OUTPATH [--parts N]
"""
import argparse
import os
import sys
import threading
import time
import urllib.request

SOCK_TIMEOUT = 30  # per-read timeout; a stalled chunk errors out and retries
MAX_ATTEMPTS = 60
DEADLINE = 3600


def head(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"]), r.headers.get("Accept-Ranges", "")


def fetch_range(url, base, end, path, stop):
    """Download absolute byte range [base, end] into path, resuming on retry."""
    expected = end - base + 1
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if stop.is_set():
            return False
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= expected:
            return True
        # Always resume from base + bytes-already-written. Never mutate base.
        req = urllib.request.Request(
            url, headers={"Range": f"bytes={base + have}-{end}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=SOCK_TIMEOUT) as r, \
                    open(path, "ab") as fh:
                while not stop.is_set():
                    try:
                        buf = r.read(1 << 20)
                    except Exception as exc:
                        print(f"    part@{base} read error: {exc}", flush=True)
                        break
                    if not buf:
                        break
                    fh.write(buf)
                    fh.flush()
        except Exception as exc:
            print(f"    part@{base} attempt {attempt}: {exc}", flush=True)
        if os.path.getsize(path) >= expected:
            return True
        time.sleep(2)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--parts", type=int, default=12)
    args = ap.parse_args()

    total, accept = head(args.url)
    print(f"size={total/1048576:.1f} MB  accept-ranges={accept!r}  parts={args.parts}",
          flush=True)
    if "bytes" not in accept:
        print("server does not advertise ranges", flush=True)
        return 1

    part_dir = args.out + ".parts"
    os.makedirs(part_dir, exist_ok=True)

    n = args.parts
    step = total // n
    bounds = []
    for i in range(n):
        lo = i * step
        hi = total - 1 if i == n - 1 else (i + 1) * step - 1
        bounds.append((lo, hi, os.path.join(part_dir, f"part{i:02d}")))

    stop = threading.Event()
    results = [None] * n
    t0 = time.time()

    def worker(i, lo, hi, path):
        try:
            results[i] = fetch_range(args.url, lo, hi, path, stop)
        except Exception as exc:  # never let a thread die silently
            print(f"    part{i} fatal: {exc}", flush=True)
            results[i] = False

    threads = [
        threading.Thread(target=worker, args=(i,) + b, daemon=True)
        for i, b in enumerate(bounds)
    ]
    for t in threads:
        t.start()

    while any(r is None for r in results):
        time.sleep(10)
        got = sum(os.path.getsize(b[2]) for b in bounds if os.path.exists(b[2]))
        rate = got / max(time.time() - t0, 1) / 1048576
        print(f"  {got/1048576:7.1f} / {total/1048576:.1f} MB  ({rate:.1f} MB/s)",
              flush=True)
        if time.time() - t0 > DEADLINE:
            print("deadline exceeded, stopping", flush=True)
            stop.set()
            break

    for t in threads:
        t.join(timeout=60)

    sizes = [os.path.getsize(b[2]) if os.path.exists(b[2]) else 0 for b in bounds]
    want = [b[1] - b[0] + 1 for b in bounds]
    if sizes != want:
        bad = [f"part{i} {s}/{w}" for i, (s, w) in enumerate(zip(sizes, want))
               if s != w]
        print(f"INCOMPLETE: {', '.join(bad)}", flush=True)
        print("re-run the same command to resume", flush=True)
        return 1

    with open(args.out, "wb") as out:
        for _, _, path in bounds:
            with open(path, "rb") as fh:
                while True:
                    buf = fh.read(1 << 22)
                    if not buf:
                        break
                    out.write(buf)

    size = os.path.getsize(args.out)
    if size != total:
        print(f"SIZE MISMATCH: got {size}, expected {total}", flush=True)
        return 1
    print(f"OK {args.out} ({size/1048576:.1f} MB) in {time.time()-t0:.0f}s", flush=True)
    for _, _, path in bounds:
        os.remove(path)
    os.rmdir(part_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
