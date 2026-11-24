"""Part 3.3 (laptop part), version 2: Polars time and memory by data size, in-memory and streaming engines.

The panel is replicated k times (CERT + i * 10,000,000). Every (mode, k) runs in its own process; a watchdog stops the process
above BUDGET_GB, so the laptop never swaps. Writes results/part3_scaling_v2.txt.
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import polars as pl
import psutil

from part3_polars_pandas import PANEL_GLOB, polars_pipeline

RES = Path("results")
KS = [1, 2, 4, 6]
BUDGET_GB = 5.0
REPEAT = 2


def collect(lf, mode):
    if mode == "streaming":
        try:
            return lf.collect(engine="streaming")
        except TypeError:
            return lf.collect(streaming=True)
    return lf.collect()


def worker(k, mode):
    proc = psutil.Process()
    gc.collect()
    base = proc.memory_info().rss
    peak = [base]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            r = proc.memory_info().rss
            peak[0] = max(peak[0], r)
            if r > BUDGET_GB * 1e9:
                print("RESULT " + json.dumps({"k": k, "mode": mode, "aborted": True, "peak_mb": (peak[0] - base) / 1e6}), flush=True)
                os._exit(3)
            time.sleep(0.02)

    threading.Thread(target=poll, daemon=True).start()
    lf = pl.scan_parquet(PANEL_GLOB)
    big = pl.concat([lf.with_columns((pl.col("CERT") + i * 10_000_000).alias("CERT")) for i in range(k)])
    times, rows = [], 0
    for _ in range(REPEAT):
        gc.collect()
        t0 = time.perf_counter()
        out = collect(polars_pipeline(big), mode)
        times.append(time.perf_counter() - t0)
        rows = out.height
        del out
    stop.set()
    print("RESULT " + json.dumps({"k": k, "mode": mode, "aborted": False, "times": times, "rows": rows,
                                  "peak_mb": (peak[0] - base) / 1e6}), flush=True)


def orchestrate():
    RES.mkdir(exist_ok=True)
    ram = psutil.virtual_memory().total / 1e9
    lines = [f"machine: {os.cpu_count()} logical CPUs, {ram:.1f} GB RAM | polars {pl.__version__} | memory guard {BUDGET_GB} GB "
             f"above baseline | {REPEAT} runs per size (best time reported)"]
    table = ["", "| Engine mode | Size | Rows | Best time (s) | Peak memory (MB) | MB per million rows |", "|---|---|---|---|---|---|"]
    for mode in ["memory", "streaming"]:
        for k in KS:
            print(f"running {mode}, k = {k} ...", flush=True)
            p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", str(k), "--mode", mode],
                               capture_output=True, text=True)
            res = None
            for ln in p.stdout.splitlines():
                if ln.startswith("RESULT "):
                    res = json.loads(ln[7:])
            if res is None:
                lines.append(f"{mode} k = {k}: worker crashed\n{p.stderr[-600:]}")
                break
            if res["aborted"]:
                lines.append(f"{mode} k = {k}: ABORTED by the memory guard at {res['peak_mb']:,.0f} MB above baseline")
                table.append(f"| {mode} | {k}x | {k * 1_681_431:,} | stopped | above guard | - |")
                break
            m = res["rows"] / 1e6
            lines.append(f"{mode} k = {k}: {res['rows']:,} rows | best {min(res['times']):.2f} s | peak {res['peak_mb']:,.0f} MB")
            table.append(f"| {mode} | {k}x | {res['rows']:,} | {min(res['times']):.2f} | {res['peak_mb']:,.0f} | {res['peak_mb'] / m:,.0f} |")
    text = "\n".join(lines + table)
    (RES / "part3_scaling_v2.txt").write_text(text, encoding="utf-8")
    print("\n" + text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--mode", default="memory")
    a = ap.parse_args()
    worker(a.worker, a.mode) if a.worker else orchestrate()