"""Part 3.3: measure Polars time and memory as the panel grows."""
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
KS = [1, 2, 3]
BUDGET_GB = 5.0
REPEAT = 2


def worker(k):
	proc = psutil.Process()
	gc.collect()
	base = proc.memory_info().rss
	peak = [base]
	stop = threading.Event()

	def poll():
		while not stop.is_set():
			rss = proc.memory_info().rss
			peak[0] = max(peak[0], rss)
			if rss > BUDGET_GB * 1e9:
				print(
					"RESULT "
					+ json.dumps(
						{"k": k, "aborted": True, "peak_mb": (peak[0] - base) / 1e6}
					),
					flush=True,
				)
				os._exit(3)
			time.sleep(0.02)

	threading.Thread(target=poll, daemon=True).start()
	lf = pl.scan_parquet(PANEL_GLOB)
	big = pl.concat(
		[
			lf.with_columns(
				(pl.col("CERT") + i * 10_000_000).alias("CERT")
			)
			for i in range(k)
		]
	)
	times, rows = [], 0
	for _ in range(REPEAT):
		gc.collect()
		t0 = time.perf_counter()
		out = polars_pipeline(big).collect()
		times.append(time.perf_counter() - t0)
		rows = out.height
		del out
	stop.set()
	print(
		"RESULT "
		+ json.dumps(
			{
				"k": k,
				"aborted": False,
				"times": times,
				"rows": rows,
				"peak_mb": (peak[0] - base) / 1e6,
			}
		),
		flush=True,
	)


def orchestrate():
	RES.mkdir(exist_ok=True)
	ram = psutil.virtual_memory().total / 1e9
	lines = [
		f"machine: {os.cpu_count()} logical CPUs, {ram:.1f} GB RAM | polars {pl.__version__} | "
		f"memory guard {BUDGET_GB} GB above baseline | {REPEAT} runs per size (best time reported)"
	]
	rows = []
	for k in KS:
		print(f"running k = {k} ...", flush=True)
		p = subprocess.run(
			[sys.executable, str(Path(__file__).resolve()), "--worker", str(k)],
			capture_output=True,
			text=True,
		)
		res = None
		for ln in p.stdout.splitlines():
			if ln.startswith("RESULT "):
				res = json.loads(ln[7:])
		if res is None:
			lines.append(f"k = {k}: worker crashed\n{p.stderr[-800:]}")
			break
		rows.append(res)
		if res["aborted"]:
			lines.append(
				f"k = {k}: ABORTED by the memory guard at {res['peak_mb']:,.0f} MB above baseline"
			)
			break
		lines.append(
			f"k = {k}: {res['rows']:,} rows | best {min(res['times']):.2f} s | "
			f"peak {res['peak_mb']:,.0f} MB"
		)

	ok = [r for r in rows if not r["aborted"]]
	lines.extend(
		[
			"",
			"| Size | Rows | Best time (s) | Peak memory (MB) | MB per million rows | Seconds per million rows |",
			"|---|---|---|---|---|---|",
		]
	)
	for r in ok:
		millions = r["rows"] / 1e6
		lines.append(
			f"| {r['k']}x | {r['rows']:,} | {min(r['times']):.2f} | {r['peak_mb']:,.0f} | "
			f"{r['peak_mb'] / millions:,.0f} | {min(r['times']) / millions:.2f} |"
		)
	for r in rows:
		if r["aborted"]:
			lines.append(
				f"| {r['k']}x | {r['k'] * 1_681_431:,} | stopped | "
				f"above guard ({r['peak_mb']:,.0f} so far) | - | - |"
			)
	if ok:
		mb_per_million = max(ok[-1]["peak_mb"] / (ok[-1]["rows"] / 1e6), 1)
		fit = ram * 1e3 * 0.8 / mb_per_million
		lines.append(
			f"\nextrapolation (not measured): at about {mb_per_million:,.0f} MB per million rows, "
			f"80% of the {ram:.1f} GB RAM holds roughly {fit:,.1f} million rows, "
			f"i.e. {fit / 1.681431:,.1f}x the current panel"
		)

	text = "\n".join(lines)
	(RES / "part3_scaling.txt").write_text(text, encoding="utf-8")
	(RES / "part3_scaling_readme_block.md").write_text(text, encoding="utf-8")
	print("\n" + text)


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--worker", type=int)
	args = ap.parse_args()
	worker(args.worker) if args.worker else orchestrate()
