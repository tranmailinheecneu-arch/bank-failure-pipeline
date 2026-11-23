"""Part 3.1: core feature transformations in Polars (eager and lazy) and in pandas.

What it does
  1. builds 18 features with the same logic in three engines: pandas, Polars eager, Polars lazy
     (safe ratios, 4-quarter calendar lags, 4-quarter rolling window, cross-sectional statistics);
  2. runs every engine in its own process (clean memory), 1 warm-up run + N timed runs;
  3. checks that the three results are the same;
  4. saves the naive and the optimised Polars query plans.
Reads  : data_storage/panel_clean/*/*.parquet (Part 2.3)
Writes : data_storage/part3/features_<engine>.parquet (not committed)
         results/part3_polars_pandas.txt, results/part3_polars_plan.txt, results/part3_1_readme_block.md
All features use only quarter q and earlier quarters (lags) or the same quarter (cross-section): no future data.
"""
import argparse
import gc
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import psutil

PANEL_GLOB = "data_storage/panel_clean/*/*.parquet"
PANEL_DIR = "data_storage/panel_clean"
OUT = Path("data_storage/part3")
RES = Path("results")
KEYS = ["CERT", "qidx"]
VALUE_COLS = ["ASSET", "DEP", "EQ", "LNLSGR", "LNATRES", "NCLNLS", "BRO", "CHBAL", "SC", "ROAQ"]
BASE_COLS = KEYS + VALUE_COLS
FEATURES = [
    "eq_assets", "npl_loans", "reserves_loans", "loans_assets", "liquid_assets", "bro_dep", "log_asset",
    "asset_yoy", "loans_yoy", "dep_yoy", "d_eq_assets_4q", "d_npl_loans_4q",
    "roa_q_mean4", "roa_q_std4", "roa_q_n4",
    "eq_assets_med_q", "npl_loans_med_q", "eq_assets_pct_q",
]
REPEAT = {"pandas": 5, "polars_eager": 5, "polars_lazy": 5}  # timed runs after 1 warm-up run, same for every engine
ENGINES = ["pandas", "polars_eager", "polars_lazy"]


# ---------------------------------------------------------------------------
# Polars pipeline: written once, runs on a DataFrame (eager) or a LazyFrame (lazy)
# ---------------------------------------------------------------------------
def sdiv(num, den):
    """Safe division: null unless the denominator is positive."""
    return pl.when(den > 0).then(num / den)


def polars_pipeline(frame):
    base = frame.select(KEYS + [pl.col(c).cast(pl.Float64) for c in VALUE_COLS])

    # T1 ratios
    base = base.with_columns(
        sdiv(pl.col("EQ"), pl.col("ASSET")).alias("eq_assets"),
        sdiv(pl.col("NCLNLS"), pl.col("LNLSGR")).alias("npl_loans"),
        sdiv(pl.col("LNATRES"), pl.col("LNLSGR")).alias("reserves_loans"),
        sdiv(pl.col("LNLSGR"), pl.col("ASSET")).alias("loans_assets"),
        sdiv(pl.col("CHBAL") + pl.col("SC"), pl.col("ASSET")).alias("liquid_assets"),
        sdiv(pl.col("BRO"), pl.col("DEP")).alias("bro_dep"),
        pl.when(pl.col("ASSET") > 0).then(pl.col("ASSET").log()).alias("log_asset"),
    )

    # T2 calendar lag of 4 quarters (self-join on qidx + 4, so a gap in a bank's history never shifts the lag)
    lag_src = ["ASSET", "LNLSGR", "DEP", "eq_assets", "npl_loans"]
    lag = base.select(
        "CERT", (pl.col("qidx") + 4).alias("qidx"),
        *[pl.col(c).alias(c + "_l4") for c in lag_src],
    )
    base = base.join(lag, on=KEYS, how="left").with_columns(
        pl.when(pl.col("ASSET_l4") > 0).then(pl.col("ASSET") / pl.col("ASSET_l4") - 1).alias("asset_yoy"),
        pl.when(pl.col("LNLSGR_l4") > 0).then(pl.col("LNLSGR") / pl.col("LNLSGR_l4") - 1).alias("loans_yoy"),
        pl.when(pl.col("DEP_l4") > 0).then(pl.col("DEP") / pl.col("DEP_l4") - 1).alias("dep_yoy"),
        (pl.col("eq_assets") - pl.col("eq_assets_l4")).alias("d_eq_assets_4q"),
        (pl.col("npl_loans") - pl.col("npl_loans_l4")).alias("d_npl_loans_4q"),
    )

    # T3 rolling window over the last 4 calendar quarters (q-3 .. q) per bank
    base = base.sort(KEYS)
    roll = base.rolling(index_column="qidx", period="4i", group_by="CERT").agg(
        pl.col("ROAQ").mean().alias("roa_q_mean4"),
        pl.col("ROAQ").std().alias("roa_q_std4"),
        pl.col("ROAQ").count().alias("roa_q_n4"),
    )
    base = base.join(roll, on=KEYS, how="left")

    # T4 cross-sectional statistics within the same quarter
    base = base.with_columns(
        pl.col("eq_assets").median().over("qidx").alias("eq_assets_med_q"),
        pl.col("npl_loans").median().over("qidx").alias("npl_loans_med_q"),
        (pl.col("eq_assets").rank("average").over("qidx") / pl.col("eq_assets").count().over("qidx"))
        .alias("eq_assets_pct_q"),
    )
    return base.select(KEYS + FEATURES).sort(KEYS)


def run_polars_eager():
    df = pl.read_parquet(PANEL_GLOB, columns=BASE_COLS)
    return polars_pipeline(df)


def run_polars_lazy():
    return polars_pipeline(pl.scan_parquet(PANEL_GLOB)).collect()


# ---------------------------------------------------------------------------
# pandas baseline: same logic, idiomatic pandas
# ---------------------------------------------------------------------------
def pandas_pipeline():
    pdf = pd.read_parquet(PANEL_DIR, columns=BASE_COLS)
    pdf[VALUE_COLS] = pdf[VALUE_COLS].astype("float64")

    def sdiv_pd(n, d):
        return (n / d).where(d > 0)

    pdf["eq_assets"] = sdiv_pd(pdf["EQ"], pdf["ASSET"])
    pdf["npl_loans"] = sdiv_pd(pdf["NCLNLS"], pdf["LNLSGR"])
    pdf["reserves_loans"] = sdiv_pd(pdf["LNATRES"], pdf["LNLSGR"])
    pdf["loans_assets"] = sdiv_pd(pdf["LNLSGR"], pdf["ASSET"])
    pdf["liquid_assets"] = sdiv_pd(pdf["CHBAL"] + pdf["SC"], pdf["ASSET"])
    pdf["bro_dep"] = sdiv_pd(pdf["BRO"], pdf["DEP"])
    pdf["log_asset"] = np.log(pdf["ASSET"].where(pdf["ASSET"] > 0))

    lag_src = ["ASSET", "LNLSGR", "DEP", "eq_assets", "npl_loans"]
    lag = pdf[KEYS + lag_src].copy()
    lag["qidx"] = lag["qidx"] + 4
    lag.columns = KEYS + [c + "_l4" for c in lag_src]
    pdf = pdf.merge(lag, on=KEYS, how="left")
    pdf["asset_yoy"] = (pdf["ASSET"] / pdf["ASSET_l4"] - 1).where(pdf["ASSET_l4"] > 0)
    pdf["loans_yoy"] = (pdf["LNLSGR"] / pdf["LNLSGR_l4"] - 1).where(pdf["LNLSGR_l4"] > 0)
    pdf["dep_yoy"] = (pdf["DEP"] / pdf["DEP_l4"] - 1).where(pdf["DEP_l4"] > 0)
    pdf["d_eq_assets_4q"] = pdf["eq_assets"] - pdf["eq_assets_l4"]
    pdf["d_npl_loans_4q"] = pdf["npl_loans"] - pdf["npl_loans_l4"]

    tmp = pdf[["CERT", "qidx", "ROAQ"]].sort_values(KEYS)
    tmp["dt"] = pd.to_datetime(tmp["qidx"].to_numpy().astype("int64"), unit="D")
    roll = (
        tmp.set_index("dt").groupby("CERT")["ROAQ"]
        .rolling("4D", min_periods=1, closed="right")
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    roll["qidx"] = (roll["dt"] - pd.Timestamp("1970-01-01")).dt.days.astype("int32")
    roll = roll.rename(columns={"mean": "roa_q_mean4", "std": "roa_q_std4", "count": "roa_q_n4"}).drop(columns="dt")
    pdf = pdf.merge(roll, on=KEYS, how="left")

    pdf["eq_assets_med_q"] = pdf.groupby("qidx")["eq_assets"].transform("median")
    pdf["npl_loans_med_q"] = pdf.groupby("qidx")["npl_loans"].transform("median")
    pdf["eq_assets_pct_q"] = pdf.groupby("qidx")["eq_assets"].rank(method="average", pct=True)
    return pdf[KEYS + FEATURES].sort_values(KEYS).reset_index(drop=True)


FUNCS = {"pandas": pandas_pipeline, "polars_eager": run_polars_eager, "polars_lazy": run_polars_lazy}


# ---------------------------------------------------------------------------
# Worker: one engine in its own process, with peak-memory sampling
# ---------------------------------------------------------------------------
def worker(name, repeat):
    OUT.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()
    gc.collect()
    base_rss = proc.memory_info().rss
    peak = [base_rss]
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            peak[0] = max(peak[0], proc.memory_info().rss)
            time.sleep(0.02)

    th = threading.Thread(target=poll, daemon=True)
    th.start()
    times, result = [], None
    for i in range(repeat + 1):  # run 0 = warm-up (file cache, imports), not counted
        result = None
        gc.collect()
        t0 = time.perf_counter()
        result = FUNCS[name]()
        dt = time.perf_counter() - t0
        if i > 0:
            times.append(dt)
    stop.set()
    th.join()
    path = OUT / f"features_{name}.parquet"
    if name == "pandas":
        result.to_parquet(path, index=False)
    else:
        result.write_parquet(path, compression="zstd")
    print("RESULT " + json.dumps({
        "engine": name, "runs": repeat, "times": times,
        "peak_delta_mb": (peak[0] - base_rss) / 1e6, "rows": int(len(result)),
    }))


# ---------------------------------------------------------------------------
# Result comparison
# ---------------------------------------------------------------------------
def compare(a, b):
    same_shape = a.shape == b.shape
    same_keys = same_shape and bool((a[KEYS].to_numpy() == b[KEYS].to_numpy()).all())
    rows = []
    if same_keys:
        for f in FEATURES:
            x = a[f].astype("float64").to_numpy()
            y = b[f].astype("float64").to_numpy()
            nan_mis = int((np.isnan(x) != np.isnan(y)).sum())
            both = ~np.isnan(x) & ~np.isnan(y)
            d = np.abs(x[both] - y[both])
            max_abs = float(d.max()) if d.size else 0.0
            max_rel = float((d / np.maximum(np.abs(x[both]), 1e-12)).max()) if d.size else 0.0
            atol = 1e-6 if f.endswith("_std4") else 1e-9  # sliding-window std: variance cancellation noise (max observed diff 1.9e-8)
            ok = nan_mis == 0 and bool(np.allclose(x[both], y[both], rtol=1e-6, atol=atol))
            rows.append((f, max_abs, max_rel, nan_mis, ok))
    table = pd.DataFrame(rows, columns=["feature", "max_abs_diff", "max_rel_diff", "null_mismatch", "match"])
    return same_shape, same_keys, table


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def orchestrate():
    RES.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s=""):
        lines.append(str(s))
        print(s, flush=True)

    ram = psutil.virtual_memory().total / 1e9
    env = (f"{os.cpu_count()} logical CPUs, {ram:.1f} GB RAM | polars {pl.__version__} | "
           f"pandas {pd.__version__} | python {sys.version.split()[0]}")
    log("machine: " + env)

    lf = polars_pipeline(pl.scan_parquet(PANEL_GLOB))
    naive = lf.explain(optimized=False)
    opt = lf.explain()
    (RES / "part3_polars_plan.txt").write_text(
        "== NAIVE PLAN (as written) ==\n" + naive + "\n\n== OPTIMISED PLAN (what runs) ==\n" + opt + "\n",
        encoding="utf-8")
    m = re.findall(r"(\d+)/(\d+) COLUMNS", opt)
    scan_cols = f"{m[0][0]} of {m[0][1]}" if m else "n/a"
    log(f"Polars plan saved (results/part3_polars_plan.txt); columns read by the optimised scan: {scan_cols}")

    res = {}
    for name in ENGINES:
        log(f"\nrunning {name} ({REPEAT[name]} timed runs + 1 warm-up; pandas can take several minutes) ...")
        p = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--worker", name, "--repeat", str(REPEAT[name])],
            capture_output=True, text=True)
        row = None
        for ln in p.stdout.splitlines():
            if ln.startswith("RESULT "):
                row = json.loads(ln[7:])
        if p.returncode != 0 or row is None:
            log(p.stderr[-2500:])
            (RES / "part3_polars_pandas.txt").write_text("\n".join(lines), encoding="utf-8")
            raise SystemExit(f"worker {name} failed")
        res[name] = row
        log(f"  best {min(row['times']):.2f} s | median {float(np.median(row['times'])):.2f} s | "
            f"peak memory above baseline {row['peak_delta_mb']:.0f} MB | rows {row['rows']:,}")

    # timing table
    best_pd = min(res["pandas"]["times"])
    trows = []
    for n in ENGINES:
        r = res[n]
        trows.append((n, r["runs"], round(min(r["times"]), 3), round(float(np.median(r["times"])), 3),
                      round(r["peak_delta_mb"]), round(best_pd / min(r["times"]), 1)))
    tdf = pd.DataFrame(trows, columns=["engine", "timed_runs", "best_s", "median_s", "peak_MB_above_baseline",
                                       "speedup_vs_pandas"])
    log("\n== Benchmark (same 18 features, same input) ==")
    log(tdf.to_string(index=False))

    # equivalence
    ref = pd.read_parquet(OUT / "features_pandas.parquet")
    lazy = pd.read_parquet(OUT / "features_polars_lazy.parquet")
    eager = pd.read_parquet(OUT / "features_polars_eager.parquet")
    s1, k1, t1 = compare(ref, lazy)
    s2, k2, t2 = compare(lazy, eager)
    log("\n== Equivalence: Polars lazy vs pandas ==")
    log(f"same shape: {s1} ({lazy.shape[0]:,} rows, {len(FEATURES)} features) | same (CERT, qidx) keys: {k1}")
    if k1:
        log(t1.to_string(index=False))
    log("\n== Equivalence: Polars eager vs Polars lazy ==")
    log(f"same shape: {s2} | same keys: {k2}")
    if k2:
        log(t2.to_string(index=False))
    n_ok = int(t1["match"].sum()) if k1 else 0
    all_ok = k1 and k2 and bool(t1["match"].all()) and bool(t2["match"].all())
    log(f"\nfeatures matching pandas within rtol 1e-6: {n_ok} of {len(FEATURES)} | "
        f"largest absolute difference: {(t1['max_abs_diff'].max() if k1 else float('nan')):.3g}")
    (RES / "part3_polars_pandas.txt").write_text("\n".join(lines), encoding="utf-8")

    # README block
    b = {r.engine: r for r in tdf.itertuples()}
    md = [
        "### Part 3.1 Polars vs pandas (`results/part3_polars_pandas.txt`)",
        "",
        "Core transformations used later as features (18 columns): safe ratios, 4-quarter calendar lags (self-join on "
        "`qidx + 4`, so gaps in a bank's history never shift a lag), a 4-quarter rolling window of quarterly ROA, "
        "and same-quarter cross-sectional median and rank. Only quarter q and earlier data (or the same quarter) are used.",
        "",
        f"Machine: {env}. Each engine runs in its own process: 1 warm-up run, then the timed runs; "
        "memory is the peak resident set size minus the size after imports.",
        "",
        "| Engine | Timed runs | Best (s) | Median (s) | Peak memory (MB) | Speedup vs pandas |",
        "|---|---|---|---|---|---|",
    ]
    for n in ENGINES:
        r = b[n]
        md.append(f"| {n} | {r.timed_runs} | {r.best_s:.2f} | {r.median_s:.2f} | {r.peak_MB_above_baseline:,} | {r.speedup_vs_pandas}x |")
    md += [
        "",
        f"- Results are identical across engines: {n_ok} of {len(FEATURES)} features match pandas within a relative "
        f"tolerance of 1e-6 (largest absolute difference {(t1['max_abs_diff'].max() if k1 else float('nan')):.3g}), "
        f"same {lazy.shape[0]:,} rows and keys, same null pattern; eager and lazy Polars agree "
        f"({'yes' if k2 and bool(t2['match'].all()) else 'no'}).",
        f"- The optimised lazy plan reads {scan_cols} columns of the panel (projection pushdown; plans in `results/part3_polars_plan.txt`).",
        "- Timings are best-of-N on one laptop and vary between runs.",
    ]
    (RES / "part3_1_readme_block.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("\nREADME block: results/part3_1_readme_block.md | plans: results/part3_polars_plan.txt")
    if not all_ok:
        raise SystemExit("results do NOT match across engines: see the tables above")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker")
    ap.add_argument("--repeat", type=int, default=3)
    args = ap.parse_args()
    if args.worker:
        worker(args.worker, args.repeat)
    else:
        orchestrate()


if __name__ == "__main__":
    main()
