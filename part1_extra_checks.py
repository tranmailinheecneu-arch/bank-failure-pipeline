"""Fair benchmark (format x engine) and float32 precision check."""
import time
from pathlib import Path

import pandas as pd
import polars as pl

S = Path("data_storage")
RES = Path("results")
RES.mkdir(exist_ok=True)
COLS = ["CERT", "REPDTE", "ASSET", "ROA"]
RATIO_COLS = ["ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ","ROAQ"]


def best_time(fn, repeat=3):
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


# ---------- 1. Fair benchmark: same file format read by both engines ----------
csv = S / "A_panel.csv"
pq = S / "D_opt_zstd.parquet"
tests = [
    ("CSV", "pandas", lambda: pd.read_csv(csv), lambda: pd.read_csv(csv, usecols=COLS)),
    ("CSV", "polars", lambda: pl.read_csv(csv, infer_schema_length=None),
     lambda: pl.read_csv(csv, columns=COLS, infer_schema_length=None)),
    ("Parquet", "pandas", lambda: pd.read_parquet(pq), lambda: pd.read_parquet(pq, columns=COLS)),
    ("Parquet", "polars", lambda: pl.read_parquet(pq), lambda: pl.read_parquet(pq, columns=COLS)),
]
rows = []
for fmt, eng, f_all, f_cols in tests:
    rows.append((fmt, eng, round(best_time(f_all), 3), round(best_time(f_cols), 3)))
bench = pd.DataFrame(rows, columns=["format", "engine", "read_all_s", "read_4cols_s"])
print(bench.to_string(index=False))
(RES / "part1_fair_benchmark.txt").write_text(bench.to_string(index=False))

# ---------- 2. float32 precision: float64 original vs float32 stored ----------
a = pl.read_parquet(S / "C_zstd.parquet")       # original precision
b = pl.read_parquet(S / "D_opt_zstd.parquet")   # ratios stored as float32
out = []
for c in RATIO_COLS:
    x = a[c].cast(pl.Float64)
    y = b[c].cast(pl.Float64)
    diff = (x - y).abs()
    rel = (diff / x.abs()).filter(x.abs() > 1e-9)
    out.append((c, float(x.abs().max()), float(diff.max()), float(rel.max()),
                int((x.is_null() != y.is_null()).sum())))
prec = pd.DataFrame(out, columns=["column", "max_abs_value", "max_abs_error", "max_rel_error", "null_mismatch"])
text = prec.to_string(index=False)
text += f"\n\nmoney columns kept at full precision: ASSET={b.schema['ASSET']}, DEP={b.schema['DEP']}"
print(text)
(RES / "part1_float32_check.txt").write_text(text)