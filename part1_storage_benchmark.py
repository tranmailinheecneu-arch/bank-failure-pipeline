import time
from pathlib import Path
import pandas as pd
import polars as pl

RAW = Path("data_raw/fdic_v2/financials")
OUT = Path("data_storage")
OUT.mkdir(exist_ok=True)


def size_mb(p):
    p = Path(p)
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def best_time(fn, repeat=3):
    ts = []
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return min(ts)


# --- 1. Gộp 170 file thành một bảng ---
files = sorted(RAW.glob("*.parquet"))
print("number of files:", len(files))

schemas = {}
for f in files:
    for col, dtype in pl.read_parquet_schema(f).items():
        schemas.setdefault(col, set()).add(str(dtype))
mixed = {c: t for c, t in schemas.items() if len(t) > 1}
print("columns with mixed dtypes across quarters:", mixed)

df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
print("shape:", df.shape)
print(df.schema)

# --- 2. Ghi các phương án ---
df.write_csv(OUT / "A_panel.csv")
df.write_parquet(OUT / "B_snappy.parquet", compression="snappy")
df.write_parquet(OUT / "C_zstd.parquet", compression="zstd")

ratio_cols = ["ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ"]
df_opt = df.with_columns(
    pl.col("STNAME").cast(pl.Categorical),
    *[pl.col(c).cast(pl.Float32) for c in ratio_cols],
)
df_opt.write_parquet(OUT / "D_opt_zstd.parquet", compression="zstd")

df_year = df_opt.with_columns(
    pl.col("REPDTE").cast(pl.Utf8).str.slice(0, 4).alias("YEAR")
)
(OUT / "E_by_year").mkdir(exist_ok=True)
for (y,) in df_year.select("YEAR").unique().rows():
    d = OUT / "E_by_year" / f"YEAR={y}"
    d.mkdir(exist_ok=True)
    df_year.filter(pl.col("YEAR") == y).drop("YEAR").write_parquet(
        d / "part.parquet", compression="zstd"
    )

# --- 3. Đo ---
cols = ["CERT", "REPDTE", "ASSET", "ROA"]
rows = []
csv_path = OUT / "A_panel.csv"

rows.append((
    "A CSV (pandas)",
    size_mb(csv_path),
    best_time(lambda: pd.read_csv(csv_path)),
    best_time(lambda: pd.read_csv(csv_path, usecols=cols)),
    best_time(lambda: pd.read_csv(csv_path).query("REPDTE >= 20090101 and REPDTE <= 20091231")),
))

for name, fname in [("B Parquet snappy", "B_snappy.parquet"),
                    ("C Parquet zstd", "C_zstd.parquet"),
                    ("D Parquet zstd + schema", "D_opt_zstd.parquet")]:
    p = OUT / fname
    rows.append((
        name,
        size_mb(p),
        best_time(lambda: pl.read_parquet(p)),
        best_time(lambda: pl.read_parquet(p, columns=cols)),
        best_time(lambda: pl.scan_parquet(p)
                  .filter(pl.col("REPDTE").cast(pl.Utf8).str.starts_with("2009"))
                  .collect()),
    ))

e = OUT / "E_by_year"
rows.append((
    "E zstd + schema + by year",
    size_mb(e),
    best_time(lambda: pl.read_parquet(str(e / "*/*.parquet"))),
    best_time(lambda: pl.read_parquet(str(e / "*/*.parquet"), columns=cols)),
    best_time(lambda: pl.read_parquet(str(e / "YEAR=2009/*.parquet"))),
))

res = pd.DataFrame(rows, columns=["scheme", "size_MB", "read_all_s",
                                  "read_4cols_s", "read_year2009_s"])
print(res.round(3).to_string(index=False))
Path("results").mkdir(exist_ok=True)
Path("results/part1_storage.txt").write_text(res.round(3).to_string(index=False))