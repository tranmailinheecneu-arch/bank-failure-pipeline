"""Build the final storage layout: Parquet + zstd + optimised schema, partitioned by year."""
import json
import shutil
from pathlib import Path

import polars as pl

RAW = Path("data_raw/fdic/financials")
OUT = Path("data_storage/panel")  # final dataset used by the later parts
RES = Path("results")
RES.mkdir(exist_ok=True)

RATIO_COLS = ["ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ"]  # ratios in %, safe for float32

files = sorted(RAW.glob("*.parquet"))
print("raw files:", len(files))

# 1. Data types of every column across the quarterly files
raw_types = {}
for file in files:
	for column, dtype in pl.read_parquet_schema(file).items():
		raw_types.setdefault(column, set()).add(str(dtype))
mixed = {column: sorted(types) for column, types in raw_types.items() if len(types) > 1}
print("columns with mixed dtypes across quarters:", mixed if mixed else "none")

df = pl.concat([pl.read_parquet(file) for file in files], how="vertical_relaxed")
print("rows:", df.height, "| columns:", df.width)

# 2. Schema optimisation: ratios as float32 and state name as categorical
df = df.with_columns(
	pl.col("STNAME").cast(pl.Categorical),
	*[pl.col(column).cast(pl.Float32) for column in RATIO_COLS],
)

# 3. Schema table: one line per column
nulls = df.null_count().row(0, named=True)
lines = [f"{'column':<10} {'raw dtype(s)':<28} {'stored dtype':<14} {'null %':>8}"]
for column in df.columns:
	raw = "/".join(sorted(raw_types.get(column, {"?"})))
	percent_null = 100 * nulls[column] / df.height
	lines.append(
		f"{column:<10} {raw:<28} {str(df.schema[column]):<14} {percent_null:8.2f}"
	)
(RES / "part1_schema_table.txt").write_text("\n".join(lines))
print("\n".join(lines))

# 4. Write partitioned by year, using zstd compression
df = df.with_columns(
	pl.col("REPDTE").cast(pl.Utf8).str.slice(0, 4).alias("YEAR")
)
if OUT.exists():
	shutil.rmtree(OUT)
for (year,) in df.select("YEAR").unique().sort("YEAR").iter_rows():
	partition = OUT / f"YEAR={year}"
	partition.mkdir(parents=True)
	df.filter(pl.col("YEAR") == year).drop("YEAR").write_parquet(
		partition / "part.parquet", compression="zstd"
	)

# 5. Verify the written dataset
check = pl.read_parquet(str(OUT / "*/*.parquet"))
duplicate_count = check.height - check.select(["CERT", "REPDTE"]).unique().height
size_mb = sum(file.stat().st_size for file in OUT.rglob("*") if file.is_file()) / 1e6
print(
	f"\nwritten rows: {check.height:,} (expected {df.height:,}) "
	f"| duplicates CERT-REPDTE: {duplicate_count} | size: {size_mb:.1f} MB"
)

# 6. Record evidence of complete download from the download manifest
manifest_path = Path("data_raw/fdic/manifest_financials.json")
manifest = json.loads(manifest_path.read_text())
shutil.copy(manifest_path, RES / "manifest_financials.json")
quarter_rows = [entry["rows"] for entry in manifest.values()]
summary = (
	f"quarters: {len(manifest)}\n"
	f"total rows: {sum(quarter_rows):,}\n"
	f"rows per quarter: min {min(quarter_rows):,} / max {max(quarter_rows):,}\n"
	f"quarters not complete: "
	f"{[quarter for quarter, entry in manifest.items() if not entry['complete']] or 'none'}\n"
	f"rows == API total in every quarter: "
	f"{all(entry['rows'] == entry['api_total'] for entry in manifest.values())}\n"
)
(RES / "part1_completeness.txt").write_text(summary)
print("\n" + summary)
