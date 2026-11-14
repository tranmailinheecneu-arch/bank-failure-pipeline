"""Evidence for the exact-zero rule in the risk-based capital ratios (Part 2.3).

Reads the stored panel BEFORE the cleaning step, so the zeros are still there.
Writes results/part2_zero_pattern.txt
"""
from pathlib import Path

import polars as pl

p = pl.read_parquet("data_storage/panel/*/*.parquet")
if "YEAR" in p.columns:
	p = p.drop("YEAR")
rep = pl.col("REPDTE").cast(pl.Utf8)
p = p.with_columns(
	rep.str.slice(0, 4).cast(pl.Int32).alias("year"),
	rep.str.slice(4, 2).cast(pl.Int32).alias("month"),
)
out = []
tot = p.group_by("year").agg(pl.len().alias("rows"))

for c in ["RBCRWAJ", "RBC1AAJ"]:
	z = p.filter(pl.col(c) == 0).group_by("year").agg(
		pl.len().alias("zero_rows"), pl.col("CERT").n_unique().alias("banks_with_zero")
	)
	t = (
		tot.join(z, on="year", how="left")
		.with_columns(pl.col("zero_rows").fill_null(0), pl.col("banks_with_zero").fill_null(0))
		.with_columns((pl.col("zero_rows") / pl.col("rows") * 100).round(2).alias("zero_pct"))
		.sort("year")
	)
	out.append(f"== {c}: rows equal to exactly 0, by year (null share of the column: "
			   f"{100 * p[c].null_count() / p.height:.2f}%) ==")
	out.append(t.to_pandas().to_string(index=False))
	out.append("")

q = (
	p.filter((pl.col("year") >= 1989) & (pl.col("year") <= 1990))
	.group_by("year", "month")
	.agg(
		pl.len().alias("rows"),
		(pl.col("RBCRWAJ") == 0).sum().alias("RBCRWAJ_zeros"),
		(pl.col("RBC1AAJ") == 0).sum().alias("RBC1AAJ_zeros"),
	)
	.sort("year", "month")
)
out.append("== Transition 1989-1990, by quarter ==")
out.append(q.to_pandas().to_string(index=False))

r1 = p.filter((pl.col("year") >= 2015) & (pl.col("RBC1AAJ") == 0))
r2 = p.filter((pl.col("year") >= 2020) & (pl.col("RBCRWAJ") == 0))
out.append("")
out.append(f"RBC1AAJ == 0 since 2015: {r1.height:,} rows, {r1['CERT'].n_unique():,} distinct banks")
out.append(f"RBCRWAJ == 0 since 2020: {r2.height:,} rows, {r2['CERT'].n_unique():,} distinct banks")

text = "\n".join(out)
Path("results").mkdir(exist_ok=True)
Path("results/part2_zero_pattern.txt").write_text(text, encoding="utf-8")
print(text[-900:])
print("\nsaved to results/part2_zero_pattern.txt")
