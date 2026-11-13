"""Part 2.1: automated data-quality report on the stored panel."""
from pathlib import Path

import polars as pl

PANEL = "data_storage/panel/*/*.parquet"
RES = Path("results")
RES.mkdir(exist_ok=True)
out = []

df = pl.read_parquet(PANEL)
rep = pl.col("REPDTE").cast(pl.Utf8)
df = df.with_columns(
	rep.str.slice(0, 4).cast(pl.Int32).alias("yr"),
	rep.str.slice(4, 2).cast(pl.Int32).alias("mo"),
).with_columns(
	(pl.col("yr") * 4 + (pl.col("mo") - 1) // 3).alias("qidx")
)
skip = {"CERT", "REPDTE", "yr", "mo", "qidx", "YEAR"}
cols = [c for c in df.columns if c not in skip]
num_cols = [c for c in cols if df.schema[c].is_numeric()]

out.append(f"rows: {df.height:,} | banks: {df['CERT'].n_unique():,} | columns: {len(cols)}")

# 1. Panel size over time
per_year = df.group_by("yr").agg(pl.col("CERT").n_unique().alias("banks"), pl.len().alias("rows")).sort("yr")
out.append("\n== Banks per year ==")
out.append(per_year.to_pandas().to_string(index=False))

# 2. Missing values: overall and first year each column has data
rows = []
for c in cols:
	nn = df.filter(pl.col(c).is_not_null())
	first = int(nn["yr"].min()) if nn.height else None
	rows.append((c, round(100 * df[c].null_count() / df.height, 2), first))
out.append("\n== Missing % overall and first year with data ==")
out.append(pl.DataFrame(rows, schema=["column", "null_pct", "first_year"], orient="row").to_pandas().to_string(index=False))

# 3. Missing % by year (wide table)
miss = df.group_by("yr").agg([(pl.col(c).is_null().mean() * 100).round(1).alias(c) for c in cols]).sort("yr")
out.append("\n== Missing % by year ==")
out.append(miss.to_pandas().to_string(index=False))

# 4. Outliers and impossible values
orow = []
for c in num_cols:
	s = df[c].drop_nulls().cast(pl.Float64)
	if s.len() == 0:
		continue
	orow.append((c, s.min(), s.quantile(0.001), s.quantile(0.5), s.quantile(0.999), s.max(),
				 int((s < 0).sum()), int((s == 0).sum())))
out.append("\n== Distribution: min / p0.1 / median / p99.9 / max / n_negative / n_zero ==")
out.append(pl.DataFrame(orow, schema=["column", "min", "p0.1", "median", "p99.9", "max", "n_neg", "n_zero"],
						orient="row").to_pandas().round(3).to_string(index=False))
out.append(f"\nASSET <= 0: {int((df['ASSET'] <= 0).sum())} rows")

# 5. Duplicates and gaps in each bank's history
out.append(f"\nduplicate (CERT, REPDTE): {df.height - df.select(['CERT', 'REPDTE']).unique().height}")
g = (df.select(["CERT", "qidx"]).group_by("CERT")
	   .agg(pl.col("qidx").min().alias("first"), pl.col("qidx").max().alias("last"), pl.len().alias("n"))
	   .with_columns(((pl.col("last") - pl.col("first") + 1) - pl.col("n")).alias("missing_q")))
out.append(f"banks with gaps inside their history: {int((g['missing_q'] > 0).sum()):,} of {g.height:,}")
out.append(f"total missing bank-quarters inside histories: {int(g['missing_q'].sum()):,}")
out.append(f"banks with a single quarter only: {int((g['n'] == 1).sum()):,}")

text = "\n".join(out)
(RES / "part2_dq_report.txt").write_text(text, encoding="utf-8")
print(text[:3000])
print("\n... full report saved to results/part2_dq_report.txt")

