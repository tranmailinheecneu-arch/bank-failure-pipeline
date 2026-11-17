"""Diagnose the roa_q_std4 mismatch between pandas and Polars (Part 3.1)."""
import numpy as np
import polars as pl

a = pl.read_parquet("data_storage/part3/features_pandas.parquet")
b = pl.read_parquet("data_storage/part3/features_polars_lazy.parquet")
x = a["roa_q_std4"].to_numpy()
y = b["roa_q_std4"].to_numpy()
d = np.abs(x - y)
m = ~np.isnan(d)
print("rows compared:", f"{m.sum():,}")
for thr in (1e-12, 1e-10, 1e-9, 1e-8, 1e-6):
	print(f"rows with |diff| > {thr:g}: {(d[m] > thr).sum():,}")

bad = m & (d > 1e-9)
print("\nrows above 1e-9:", f"{bad.sum():,}")
if bad.sum():
	print("pandas value exactly 0 in those rows :", int((x[bad] == 0).sum()))
	print("polars value exactly 0 in those rows :", int((y[bad] == 0).sum()))
	print("pandas value range in those rows     :", x[bad].min(), x[bad].max())
	print("polars value range in those rows     :", y[bad].min(), y[bad].max())
	i = int(np.argmax(np.where(bad, d, -1)))
	cert, q = int(a["CERT"][i]), int(a["qidx"][i])
	print(f"\nworst row: CERT {cert}, qidx {q}: pandas {x[i]!r} vs polars {y[i]!r}")
	p = pl.read_parquet("data_storage/panel_clean/*/*.parquet", columns=["CERT", "qidx", "ROAQ"])
	w = p.filter((pl.col("CERT") == cert) & (pl.col("qidx") >= q - 4) & (pl.col("qidx") <= q)).sort("qidx")
	print("ROAQ in and just before the 4-quarter window:")
	print(w.to_pandas().to_string(index=False))
