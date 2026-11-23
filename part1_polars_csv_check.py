import time
from pathlib import Path
import polars as pl

csv = max(Path("data_storage").glob("*.csv"), key=lambda p: p.stat().st_size)
cols4 = pl.read_csv(csv, n_rows=5).columns[:4]
schema = pl.read_csv(csv, infer_schema_length=None).schema   # lấy schema đầy đủ một lần

def best(fn, n=3):
	ts = []
	for _ in range(n):
		s = time.perf_counter(); fn(); ts.append(time.perf_counter() - s)
	return min(ts)

cases = {
	"infer schema from all rows": dict(infer_schema_length=None),
	"infer schema from 10,000 rows": dict(infer_schema_length=10_000),
	"explicit schema (no inference)": dict(schema=schema),
}
lines = [f"file: {csv.name} ({csv.stat().st_size / 1e6:.0f} MB); best of 3, seconds"]
for name, kw in cases.items():
	try:
		full = best(lambda: pl.read_csv(csv, **kw))
		four = best(lambda: pl.read_csv(csv, columns=cols4, **kw))
		lines.append(f"{name}: read all {full:.3f} | read 4 columns {four:.3f}")
	except Exception as exc:
		lines.append(f"{name}: failed ({type(exc).__name__}: {str(exc)[:120]})")
Path("results/part1_polars_csv_check.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
