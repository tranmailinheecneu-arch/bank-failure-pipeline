"""Write the 12-column input file used by the Spark notebook (Part 3.2)."""
from pathlib import Path

import polars as pl

from part3_polars_pandas import BASE_COLS, PANEL_GLOB

out = Path("data_storage/part3")
out.mkdir(parents=True, exist_ok=True)
df = pl.read_parquet(PANEL_GLOB, columns=BASE_COLS)
path = out / "spark_input.parquet"
df.write_parquet(path, compression="snappy")  # snappy: the safest codec for Spark
print(df.shape, f"{path.stat().st_size / 1e6:.1f} MB ->", path)
