"""Export the development rows (24 features, label, timestamp flag) for the Spark MLlib demo. No test rows."""
from pathlib import Path

import pandas as pd

from bank_common import FEATURES_MODEL

Path("data_storage/part5").mkdir(parents=True, exist_ok=True)
cols = ["CERT", "qidx", "y", "y_before_public"] + FEATURES_MODEL
df = pd.read_parquet("data_storage/part4/features.parquet", filters=[("period", "==", "dev")], columns=cols)
df[FEATURES_MODEL] = df[FEATURES_MODEL].astype("float32")
df["y"] = df["y"].astype("float64")
df.to_parquet("data_storage/part5/mllib_input.parquet", index=False, compression="snappy")
print(df.shape, f"{Path('data_storage/part5/mllib_input.parquet').stat().st_size / 1e6:.0f} MB")