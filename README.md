 # Bank Failure Early-Warning Pipeline (BFT65B)

Individual final report, National Economics University, BFT65B Big Data Analytics in Finance.
Author: Tran Mai Linh (MSV: 11230990). Data source: [US FDIC BankFind API](https://api.fdic.gov/banks). Data files are not committed.

## Setup
Python 3.13 (Windows tested). Disk space needed: about 1 GB.

```
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Repository layout
- `fdic_download_starter.py`: download script (FDIC API)
- `part1_storage_benchmark.py`, `build_storage.py`, `part1_extra_checks.py`: Part 1
- `results/`: small result files (tables, completeness evidence)
- `data_raw/`, `data_storage/`: generated locally, not committed

## Part 1: Acquisition and storage
Quick test (4 quarters, 2-3 minutes) instead of step 1:
`python fdic_download_starter.py --start 2023-03-31 --end 2023-12-31`

Run in this order (steps 3 and 4 need the files created by step 2):

1. Download the full data (about one hour, resumable; rerun the same command if interrupted): `python fdic_download_starter.py`
2. CSV vs. Parquet storage benchmark (creates `data_storage/*.csv` and `*.parquet`): `python part1_storage_benchmark.py` (output: `results/part1_storage.txt`)
3. Build the final storage layout (Parquet + zstd, partitioned by year): `python build_storage.py`
4. Fair benchmark and float32 check: `python part1_extra_checks.py` (outputs in `results/`: schema table, completeness evidence, benchmark tables)

## Data download record
- Method: public FDIC BankFind API (no key), script `fdic_download_starter.py`, range 1984-03-31 to 2026-06-30
- Downloaded on: 2026-11-07
- Quarters: 170 (1984Q1 to 2026Q2)
- Panel rows: 1,682,616
- Distinct banks: 24,064
- Failures table: 4,117 rows
- Institutions table: 27,835 rows
- Completeness: rows equal the API total in every quarter, no duplicate (CERT, REPDTE) (see `results/part1_completeness.txt`)

## Sources
- `fdic_download_starter.py`: starter script provided by the course instructor (used as given).
- FDIC BankFind API and field dictionary: https://api.fdic.gov/banks/docs/
- AI assistance: Claude (Anthropic) was used for guidance on setup, code and report structure.