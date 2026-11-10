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
- `fdic_download_starter.py`: original download script provided by the instructor (unchanged, kept for reference)
- `download_fdic.py`: my modified copy of the starter script (extended field list, see below)
- `part1_storage_benchmark.py`, `build_storage.py`, `part1_extra_checks.py`: Part 1
- `results/`: small result files (tables, completeness evidence)
- `data_raw/`, `data_storage/`: generated locally, not committed

## Part 1: Acquisition and storage
Quick test (4 quarters, 2-3 minutes) instead of step 1:
`python download_fdic.py --start 2023-03-31 --end 2023-12-31 --out data_raw/test_v2`

Run in this order (steps 3 and 4 need the files created by step 2):

1. Download the full data (about one hour, resumable; rerun the same command if interrupted): `python download_fdic.py`
2. CSV vs. Parquet storage benchmark (creates `data_storage/*.csv` and `*.parquet`): `python part1_storage_benchmark.py` (output: `results/part1_storage.txt`)
3. Build the final storage layout (Parquet + zstd, partitioned by year): `python build_storage.py`
4. Fair benchmark and float32 check: `python part1_extra_checks.py` (outputs in `results/`: schema table, completeness evidence, benchmark tables)

## Data download record
- Method: public FDIC BankFind API (no key), script `download_fdic.py`, range 1984-03-31 to 2026-06-30
- Downloaded on: 2026-11-09
- Fields: 51 columns = 3 keys + 30 core fields from the starter + 18 extra fields (quarterly flows, real-estate loan concentration, funding and capital-quality items)
- Quarters: 170 (1984Q1 to 2026Q2)
- Panel rows: 1,682,616
- Distinct banks: 24,064
- Failures table: 4,117 rows
- Institutions table: 27,835 rows
- Completeness: rows equal the API total in every quarter, no duplicate (CERT, REPDTE) (see `results/part1_completeness.txt`)

## Sources
- `fdic_download_starter.py`: starter script provided by the course instructor (original, unchanged).
- `download_fdic.py`: copy of the starter script, modified by me only in the field list `FIN_FIELDS` and the default output folder.
- FDIC BankFind API and field dictionary: https://api.fdic.gov/banks/docs/
- AI assistance: Claude (Anthropic) was used for guidance on setup, code and report structure.
### Part 1 results
| Scheme | Size (MB) | Read all (s) | Read 4 cols (s) | Read one year (s) |
|---|---|---|---|---|
| A CSV (pandas) | 615.6 | 1.390 | 1.060 | 1.262 |
| B Parquet snappy | 318.1 | 0.334 | 0.035 | 0.122 |
| C Parquet zstd | 225.2 | 0.324 | 0.030 | 0.096 |
| D zstd + schema | 200.3 | 0.318 | 0.024 | 0.082 |
| E zstd + schema + partition by year (final) | 201.8 | 0.275 | 0.036 | 0.029 |

- Final layout E is 3.1x smaller than CSV; a one-year query is 43x faster (0.029 s vs 1.262 s) because partition pruning reads only one folder.
- Trade-off: E is slightly slower than D for a 4-column scan (0.036 s vs 0.024 s), and adds 1.5 MB. Accepted because filtering by period is the dominant access pattern in a time-ordered panel.
- Float32 for ratio columns (ROA, ROE, NIMY, RBC1AAJ, RBCRWAJ, ROAQ): max relative error 6e-8, no null mismatches (results/part1_float32_check.txt). Money columns (ASSET, DEP) stay at full precision.
- Completeness: 170 quarters, 1,682,616 rows, every quarter equals the API total; per-year row counts in the stored panel sum to the same total (results/part1_completeness.txt).
### Part 2.1 data-quality findings (results/part2_dq_report.txt)
- Panel: 1,682,616 rows, 24,064 banks, 49 variables; no duplicate (CERT, REPDTE); no ASSET <= 0.
- Banks per year fall from 18,330 (1984) to 4,539 (2025).
- Structural missingness: ~18% of rows miss charge-off and nonaccrual fields in 1984-1989; P3ASSET exists only from 2001.
- RBCRWAJ holds 468,358 zeros (27.8% of rows) that are placeholders before risk-based capital reporting, not real values.
- Extreme ratios (ROA to 861%, ROE to 354,658%) come from tiny denominators; legitimate negatives (negative equity, net loss) are kept.
- 32 banks have gaps inside their history (92 missing bank-quarters); 204 banks have a single quarter.