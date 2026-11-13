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

Quick test (4 quarters, 2-3 minutes) instead of step 1: `python download_fdic.py --start 2023-03-31 --end 2023-12-31 --out data_raw/test_v2`

Run in this order (steps 3 and 4 need the files created by step 2):

1. Download the full data (about one hour, resumable; rerun the same command if interrupted): `python download_fdic.py`
2. CSV vs. Parquet storage benchmark (creates `data_storage/*.csv` and `*.parquet`): `python part1_storage_benchmark.py` (output: `results/part1_storage.txt`)
3. Build the final storage layout (Parquet + zstd, partitioned by year): `python build_storage.py`
4. Fair benchmark and float32 check: `python part1_extra_checks.py` (outputs in `results/`: schema table, completeness evidence, benchmark tables)

### Data download record

- Method: public FDIC BankFind API (no key), script `download_fdic.py`, range 1984-03-31 to 2026-06-30
- Downloaded on: 2026-11-09
- Fields: 51 columns = 3 keys + 30 core fields from the starter + 18 extra fields (quarterly flows, real-estate loan concentration, funding and capital-quality items)
- Quarters: 170 (1984Q1 to 2026Q2), between 4,313 and 18,133 rows per quarter
- Panel rows: 1,682,616
- Distinct banks: 24,064
- Failures table: 4,117 rows
- Institutions table: 27,835 rows
- Completeness: rows equal the API total in all 170 quarters, no duplicate (CERT, REPDTE) (see `results/part1_completeness.txt`)
- Disk usage measured after a full run: `data_raw` 575 MB, `data_storage` 1,681 MB (includes the CSV benchmark copies)

### Part 1 results

Storage schemes compared (`results/part1_storage.txt`):

| Scheme | Size (MB) | Read all (s) | Read 4 cols (s) | Read one year (s) |
|---|---|---|---|---|
| A CSV (pandas) | 615.6 | 1.390 | 1.060 | 1.262 |
| B Parquet snappy | 318.1 | 0.334 | 0.035 | 0.122 |
| C Parquet zstd | 225.2 | 0.324 | 0.030 | 0.096 |
| D zstd + schema | 200.3 | 0.318 | 0.024 | 0.082 |
| E zstd + schema + partition by year (final) | 201.8 | 0.275 | 0.036 | 0.029 |

- Final layout E is 3.05x smaller than CSV (-67%). Zstd is 29% smaller than snappy (225.2 vs 318.1 MB); the explicit schema removes another 11% (225.2 to 200.3 MB).
- A one-year query is 43.5x faster on E than on CSV (0.029 s vs 1.262 s) and 2.8x faster than on D (0.029 s vs 0.082 s), because partition pruning reads only one folder. A 4-column scan is 29x faster than CSV (0.036 s vs 1.060 s).
- Trade-off: E is 1.5 MB larger than D and slower for a 4-column scan (0.036 s vs 0.024 s). It was chosen because filtering by period is the main access pattern for a time-ordered panel, and its full read is also faster (0.275 s vs 0.318 s).
- Stored schema (`results/part1_schema_table.txt`): of 51 columns, 39 are Float64, 6 ratio columns are Float32 (NIMY, ROAQ, ROA, ROE, RBC1AAJ, RBCRWAJ), 4 are Int64 (CERT, ASSET, FREPP, P3ASSET), STNAME is Categorical and REPDTE is String.
- Float32 check (`results/part1_float32_check.txt`): maximum relative error 6e-8 on the six ratio columns (largest absolute error 0.0138, on ROE, whose maximum is 354,657.8), no null mismatches. Money columns (ASSET, DEP) stay at full precision.
- Engine comparison (`results/part1_fair_benchmark.txt`, seconds, read all / read 4 columns): CSV pandas 1.199 / 1.067, CSV Polars 17.509 / 19.929, Parquet pandas 0.639 / 0.044, Parquet Polars 0.304 / 0.031. On Parquet, Polars is 2.1x faster than pandas for a full read and 1.4x for a 4-column read. For pandas, Parquet is 1.9x faster than CSV for a full read and 24x faster for a 4-column read.
- Timings come from single runs on one laptop and vary between runs (pandas CSV full read: 1.390 s in the storage benchmark, 1.199 s here). Polars was the slowest on CSV (17.5 s), which was not investigated, so conclusions rely on the Parquet rows.
- Completeness: the per-year row counts of the stored panel sum to 1,682,616, equal to the sum of the API totals (e.g. 2026: 8,666 rows = 4,353 + 4,313).

## Part 2: Data quality and clean panel

Run after Part 1 (reads `data_storage/panel/`):

1. Automated data-quality report: `python part2_dq_report.py` (output: `results/part2_dq_report.txt`: banks per year, missing values by column and by year, outliers and impossible values, gaps in each bank's history)

### Part 2.1 data-quality findings (`results/part2_dq_report.txt`)

Panel audited: 1,682,616 rows, 24,064 banks, 49 variables (the 51 stored columns without CERT and REPDTE).

| Check | Result |
|---|---|
| Duplicate (CERT, REPDTE) | 0 |
| ASSET <= 0 | 0 rows (but minimum ASSET is 1, i.e. USD 1 thousand; 0.1th percentile is 1,597) |
| Banks per year | peak 18,502 (1985), 4,539 in 2025 (-75%); 2026 holds only 2 quarters (8,666 rows) |
| Gaps inside a bank's history | 32 of 24,064 banks, 92 missing bank-quarters in total |
| Banks with a single quarter | 204 |

Missing values:

| Problem | Evidence |
|---|---|
| Structural gap 1984-1989 | CRLNLS, DRLNLS, NTLNLS, NTLNLSQ miss 18.0-18.5% of rows each year and NAASSET, NALNLS, P9ASSET miss 17.9-19.7%; all fall to 0.0-0.4% from 1990. ELNATQ and ELNATR miss 18.0-18.5% in 1984-1986, then 0.2% |
| P3ASSET reported only from 2001 | 100% null in 1984-2000 (971,925 rows), 0% from 2001; overall 57.76% null |
| One-off gap in 1984 | BRO is 17.7% null in 1984, then 0.2% |
| Mostly complete columns | overall null share is below 5% for every other column (ROE 1.11%, BRO 1.00%) |

Impossible or suspicious values:

| Problem | Evidence |
|---|---|
| Zero used as missing for risk-based capital | RBCRWAJ has 468,358 zeros (27.8% of rows) although its null share is 0.00%. The panel has 466,757 rows up to 1990Q3, within 1,601 rows of that count, so the zeros match the period before the ratio was reported (to be confirmed by counting zeros per quarter). RBC1AAJ has 79,496 zeros (4.7%) |
| Extreme ratios | ROA ranges from -761.9 to 860.6 (0.1th to 99.9th percentile: -22.4 to 33.1, median 0.975); ROE from -354,657.8 to 301,050 (-375.9 to 125.5, median 10.34); RBC1AAJ up to 466,500; RBCRWAJ from -16,629.6 to 274,636.4; NIMY from -4,515.2 to 1,125 |
| Negative values in items that should not be negative | ORE 646 rows, SC 96, LNLSNET 50, LNLSGR 32, INTAN 13, INTINC 12, LNATRES 9, NCLNLS 3 |
| Negative but legitimate | EQ negative in 15,659 rows (0.93%); NETINC negative in 172,724 rows (10.3%) |
| Zero denominators | DEP = 0 in 12,141 rows, DEPDOM = 0 in 12,178, LIAB = 0 in 285 |
| Legitimate zeros (not missing) | BRO 1,321,289 zero rows (78.5%), INTAN 71.3%, FREPP 68.9%, ORE 39.8% |

Open items for the next Part 2 steps:
- Failures table has 4,117 rows and institutions has 27,835 rows, 3,771 more than the 24,064 banks in the panel; to be reconciled in the exit classification (Part 2.2).
- Handling of each problem above (nulls, flags, exclusions) and the clean labelled panel will be added when the corresponding scripts are done.