# Bank Failure Early-Warning Pipeline (BFT65B)

Individual final report, National Economics University, BFT65B Big Data Analytics in Finance.
Author: Tran Mai Linh (MSV: 11230990).

Data source: [US FDIC BankFind API](https://api.fdic.gov/banks). Data files are not committed. All other sources, including AI tools, are listed in the Sources section at the end.

## Setup
Python 3.13 (Windows tested). Disk space needed: about 2.3 GB (`data_raw` 575 MB + `data_storage` 1,681 MB, mostly the CSV benchmark copies).
```
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Repository layout
- `fdic_download_starter.py`: original download script provided by the instructor (unchanged, kept for reference)
- `download_fdic.py`: my modified copy of the starter script (extended field list, see below)
- `part1_storage_benchmark.py`, `build_storage.py`, `part1_extra_checks.py`: Part 1
- `part2_dq_report.py`, `part2_exits.py`, `part2_clean_panel.py`, `part2_zero_pattern.py`: Part 2
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
- Each timing is the best of 3 repeats within one script run on one laptop and varies between runs (pandas CSV full read: 1.390 s in the storage benchmark, 1.199 s here). Polars was the slowest on CSV (17.5 s); a likely cause is the full-file schema inference (`infer_schema_length=None`) used in the benchmark, which was not tested separately, so conclusions rely on the Parquet rows.
- Completeness: the per-year row counts of the stored panel sum to 1,682,616, equal to the sum of the API totals (e.g. 2026: 8,666 rows = 4,353 + 4,313).

## Part 2: Data quality and clean panel

Run after Part 1 (reads `data_storage/panel/` and the failures and institutions tables in `data_raw/fdic_v2/`):

1. Automated data-quality report: `python part2_dq_report.py` (output: `results/part2_dq_report.txt`: banks per year, missing values by column and by year, outliers and impossible values, gaps in each bank's history)
2. Exit classification (failure vs other exits): `python part2_exits.py` (outputs: `results/part2_exits.txt`, `results/fig_part2_exits_by_year.png`, `results/part2_exits_plan.txt`, `data_storage/exits/bank_exits.parquet`). The first run needs internet to fetch FDIC change-code descriptions; they are cached in `results/part2_changecodes.json` (committed), so reruns work offline.
3. Clean point-in-time panel and label: `python part2_clean_panel.py` (outputs: `results/part2_clean_panel.txt`, `data_storage/panel_clean/`)
4. Evidence for the zero rule in the capital ratios: `python part2_zero_pattern.py` (output: `results/part2_zero_pattern.txt`)

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
- Failures table has 4,117 rows and institutions has 27,835 rows. Reconciliation (Part 2.2): 4,006 institutions never appear in the panel and 235 panel banks are missing from the institutions table (27,835 − 4,006 + 235 = 24,064).

### Part 2.2 exit classification (`results/part2_exits.txt`)

Every one of the 24,064 banks in the panel is assigned one exit class (last quarter of the panel: 2026Q2). Only the failures table decides what is a failure; a merger is never a failure.

| Exit class | Banks | Share |
|---|---|---|
| other_exit | 16,910 | 70.27% |
| survivor | 4,306 | 17.89% |
| failure | 2,662 | 11.06% |
| reported_after_failure_date | 177 | 0.74% |
| survivor_listed_in_failures | 7 | 0.03% |
| failure_date_before_first_report | 2 | 0.01% |

- Failures table: 4,117 rows; 488 have no usable CERT (0 dated 1984 or later) and 344 linked banks never appear in the panel (22 of them dated 1984 or later; the rest failed before the panel starts).
- 400 banks have only an open-bank ASSISTANCE event (RESTYPE = ASSISTANCE): they kept operating, so they are not labelled as failures; only RESTYPE = FAILURE counts.
- Banks with a FAILURE event in the panel: 2,841 (2,662 clean failures, 177 that kept reporting after FAILDATE, 2 with FAILDATE before their first report).
- Failures by year of last report peak in 1989 (416 banks); other exits peak in 1994 (840 banks); 2026 covers the first half-year only (`results/fig_part2_exits_by_year.png`).
- No classified failure has its last report more than four quarters before FAILDATE, so every one of them can receive a positive four-quarter-ahead label.
- 841 of 2,662 classified failures may have a final report that was not public before failure under the 35-day publication-lag assumption (an assumption, not measured).
- 865 other exits carry a receivership, bridge-bank, payoff or assisted-merger FDIC change code but are not linked to a FAILURE row (candidates for failures missing from the failures table). Decision (Part 2.3): they stay other exits with y = 0 in the main label; the sensitivity label `y_alt` treats them as failures one quarter after their last report.
- 186 banks fall in anomaly classes: 177 still reported after their FAILURE date, 2 have a FAILURE date before their first report, and 7 are listed only with an ASSISTANCE event but still report.
- Built with a lazy Polars scan that reads 3 of 51 columns in 0.2 s (`results/part2_exits_plan.txt`).

### Part 2.3 clean panel and label (`results/part2_clean_panel.txt`)

Output `data_storage/panel_clean/` (1,681,431 rows, 24,062 banks, 61 columns, 193 MB). Label: `y = 1` if the bank fails in (q, q + 4 quarters]; only RESTYPE = FAILURE counts and a merger is never a failure. `y` is null after 2025Q2 because those labels are incomplete.

| Problem | Handling | Effect |
|---|---|---|
| Rows of failed banks dated in or after the failure quarter | dropped (they would not exist in real time) | 1,069 rows, 268 banks |
| FAILDATE before the first report | bank excluded | 2 banks, 116 rows |
| Exact zero used as "not reported" in RBCRWAJ and RBC1AAJ (whole eras and a fixed set of banks, see `results/part2_zero_pattern.txt`) | set to null in every quarter and flagged (`flag_rbc_zero`) | RBCRWAJ 467,509 values (20,909 banks), RBC1AAJ 78,650 values (3,878 banks); 467,522 rows flagged |
| Negative values in items that cannot be negative | set to null and flagged (`flag_neg_fixed`) | 814 rows (ORE 641, SC 87, LNLSNET 50, LNLSGR 32, INTAN 13, INTINC 12, LNATRES 9, NCLNLS 3) |
| Zero denominator (DEP = 0) | kept, flagged (`flag_dep_zero`); safe division in Part 4 | 12,141 rows |
| Missing quarter inside a bank's history | flagged (`gap_before`); lags use the calendar quarter index `qidx`, not the row position | 33 rows, 32 banks |
| Structural nulls (1984-1989 items, P3ASSET before 2001) and extreme ratios | left as they are: imputation and clipping bounds are learned on training data only, inside the Part 4 pipeline | none |
| Report not yet public when the bank failed | `avail_date` = quarter end + 35 days (assumed lag); `y_before_public` flags the affected positives | 798 of 10,860 positive rows (one per bank) |

- Labelled rows: 1,663,902; positive rows: 10,860 (0.653%) from 2,837 failed banks. 2 of the 2,839 banks with a FAILURE event have no positive row: they failed in July 2026, after the last panel quarter, so their labelled rows correctly have y = 0.
- Sensitivity label `y_alt` (also counts the open-bank assistance banks and the failure-like change-code exits): 14,181 positive rows from 3,812 banks.
- `y`, `y_alt`, `y_before_public` and `avail_date` are label or timing columns and must not be used as features.
- Verification on the written data: 8 of 8 checks passed (row count, no duplicates, null labels only after 2025Q2, independent recount of positives, no label for non-failed banks, no row after the failure quarter, no future-revealing column, no exact zero left in the capital ratios).

## Sources
- `fdic_download_starter.py`: starter script provided by the course instructor (original, unchanged).
- `download_fdic.py`: copy of the starter script, modified by me only in the field list `FIN_FIELDS` and the default output folder.
- FDIC BankFind API and field dictionary: https://api.fdic.gov/banks/docs/ (failures, institutions and history endpoints).
- Course materials: Lecture 3 "Financial Time Series" (07_L03_TimeSeries_ML) for the point-in-time rule and the available-timestamp idea; Lessons 12-14 (visualization) for the figure design.
- AI tools: Claude (Anthropic) and ChatGPT (OpenAI) were used for guidance on environment setup, debugging, code drafting (storage benchmark, data-quality report, exit classification, clean panel and label construction, zero-pattern evidence), choosing extra FDIC fields and structuring the README and report. I ran all code myself, checked the outputs against the generated result files, and take responsibility for the final content.