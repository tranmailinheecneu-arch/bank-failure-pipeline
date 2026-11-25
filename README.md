# Bank Failure Early-Warning Pipeline (BFT65B)

Individual final report, National Economics University, BFT65B Big Data Analytics in Finance. Author: Tran Mai Linh (MSV: 11230990).

Data source: [US FDIC BankFind API](https://api.fdic.gov/banks). Data files are not committed. All other sources, including AI tools, are listed in the Sources section at the end.

## Setup

Python 3.13 (Windows tested). Disk space: Part 1 uses about 2.3 GB (data_raw 575 MB + data_storage 1,681 MB, mostly the CSV benchmark copies); Parts 2 to 4 add the clean panel (193 MB) and the feature tables in `data_storage/part3/` and `data_storage/part4/`. Allow about 3 GB in total.

```
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Repository layout

- `fdic_download_starter.py`: original download script provided by the instructor (unchanged, kept for reference)
- `download_fdic.py`: my modified copy of the starter script (extended field list, see below)
- `part1_storage_benchmark.py`, `build_storage.py`, `part1_extra_checks.py`, `part1_polars_csv_check.py`: Part 1
- `part2_dq_report.py`, `part2_exits.py`, `part2_clean_panel.py`, `part2_zero_pattern.py`: Part 2
- `part3_polars_pandas.py`, `part3_check_std.py`, `part3_make_spark_input.py`, `part3_scaling.py`, `notebooks/part3_spark.ipynb`: Part 3
- `bank_common.py`, `part4_features.py`: Part 4
- `bank_models.py`, `part5_validation.py`, `part5_test.py`, `part5_export_mllib.py`, `notebooks/part5_mllib.ipynb`, `part5_dev_extras.py`: Part 5
- `check_report_numbers.py`: checks that the key numbers quoted in the report and in this README are present in `results/`
- `results/`: small result files (tables, completeness evidence)
- `data_raw/`, `data_storage/`: generated locally, not committed

## Part 1: Acquisition and storage

Quick test (4 quarters, 2-3 minutes) instead of step 1: `python download_fdic.py --start 2023-03-31 --end 2023-12-31 --out data_raw/test_v2`

Run in this order (steps 3 to 5 need the files created by step 2):

1. Download the full data (about one hour, resumable; rerun the same command if interrupted): `python download_fdic.py`
2. CSV vs. Parquet storage benchmark (creates `data_storage/*.csv` and `*.parquet`): `python part1_storage_benchmark.py` (output: `results/part1_storage.txt`)
3. Build the final storage layout (Parquet + zstd, partitioned by year): `python build_storage.py`
4. Fair benchmark and float32 check: `python part1_extra_checks.py` (outputs in `results/`: schema table, completeness evidence, benchmark tables)
5. Polars CSV schema-inference check: `python part1_polars_csv_check.py` (output: `results/part1_polars_csv_check.txt`)

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
- Disk usage measured after a full run of Part 1: `data_raw` 575 MB, `data_storage` 1,681 MB (includes the CSV benchmark copies)

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
- Polars read the CSV in 17.5 s in the first benchmark because it inferred the schema from all rows; `results/part1_polars_csv_check.txt` confirms it: the same 616 MB CSV takes 17.3 s with inference over all rows, 1.10 s over 10,000 rows and 0.648 s with an explicit schema (four columns: 16.4, 0.43 and 0.33 s). With an explicit schema Polars reads the CSV 1.9 times faster than pandas (0.648 against 1.199 s); Parquet is still 2.1 times faster for a full read (0.304 s) and 10.7 times for four columns (0.031 s).
- Each timing is the best of 3 repeats within one script run on one laptop and varies between runs (pandas CSV full read: 1.390 s in the storage benchmark, 1.199 s here).
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
| Zero used as missing for risk-based capital | RBCRWAJ is exactly 0 in every row of 1984-1989 (419,414 rows; 16,152 zeros in 1989Q4, 83 in 1990Q1), in 0.2-0.4% of rows in 1990-2019 and in 35-42% of rows from 2020 (45,865 rows, 2,621 banks). RBC1AAJ is 0 in about 18% of rows in 1984-1989 and in 68 rows a year since 2015 (17 banks). Both columns have a null share of 0.00%, so these zeros hide missing values (`results/part2_zero_pattern.txt`); handling in Part 2.3 |
| Extreme ratios | ROA ranges from -761.9 to 860.6 (0.1th to 99.9th percentile: -22.4 to 33.1, median 0.975); ROE from -354,657.8 to 301,050 (-375.9 to 125.5, median 10.34); RBC1AAJ up to 466,500; RBCRWAJ from -16,629.6 to 274,636.4; NIMY from -4,515.2 to 1,125 |
| Negative values in items that should not be negative | ORE 646 rows, SC 96, LNLSNET 50, LNLSGR 32, INTAN 13, INTINC 12, LNATRES 9, NCLNLS 3 |
| Negative but legitimate | EQ negative in 15,659 rows (0.93%); NETINC negative in 172,724 rows (10.3%) |
| Zero denominators | DEP = 0 in 12,141 rows, DEPDOM = 0 in 12,178, LIAB = 0 in 285 |
| Legitimate zeros (not missing) | BRO 1,321,289 zero rows (78.5%), INTAN 71.3%, FREPP 68.9%, ORE 39.8% |

Open items:

- Failures table has 4,117 rows and institutions has 27,835 rows. Reconciliation (Part 2.2): 4,006 institutions never appear in the panel and 235 panel banks are missing from the institutions table (27,835 âˆ’ 4,006 + 235 = 24,064).
- RBCRWAJ is unavailable in 1984-1989 (all rows) and in 35-42% of rows from 2020, so it cannot be the main capital measure of the model; this is handled in Part 4.

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

- The Failures table has 4,117 rows: 488 have no usable CERT (all dated before 1984), 344 linked banks never appear in the panel (22 dated 1984 or later), and 400 banks have only an ASSISTANCE event (393 later left the panel, 7 still report). The table does not separate open-bank assistance from assisted mergers (242 exits carry the change code "Merger - Assisted"), but 37 banks were assisted and failed later, so some kept operating. The main label therefore counts only RESTYPE = FAILURE; the sensitivity label `y_alt` adds the assistance banks and the 865 failure-like exits.
- Banks with a FAILURE event in the panel: 2,841 (2,662 clean failures, 177 that kept reporting after FAILDATE, 2 with FAILDATE before their first report).
- Failures by year of last report peak in 1989 (416 banks); other exits peak in 1994 (840 banks); 2026 covers the first half-year only (`results/fig_part2_exits_by_year.png`).
- No classified failure has its last report more than four quarters before FAILDATE, so every one of them can receive a positive four-quarter-ahead label.
- 841 of 2,662 classified failures may have a final report that was not public before failure under the 35-day publication-lag assumption (an assumption, not measured). This count is per bank and by the date of the last report (91 reports dated on or after the failure date, 750 within 35 days before it). Part 2.3 counts the same effect on the positive rows of the clean panel, after the rows dated on or after the failure quarter are dropped: 798 rows of 2,837 failing banks (28.1%).
- 865 other exits carry a receivership, bridge-bank, payoff or assisted-merger FDIC change code but are not linked to a FAILURE row (candidates for failures missing from the failures table). Decision (Part 2.3): they stay other exits with y = 0 in the main label; the sensitivity label `y_alt` treats them as failures one quarter after their last report.
- 186 banks fall in anomaly classes: 177 still reported after their FAILURE date, 2 have a FAILURE date before their first report, and 7 are listed only with an ASSISTANCE event but still report.
- Built with a lazy Polars scan that reads 3 of 51 columns in 0.2 s (`results/part2_exits_plan.txt`).

### Part 2.3 clean panel and label (`results/part2_clean_panel.txt`)

Output `data_storage/panel_clean/` (1,681,431 rows, 24,062 banks, 61 columns, 193 MB). Label: y = 1 if the bank fails in (q, q + 4 quarters]; only RESTYPE = FAILURE counts and a merger is never a failure. y is null after 2025Q2 because those labels are incomplete.

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

- Labelled rows: 1,663,902; positive rows: 10,860 (0.653%) from 2,837 failed banks. 2 of the 2,839 banks with a FAILURE event (the 2,841 of Part 2.2 minus the 2 excluded banks) have no positive row: they failed in July 2026, after the last panel quarter, so their labelled rows correctly have y = 0.
- Sensitivity label `y_alt` (also counts the open-bank assistance banks and the failure-like change-code exits): 14,181 positive rows from 3,812 banks.
- `y`, `y_alt`, `y_before_public` and `avail_date` are label or timing columns and must not be used as features.
- Verification on the written data: 8 of 8 checks passed (row count, no duplicates, null labels only after 2025Q2, independent recount of positives, no label for non-failed banks, no row after the failure quarter, no future-revealing column, no exact zero left in the capital ratios).

## Part 3: Scalable processing

Run after Part 2.3 (reads `data_storage/panel_clean/`):

1. Polars vs pandas benchmark and equivalence check: `python part3_polars_pandas.py` (a few minutes; outputs: `results/part3_polars_pandas.txt`, `results/part3_polars_plan.txt`; the feature tables go to `data_storage/part3/`, not committed)
2. Diagnosis of the only numerical difference (the script prints its report; save it to a file):
   - Windows PowerShell: `python part3_check_std.py | Out-File results\part3_std_check.txt -Encoding utf8`
   - Mac/Linux: `python part3_check_std.py > results/part3_std_check.txt`
3. Input file for the Spark notebook (12 columns): `python part3_make_spark_input.py`
4. Spark on Google Colab: run `notebooks/part3_spark.ipynb` (upload `data_storage/part3/spark_input.parquet` and `part3_polars_pandas.py`; Java 17, pyspark 4.0.1; outputs `results/part3_spark_results.txt`, `results/part3_spark_plan.txt`)
5. Scaling experiment on the laptop (in-memory and streaming engine, 1x to 6x, memory guard 5 GB): `python part3_scaling.py` (output: `results/part3_scaling_v2.txt`)

### Part 3.1 Polars vs pandas (`results/part3_polars_pandas.txt`)

The same 18 features are built in pandas, Polars eager and Polars lazy from the clean panel (1,681,431 rows): safe ratios, 4-quarter calendar lags (self-join on qidx + 4, so a gap in a bank's history never shifts a lag), a 4-quarter rolling window of quarterly ROA, and same-quarter cross-sectional median and rank. Only quarter q and earlier data (or the same quarter) are used.

Machine: 12 logical CPUs, 8.2 GB RAM, polars 2.1.0, pandas 3.1.1, Python 3.13.7. Each engine runs in its own process with one warm-up run; memory is the peak resident set size (sampled every 20 ms) minus the size after imports.

| Engine | Timed runs | Best (s) | Median (s) | Peak memory (MB) | Speedup vs pandas (best / median) |
|---|---|---|---|---|---|
| pandas | 5 | 5.46 | 11.02 | 1,265 | 1.0x |
| polars_eager | 5 | 2.01 | 2.51 | 1,784 | 2.7x / 4.4x |
| polars_lazy | 5 | 2.23 | 2.59 | 1,703 | 2.4x / 4.3x |

- Equivalence: 18 of 18 features match pandas (relative tolerance 1e-6, absolute 1e-9, and 1e-6 for the rolling standard deviation), same 1,681,431 rows, keys and null pattern; eager and lazy Polars are identical. The only difference is `roa_q_std4`: 5 of 1,652,719 comparable rows differ by more than 1e-9 (largest 1.86e-8), none by more than 1e-6; in the worst row (CERT 28535) the four ROAQ values in the window are all 0.69, so the true standard deviation is exactly 0: Polars returns 0 and pandas leaves a rounding residue.
- Every engine has one warm-up and five timed runs. Polars is 2.4 to 2.7 times faster than pandas by best time and 4.3 to 4.4 times by median; the median of pandas (11.0 s) is twice its best time (5.5 s), so pandas varies a lot between runs (not investigated). Polars uses 1.3 to 1.4 times the memory of pandas. Lazy is not faster than eager (2.23 against 2.01 s) because the eager code already reads only the 12 needed columns; the lazy engine matters when the column choice is not written by hand, and for filters (Part 4). All timings come from one laptop and vary between runs.
- Query plan (`results/part3_polars_plan.txt`): the unoptimised plan scans the panel 4 times with all 61 columns (244 column reads); the optimised plan scans it twice, reading 12 and 7 of 61 columns (19 column reads), computes the shared conditions ASSET > 0 and LNLSGR > 0 once, and caches the shared ratio-and-lag sub-plan so that the rolling window reuses it. No filter appears in the query, so predicate pushdown does not apply yet.

### Part 3.2 Spark (`results/part3_spark_results.txt`, `results/part3_spark_plan.txt`, `notebooks/part3_spark.ipynb`)

The same 18 features are rebuilt in PySpark 4.0.1 (local mode, Google Colab: 2 vCPU, 13.6 GB RAM, Polars 1.35.2, Python 3.13, adaptive query execution on) and compared with Polars on the same input file (12 columns of the clean panel, 1,681,431 rows). Spark local mode is not a cluster: it measures the engine, not the speed-up from adding machines.

- Equivalence: 18 of 18 features match Polars (same rows and keys). The first comparison gave 17 of 18 with a 1e-7 tolerance for `roa_q_std4`; the diagnosis found 15 of 1,652,719 rows above 1e-7 and none above 1e-6 (largest 6.74e-7). In the worst row (CERT 27229) the four ROAQ values of the window are all 0.0, so the exact standard deviation (NumPy, two-pass) is 0: Spark returns 0 and Polars 6.74e-7, and Spark is closer to the exact value in all 15 rows. The tolerance was then set to 1e-6 for this feature (1e-9 for the others). The rounding residue depends on engine, version and the values before the window: on the laptop run (Polars 2.1.0) it was pandas that left a residue (1.9e-8). `log_asset` differs by 3.6e-15 (different log implementations).
- Plan (formatted physical plan before execution, so adaptive execution may change partition counts at run time): 6 parquet scans reading 12, 7, 6, 2, 4 and 2 of the 12 columns (column pruning, and a pushed IsNotNull filter on five of the six scans), 12 shuffles (Exchange), 12 sorts, 4 window operators, 4 sort-merge joins, 1 broadcast hash join (the 170-row median table) and 2 object hash aggregates (exact median). The first "Plan facts" block in the results file counts every node twice (tree plus detail section); the second block, "each node counted once", is the correct one.
- Why 6 scans against 2 in Polars: the sub-plan with ratios and the 4-quarter lag is used by three branches (main result, medians, ranks); in this plan Spark recomputes it for each branch (no ReusedExchange node), while Polars caches it (CACHE node) and scans twice. The cost of Spark is mainly the 12 shuffles and sorts (hash partitioning on CERT, qidx) and JVM scheduling overhead.
- Time on the same Colab machine (best / median of three after one warm-up): Polars lazy 2.89 / 3.02 s, Spark 52.51 / 55.37 s, so Polars is 18.2 times faster. Spark's output is not collected to the driver, while Polars materialises its result, so the comparison slightly favours Spark.

### Part 3.3 When is a cluster justified? (`results/part3_spark_results.txt`, `results/part3_scaling_v2.txt`)

The panel is replicated k times (CERT + i * 10,000,000, so each copy is an independent set of banks).

| Colab (2 vCPU), one run per size | Rows | Polars (s) | Spark (s) | Spark s per million rows |
|---|---|---|---|---|
| 1x | 1,681,431 | 3.8 | 55.4 | 33.0 |
| 2x | 3,362,862 | 8.2 | 120.6 | 35.9 |
| 4x | 6,725,724 | not run | 244.3 | 36.3 |

Laptop (12 CPUs, 8.2 GB), best of 2 runs, memory guard 5 GB (`results/part3_scaling_v2.txt`):

| Engine mode | Size | Rows | Best time (s) | Peak memory (MB) | MB per million rows |
|---|---|---|---|---|---|
| in-memory | 1x | 1,681,431 | 2.14 | 1,678 | 998 |
| in-memory | 2x | 3,362,862 | 3.45 | 2,104 | 626 |
| in-memory | 4x | 6,725,724 | 11.85 | 1,990 | 296 |
| in-memory | 6x | 10,088,586 | 13.96 | 3,130 | 310 |
| streaming | 1x | 1,681,431 | 1.45 | 2,000 | 1,189 |
| streaming | 2x | 3,362,862 | 2.81 | 3,247 | 966 |
| streaming | 4x | 6,725,724 | 6.33 | 3,446 | 512 |
| streaming | 6x | 10,088,586 | 18.91 | 3,100 | 307 |

- Spark in local mode is 15 to 18 times slower than Polars and grows linearly (33 to 36 s per million rows). One run per size cannot separate the shape of the Polars curve from noise.
- A panel 6 times larger (10.1 million rows) needed 3.1 GB at its peak (38% of the 8.2 GB RAM) and the 5 GB guard was never reached, so the single-machine limit is above 6 times the panel. Extrapolating the slope between 4x and 6x (about 340 MB per extra million rows) gives about 20 million rows (12x) at 80% of the RAM; this is a rough extrapolation, not a measurement.
- The streaming engine did not lower the peak memory (3,100 against 3,130 MB at 6x) and was slower at 6x (18.9 against 14.0 s); which operations ran in the streaming engine was not checked.
- Conclusion: for a 1.7 million row panel (about 200 MB of compressed Parquet, about 2 seconds on one machine) a cluster is not justified. It would be justified when the data no longer fits in one machine's memory (here beyond at least 6 times the panel, which fitted in 3.1 GB; for instance loan-level data), when a time limit cannot be met on one machine, when the work splits by independent keys without cross-sectional shuffles, or when the data already lives in distributed storage.

## Part 4: Feature pipeline

Run after Part 2.3 and Part 3 (reads `data_storage/panel_clean/`, imports `part3_polars_pandas.py`):

1. Feature table, split definition, preprocessing Pipeline and leakage tests: `python part4_features.py` (output: `results/part4_features.txt`; the feature table `data_storage/part4/features.parquet` is not committed). Shared definitions (periods, feature list, preprocessing) are in `bank_common.py`.

### Part 4 feature table, split and leakage control (`results/part4_features.txt`)

Feature table `data_storage/part4/features.parquet` (1,681,431 rows): 24 model features in 7 groups (17 from Part 3 plus 7 same-quarter ratios) joined with the labels `y`, `y_alt`, `y_before_public` and the timestamp `avail_date` of Part 2.3. Every feature uses quarter q and earlier data only (lags and rolling windows look backwards) or the same quarter (cross-sectional median and rank). Not used: `roa_q_n4` (a count of quarters, a data-availability artefact), the year or quarter index (a model would learn the era), the regulatory capital ratios RBC1AAJ and RBCRWAJ (RBCRWAJ is zero in every row of 1984-1989 and again in 35-42% of rows from 2020; RBC1AAJ is zero in about 18% of rows in 1984-1989) and the ACTIVE flag of the institutions table (it knows the future). The single-feature AUC and the missing shares below use development rows only.

| Group | Feature | Why it matters | Null % dev | Null % 1984-89 | Dev AUC (single feature) |
|---|---|---|---|---|---|
| capital | eq_assets | Capital buffer: equity / assets; thin capital absorbs fewer losses | 0.24 | 0.19 | 0.053 |
| capital | d_eq_assets_4q | 4-quarter change of the capital ratio: erosion is a warning sign | 8.00 | 19.38 | 0.137 |
| capital | eq_assets_med_q | Median capital ratio of all banks in the same quarter (system-wide context) | 0.00 | 0.00 | 0.275 |
| capital | eq_assets_pct_q | Rank of the bank's capital ratio within the quarter (0 to 1) | 0.24 | 0.19 | 0.080 |
| asset quality | npl_loans | Non-current loans / gross loans: credit problems already visible | 1.12 | 1.02 | 0.908 |
| asset quality | reserves_loans | Loan-loss reserves / gross loans: cushion against bad loans | 1.15 | 0.61 | 0.751 |
| asset quality | d_npl_loans_4q | 4-quarter change of the non-current loan ratio | 8.81 | 20.15 | 0.656 |
| asset quality | npl_loans_med_q | Median non-current loan ratio in the quarter (system-wide stress) | 0.00 | 0.00 | 0.715 |
| asset quality | ore_assets | Foreclosed real estate / assets: assets taken over from defaulted loans | 0.29 | 0.24 | 0.885 |
| earnings | roaq | Quarterly return on assets (earnings power) | 0.28 | 0.32 | 0.101 |
| earnings | roa_q_mean4 | Average quarterly ROA over the last 4 quarters | 0.26 | 0.25 | 0.064 |
| earnings | roa_q_std4 | Volatility of quarterly ROA over the last 4 quarters | 2.26 | 5.20 | 0.910 |
| earnings | nimy | Net interest margin (core profitability) | 0.00 | 0.00 | 0.255 |
| liquidity and funding | liquid_assets | (Cash balances + securities) / assets: liquidity buffer | 0.01 | 0.00 | 0.276 |
| liquidity and funding | bro_dep | Brokered deposits / deposits: reliance on volatile funding | 1.90 | 3.52 | 0.610 |
| liquidity and funding | dep_assets | Deposits / assets: funding structure | 0.00 | 0.00 | 0.773 |
| loan book | loans_assets | Loans / assets: share of the balance sheet exposed to credit risk | 0.00 | 0.01 | 0.621 |
| loan book | lnre_assets | Real-estate loans / assets: concentration in real estate | 0.00 | 0.00 | 0.566 |
| loan book | lnci_assets | Commercial and industrial loans / assets | 0.00 | 0.00 | 0.522 |
| loan book | lncon_assets | Consumer loans / assets | 0.22 | 0.14 | 0.489 |
| growth | asset_yoy | Asset growth over 4 quarters: very fast growth can signal risk-taking | 7.78 | 19.22 | 0.197 |
| growth | loans_yoy | Loan growth over 4 quarters | 8.56 | 19.55 | 0.203 |
| growth | dep_yoy | Deposit growth over 4 quarters | 8.30 | 19.46 | 0.265 |
| size | log_asset | Size (log of assets) | 0.00 | 0.00 | 0.478 |

- Redundancy: 3 feature pairs have a Spearman correlation above 0.9 in development rows (eq_assets ~ eq_assets_pct_q; eq_assets_med_q ~ npl_loans_med_q; asset_yoy ~ dep_yoy). They are kept: the tree model is insensitive to it and the logistic model is regularised.
- Split (decided from the failure counts of Part 2.3 before any model was fitted; the test period is used once, in Part 5): development 1984Q1 to 2006Q4, gap 2007Q1 to 2007Q4 (four quarters, because a label looks four quarters ahead), test 2008Q1 to 2014Q4.

| Period | Rows | Labelled rows | Positive rows | Failing banks | Positive rows (actionable) | Positive % |
|---|---|---|---|---|---|---|
| dev | 1,193,702 | 1,193,702 | 8,678 | 2,290 | 8,140 | 0.727 |
| gap | 34,754 | 34,754 | 48 | 27 | 47 | 0.138 |
| test | 212,645 | 212,645 | 2,005 | 513 | 1,764 | 0.943 |
| after | 240,330 | 222,801 | 129 | 36 | 111 | 0.058 |

- All preprocessing lives in a scikit-learn Pipeline (`bank_common.py`, function `preprocessing`) fitted on training rows only: clipping to the 0.5% and 99.5% quantiles, median imputation and standardisation for the linear model; clipping only for the tree model (native missing-value handling). Demonstration: fitting the clip bounds on dev + test rows instead of dev rows changes them (largest relative shift of an upper bound 0.19), which is exactly the leak the Pipeline avoids.
- Rows with `y_before_public` = 1 (the report was not public before the bank failed) are flagged here; their treatment in training and evaluation is decided in Part 5.
- Leakage tests: 8 of 8 safety tests passed (features of earlier rows do not change when later rows are removed, at two cut-offs, 2000Q4 and 2010Q4; four-quarter dev/test gap; no label column among the features; no failing bank in both periods; test period ends before labels become incomplete; clip bounds equal to the development quantiles; unchanged after transforming test rows), plus one demonstration that fitting the clip bounds on dev + test rows would change them.

## Part 5: Modelling and validation

Run after Part 4 (reads `data_storage/part4/features.parquet`), in this order:

1. Validation, model selection and alarm thresholds on the development period only: `python part5_validation.py` (outputs: `results/part5_validation.txt`, `results/part5_selection.json`)
2. Commit the selection before the test period is opened: `git add results/part5_selection.json` then `git commit -m "Part 5 selection"` (the test script refuses to run if this file is not committed)
3. Single test evaluation: `python part5_test.py` (outputs: `results/part5_test.txt`, `results/part5_test_LOCK.json`). The committed lock file documents the original single run, so the script refuses to run again in this repository; to reproduce the evaluation, work in a scratch copy of the repository and move `results/part5_test_LOCK.json` away first.
4. Export of the development rows for the MLlib demo: `python part5_export_mllib.py` (development rows only, no test row)
5. Spark MLlib on Google Colab: run notebooks/part5_mllib.ipynb after uploading data_storage/part5/mllib_input.parquet (output: `results/part5_mllib_results.txt`)
6. Post-hoc robustness checks on the development period (run after the test, changes no committed choice): `python part5_dev_extras.py` (output: `results/part5_dev_extras.txt`)
7. Optional check that the numbers quoted in the report and in this README are present in `results/`: `python check_report_numbers.py` (expected: all checks passed)

### Part 5.1 validation, model choice and alarm threshold (`results/part5_validation.txt`, `results/part5_selection.json`)

Validation: four expanding-window folds inside the development period, with four quarters left out between the end of training and the start of validation (a label looks four quarters ahead; asserted in the code). Random cross-validation is not used: it would mix quarters and put the same bank on both sides. This script never reads the test rows. Metric: Average Precision (area under the precision-recall curve), because about 0.7% of development bank-quarters are positive (between 0.04% and 2.4% inside the validation folds) and accuracy would reward never raising an alarm; recall and lift at a fixed alarm capacity (1% or 5% of bank-quarters flagged) and the share of failing banks caught are reported next to it, ROC-AUC only as a secondary figure. Rows with `y_before_public` = 1 stay in training but are excluded from evaluation (nobody could have acted on them); the last column of the model table shows the pooled AP with them included.

| Fold | Train | Validation | Purged quarters | Train positive rows | Validation positive rows | Validation failing banks | Validation base rate % |
|---|---|---|---|---|---|---|---|
| fold1 | 1984Q1-1986Q4 | 1988Q1-1989Q4 | 4 | 1,758 | 3,226 | 1,076 | 2.441 |
| fold2 | 1984Q1-1988Q4 | 1990Q1-1991Q4 | 4 | 4,463 | 1,790 | 673 | 1.464 |
| fold3 | 1984Q1-1990Q4 | 1992Q1-1995Q4 | 4 | 7,200 | 473 | 191 | 0.223 |
| fold4 | 1984Q1-1994Q4 | 1996Q1-2006Q4 | 4 | 8,488 | 163 | 49 | 0.037 |

| Model | Mean fold AP | AP fold 1 | AP fold 2 | AP fold 3 | AP fold 4 | Pooled AP | Pooled AUC | Recall @1% | Recall @5% | Lift @5% | Pooled AP, all rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 npl_loans only | 0.1043 | 0.2559 | 0.1065 | 0.0415 | 0.0133 | 0.136 | 0.9182 | 0.33 | 0.677 | 13.5 | 0.1441 |
| B1 logistic, 3 features | 0.2184 | 0.4505 | 0.2249 | 0.1108 | 0.0875 | 0.2868 | 0.9789 | 0.536 | 0.941 | 18.8 | 0.3048 |
| M1 logistic C=0.01 | 0.2091 | 0.446 | 0.2145 | 0.112 | 0.064 | 0.2738 | 0.971 | 0.523 | 0.866 | 17.3 | 0.2905 |
| M1 logistic C=0.1 | 0.2087 | 0.4463 | 0.2136 | 0.1115 | 0.0633 | 0.2733 | 0.971 | 0.522 | 0.866 | 17.3 | 0.2901 |
| M1 logistic C=1 | 0.2086 | 0.4463 | 0.2135 | 0.1114 | 0.0632 | 0.2733 | 0.971 | 0.522 | 0.866 | 17.3 | 0.2901 |
| M2 gbm 150 iter, lr 0.10, 15 leaves | 0.2907 | 0.4316 | 0.3551 | 0.2774 | 0.0986 | 0.3018 | 0.9761 | 0.507 | 0.878 | 17.6 | 0.3239 |
| M2 gbm 300 iter, lr 0.05, 15 leaves | 0.299 | 0.4353 | 0.3306 | 0.3245 | 0.1056 | 0.2943 | 0.9773 | 0.511 | 0.893 | 17.9 | 0.3165 |
| M2 gbm 300 iter, lr 0.05, 31 leaves | 0.2992 | 0.4328 | 0.3307 | 0.3295 | 0.1036 | 0.2942 | 0.9748 | 0.506 | 0.884 | 17.7 | 0.3163 |

- Selection rule, fixed before any result was seen: among the models with 24 features (M1 logistic, M2 gradient boosting) the highest mean fold AP. Selected: M2 gbm 300 iter, lr 0.05, 31 leaves. Baselines (fixed): B0 ranks by the non-current loan ratio alone, B1 is a logistic model on three features (eq_assets, npl_loans, roaq). All preprocessing is inside the Pipeline fitted on the training rows of each fold. M2 wins by mean fold AP (0.2992 against 0.2184 for B1), driven by folds 2 and 3, while the pooled AP is almost equal (0.2942 against 0.2868) and B1 saves more cost at its cost-optimal threshold (86% against 69% at kappa 0.02).
- Cost of a missed failure, from the failures table (failures of 1984-2007, 2,291 with COST and assets at failure): asset-weighted loss rate 0.1959 (sum of COST over sum of assets at failure), times the bank's assets, spread over the up to four positive quarters of a failure: 15.3 million USD per positive bank-quarter on average.
- Cost of a false alarm: not observable in FDIC data. Assumed to be kappa times the average missed-failure cost, main kappa = 0.02 (a false alarm costs one fiftieth of a missed failure), with a sensitivity analysis from 0.005 to 0.1. This is an assumption, not an estimate; Part 5.4 restates it as a supervisory capacity.

The alarm threshold minimises the total cost on the out-of-fold predictions of the development period (no test data):

| kappa | Model | Alarm rate % | Recall (bank-quarters) | Recall (failing banks) | Cost saving vs no alarm | Cost (bn USD) | No-alarm cost (bn USD) |
|---|---|---|---|---|---|---|---|
| 0.005 | B1 logistic, 3 features | 5.96 | 0.953 | 0.981 | 0.944 | 6.26 | 111.91 |
| 0.005 | M2 gbm 300 iter, lr 0.05, 31 leaves | 15.27 | 0.968 | 0.982 | 0.853 | 16.44 | 111.91 |
| 0.01 | B1 logistic, 3 features | 4.97 | 0.941 | 0.978 | 0.913 | 9.7 | 111.91 |
| 0.01 | M2 gbm 300 iter, lr 0.05, 31 leaves | 9.09 | 0.944 | 0.971 | 0.794 | 23.08 | 111.91 |
| 0.02 | B1 logistic, 3 features | 4.58 | 0.933 | 0.975 | 0.86 | 15.61 | 111.91 |
| 0.02 | M2 gbm 300 iter, lr 0.05, 31 leaves | 6.33 | 0.914 | 0.957 | 0.694 | 34.29 | 111.91 |
| 0.05 | B1 logistic, 3 features | 3.14 | 0.882 | 0.956 | 0.742 | 28.9 | 111.91 |
| 0.05 | M2 gbm 300 iter, lr 0.05, 31 leaves | 3.37 | 0.826 | 0.903 | 0.545 | 50.94 | 111.91 |
| 0.1 | B1 logistic, 3 features | 2.8 | 0.859 | 0.947 | 0.587 | 46.23 | 111.91 |
| 0.1 | M2 gbm 300 iter, lr 0.05, 31 leaves | 1.74 | 0.689 | 0.824 | 0.398 | 67.41 | 111.91 |

Second alarm rule (capacity rule), added after the development results above had been seen and before the test period was opened: flag the top r% of banks of every quarter, with r the cost-optimal alarm rate of the table at kappa 0.02 (B1 4.58%, M2 gbm 300 iter, lr 0.05, 31 leaves 6.33%). It does not depend on the score scale, which differs between models fitted on different training windows. Both rules are evaluated once on the test period (Part 5.2).

### Part 5.2 test evaluation, used once (`results/part5_test.txt`)

The test period (2008Q1 to 2014Q4) was opened exactly once, after `results/part5_selection.json` was committed; `results/part5_test_LOCK.json` records the time, the hash of that file and the git commit, and the script refuses to run a second time or when the selection file is not committed. All models are refitted on all development rows; thresholds and alarm rates come from the development period. Evaluation uses the actionable rows (reports that were public before the failure). To reproduce the evaluation in a scratch copy of the repository, move `results/part5_test_LOCK.json` away first (the committed lock documents the original single run).

| Model | AP | ROC-AUC | Recall @1% | Recall @5% | Lift @5% | AP, all rows | AP vs y_alt |
|---|---|---|---|---|---|---|---|
| B0 npl_loans only | 0.1748 | 0.9376 | 0.293 | 0.72 | 14.4 | 0.2063 | 0.2022 |
| B1 logistic, 3 features | 0.4946 | 0.976 | 0.573 | 0.904 | 18.1 | 0.5505 | 0.5426 |
| M2 gbm 300 iter, lr 0.05, 31 leaves | 0.525 | 0.9843 | 0.608 | 0.923 | 18.5 | 0.5749 | 0.5625 |

Rule 1 (score threshold from the development period):

| kappa | Model | Alarm rate % | Recall (bank-quarters) | Recall (failing banks) | Cost saving vs no alarm | Cost (bn USD) | No-alarm cost (bn USD) |
|---|---|---|---|---|---|---|---|
| 0.005 | B1 logistic, 3 features | 7.92 | 0.94 | 0.975 | 0.782 | 21.44 | 98.16 |
| 0.005 | M2 gbm 300 iter, lr 0.05, 31 leaves | 17.73 | 0.984 | 0.996 | 0.966 | 3.32 | 98.16 |
| 0.01 | B1 logistic, 3 features | 7.04 | 0.929 | 0.973 | 0.764 | 23.18 | 98.16 |
| 0.01 | M2 gbm 300 iter, lr 0.05, 31 leaves | 10.54 | 0.968 | 0.99 | 0.932 | 6.63 | 98.16 |
| 0.02 | B1 logistic, 3 features | 6.68 | 0.926 | 0.973 | 0.744 | 25.11 | 98.16 |
| 0.02 | M2 gbm 300 iter, lr 0.05, 31 leaves | 7.29 | 0.948 | 0.981 | 0.585 | 40.75 | 98.16 |
| 0.05 | B1 logistic, 3 features | 5.19 | 0.907 | 0.969 | 0.539 | 45.26 | 98.16 |
| 0.05 | M2 gbm 300 iter, lr 0.05, 31 leaves | 3.81 | 0.9 | 0.975 | 0.508 | 48.25 | 98.16 |
| 0.1 | B1 logistic, 3 features | 4.79 | 0.902 | 0.969 | 0.477 | 51.35 | 98.16 |
| 0.1 | M2 gbm 300 iter, lr 0.05, 31 leaves | 1.69 | 0.76 | 0.928 | 0.37 | 61.88 | 98.16 |

Rule 2 (capacity rule: top r% of banks flagged in each quarter, r from the development optimum):

| kappa | Model | Alarm rate % | Recall (bank-quarters) | Recall (failing banks) | Cost saving vs no alarm | Cost (bn USD) |
|---|---|---|---|---|---|---|
| 0.005 | B1 logistic, 3 features | 5.85 | 0.914 | 0.973 | 0.92 | 7.87 |
| 0.005 | M2 gbm 300 iter, lr 0.05, 31 leaves | 15.17 | 0.98 | 0.994 | 0.957 | 4.2 |
| 0.01 | B1 logistic, 3 features | 4.86 | 0.891 | 0.967 | 0.898 | 10.0 |
| 0.01 | M2 gbm 300 iter, lr 0.05, 31 leaves | 8.98 | 0.95 | 0.984 | 0.913 | 8.53 |
| 0.02 | B1 logistic, 3 features | 4.47 | 0.875 | 0.963 | 0.875 | 12.29 |
| 0.02 | M2 gbm 300 iter, lr 0.05, 31 leaves | 6.22 | 0.926 | 0.979 | 0.87 | 12.74 |
| 0.05 | B1 logistic, 3 features | 3.03 | 0.807 | 0.947 | 0.8 | 19.65 |
| 0.05 | M2 gbm 300 iter, lr 0.05, 31 leaves | 3.26 | 0.842 | 0.957 | 0.466 | 52.47 |
| 0.1 | B1 logistic, 3 features | 2.69 | 0.777 | 0.942 | 0.588 | 40.48 |
| 0.1 | M2 gbm 300 iter, lr 0.05, 31 leaves | 1.64 | 0.679 | 0.904 | 0.356 | 63.23 |

Average Precision of M2 gbm 300 iter, lr 0.05, 31 leaves by test year:

| Year | Rows | Positive rows | AP | Base rate % |
|---|---|---|---|---|
| 2008 | 33,991 | 343 | 0.449 | 1.009 |
| 2009 | 32,846 | 600 | 0.6461 | 1.827 |
| 2010 | 31,474 | 386 | 0.562 | 1.226 |
| 2011 | 30,184 | 223 | 0.4998 | 0.739 |
| 2012 | 29,133 | 105 | 0.5066 | 0.36 |
| 2013 | 27,978 | 68 | 0.4352 | 0.243 |
| 2014 | 26,798 | 39 | 0.3389 | 0.146 |

- Reading of the test results (actionable rows, chance level of AP 0.83%): the selected gradient-boosting model reaches AP 0.525 (63 times chance), ROC-AUC 0.984 and recall 0.923 when 5% of bank-quarters are flagged; the three-feature logistic baseline B1 reaches 0.495, 0.976 and 0.904, and B0 (non-current loans only) 0.175, 0.938 and 0.720. The gain of the larger model over the simple one is small.
- Costs are less stable than rankings: at kappa 0.02 the score-threshold rule saves 58.5% of the no-alarm cost with M2 and 74.4% with B1, the capacity rule 87.0% and 87.5%; at kappa 0.05 the capacity rule saves 46.6% with M2 and 80.0% with B1 although M2 flags more failing bank-quarters. Costs are weighted by assets, so a few very large banks can dominate them (not tested). Ranking metrics are therefore the main result and costs are secondary.
- Removing the 241 non-actionable positive rows lowers the AP from 0.575 to 0.525: the quarter just before a failure is the easiest to flag. AP falls from 0.646 (2009) to 0.339 (2014) while the share of positive rows falls from 1.83% to 0.15%, so the gain over chance rises from about 35 to about 232 times. AP on the test period is not comparable with the validation folds (different base rates).
- A first run of `part5_test.py` stopped on a missing key of `results/part5_selection.json` before any model was fitted or any test metric was computed (the lock had been written earlier in that version of the script). I removed that lock, regenerated the selection file, moved the lock after the input checks, committed the selection file and only then ran the single evaluation reported here.

### Part 5.3 Spark MLlib demo (`results/part5_mllib_results.txt`, `notebooks/part5_mllib.ipynb`)

A short MLlib demo on fold 4 of the purged validation (Google Colab, 2 vCPU, pyspark 4.0.1; input exported from the development rows only by `part5_export_mllib.py`, so no test row is involved). Pipeline: Imputer (median), VectorAssembler, StandardScaler (logistic model only), LogisticRegression or GBTClassifier with a balanced class weight; the clip bounds (0.5% and 99.5% quantiles, approxQuantile) are learned on the training rows only. Training: 1984Q1 to 1994Q4 (704,362 rows, 8,488 positive); validation: 1996Q1 to 2006Q4 without the non-actionable rows (440,235 rows, 163 positive, base rate 0.037%). These counts equal those of fold 4 in Part 5.1, so the demo reproduces the same split.

| Model | Area under PR | Area under ROC | Fit time (s) | Fold-4 AP of the same family in scikit-learn |
|---|---|---|---|---|
| MLlib logistic regression (24 features) | 0.0400 | 0.8810 | 90 | 0.0640 (M1 logistic, C = 0.01) |
| MLlib gradient-boosted trees (30 iterations, depth 4) | 0.0826 | 0.8963 | 116 | 0.0986 to 0.1056 (M2 gradient boosting, three settings) |

- The ranking of the two families is the same as in scikit-learn (trees above logistic regression). MLlib is lower for each family: its GBT is much smaller (30 iterations, depth 4, against 150 to 300 iterations), the regularisation differs and Imputer and approxQuantile are approximate. The demo shows the tool, it is not a replacement for the scikit-learn models reported in Part 5.1 and 5.2, and it played no role in the model selection.
- Spark's areaUnderPR uses trapezoidal interpolation, so it is close to but not identical with scikit-learn's Average Precision. The ROC areas are not comparable with the pooled AUC column of Part 5.1, which pools the four folds (the note at the end of the results file refers to that column; only the AP of fold 4 is comparable).
- With 163 positive rows from 49 banks the fold-4 figures are noisy; a difference of 0.02 in AP should not be over-interpreted.

### Part 5.4 development-period robustness analyses (`results/part5_dev_extras.txt`)

Run after the test evaluation; it reads only development rows, refits the committed models and changes no committed choice. It stops unless the out-of-fold AP of B1 and of the selected model reproduces the Part 5.1 model table to four decimals (it does, difference 0.0000).

- **Implied kappa (asset-weighted).** A capacity of 5% of bank-quarters is cost-optimal at kappa 0.0106 (B1) and 0.0328 (selected); the main kappa of 0.02 lies between them (alarm rates 4.58% and 6.33%). Capacity 2%: 0.1663 and 0.0964; capacity 10%: 0.0007 and 0.0066. This replaces one assumption by another that is easier to state; it does not make the cost observable.
- **Calibration.** B1 (balanced class weight) outputs scores: mean predicted rate 0.199, 0.152, 0.080, 0.053 against observed 0.0244, 0.0146, 0.0022, 0.0004; isotonic recalibration fitted on earlier folds lowers its Brier score from 0.0491 to 0.0130, from 0.0182 to 0.0024 and from 0.00678 to 0.00042 (folds 2 to 4). For the selected model (no class weight) the same recalibration does not help (Brier 0.01351 to 0.01408, 0.00263 to 0.00334, 0.00035 to 0.00036): the base rate falls from 2.4% to 0.04% between folds, so a mapping learned in one regime does not transfer.
- **Is the larger model clearly better?** Bank-level bootstrap (500 resamples): selected minus B1 pooled AP +0.0092, 95% interval [-0.0070, +0.0248] (contains 0); mean fold AP +0.0793, interval [+0.0594, +0.1005] (excludes 0; the selection rule used this metric). The bootstrap's own weighted AP differs from scikit-learn's by 0.0017 on B1.
- **Label noise.** Against `y_alt` (7,856 positives instead of 5,652) the pooled AP is B1 0.589 and selected 0.522 (against `y`: 0.287 and 0.294), so the pooled order of the two changes; by mean fold AP against `y_alt`, B1 (0.465) is ahead of the selected model (0.430). On the test period the order is unchanged (0.543 against 0.563).

| Fold | Positives y | Positives y_alt | B0 AP(y) / AP(y_alt) | B1 AP(y) / AP(y_alt) | Selected AP(y) / AP(y_alt) |
|---|---|---|---|---|---|
| 1 | 3226 | 3911 | 0.256 / 0.364 | 0.451 / 0.645 | 0.433 / 0.509 |
| 2 | 1790 | 2933 | 0.106 / 0.296 | 0.225 / 0.668 | 0.331 / 0.737 |
| 3 | 473 | 849 | 0.041 / 0.171 | 0.111 / 0.460 | 0.330 / 0.369 |
| 4 | 163 | 163 | 0.013 / 0.013 | 0.088 / 0.088 | 0.104 / 0.104 |
| pooled | 5652 | 7856 | 0.136 / 0.257 | 0.287 / 0.589 | 0.294 / 0.522 |

## Sources

- `fdic_download_starter.py`: starter script provided by the course instructor (original, unchanged).
- `download_fdic.py`: copy of the starter script, modified by me only in the field list `FIN_FIELDS` and the default output folder.
- FDIC BankFind API and field dictionary: https://api.fdic.gov/banks/docs/ (failures, institutions and history endpoints).
- Course materials: Lecture 3 "Financial Time Series" (07_L03_TimeSeries_ML) for the point-in-time rule and the available-timestamp idea; Lessons 12-14 (visualization) for the figure design.
- Apache Spark / PySpark installation documentation (Java 17 or later is required): https://spark.apache.org/docs/latest/api/python/getting_started/install.html
- AI tools: Claude (Anthropic) and ChatGPT (OpenAI) were used for guidance on environment setup, debugging, code drafting (storage benchmark, data-quality report, exit classification, clean panel and label construction, zero-pattern evidence, Polars, pandas and Spark benchmarks, Spark notebook and scaling experiment, feature pipeline, validation, test and MLlib scripts, development-period robustness analyses, number check), choosing extra FDIC fields and structuring the README and report. I ran all code myself, checked the outputs against the generated result files, and take responsibility for the final content.

