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
- `part3_polars_pandas.py`, `part3_check_std.py`, `part3_make_spark_input.py`, `part3_scaling.py`, `notebooks/part3_spark.ipynb`: Part 3
- `bank_common.py`, `part4_features.py`: Part 4
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
| Zero used as missing for risk-based capital | RBCRWAJ is exactly 0 in every row of 1984-1989 (419,414 rows of the stored panel), falls to 0.2-0.4% of rows in 1990-2019 (67-225 rows per year) and is 0 again in about 35-42% of rows from 2020 (45,865 rows, 2,621 banks). RBC1AAJ is 0 in about 18% of rows in 1984-1989 (76,038 rows) and in 0.2-0.4% afterwards; since 2015 it is 68 rows per year, a fixed set of 17 banks (782 rows). Both columns have a null share of 0.00%, so these zeros hide missing values (`results/part2_zero_pattern.txt`); handling in Part 2.3 |
| Extreme ratios | ROA ranges from -761.9 to 860.6 (0.1th to 99.9th percentile: -22.4 to 33.1, median 0.975); ROE from -354,657.8 to 301,050 (-375.9 to 125.5, median 10.34); RBC1AAJ up to 466,500; RBCRWAJ from -16,629.6 to 274,636.4; NIMY from -4,515.2 to 1,125 |
| Negative values in items that should not be negative | ORE 646 rows, SC 96, LNLSNET 50, LNLSGR 32, INTAN 13, INTINC 12, LNATRES 9, NCLNLS 3 |
| Negative but legitimate | EQ negative in 15,659 rows (0.93%); NETINC negative in 172,724 rows (10.3%) |
| Zero denominators | DEP = 0 in 12,141 rows, DEPDOM = 0 in 12,178, LIAB = 0 in 285 |
| Legitimate zeros (not missing) | BRO 1,321,289 zero rows (78.5%), INTAN 71.3%, FREPP 68.9%, ORE 39.8% |

Open items:
- Failures table has 4,117 rows and institutions has 27,835 rows. Reconciliation (Part 2.2): 4,006 institutions never appear in the panel and 235 panel banks are missing from the institutions table (27,835 − 4,006 + 235 = 24,064).
- RBCRWAJ is unavailable in 1984-1989 (all rows) and in about 37% of rows from 2020, so it cannot be the main capital measure of the model; this is handled in Part 4.

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

- Labelled rows: 1,663,902; positive rows: 10,860 (0.653%) from 2,837 failed banks. 2 of the 2,839 banks with a FAILURE event (the 2,841 of Part 2.2 minus the 2 excluded banks) have no positive row: they failed in July 2026, after the last panel quarter, so their labelled rows correctly have y = 0.
- Sensitivity label `y_alt` (also counts the open-bank assistance banks and the failure-like change-code exits): 14,181 positive rows from 3,812 banks.
- `y`, `y_alt`, `y_before_public` and `avail_date` are label or timing columns and must not be used as features.
- Verification on the written data: 8 of 8 checks passed (row count, no duplicates, null labels only after 2025Q2, independent recount of positives, no label for non-failed banks, no row after the failure quarter, no future-revealing column, no exact zero left in the capital ratios).


## Part 3: Scalable processing

Run after Part 2.3 (reads `data_storage/panel_clean/`):

1. Polars vs pandas benchmark and equivalence check: `python part3_polars_pandas.py` (a few minutes; outputs: `results/part3_polars_pandas.txt`, `results/part3_polars_plan.txt`; the feature tables go to `data_storage/part3/`, not committed)
2. Diagnosis of the only numerical difference: `python part3_check_std.py | Out-File results\part3_std_check.txt -Encoding utf8`
3. Input file for the Spark notebook (12 columns): `python part3_make_spark_input.py`
4. Spark on Google Colab: run `notebooks/part3_spark.ipynb` (upload `data_storage/part3/spark_input.parquet` and `part3_polars_pandas.py`; Java 17, pyspark 4.0.1; outputs `results/part3_spark_results.txt`, `results/part3_spark_plan.txt`)
5. Scaling experiment on the laptop: `python part3_scaling.py` (output: `results/part3_scaling.txt`)

### Part 3.1 Polars vs pandas (`results/part3_polars_pandas.txt`)

The same 18 features are built in pandas, Polars eager and Polars lazy from the clean panel (1,681,431 rows): safe ratios, 4-quarter calendar lags (self-join on `qidx + 4`, so a gap in a bank's history never shifts a lag), a 4-quarter rolling window of quarterly ROA, and same-quarter cross-sectional median and rank. Only quarter q and earlier data (or the same quarter) are used.

Machine: 12 logical CPUs, 8.2 GB RAM, polars 2.1.0, pandas 3.1.1, Python 3.13.7. Each engine runs in its own process with one warm-up run; memory is the peak resident set size (sampled every 20 ms) minus the size after imports.

| Engine | Timed runs | Best (s) | Median (s) | Peak memory (MB) | Speedup vs pandas (best / median) |
|---|---|---|---|---|---|
| pandas | 2 | 5.73 | 6.16 | 1,253 | 1.0x |
| polars_eager | 5 | 1.96 | 2.36 | 1,991 | 2.9x / 2.6x |
| polars_lazy | 5 | 2.01 | 2.65 | 2,443 | 2.9x / 2.3x |

- Equivalence: 18 of 18 features match pandas (relative tolerance 1e-6, absolute tolerance 1e-9, and 1e-6 for the rolling standard deviation), same 1,681,431 rows and keys, same null pattern; eager and lazy Polars are identical.
- The only difference is `roa_q_std4`: 5 of 1,652,719 comparable rows differ by more than 1e-9 (largest 1.86e-8), none by more than 1e-6. In the worst row (CERT 28535) the four ROAQ values in the window are all 0.69, so the true standard deviation is exactly 0: Polars returns 0 and pandas leaves a rounding residue.
- Polars is about 2.3-2.9x faster than pandas but uses more memory (1.6x for eager, 1.9x for lazy), because it runs multithreaded and keeps more intermediate data at once.
- Lazy is not faster than eager here (2.01 s vs 1.96 s) because the eager code already reads only the 12 needed columns. The lazy engine matters when the column choice is not hand-written, and for filters (Part 4).
- Query plan (`results/part3_polars_plan.txt`): the unoptimised plan scans the panel 4 times with all 61 columns (244 column reads); the optimised plan scans it twice, reading 12 and 7 of 61 columns (19 column reads), computes the shared conditions `ASSET > 0` and `LNLSGR > 0` once, and caches the shared ratio-and-lag sub-plan so that the rolling window reuses it. No filter appears in the query, so predicate pushdown does not apply yet.
- Timings are best-of-N on one laptop and vary between runs; pandas has fewer timed runs because it is slower.

### Part 3.2 Spark (`results/part3_spark_results.txt`, `results/part3_spark_plan.txt`, `notebooks/part3_spark.ipynb`)

The same 18 features are rebuilt in PySpark 4.0.1 (local mode, Google Colab: 2 vCPU, 13.6 GB RAM, Polars 1.35.2, Python 3.13, adaptive query execution on, 16 shuffle partitions) and compared with Polars on the same input file (12 columns of the clean panel, 1,681,431 rows). Spark local mode is not a cluster: it measures the engine, not the speed-up from adding machines.

- Equivalence: 18 of 18 features match Polars (same rows and keys). The first comparison gave 17 of 18 with a 1e-7 tolerance for `roa_q_std4`; the diagnosis found 15 of 1,652,719 rows above 1e-7 and none above 1e-6 (largest 6.74e-7). In the worst row (CERT 27229) the four ROAQ values of the window are all 0.0, so the exact standard deviation (NumPy, two-pass) is 0: Spark returns 0 and Polars 6.74e-7, and Spark is closer to the exact value in all 15 rows. The tolerance was then set to 1e-6 for this feature (1e-9 for the others). The rounding residue depends on engine, version and the values before the window: on the laptop run (Polars 2.1.0) it was pandas that left a residue (1.9e-8). `log_asset` differs by 3.6e-15 (different `log` implementations).
- Plan (formatted physical plan before execution, so adaptive execution may change partition counts at run time): 6 parquet scans reading 12, 7, 6, 2, 4 and 2 of the 12 columns (column pruning; one scan also carries a pushed `IsNotNull(CERT)` filter), 12 shuffles (`Exchange`), 12 sorts, 4 window operators, 4 sort-merge joins, 1 broadcast hash join (the 170-row median table) and 2 object hash aggregates (exact median). The first "Plan facts" block in the results file counts every node twice (tree plus detail section); the second block, "each node counted once", is the correct one.
- Why 6 scans against 2 in Polars: the sub-plan with ratios and the 4-quarter lag is used by three branches (main result, medians, ranks); in this plan Spark recomputes it for each branch (no `ReusedExchange` node), while Polars caches it (`CACHE` node) and scans twice. The cost of Spark is mainly the 12 shuffles and sorts (hash partitioning on `CERT, qidx`) and JVM scheduling overhead.
- Time on the same Colab machine (best / median of 3 after one warm-up): Polars lazy 3.18 / 3.67 s, Spark 58.60 / 61.67 s, so Polars is 18.4x faster by best time. Spark's output is not collected to the driver, while Polars materialises its result, so the comparison slightly favours Spark.

### Part 3.3 When is a cluster justified? (`results/part3_spark_results.txt`, `results/part3_scaling.txt`)

The panel is replicated k times (`CERT + i * 10,000,000`, so each copy is an independent set of banks).

| Colab (2 vCPU), one run per size | Rows | Polars (s) | Spark (s) | Spark s per million rows |
|---|---|---|---|---|
| 1x | 1,681,431 | 5.2 | 60.5 | 36.0 |
| 2x | 3,362,862 | 7.1 | 132.8 | 39.5 |
| 4x | 6,725,724 | not run | 244.2 | 36.3 |

| Laptop (12 CPUs, 8.2 GB RAM), Polars lazy | Rows | Best of 2 (s) | Peak memory (MB) | MB per million rows |
|---|---|---|---|---|
| 1x | 1,681,431 | 2.85 | 1,165 | 693 |
| 2x | 3,362,862 | 5.46 | 2,127 | 633 |
| 3x | 5,044,293 | 6.22 | 2,428 | 481 |

- The 1x Polars time here (5.2 s) differs from the 3.18 s of Part 3.2: this loop times a single run of the replicated query instead of the best of three (cause not isolated).
- Spark time grows about linearly with the data (36-40 s per million rows); Polars grows less than linearly because fixed costs are shared (3.09 then 2.11 s per million rows on Colab).
- Memory is not linear in the data (693 to 481 MB per million rows), and the same pipeline peaked at 2,443 MB in Part 3.1 against 1,165 MB here (different query structure and number of runs; not explained). The 5 GB guard never triggered, so the real single-machine limit was not reached. Extrapolating 481 to 1,453 MB per million rows, 80% of 8.2 GB holds roughly 4.5 to 13.6 million rows, i.e. 3x to 8x the current panel (an extrapolation, not a measurement).
- Even if Spark scaled perfectly with cores and shuffles over a network were free, it would need about 18x more cores than Polars to tie on this workload (a calculation under an optimistic assumption, not a measurement).
- Conclusion: for a 1.7 million row panel (about 200 MB of compressed Parquet, 3 seconds on one machine) a cluster is not justified. It would be justified when the data no longer fits in one machine's memory (here beyond roughly 3x to 8x the panel, e.g. loan-level data), when a time limit cannot be met on one machine, when the work splits by independent keys without cross-sectional shuffles, or when the data already lives in distributed storage. 

## Part 4: Feature pipeline

Run after Part 2.3 and Part 3 (reads `data_storage/panel_clean/`, imports `part3_polars_pandas.py`):

1. Feature table, split definition, preprocessing Pipeline and leakage tests: `python part4_features.py` (output: `results/part4_features.txt`; the feature table `data_storage/part4/features.parquet` is not committed). Shared definitions (periods, feature list, preprocessing) are in `bank_common.py`.

### Part 4 feature table, split and leakage control (`results/part4_features.txt`)

Feature table `data_storage/part4/features.parquet` (1,681,431 rows): 24 model features in 7 groups (17 from Part 3 plus 7 same-quarter ratios) joined with the labels `y`, `y_alt`, `y_before_public` and the timestamp `avail_date` of Part 2.3. Every feature uses quarter q and earlier data only (lags and rolling windows look backwards) or the same quarter (cross-sectional median and rank). Not used: `roa_q_n4` (a count of quarters, a data-availability artefact), the year or quarter index (a model would learn the era), the regulatory capital ratios RBC1AAJ and RBCRWAJ (not reported in 1984-1989, zero in about 37% of rows from 2020) and the `ACTIVE` flag of the institutions table (it knows the future). The single-feature AUC and the missing shares below use development rows only.

| Group | Feature | Why it matters | Null % dev | Null % 1984-89 | Dev AUC (single feature) |
|---|---|---|---|---|---|
| capital | `eq_assets` | Capital buffer: equity / assets; thin capital absorbs fewer losses | 0.24 | 0.19 | 0.053 |
| capital | `d_eq_assets_4q` | 4-quarter change of the capital ratio: erosion is a warning sign | 8.00 | 19.38 | 0.137 |
| capital | `eq_assets_med_q` | Median capital ratio of all banks in the same quarter (system-wide context) | 0.00 | 0.00 | 0.275 |
| capital | `eq_assets_pct_q` | Rank of the bank's capital ratio within the quarter (0 to 1) | 0.24 | 0.19 | 0.080 |
| asset quality | `npl_loans` | Non-current loans / gross loans: credit problems already visible | 1.12 | 1.02 | 0.908 |
| asset quality | `reserves_loans` | Loan-loss reserves / gross loans: cushion against bad loans | 1.15 | 0.61 | 0.751 |
| asset quality | `d_npl_loans_4q` | 4-quarter change of the non-current loan ratio | 8.81 | 20.15 | 0.656 |
| asset quality | `npl_loans_med_q` | Median non-current loan ratio in the quarter (system-wide stress) | 0.00 | 0.00 | 0.715 |
| asset quality | `ore_assets` | Foreclosed real estate / assets: assets taken over from defaulted loans | 0.29 | 0.24 | 0.885 |
| earnings | `roaq` | Quarterly return on assets (earnings power) | 0.28 | 0.32 | 0.101 |
| earnings | `roa_q_mean4` | Average quarterly ROA over the last 4 quarters | 0.26 | 0.25 | 0.064 |
| earnings | `roa_q_std4` | Volatility of quarterly ROA over the last 4 quarters | 2.26 | 5.20 | 0.910 |
| earnings | `nimy` | Net interest margin (core profitability) | 0.00 | 0.00 | 0.255 |
| liquidity and funding | `liquid_assets` | (Cash balances + securities) / assets: liquidity buffer | 0.01 | 0.00 | 0.276 |
| liquidity and funding | `bro_dep` | Brokered deposits / deposits: reliance on volatile funding | 1.90 | 3.52 | 0.610 |
| liquidity and funding | `dep_assets` | Deposits / assets: funding structure | 0.00 | 0.00 | 0.773 |
| loan book | `loans_assets` | Loans / assets: share of the balance sheet exposed to credit risk | 0.00 | 0.01 | 0.621 |
| loan book | `lnre_assets` | Real-estate loans / assets: concentration in real estate | 0.00 | 0.00 | 0.566 |
| loan book | `lnci_assets` | Commercial and industrial loans / assets | 0.00 | 0.00 | 0.522 |
| loan book | `lncon_assets` | Consumer loans / assets | 0.22 | 0.14 | 0.489 |
| growth | `asset_yoy` | Asset growth over 4 quarters: very fast growth can signal risk-taking | 7.78 | 19.22 | 0.197 |
| growth | `loans_yoy` | Loan growth over 4 quarters | 8.56 | 19.55 | 0.203 |
| growth | `dep_yoy` | Deposit growth over 4 quarters | 8.30 | 19.46 | 0.265 |
| size | `log_asset` | Size (log of assets) | 0.00 | 0.00 | 0.478 |

- Redundancy: 3 feature pairs have a Spearman correlation above 0.9 in development rows (`eq_assets` ~ `eq_assets_pct_q`; `eq_assets_med_q` ~ `npl_loans_med_q`; `asset_yoy` ~ `dep_yoy`). They are kept: the tree model is insensitive to it and the logistic model is regularised.

Split (decided from the failure counts of Part 2.3 before any model was fitted; the test period is used once, in Part 5): development 1984Q1 to 2006Q4, gap 2007Q1 to 2007Q4 (four quarters, because a label looks four quarters ahead), test 2008Q1 to 2014Q4.

| Period | Rows | Labelled rows | Positive rows | Failing banks | Positive rows (actionable) | Positive % |
|---|---|---|---|---|---|---|
| dev | 1,193,702 | 1,193,702 | 8,678 | 2,290 | 8,140 | 0.727 |
| gap | 34,754 | 34,754 | 48 | 27 | 47 | 0.138 |
| test | 212,645 | 212,645 | 2,005 | 513 | 1,764 | 0.943 |
| after | 240,330 | 222,801 | 129 | 36 | 111 | 0.058 |

- All preprocessing lives in a scikit-learn `Pipeline` (`bank_common.py`, function `preprocessing`) fitted on training rows only: clipping to the 0.5% and 99.5% quantiles, median imputation and standardisation for the linear model; clipping only for the tree model (native missing-value handling). Demonstration: fitting the clip bounds on dev + test rows instead of dev rows changes them (largest relative shift of an upper bound 0.19), which is exactly the leak the Pipeline avoids.
- Rows with `y_before_public = 1` (the report was not public before the bank failed) are flagged here; their treatment in training and evaluation is decided in Part 5.
- Leakage tests: 8 of 8 safety tests passed (features of earlier rows do not change when later rows are removed, at two cut-offs; four-quarter dev/test gap; no label column among the features; no failing bank in both periods; test period ends before labels become incomplete; clip bounds equal to the development quantiles; unchanged after transforming test rows), plus one demonstration that fitting the clip bounds on dev + test rows would change them.

## Sources
- `fdic_download_starter.py`: starter script provided by the course instructor (original, unchanged).
- `download_fdic.py`: copy of the starter script, modified by me only in the field list `FIN_FIELDS` and the default output folder.
- FDIC BankFind API and field dictionary: https://api.fdic.gov/banks/docs/ (failures, institutions and history endpoints).
- Course materials: Lecture 3 "Financial Time Series" (07_L03_TimeSeries_ML) for the point-in-time rule and the available-timestamp idea; Lessons 12-14 (visualization) for the figure design.
- Apache Spark / PySpark installation documentation (Java 17 or later is required): https://spark.apache.org/docs/latest/api/python/getting_started/install.html
- AI tools: Claude (Anthropic) and ChatGPT (OpenAI) were used for guidance on environment setup, debugging, code drafting (storage benchmark, data-quality report, exit classification, clean panel and label construction, zero-pattern evidence, Polars/pandas/Spark benchmarks, Spark notebook and scaling experiment), feature pipeline (Part 4), choosing extra FDIC fields and structuring the README and report. I ran all code myself, checked the outputs against the generated result files, and take responsibility for the final content.