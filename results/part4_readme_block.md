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
- Leakage tests: 9 of 9 passed (features of earlier rows do not change when later rows are removed at two cut-offs, four-quarter dev/test gap, no label column among the features, no failing bank in both periods, clip bounds equal to the development quantiles, unchanged after transforming test rows).
