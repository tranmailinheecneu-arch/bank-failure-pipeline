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
