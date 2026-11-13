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
- Failures by year of last report peak in 1989 (416 banks); other exits peak in 1994 (840 banks) (`results/fig_part2_exits_by_year.png`).
- 0 failures have their last report more than four quarters before FAILDATE; these cannot receive a positive four-quarter-ahead label.
- 841 of 2,662 classified failures may have a final report that was not public before failure under the 35-day publication-lag assumption.
- 865 other exits carry a receivership, bridge-bank, payoff or assisted-merger FDIC change code but are not linked to a FAILURE row (candidates for failures missing from the failures table; reviewed in Part 2.3).
- 186 banks fall in anomaly classes (FAILURE event but still reporting, or FAILDATE before the first report).
- Built with a lazy Polars scan that reads 3 of 51 columns in 0.2 s (`results/part2_exits_plan.txt`).
