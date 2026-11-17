machine: 12 logical CPUs, 8.2 GB RAM | polars 2.1.0 | memory guard 5.0 GB above baseline | 2 runs per size (best time reported)
k = 1: 1,681,431 rows | best 2.85 s | peak 1,165 MB
k = 2: 3,362,862 rows | best 5.46 s | peak 2,127 MB
k = 3: 5,044,293 rows | best 6.22 s | peak 2,428 MB

| Size | Rows | Best time (s) | Peak memory (MB) | MB per million rows | Seconds per million rows |
|---|---|---|---|---|---|
| 1x | 1,681,431 | 2.85 | 1,165 | 693 | 1.69 |
| 2x | 3,362,862 | 5.46 | 2,127 | 633 | 1.62 |
| 3x | 5,044,293 | 6.22 | 2,428 | 481 | 1.23 |

extrapolation (not measured): at about 481 MB per million rows, 80% of the 8.2 GB RAM holds roughly 13.7 million rows, i.e. 8.1x the current panel