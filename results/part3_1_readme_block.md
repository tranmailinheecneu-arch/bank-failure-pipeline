### Part 3.1 Polars vs pandas (`results/part3_polars_pandas.txt`)

Core transformations used later as features (18 columns): safe ratios, 4-quarter calendar lags (self-join on `qidx + 4`, so gaps in a bank's history never shift a lag), a 4-quarter rolling window of quarterly ROA, and same-quarter cross-sectional median and rank. Only quarter q and earlier data (or the same quarter) are used.

Machine: 12 logical CPUs, 8.2 GB RAM | polars 2.1.0 | pandas 3.1.1 | python 3.13.7. Each engine runs in its own process: 1 warm-up run, then the timed runs; memory is the peak resident set size minus the size after imports.

| Engine | Timed runs | Best (s) | Median (s) | Peak memory (MB) | Speedup vs pandas |
|---|---|---|---|---|---|
| pandas | 2 | 5.73 | 6.16 | 1,253 | 1.0x |
| polars_eager | 5 | 1.96 | 2.36 | 1,991 | 2.9x |
| polars_lazy | 5 | 2.01 | 2.65 | 2,443 | 2.9x |

- Results are identical across engines: 18 of 18 features match pandas within a relative tolerance of 1e-6 (largest absolute difference 1.86e-08), same 1,681,431 rows and keys, same null pattern; eager and lazy Polars agree (yes).
- The optimised lazy plan reads 12 of 61 columns of the panel (projection pushdown; plans in `results/part3_polars_plan.txt`).
- Timings are best-of-N on one laptop and vary between runs.
