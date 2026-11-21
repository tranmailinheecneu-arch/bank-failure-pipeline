### Part 5.2 test evaluation, used once (`results/part5_test.txt`)

The test period (2008Q1 to 2014Q4) was opened exactly once, after `results/part5_selection.json` was committed; `results/part5_test_LOCK.json` records the time, the hash of that file and the git commit, and the script refuses to run a second time or when the selection file is not committed. All models are refitted on all development rows; thresholds and alarm rates come from the development period. Evaluation uses the actionable rows (reports that were public before the failure).

| Model | AP | ROC-AUC | Recall @1% | Recall @5% | Lift @5% | AP, all rows | AP vs `y_alt` |
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
