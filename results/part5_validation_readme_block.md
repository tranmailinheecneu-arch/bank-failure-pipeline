### Part 5.1 validation, model choice and alarm threshold (`results/part5_validation.txt`, `results/part5_selection.json`)

Validation: four expanding-window folds inside the development period, with four quarters left out between the end of training and the start of validation (a label looks four quarters ahead; asserted in the code). Random cross-validation is not used: it would mix quarters and put the same bank on both sides. This script never reads the test rows. Metric: Average Precision (area under the precision-recall curve), because only about 0.7% of bank-quarters are positive and accuracy would reward never raising an alarm; recall and lift at a fixed alarm capacity (1% or 5% of bank-quarters flagged) and the share of failing banks caught are reported next to it, ROC-AUC only as a secondary figure. Rows with `y_before_public = 1` stay in training but are excluded from evaluation (nobody could have acted on them); the last column shows the pooled AP with them included.

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

- Selection rule, fixed before any result was seen: among the models with 24 features (M1 logistic, M2 gradient boosting) the highest mean fold AP. Selected: **M2 gbm 300 iter, lr 0.05, 31 leaves**. Baselines (fixed): B0 ranks by the non-current loan ratio alone, B1 is a logistic model on three features (`eq_assets`, `npl_loans`, `roaq`). All preprocessing is inside the Pipeline fitted on the training rows of each fold.
- Cost of a missed failure, from the failures table (failures of 1984-2007, 2,291 with COST and assets at failure): asset-weighted loss rate 0.1959 (sum of COST over sum of assets at failure), times the bank's assets, spread over the up to four positive quarters of a failure: 15.3 million USD per positive bank-quarter on average.
- Cost of a false alarm: not observable in FDIC data. Assumed to be kappa times the average missed-failure cost, main kappa = 0.02 (a false alarm costs one fiftieth of a missed failure), with a sensitivity analysis from 0.005 to 0.1. This is an assumption, not an estimate.
- The alarm threshold minimises the total cost on the out-of-fold predictions of the development period (no test data):

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
