### Part 5.4 development-period robustness analyses (`results/part5_dev_extras.txt`)

Script `part5_dev_extras.py`; it reads only development rows and refits the committed models, and it stops unless the out-of-fold Average Precision of B1 and of the selected model reproduces Table 12 to four decimals (it does).

- **False-alarm cost: implied kappa.** The cost of a false alarm cannot be observed, but the number of banks that supervisors can review can be assumed. The kappa at which the cost-optimal alarm rate equals a capacity of 1%, 2%, 5% and 10% of bank-quarters is (asset-weighted costs, as in Part 5.3): 1%: B1 0.4320, selected 0.2257; 2%: B1 0.1663, selected 0.0964; 5%: B1 0.0106, selected 0.0328; 10%: B1 0.0007, selected 0.0066. The main kappa of 0.02 corresponds to an alarm rate of 4.58% for B1 and 6.33% for the selected model. This replaces one assumption (kappa) by another that is easier to state (capacity); it does not make the cost observable.
- **Calibration.** B1 uses a balanced class weight, so its probabilities are scores: mean predicted rate 0.199 against observed 0.0244 in fold 1; isotonic recalibration fitted on earlier folds lowers its Brier score from 0.0491 to 0.0130 (fold 2), from 0.0182 to 0.0024 (fold 3) and from 0.00678 to 0.00042 (fold 4). The selected model is trained without class weight: its mean predicted rate is 0.0085, 0.0181, 0.0061 and 0.00031 against observed 0.0244, 0.0146, 0.0022 and 0.00037; the gap between predicted and observed rate changes from fold to fold with the base rate of the training window, which is consistent with the instability of its score threshold (Part 5.3) but does not prove it.
- **Is the larger model clearly better?** Bank-level bootstrap (500 resamples): selected minus B1 pooled AP +0.0092, 95% interval [-0.0070, +0.0248] (contains 0: not clearly better); mean fold AP +0.0793, 95% interval [+0.0594, +0.1005] (excludes 0; the selection rule used this metric).
- **Label noise.** Against the sensitivity label y_alt (positives 7,856 against 5,652), the pooled ranking of the three models by AP is ['B1', 'M2', 'B0'] against ['M2', 'B1', 'B0'] for y (the order changes): the models were trained on y only, so this measures how much the evaluation depends on the label definition.

| Fold | Positives y | Positives y_alt | B0 AP(y) / AP(y_alt) | B1 AP(y) / AP(y_alt) | Selected AP(y) / AP(y_alt) |
|---|---|---|---|---|---|
| 1 | 3226 | 3911 | 0.256 / 0.364 | 0.451 / 0.645 | 0.433 / 0.509 |
| 2 | 1790 | 2933 | 0.106 / 0.296 | 0.225 / 0.668 | 0.331 / 0.737 |
| 3 | 473 | 849 | 0.041 / 0.171 | 0.111 / 0.460 | 0.330 / 0.369 |
| 4 | 163 | 163 | 0.013 / 0.013 | 0.088 / 0.088 | 0.104 / 0.104 |
| pooled | 5652 | 7856 | 0.136 / 0.257 | 0.287 / 0.589 | 0.294 / 0.522 |
