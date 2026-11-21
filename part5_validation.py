"""Part 5 (development rows only): purged time-ordered validation, model comparison, cost-based alarm threshold.

Reads  : data_storage/part4/features.parquet (ONLY the development rows are loaded; the test rows are never read here)
         data_raw/fdic_v2/failures.parquet (COST and assets at failure; failures of 1984-2007 only)
Writes : results/part5_validation.txt, results/part5_validation_readme_block.md, results/part5_selection.json
Rules fixed before any result was seen:
  - selection: among the models M1 (logistic, 24 features) and M2 (gradient boosting) the highest mean fold Average
    Precision on actionable rows; baselines B0 and B1 are fixed
  - actionable rows: rows without y_before_public = 1 (the bank failed before its report was public)
  - main false-alarm cost ratio kappa = 0.02, sensitivity 0.005 ... 0.1
  Added AFTER the development results were seen and BEFORE the test period was opened: the capacity rule (flag the
  top r% of banks of each quarter, r = cost-optimal alarm rate of the table), because the score threshold of the
  gradient-boosting model is unstable across training windows. Both rules are evaluated once in part5_test.py.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from bank_common import FEATURES_MODEL, HORIZON, DEV, ql
from bank_models import (BASELINE_FEATURES, FOLDS, KAPPA_MAIN, KAPPAS, RATES, bank_recall, best_threshold, cost_at,
                         eval_metrics, fold_masks, make_model, purge_gap)

RES = Path("results")
RES.mkdir(exist_ok=True)
out = []


def log(s=""):
    out.append(str(s))


CANDIDATES = [
    ("B0 npl_loans only", "single", ["npl_loans"], {}),
    ("B1 logistic, 3 features", "logit", BASELINE_FEATURES, {"C": 1.0}),
    ("M1 logistic C=0.01", "logit", FEATURES_MODEL, {"C": 0.01}),
    ("M1 logistic C=0.1", "logit", FEATURES_MODEL, {"C": 0.1}),
    ("M1 logistic C=1", "logit", FEATURES_MODEL, {"C": 1.0}),
    ("M2 gbm 150 iter, lr 0.10, 15 leaves", "gbm", FEATURES_MODEL, {"max_iter": 150, "lr": 0.1, "leaves": 15}),
    ("M2 gbm 300 iter, lr 0.05, 15 leaves", "gbm", FEATURES_MODEL, {"max_iter": 300, "lr": 0.05, "leaves": 15}),
    ("M2 gbm 300 iter, lr 0.05, 31 leaves", "gbm", FEATURES_MODEL, {"max_iter": 300, "lr": 0.05, "leaves": 31}),
]
B0, B1 = CANDIDATES[0][0], CANDIDATES[1][0]

t0 = time.time()
feat = pd.read_parquet("data_storage/part4/features.parquet", filters=[("period", "==", "dev")]).reset_index(drop=True)
q = feat["qidx"].to_numpy()
y = feat["y"].to_numpy(float)
cert = feat["CERT"].to_numpy()
assert not np.isnan(y).any() and q.max() <= DEV[1]
actionable = ~((y == 1) & (feat["y_before_public"].to_numpy() == 1))
folds = fold_masks(q, y)
gaps = [purge_gap(q, tr, va) for _, tr, va in folds]
assert min(gaps) >= HORIZON, gaps
log(f"development rows: {len(feat):,} ({ql(q.min())} to {ql(q.max())}), positive rows {int(y.sum()):,}, "
    f"actionable positive rows {int((y[actionable] == 1).sum()):,}")

# ------------------------------------------------------------------ cost estimate from the failures table
f = pl.read_parquet("data_raw/fdic_v2/failures.parquet").with_columns(
    pl.col("COST").cast(pl.Float64, strict=False), pl.col("QBFASSET").cast(pl.Float64, strict=False),
    pl.col("FAILYR").cast(pl.Int64, strict=False))
f = f.filter((pl.col("RESTYPE").cast(pl.Utf8).str.to_uppercase() == "FAILURE")
             & (pl.col("FAILYR") >= 1984) & (pl.col("FAILYR") <= 2007))
n_f = f.height
f = f.filter(pl.col("COST").is_not_null() & pl.col("QBFASSET").is_not_null() & (pl.col("QBFASSET") > 0))
loss_rate = float(f["COST"].sum() / f["QBFASSET"].sum())
ratio = (f["COST"] / f["QBFASSET"]).to_numpy()
asset = np.exp(feat["log_asset"].to_numpy(float))
asset = np.where(np.isnan(asset), np.nanmedian(asset), asset)            # thousands of USD
cfn = loss_rate * asset / HORIZON       # each failure is spread over its (up to) 4 positive rows
mean_cfn = float(cfn[(y == 1) & actionable].mean())
log("\n== Cost of a missed failure, estimated from the failures table (failures of 1984-2007 only) ==")
log(f"failures in the table: {n_f:,}; with COST and assets at failure: {f.height:,}")
log(f"asset-weighted loss rate = sum(COST) / sum(assets at failure) = {loss_rate:.4f} "
    f"(median bank {np.median(ratio):.4f}, 90th percentile {np.quantile(ratio, 0.9):.4f})")
log(f"average cost of a missed failure per positive bank-quarter: {mean_cfn / 1e3:,.1f} million USD "
    f"(loss rate x assets / {HORIZON})")
log(f"false-alarm cost = kappa x {mean_cfn / 1e3:,.1f} million USD; main kappa = {KAPPA_MAIN} "
    "(NOT observable in the FDIC data: an assumption, tested with kappa from 0.005 to 0.1)")

# ------------------------------------------------------------------ folds
log("\n== Purged expanding-window folds ==")
frow = []
for (fname, tr, va), (te, vs, ve) in zip(folds, FOLDS):
    ev = va & actionable
    frow.append((fname, f"{ql(q[tr].min())}-{ql(te)}", f"{ql(vs)}-{ql(ve)}", purge_gap(q, tr, va),
                 int(tr.sum()), int((y[tr] == 1).sum()), int(ev.sum()), int((y[ev] == 1).sum()),
                 len(set(cert[ev & (y == 1)])), round(100 * y[ev].mean(), 3)))
ftab = pd.DataFrame(frow, columns=["fold", "train", "validation", "purged_q", "train_rows", "train_pos_rows",
                                   "val_rows", "val_pos_rows", "val_failing_banks", "val_base_rate_%"])
log(ftab.to_string(index=False))

# ------------------------------------------------------------------ out-of-fold predictions
OOF = {}
for name, kind, cols, params in CANDIDATES:
    t1 = time.time()
    oof = np.full(len(feat), np.nan)
    X = feat[cols].to_numpy(float)
    for fname, tr, va in folds:
        m = make_model(kind, **params).fit(X[tr], y[tr].astype(int))
        oof[va] = m.predict_proba(X[va])[:, 1]
    OOF[name] = oof
    print(f"{name}: {time.time() - t1:.0f} s", flush=True)

rows, fold_ap = [], {}
for name, kind, cols, params in CANDIDATES:
    aps = [eval_metrics(y[va & actionable], OOF[name][va & actionable])["AP"] for _, _, va in folds]
    fold_ap[name] = aps
    pm = ~np.isnan(OOF[name]) & actionable
    p = eval_metrics(y[pm], OOF[name][pm])
    pa = eval_metrics(y[~np.isnan(OOF[name])], OOF[name][~np.isnan(OOF[name])])
    rows.append((name, round(float(np.mean(aps)), 4), *[round(a, 4) for a in aps], round(p["AP"], 4),
                 round(p["AUC"], 4), round(p["recall@1%"], 3), round(p["recall@5%"], 3), round(p["lift@5%"], 1),
                 round(pa["AP"], 4)))
mtab = pd.DataFrame(rows, columns=["model", "mean_fold_AP", "AP_f1", "AP_f2", "AP_f3", "AP_f4", "pooled_AP",
                                   "pooled_AUC", "recall@1%", "recall@5%", "lift@5%", "pooled_AP_all_rows"])
log("\n== Model comparison on out-of-fold predictions (actionable rows; Average Precision is the main metric) ==")
log("chance level of AP = positive rate of the fold (column val_base_rate_% above)")
log(mtab.to_string(index=False))

cands_m = [c for c in CANDIDATES if c[0].startswith("M")]
sel = max(cands_m, key=lambda c: np.mean(fold_ap[c[0]]))
SEL = sel[0]
log(f"\nselected by the fixed rule (highest mean fold AP among M1/M2): {SEL}")

# ------------------------------------------------------------------ cost-based threshold on out-of-fold predictions
mask = {n: ~np.isnan(OOF[n]) & actionable for n in (B1, SEL)}
thresholds = {B1: {}, SEL: {}}
srow, mainrows = [], []
for kappa in KAPPAS:
    cfp = kappa * mean_cfn
    for n in (B1, SEL):
        m = mask[n]
        s, yy, cc = OOF[n][m], y[m], cfn[m]
        t, _ = best_threshold(s, yy, cc, cfp)
        thresholds[n][str(kappa)] = None if np.isinf(t) else t
        c, fn, fp, alarm = cost_at(s, yy, cc, cfp, t)
        no_alarm = float((yy * cc).sum())
        all_alarm = float((yy == 0).sum() * cfp)
        br, nb = bank_recall(cert[m], yy, alarm)
        rec = float(alarm[yy == 1].mean())
        row = (kappa, n, round(100 * alarm.mean(), 2), round(rec, 3), round(br, 3), round(1 - c / no_alarm, 3),
               round(c / 1e6, 2), round(no_alarm / 1e6, 2))
        srow.append(row)
stab = pd.DataFrame(srow, columns=["kappa", "model", "alarm_rate_%", "recall_rows", "recall_banks",
                                   "cost_saving_vs_no_alarm", "cost_bn_USD", "no_alarm_cost_bn_USD"])
log("\n== Cost-optimal threshold chosen on out-of-fold predictions, for several kappa (kappa 0.02 is the main case) ==")
log(stab.to_string(index=False))
m = mask[SEL]
cfp_main = KAPPA_MAIN * mean_cfn
all_alarm_cost = float((y[m] == 0).sum() * cfp_main)
no_alarm_cost = float((y[m] * cfn[m]).sum())
log(f"\nmain case: cost without any alarm {no_alarm_cost / 1e6:.2f} bn USD; alarming every bank-quarter "
    f"{all_alarm_cost / 1e6:.2f} bn USD (the alarm model must beat the smaller of the two)")
log(f"\nbuilt in {time.time() - t0:.0f} s")
(RES / "part5_validation.txt").write_text("\n".join(out), encoding="utf-8")

selection = {
    "selected": SEL, "baselines": [B0, B1], "kappa_main": KAPPA_MAIN, "kappas": KAPPAS, "loss_rate": loss_rate,
    "mean_cfn": mean_cfn, "models": {c[0]: {"kind": c[1], "cols": c[2], "params": c[3]} for c in CANDIDATES
                                      if c[0] in (B0, B1, SEL)},
    "thresholds": thresholds,
    "alarm_rates_pct": {n: {str(r[0]): r[2] for r in srow if r[1] == n} for n in (B1, SEL)},
    "dev_median_asset": float(np.median(np.exp(feat["log_asset"].dropna()))),
}
(RES / "part5_selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
sv = {(r[0], r[1]): r[5] for r in srow}              # cost saving vs no alarm, by (kappa, model)
ar = {(r[0], r[1]): r[2] for r in srow}              # cost-optimal alarm rate in %, by (kappa, model)
mean_ap = mtab.set_index("model")["mean_fold_AP"]

# ------------------------------------------------------------------ README block
md = [
    "### Part 5.1 validation, model choice and alarm threshold (`results/part5_validation.txt`, `results/part5_selection.json`)",
    "",
    "Validation: four expanding-window folds inside the development period, with four quarters left out between the end of "
    "training and the start of validation (a label looks four quarters ahead; asserted in the code). Random cross-validation is not "
    "used: it would mix quarters and put the same bank on both sides. This script never reads the test rows. Metric: Average Precision "
    "(area under the precision-recall curve), because about 0.7% of development bank-quarters are positive (between 0.04% and 2.4% inside the validation folds) and accuracy would reward never "
    "raising an alarm; recall and lift at a fixed alarm capacity (1% or 5% of bank-quarters flagged) and the share of failing banks "
    "caught are reported next to it, ROC-AUC only as a secondary figure. Rows with `y_before_public = 1` stay in training but are "
    "excluded from evaluation (nobody could have acted on them); the last column shows the pooled AP with them included.",
    "",
    "| Fold | Train | Validation | Purged quarters | Train positive rows | Validation positive rows | Validation failing banks | Validation base rate % |",
    "|---|---|---|---|---|---|---|---|",
]
for r in ftab.itertuples():
    md.append(f"| {r.fold} | {r.train} | {r.validation} | {r.purged_q} | {r.train_pos_rows:,} | {r.val_pos_rows:,} | "
              f"{r.val_failing_banks:,} | {r._10} |")
md += ["", "| Model | Mean fold AP | AP fold 1 | AP fold 2 | AP fold 3 | AP fold 4 | Pooled AP | Pooled AUC | Recall @1% | Recall @5% | Lift @5% | Pooled AP, all rows |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in mtab.itertuples():
    md.append(f"| {r.model} | {r.mean_fold_AP} | {r.AP_f1} | {r.AP_f2} | {r.AP_f3} | {r.AP_f4} | {r.pooled_AP} | {r.pooled_AUC} | "
               f"{r._9} | {r._10} | {r._11} | {r.pooled_AP_all_rows} |")
md += [
    "",
    f"- Selection rule, fixed before any result was seen: among the models with 24 features (M1 logistic, M2 gradient boosting) the "
    f"highest mean fold AP. Selected: **{SEL}**. Baselines (fixed): B0 ranks by the non-current loan ratio alone, B1 is a logistic model on "
    "three features (`eq_assets`, `npl_loans`, `roaq`). All preprocessing is inside the Pipeline fitted on the training rows of each fold. "
    f"The two are close: M2 wins by mean fold AP ({mean_ap[SEL]} against {mean_ap[B1]} for B1) but B1 saves more cost at its "
    f"cost-optimal threshold ({100 * sv[(KAPPA_MAIN, B1)]:.0f}% against {100 * sv[(KAPPA_MAIN, SEL)]:.0f}% at kappa {KAPPA_MAIN}).",
    f"- Cost of a missed failure, from the failures table (failures of 1984-2007, {f.height:,} with COST and assets at failure): asset-weighted "
    f"loss rate {loss_rate:.4f} (sum of COST over sum of assets at failure), times the bank's assets, spread over the up to four positive "
    f"quarters of a failure: {mean_cfn / 1e3:,.1f} million USD per positive bank-quarter on average.",
    f"- Cost of a false alarm: not observable in FDIC data. Assumed to be kappa times the average missed-failure cost, main kappa = {KAPPA_MAIN} "
    "(a false alarm costs one fiftieth of a missed failure), with a sensitivity analysis from 0.005 to 0.1. This is an assumption, not an estimate.",
    "- The alarm threshold minimises the total cost on the out-of-fold predictions of the development period (no test data):",
    "",
    "| kappa | Model | Alarm rate % | Recall (bank-quarters) | Recall (failing banks) | Cost saving vs no alarm | Cost (bn USD) | No-alarm cost (bn USD) |",
    "|---|---|---|---|---|---|---|---|",
]
for r in stab.itertuples():
    md.append(f"| {r.kappa} | {r.model} | {r._3} | {r.recall_rows} | {r.recall_banks} | {r.cost_saving_vs_no_alarm} | "
              f"{r.cost_bn_USD} | {r.no_alarm_cost_bn_USD} |")
md += [
    "",
    f"- Second alarm rule (capacity rule), added after the development results above had been seen and before the test period was "
    f"opened: flag the top r% of banks of every quarter, with r the cost-optimal alarm rate of the table at kappa {KAPPA_MAIN} "
    f"(B1 {ar[(KAPPA_MAIN, B1)]}%, {SEL} {ar[(KAPPA_MAIN, SEL)]}%). It does not depend on the score scale, which differs between "
    "models fitted on different training windows. Both rules are evaluated once on the test period (Part 5.2).",
]
(RES / "part5_validation_readme_block.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(out)[-3500:])
print("\nfull report: results/part5_validation.txt | selection: results/part5_selection.json")
