"""Part 4: leak-free feature table joined with the label, split definition, preprocessing Pipeline and leakage tests.

Reads  : data_storage/panel_clean/*/*.parquet (Part 2.3)
Writes : data_storage/part4/features.parquet (not committed)
         results/part4_features.txt, results/part4_readme_block.md
No model is fitted here. Test-period rows are used only for counts and to demonstrate what a leak would change.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import roc_auc_score

from bank_common import (DEV, FEATURES_ALL, FEATURES_MODEL, FORBIDDEN, GAP, GROUPS, HORIZON, SEED, TEST,
                         QuantileClipper, Q, dev_mask, ql, preprocessing, test_mask)
from part3_polars_pandas import KEYS, PANEL_GLOB, polars_pipeline, sdiv

OUT = Path("data_storage/part4")
OUT.mkdir(parents=True, exist_ok=True)
RES = Path("results")
RES.mkdir(exist_ok=True)

WHY = {
    "eq_assets": "Capital buffer: equity / assets; thin capital absorbs fewer losses",
    "d_eq_assets_4q": "4-quarter change of the capital ratio: erosion is a warning sign",
    "eq_assets_med_q": "Median capital ratio of all banks in the same quarter (system-wide context)",
    "eq_assets_pct_q": "Rank of the bank's capital ratio within the quarter (0 to 1)",
    "npl_loans": "Non-current loans / gross loans: credit problems already visible",
    "reserves_loans": "Loan-loss reserves / gross loans: cushion against bad loans",
    "d_npl_loans_4q": "4-quarter change of the non-current loan ratio",
    "npl_loans_med_q": "Median non-current loan ratio in the quarter (system-wide stress)",
    "ore_assets": "Foreclosed real estate / assets: assets taken over from defaulted loans",
    "roaq": "Quarterly return on assets (earnings power)",
    "roa_q_mean4": "Average quarterly ROA over the last 4 quarters",
    "roa_q_std4": "Volatility of quarterly ROA over the last 4 quarters",
    "nimy": "Net interest margin (core profitability)",
    "liquid_assets": "(Cash balances + securities) / assets: liquidity buffer",
    "bro_dep": "Brokered deposits / deposits: reliance on volatile funding",
    "dep_assets": "Deposits / assets: funding structure",
    "loans_assets": "Loans / assets: share of the balance sheet exposed to credit risk",
    "lnre_assets": "Real-estate loans / assets: concentration in real estate",
    "lnci_assets": "Commercial and industrial loans / assets",
    "lncon_assets": "Consumer loans / assets",
    "asset_yoy": "Asset growth over 4 quarters: very fast growth can signal risk-taking",
    "loans_yoy": "Loan growth over 4 quarters",
    "dep_yoy": "Deposit growth over 4 quarters",
    "log_asset": "Size (log of assets)",
}

out = []


def log(s=""):
    out.append(str(s))


def build(lf):
    """18 features of Part 3 plus 7 extra same-quarter ratios. Uses quarter q and earlier data only."""
    f = lambda c: pl.col(c).cast(pl.Float64)
    base = polars_pipeline(lf)
    extra = lf.select(
        "CERT", "qidx",
        f("ROAQ").alias("roaq"), f("NIMY").alias("nimy"),
        sdiv(f("ORE"), f("ASSET")).alias("ore_assets"),
        sdiv(f("LNRE"), f("ASSET")).alias("lnre_assets"),
        sdiv(f("LNCI"), f("ASSET")).alias("lnci_assets"),
        sdiv(f("LNCON"), f("ASSET")).alias("lncon_assets"),
        sdiv(f("DEP"), f("ASSET")).alias("dep_assets"),
    )
    return base.join(extra, on=KEYS, how="left").sort(KEYS)


t0 = time.time()
lf = pl.scan_parquet(PANEL_GLOB)
labels = lf.select("CERT", "qidx", "y", "y_alt", "y_before_public", "avail_date")
feat = build(lf).join(labels, on=KEYS, how="left").sort(KEYS).collect().to_pandas()
q = feat["qidx"].to_numpy()
y = feat["y"].to_numpy(dtype=float)
feat["period"] = np.select(
    [(q >= DEV[0]) & (q <= DEV[1]), (q >= GAP[0]) & (q <= GAP[1]), (q >= TEST[0]) & (q <= TEST[1])],
    ["dev", "gap", "test"], default="after")
feat.to_parquet(OUT / "features.parquet", index=False, compression="zstd")
log(f"feature table: {len(feat):,} rows, {len(FEATURES_MODEL)} model features (+ roa_q_n4 built but unused), "
    f"labels y, y_alt, y_before_public and timestamp avail_date | built in {time.time() - t0:.1f} s")

# ---------------------------------------------------------------- periods
log(f"\n== Periods: development {ql(DEV[0])}-{ql(DEV[1])}, gap {ql(GAP[0])}-{ql(GAP[1])}, test {ql(TEST[0])}-{ql(TEST[1])} ==")
log("(row and failure counts only; the test period is not used for any decision)")
prow = []
for p in ["dev", "gap", "test", "after"]:
    d = feat[feat.period == p]
    lab = d[d.y.notna()]
    pos = lab[lab.y == 1]
    act = pos[pos.y_before_public == 0]
    prow.append((p, len(d), len(lab), len(pos), pos.CERT.nunique(), len(act), round(100 * len(pos) / max(len(lab), 1), 3)))
ptab = pd.DataFrame(prow, columns=["period", "rows", "labelled_rows", "positive_rows", "failing_banks",
                                   "positive_rows_actionable", "positive_pct"])
log(ptab.to_string(index=False))
dev_banks = set(feat[(feat.period == "dev") & (feat.y == 1)].CERT)
test_banks = set(feat[(feat.period == "test") & (feat.y == 1)].CERT)

# ---------------------------------------------------------------- feature table (development rows only)
dm = dev_mask(q, y)
era = dm & (q <= Q(1989, 4))
yd = y[dm].astype(int)
frows = []
for grp, cols in GROUPS.items():
    for c in cols:
        x = feat[c].to_numpy(float)
        m = ~np.isnan(x[dm])
        auc = roc_auc_score(yd[m], x[dm][m]) if m.sum() > 100 and yd[m].min() != yd[m].max() else float("nan")
        frows.append((grp, c, round(100 * np.isnan(x[dm]).mean(), 2), round(100 * np.isnan(x[era]).mean(), 2), round(auc, 3)))
ft = pd.DataFrame(frows, columns=["group", "feature", "null_pct_dev", "null_pct_1984_89", "auc_dev"])
log("\n== Model features (development rows only): missing share and single-feature AUC ==")
log("AUC > 0.5: a higher value goes with failure within 4 quarters; AUC < 0.5: a lower value does")
log(ft.to_string(index=False))

samp = feat.loc[dm, FEATURES_MODEL].sample(150000, random_state=SEED)
corr = samp.corr(method="spearman")
pairs = [(a, b, round(float(corr.loc[a, b]), 3)) for i, a in enumerate(FEATURES_MODEL) for b in FEATURES_MODEL[i + 1:]
         if abs(corr.loc[a, b]) > 0.9]
log("\n== Redundancy: feature pairs with |Spearman| > 0.9 (150,000 development rows; kept, see report) ==")
log("\n".join(f"{a} ~ {b}: {r}" for a, b, r in pairs) if pairs else "none")

# ---------------------------------------------------------------- preprocessing pipeline demonstration
Xdev = feat.loc[dm, FEATURES_MODEL].to_numpy(float)
tmask = test_mask(q, y)
Xtest = feat.loc[tmask, FEATURES_MODEL].to_numpy(float)
pipe = preprocessing("linear").fit(Xdev)
lo, hi = pipe.named_steps["clip"].lo_.copy(), pipe.named_steps["clip"].hi_.copy()
contam = QuantileClipper().fit(np.vstack([Xdev, Xtest]))
pipe.transform(Xtest)
log("\n== Preprocessing Pipeline (clip -> median impute -> standardise), fitted on development rows only ==")
show_cols = ["eq_assets", "npl_loans", "roaq", "asset_yoy", "log_asset"]
brow = []
for c in show_cols:
    i = FEATURES_MODEL.index(c)
    brow.append((c, lo[i], hi[i], contam.lo_[i], contam.hi_[i]))
btab = pd.DataFrame(brow, columns=["feature", "dev_p0.5", "dev_p99.5", "dev+test_p0.5", "dev+test_p99.5"])
log(btab.round(5).to_string(index=False))
log("The right-hand columns show what fitting on dev + test rows would change (a leak); the Pipeline never sees them.")

# ---------------------------------------------------------------- leakage tests
log("\n== Leakage tests ==")
tests = []


def trunc_check(q0):
    part = build(lf.filter(pl.col("qidx") <= q0)).collect().to_pandas()
    full = feat.loc[feat.qidx <= q0, ["CERT", "qidx"] + FEATURES_ALL].reset_index(drop=True)
    if len(part) != len(full) or not (part[["CERT", "qidx"]].to_numpy() == full[["CERT", "qidx"]].to_numpy()).all():
        return False, float("nan"), len(part)
    worst, ok = 0.0, True
    for c in FEATURES_ALL:
        a, b = part[c].to_numpy(float), full[c].to_numpy(float)
        if (np.isnan(a) != np.isnan(b)).any():
            ok = False
        m = ~np.isnan(a) & ~np.isnan(b)
        worst = max(worst, float(np.abs(a[m] - b[m]).max()) if m.any() else 0.0)
    return ok and worst < 1e-9, worst, len(part)


for yy, qq in [(2000, 4), (2010, 4)]:
    ok, worst, n = trunc_check(Q(yy, qq))
    tests.append((f"truncation test at {yy}Q{qq}: features of {n:,} rows unchanged when later rows are removed",
                  ok, f"largest difference {worst:.3g}"))
dev_last = int(q[dm].max())
test_first = int(q[tmask].min())
tests.append(("dev and test are separated by at least 4 quarters", test_first - dev_last - 1 >= HORIZON,
              f"{ql(dev_last)} -> {ql(test_first)}"))
tests.append(("no label or future-revealing column among the model features",
              not any(c in FORBIDDEN for c in FEATURES_MODEL), ""))
tests.append(("no failing bank has positive rows in both dev and test", len(dev_banks & test_banks) == 0, ""))
tests.append(("the test period ends before the labels become incomplete (2025Q2)", TEST[1] <= Q(2025, 2), ""))
direct = np.nanquantile(Xdev, 0.005, axis=0)
tests.append(("clip bounds of the fitted Pipeline equal the quantiles of the development rows",
              bool(np.allclose(lo, direct, equal_nan=True)), ""))
shift = float(np.nanmax(np.abs(contam.hi_ - hi) / np.maximum(np.abs(hi), 1e-12)))
tests.append(("fitting on dev + test rows WOULD change the clip bounds (so the leak is real and the Pipeline avoids it)",
              shift > 0, f"largest relative shift of an upper bound {shift:.3g}"))
tests.append(("transform on test rows leaves the fitted bounds unchanged",
              bool(np.array_equal(lo, pipe.named_steps['clip'].lo_, equal_nan=True)
                   and np.array_equal(hi, pipe.named_steps['clip'].hi_, equal_nan=True)), ""))
all_ok = True
for name, ok, detail in tests:
    all_ok &= bool(ok)
    log(f"{'PASS' if ok else 'FAIL'}  {name} {detail}".rstrip())

(RES / "part4_features.txt").write_text("\n".join(out), encoding="utf-8")

# ---------------------------------------------------------------- README block
md = [
    "### Part 4 feature table, split and leakage control (`results/part4_features.txt`)",
    "",
    f"Feature table `data_storage/part4/features.parquet` ({len(feat):,} rows): {len(FEATURES_MODEL)} model features in 7 groups "
    "(17 from Part 3 plus 7 same-quarter ratios) joined with the labels `y`, `y_alt`, `y_before_public` and the timestamp `avail_date` of Part 2.3. "
    "Every feature uses quarter q and earlier data only (lags and rolling windows look backwards) or the same quarter "
    "(cross-sectional median and rank). Not used: `roa_q_n4` (a count of quarters, a data-availability artefact), the year or quarter index "
    "(a model would learn the era), the regulatory capital ratios RBC1AAJ and RBCRWAJ (not reported in 1984-1989, zero in about 37% of rows "
    "from 2020) and the `ACTIVE` flag of the institutions table (it knows the future). The single-feature AUC and the missing shares below "
    "use development rows only.",
    "",
    "| Group | Feature | Why it matters | Null % dev | Null % 1984-89 | Dev AUC (single feature) |",
    "|---|---|---|---|---|---|",
]
for (grp, cols), _ in zip(GROUPS.items(), range(len(GROUPS))):
    for c in cols:
        r = ft[ft.feature == c].iloc[0]
        md.append(f"| {grp} | `{c}` | {WHY[c]} | {r.null_pct_dev:.2f} | {r.null_pct_1984_89:.2f} | {r.auc_dev:.3f} |")
md += [
    "",
    f"- Redundancy: {len(pairs)} feature pairs have a Spearman correlation above 0.9 in development rows"
    + (" (" + "; ".join(f"`{a}` ~ `{b}`" for a, b, _ in pairs) + ")" if pairs else "")
    + ". They are kept: the tree model is insensitive to it and the logistic model is regularised.",
    "",
    f"Split (decided from the failure counts of Part 2.3 before any model was fitted; the test period is used once, in Part 5): "
    f"development {ql(DEV[0])} to {ql(DEV[1])}, gap {ql(GAP[0])} to {ql(GAP[1])} (four quarters, because a label looks four quarters ahead), "
    f"test {ql(TEST[0])} to {ql(TEST[1])}.",
    "",
    "| Period | Rows | Labelled rows | Positive rows | Failing banks | Positive rows (actionable) | Positive % |",
    "|---|---|---|---|---|---|---|",
]
for r in ptab.itertuples():
    md.append(f"| {r.period} | {r.rows:,} | {r.labelled_rows:,} | {r.positive_rows:,} | {r.failing_banks:,} | "
              f"{r.positive_rows_actionable:,} | {r.positive_pct:.3f} |")
md += [
    "",
    "- All preprocessing lives in a scikit-learn `Pipeline` (`bank_common.py`, function `preprocessing`) fitted on training rows only: "
    "clipping to the 0.5% and 99.5% quantiles, median imputation and standardisation for the linear model; clipping only for the tree "
    "model (native missing-value handling). Demonstration: fitting the clip bounds on dev + test rows instead of dev rows changes them "
    f"(largest relative shift of an upper bound {shift:.2g}), which is exactly the leak the Pipeline avoids.",
    "- Rows with `y_before_public = 1` (the report was not public before the bank failed) are flagged here; their treatment in training and "
    "evaluation is decided in Part 5.",
     f"- Leakage tests: {sum(1 for n, ok, _ in tests if ok and 'WOULD change' not in n)} of {len(tests) - 1} safety tests passed "
    "(features of earlier rows do not change when later rows are removed, at two cut-offs; four-quarter dev/test gap; no label column "
    "among the features; no failing bank in both periods; test period ends before labels become incomplete; clip bounds equal to the "
    "development quantiles; unchanged after transforming test rows), plus one demonstration that fitting the clip bounds on dev + test "
    "rows would change them.",
]
(RES / "part4_readme_block.md").write_text("\n".join(md) + "\n", encoding="utf-8")

print("\n".join(out)[-3800:])
print("\nfull report: results/part4_features.txt | README block: results/part4_readme_block.md")
if not all_ok:
    raise SystemExit("a leakage test FAILED: do not model before fixing it")
