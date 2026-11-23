"""Part 5 extensions on the DEVELOPMENT period only (the test rows are never read).

Reads  : data_storage/part4/features.parquet (only rows up to 2006Q4), results/part5_selection.json
Writes : results/part5_dev_extras.txt, results/fig_part5_dev_calibration.png

Four analyses on out-of-fold (OOF) predictions of the same purged folds as Table 11:
  D  implied kappa: which false-alarm cost makes a given alarm capacity (1%, 2%, 5%, 10%) optimal
  E  calibration: Brier score, reliability curve, isotonic recalibration using earlier folds only
  F  bank-level bootstrap: 95% interval for AP(M2) - AP(B1)
  G  label noise: AP against y (main label) and y_alt (sensitivity label)
The models are refitted here with the committed settings; the first printed block compares the
reproduced fold APs with Table 12 so you can see whether the reproduction is faithful.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
N_BOOT = 200
B1_CLASS_WEIGHT = "balanced"  # if the reproduction check differs from Table 12, try None
M2_CLASS_WEIGHT = None  # idem
M2_EARLY_STOPPING = "auto"  # scikit-learn default; try False if M2 does not reproduce
RES = Path("results")
RES.mkdir(exist_ok=True)


def q(year, quarter):
    return year * 4 + (quarter - 1)


DEV_END = q(2006, 4)
# (train_from, train_to, valid_from, valid_to): the 4 quarters between train_to and valid_from are purged
FOLDS = [
    (q(1984, 1), q(1986, 4), q(1988, 1), q(1989, 4)),
    (q(1984, 1), q(1988, 4), q(1990, 1), q(1991, 4)),
    (q(1984, 1), q(1990, 4), q(1992, 1), q(1995, 4)),
    (q(1984, 1), q(1994, 4), q(1996, 1), q(2006, 4)),
]
EXPECT = {"B1": [0.4505, 0.2249, 0.1108, 0.0875], "M2": [0.4328, 0.3307, 0.3295, 0.1036]}  # Table 12


class Clip(BaseEstimator, TransformerMixin):
    """Clip to the 0.5% and 99.5% quantiles of the TRAINING rows."""

    def __init__(self, lo=0.5, hi=99.5):
        self.lo = lo
        self.hi = hi

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, self.lo, axis=0)
        self.hi_ = np.nanpercentile(X, self.hi, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def make_b1():
    return make_pipeline(Clip(), SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=1.0, max_iter=1000, class_weight=B1_CLASS_WEIGHT))


def make_m2(params):
    return make_pipeline(Clip(), HistGradientBoostingClassifier(
        max_iter=params["max_iter"], learning_rate=params["lr"], max_leaf_nodes=params["leaves"],
        class_weight=M2_CLASS_WEIGHT, early_stopping=M2_EARLY_STOPPING, random_state=SEED))


def load_dev():
    sel = json.loads(Path("results/part5_selection.json").read_text())
    cols = sorted(set(sel["models"][sel["selected"]]["cols"]) | set(sel["models"]["B1 logistic, 3 features"]["cols"]))
    need = ["CERT", "qidx", "y", "y_alt", "y_before_public"] + cols
    # the filter keeps the test rows out of memory: only qidx <= 2006Q4 is read
    df = pd.read_parquet("data_storage/part4/features.parquet", columns=need, filters=[("qidx", "<=", DEV_END)])
    return df, sel


def ap(y, s, w=None):
    return average_precision_score(y, s, sample_weight=w) if y.sum() > 0 else np.nan


def optimal_alarm_rate(score, y, kappa):
    """Alarm rate that minimises: 1 per missed positive row + kappa per false alarm."""
    order = np.argsort(-score, kind="stable")
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    cost = (ys.sum() - tp) + kappa * fp
    i = int(np.argmin(cost))
    return (i + 1) / len(y) if cost[i] < ys.sum() else 0.0


def run(df, sel):
    out = []

    def log(s=""):
        out.append(str(s))

    m2_name = sel["selected"]
    b1_cols = sel["models"]["B1 logistic, 3 features"]["cols"]
    m2_cols = sel["models"][m2_name]["cols"]
    m2_params = sel["models"][m2_name]["params"]

    df = df[df["qidx"] <= DEV_END]
    df = df[df["y"].notna()].copy()
    df["y"] = df["y"].astype(int)
    df["y_alt"] = df["y_alt"].fillna(0).astype(int)
    df["pub_bad"] = df["y_before_public"].fillna(False).astype(bool)
    log(f"development rows: {len(df):,} (qidx <= {DEV_END}) | positive y {int(df['y'].sum()):,} | positive y_alt {int(df['y_alt'].sum()):,}")

    parts = []
    for k, (a, b, c, d) in enumerate(FOLDS, 1):
        tr = df[(df.qidx >= a) & (df.qidx <= b)]
        va = df[(df.qidx >= c) & (df.qidx <= d) & (~df.pub_bad)].copy()  # actionable validation rows
        b1 = make_b1().fit(tr[b1_cols], tr["y"])
        m2 = make_m2(m2_params).fit(tr[m2_cols], tr["y"])
        va["p_b1"] = b1.predict_proba(va[b1_cols])[:, 1]
        va["p_m2"] = m2.predict_proba(va[m2_cols])[:, 1]
        va["p_b0"] = va["npl_loans"].fillna(-1.0) if "npl_loans" in va.columns else 0.0
        va["fold"] = k
        parts.append(va[["fold", "CERT", "y", "y_alt", "p_b0", "p_b1", "p_m2"]])
        print(f"fold {k} done", flush=True)
    ev = pd.concat(parts, ignore_index=True)

    # ---- reproduction check ----
    log("\n== Reproduction check: fold AP of y against Table 12 ==")
    for name, col in (("B1", "p_b1"), ("M2", "p_m2")):
        got = [ap(g["y"].values, g[col].values) for _, g in ev.groupby("fold")]
        log(f"{name}: reproduced " + " ".join(f"{x:.4f}" for x in got) + " | Table 12 " + " ".join(f"{x:.4f}" for x in EXPECT[name])
            + f" | max abs difference {max(abs(x - y) for x, y in zip(got, EXPECT[name])):.4f}")
    log("(a difference above 0.01 means the refit differs from the committed models: change B1_CLASS_WEIGHT / M2_CLASS_WEIGHT / M2_EARLY_STOPPING at the top, or send bank_common.py)")

    y = ev["y"].values
    # ---- D: implied kappa ----
    log("\n== D. Implied kappa (unit cost per missed positive row, kappa per false alarm; pooled OOF) ==")
    log("optimal alarm rate (%) by kappa:")
    for kappa in (0.005, 0.01, 0.02, 0.05, 0.1):
        r1 = optimal_alarm_rate(ev["p_b1"].values, y, kappa) * 100
        r2 = optimal_alarm_rate(ev["p_m2"].values, y, kappa) * 100
        log(f"  kappa {kappa:<6}: B1 {r1:5.2f} | M2 {r2:5.2f}")
    grid = np.geomspace(0.001, 0.5, 80)
    for name, col in (("B1", "p_b1"), ("M2", "p_m2")):
        rates = np.array([optimal_alarm_rate(ev[col].values, y, k) * 100 for k in grid])
        rates = np.minimum.accumulate(rates)  # rates cannot rise with kappa
        for cap in (1, 2, 5, 10):
            if rates.min() <= cap <= rates.max():
                ks = float(np.exp(np.interp(cap, rates[::-1], np.log(grid)[::-1])))
                log(f"  {name}: a supervisor capacity of {cap}% of bank-quarters is optimal at kappa = {ks:.4f}")
            else:
                log(f"  {name}: capacity {cap}% outside the grid (alarm rates {rates.min():.2f}-{rates.max():.2f}%)")
    log("note: unit costs ignore the asset weighting of Part 5.3, so the rates differ from Table C1; the point is the mapping capacity <-> kappa.")

    # ---- E: calibration ----
    log("\n== E. Calibration (Brier = mean squared error of the probability; lower is better) ==")
    base = ev.groupby("fold")["y"].mean()
    for name, col in (("B1", "p_b1"), ("M2", "p_m2")):
        log(f"{name}: pooled Brier {brier_score_loss(y, ev[col]):.5f} | per fold: mean predicted vs observed rate")
        for k, g in ev.groupby("fold"):
            log(f"   fold {k}: mean predicted {g[col].mean():.4f} | observed {base[k]:.4f}")
        # isotonic recalibration, fitted on earlier folds only (time order is respected)
        for k in (2, 3, 4):
            past = ev[ev.fold < k]
            cur = ev[ev.fold == k]
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(past[col], past["y"])
            raw = brier_score_loss(cur["y"], cur[col])
            cal = brier_score_loss(cur["y"], iso.predict(cur[col]))
            log(f"   fold {k}: Brier raw {raw:.5f} -> isotonic fitted on folds < {k}: {cal:.5f}")
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, (name, col) in zip(axes, (("B1 logistic, 3 features", "p_b1"), ("M2 gradient boosting", "p_m2"))):
        bins = pd.qcut(ev[col].rank(method="first"), 10, labels=False)
        t = ev.groupby(bins).agg(pred=(col, "mean"), obs=("y", "mean"))
        ax.plot(t["pred"], t["obs"], "o-", color="#0072B2")
        ax.plot([1e-4, 1], [1e-4, 1], "--", color="gray", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(name, loc="left", fontsize=10, fontweight="bold")
        ax.set_xlabel("mean predicted probability (decile)")
        ax.set_ylabel("observed positive rate")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Development period, out-of-fold: predicted probability against observed rate", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RES / "fig_part5_dev_calibration.png", dpi=200)

    # ---- F: bootstrap over banks ----
    log(f"\n== F. Bank-level bootstrap ({N_BOOT} resamples of banks; AP(M2) - AP(B1)) ==")
    codes, uniq = pd.factorize(ev["CERT"])
    nb = len(uniq)
    rng = np.random.default_rng(SEED)
    p1, p2, fold = ev["p_b1"].values, ev["p_m2"].values, ev["fold"].values
    d_pool, d_mean = [], []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb)[codes]
        m = w > 0
        d_pool.append(ap(y[m], p2[m], w[m]) - ap(y[m], p1[m], w[m]))
        fd = []
        for k in range(1, 5):
            mk = m & (fold == k)
            fd.append(ap(y[mk], p2[mk], w[mk]) - ap(y[mk], p1[mk], w[mk]))
        d_mean.append(np.nanmean(fd))
    obs_pool = ap(y, p2) - ap(y, p1)
    obs_mean = np.nanmean([ap(y[fold == k], p2[fold == k]) - ap(y[fold == k], p1[fold == k]) for k in range(1, 5)])
    for lab, obs, ds in (("pooled AP", obs_pool, d_pool), ("mean fold AP", obs_mean, d_mean)):
        lo, hi = np.percentile(ds, [2.5, 97.5])
        log(f"  {lab}: observed difference {obs:+.4f} | 95% interval [{lo:+.4f}, {hi:+.4f}] | share of resamples with M2 > B1: {np.mean(np.array(ds) > 0):.2f}")
    log("  an interval that contains 0 means the larger model is not clearly better than the three-feature baseline.")

    # ---- G: label noise ----
    log("\n== G. Label noise: AP against the main label y and the sensitivity label y_alt (development folds) ==")
    log("fold | positives y | positives y_alt | " + " | ".join(f"{m} AP(y) / AP(y_alt)" for m in ("B0", "B1", "M2")))
    for k, g in list(ev.groupby("fold")) + [("pooled", ev)]:
        cells = []
        for col in ("p_b0", "p_b1", "p_m2"):
            cells.append(f"{ap(g['y'].values, g[col].values):.3f} / {ap(g['y_alt'].values, g[col].values):.3f}")
        log(f"{k} | {int(g['y'].sum())} | {int(g['y_alt'].sum())} | " + " | ".join(cells))
    order_y = sorted(("p_b0", "p_b1", "p_m2"), key=lambda c: -ap(ev["y"].values, ev[c].values))
    order_a = sorted(("p_b0", "p_b1", "p_m2"), key=lambda c: -ap(ev["y_alt"].values, ev[c].values))
    log(f"pooled ranking of the models by AP: y {order_y} | y_alt {order_a} | same order: {order_y == order_a}")

    (RES / "part5_dev_extras.txt").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print("\nsaved results/part5_dev_extras.txt and results/fig_part5_dev_calibration.png")


if __name__ == "__main__":
    d, s = load_dev()
    run(d, s)
