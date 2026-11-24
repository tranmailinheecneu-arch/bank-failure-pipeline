"""Part 5.4: development-period robustness analyses (no test row is read).

Uses EXACTLY the committed definitions: bank_models.make_model, bank_models.fold_masks and the specifications of
results/part5_selection.json. Reproduction check first: the out-of-fold Average Precision of B1 and of the selected model must
equal Table 12 of the README (4 decimals), otherwise the script stops.
A. implied kappa: the false-alarm cost ratio at which the cost-optimal alarm rate equals a supervisor's capacity
B. calibration: Brier score, predicted against observed rate per fold, isotonic recalibration fitted on earlier folds only
C. bank-level bootstrap of the Average Precision difference between the selected model and B1
D. label noise: model ranking against y and against the sensitivity label y_alt
Writes results/part5_dev_extras.txt and results/part5_dev_extras_readme_block.md.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

from bank_common import DEV, HORIZON, SEED, ql
from bank_models import fold_masks, make_model, purge_gap

REF = {"B1": [0.4505, 0.2249, 0.1108, 0.0875], "SEL": [0.4328, 0.3307, 0.3295, 0.1036]}   # Table 12 of the README
CAPS = [0.01, 0.02, 0.05, 0.10]
KAPPAS = [0.005, 0.01, 0.02, 0.05, 0.1]
NBOOT = 500
RES = Path("results")
out, md = [], []


def log(s=""):
    out.append(str(s))
    print(s, flush=True)


t0 = time.time()
sel = json.loads((RES / "part5_selection.json").read_text(encoding="utf-8"))
B0n, B1n = sel["baselines"]
SELN = sel["selected"]
names = [B0n, B1n, SELN]
need = sorted(set(sum([sel["models"][n]["cols"] for n in names], [])) | {"CERT", "qidx", "y", "y_alt", "y_before_public", "period", "log_asset"})
feat = pd.read_parquet("data_storage/part4/features.parquet", columns=need, filters=[("period", "==", "dev")]).reset_index(drop=True)
q = feat["qidx"].to_numpy()
y = feat["y"].to_numpy(float)
ya = feat["y_alt"].to_numpy(float)
cert = feat["CERT"].to_numpy()
assert not np.isnan(y).any() and q.max() <= DEV[1]
actionable = ~((y == 1) & (feat["y_before_public"].to_numpy() == 1))
folds = fold_masks(q, y)
assert min(purge_gap(q, tr, va) for _, tr, va in folds) >= HORIZON
log(f"development rows: {len(feat):,} ({ql(q.min())} to {ql(q.max())}); positive y {int(y.sum()):,}; positive y_alt {int(ya.sum()):,}")

OOF = {}
for n in names:
    spec = sel["models"][n]
    X = feat[spec["cols"]].to_numpy(float)
    o = np.full(len(feat), np.nan)
    for _, tr, va in folds:
        o[va] = make_model(spec["kind"], **spec["params"]).fit(X[tr], y[tr].astype(int)).predict_proba(X[va])[:, 1]
    OOF[n] = o
    print(f"fitted {n} ({time.time() - t0:.0f} s)", flush=True)


def fold_ap(o, label):
    return [average_precision_score(label[va & actionable].astype(int), o[va & actionable]) for _, _, va in folds]


log("\n== Reproduction check against Table 12 (fold AP of y) ==")
for n, key in [(B1n, "B1"), (SELN, "SEL")]:
    got = [round(a, 4) for a in fold_ap(OOF[n], y)]
    d = max(abs(a - b) for a, b in zip(got, REF[key]))
    log(f"{key}: reproduced {got} | Table 12 {REF[key]} | max abs difference {d:.4f}")
    if d > 2e-4:
        (RES / "part5_dev_extras.txt").write_text("\n".join(out), encoding="utf-8")
        raise SystemExit(f"{key} does not reproduce Table 12: the analyses below would describe another model. Stopped.")
log("reproduction OK: the models of this script are the committed models")

m_all = ~np.isnan(OOF[B1n]) & actionable
asset = np.exp(feat["log_asset"].to_numpy(float))
asset = np.where(np.isnan(asset), np.nanmedian(asset), asset)
cfn_all = sel["loss_rate"] * asset / HORIZON
mean_cfn = sel["mean_cfn"]

# ---------------------------------------------------------------- A. implied kappa
def curve(s, yy, cc):
    order = np.argsort(-s, kind="mergesort")
    ys, cs = yy[order], cc[order]
    miss = np.concatenate([[(ys * cs).sum()], (ys * cs).sum() - np.cumsum(ys * cs)])   # missed cost when the top k are flagged
    fa = np.concatenate([[0.0], np.cumsum(1 - ys)])                                      # false alarms among the top k
    return miss, fa


def rate_at(miss, fa, cfp, n):
    return int(np.argmin(miss + cfp * fa)) / n


grid = np.logspace(-4, 0.3, 90)
res_a, rows_a = {}, []
for n in (B1n, SELN):
    s, yy = OOF[n][m_all], y[m_all]
    for label, cc, unit in [("asset-weighted", cfn_all[m_all], mean_cfn), ("unit", np.ones(m_all.sum()), 1.0)]:
        miss, fa = curve(s, yy, cc)
        rates = np.array([rate_at(miss, fa, k * unit, len(s)) for k in grid])
        res_a[(n, label)] = (rates, [rate_at(miss, fa, k * unit, len(s)) for k in KAPPAS])
rows = []
for cap in CAPS:
    row = [f"{int(cap * 100)}%"]
    for n in (B1n, SELN):
        for label in ("asset-weighted", "unit"):
            rates = res_a[(n, label)][0]
            row.append(f"{float(np.exp(np.interp(cap, rates[::-1], np.log(grid)[::-1]))):.4f}")
    rows.append(row)
ta = pd.DataFrame(rows, columns=["capacity", "B1 asset-weighted", "B1 unit", "selected asset-weighted", "selected unit"])
log("\n== A. Implied kappa: kappa at which the cost-optimal alarm rate equals a supervisor capacity (pooled out-of-fold) ==")
log(ta.to_string(index=False))
ra = pd.DataFrame({"kappa": KAPPAS, "B1 alarm % (asset-weighted)": [round(100 * r, 2) for r in res_a[(B1n, 'asset-weighted')][1]],
                   "selected alarm % (asset-weighted)": [round(100 * r, 2) for r in res_a[(SELN, 'asset-weighted')][1]]})
log("optimal alarm rate by kappa (asset-weighted; should match the cost table of Part 5.1):")
log(ra.to_string(index=False))

# ---------------------------------------------------------------- B. calibration
def brier(p, yy):
    return float(np.mean((p - yy) ** 2))


log("\n== B. Calibration (Brier = mean squared error of the probability, lower is better) ==")
cal = []
for n, key in [(B1n, "B1"), (SELN, "selected")]:
    o = OOF[n]
    log(f"{key}: pooled Brier {brier(o[m_all], y[m_all]):.5f}")
    for j, (_, _, va) in enumerate(folds):
        mf = va & actionable
        line = f"  fold {j + 1}: mean predicted {o[mf].mean():.4f} | observed {y[mf].mean():.4f}"
        raw = brier(o[mf], y[mf])
        iso_b = np.nan
        if j >= 1:
            prev = np.zeros(len(feat), bool)
            for i in range(j):
                prev |= folds[i][2] & actionable
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(o[prev], y[prev])
            iso_b = brier(iso.predict(o[mf]), y[mf])
            line += f" | Brier raw {raw:.5f} -> isotonic fitted on folds < {j + 1}: {iso_b:.5f}"
        log(line)
        cal.append((key, j + 1, o[mf].mean(), y[mf].mean(), raw, iso_b))
tcal = pd.DataFrame(cal, columns=["model", "fold", "mean_pred", "observed", "brier_raw", "brier_isotonic"])

# ---------------------------------------------------------------- C. bank-level bootstrap
codes, uniq = pd.factorize(cert)
nb = len(uniq)
sets = {"pooled": m_all, **{f"f{j + 1}": va & actionable for j, (_, _, va) in enumerate(folds)}}
pre = {}
for n in (B1n, SELN):
    for k, msk in sets.items():
        idx = np.flatnonzero(msk)
        order = idx[np.argsort(-OOF[n][idx], kind="mergesort")]
        pre[(n, k)] = (y[order], codes[order])


def wap(ys, w):
    wy = w * ys
    tp = np.cumsum(wy)
    seen = np.cumsum(w)
    prec = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
    P = tp[-1]
    return float((prec * wy).sum() / P) if P > 0 else np.nan


def metrics(n, counts):
    p = wap(pre[(n, "pooled")][0], counts[pre[(n, "pooled")][1]].astype(float))
    f = [wap(pre[(n, f"f{j}")][0], counts[pre[(n, f"f{j}")][1]].astype(float)) for j in range(1, 5)]
    return p, float(np.mean(f))


ones = np.ones(nb)
obs = {n: metrics(n, ones) for n in (B1n, SELN)}
chk = abs(obs[B1n][0] - average_precision_score(y[m_all].astype(int), OOF[B1n][m_all]))
log("\n== C. Bank-level bootstrap (resample banks with replacement) ==")
log(f"own weighted AP against scikit-learn on B1 (pooled): difference {chk:.5f}")
rng = np.random.default_rng(SEED)
dp, dm = [], []
for _ in range(NBOOT):
    c = np.bincount(rng.integers(0, nb, nb), minlength=nb)
    a, b = metrics(SELN, c), metrics(B1n, c)
    dp.append(a[0] - b[0])
    dm.append(a[1] - b[1])
dp, dm = np.array(dp), np.array(dm)
def ci(v):
    return np.nanpercentile(v, [2.5, 97.5])
cp, cm = ci(dp), ci(dm)
log(f"  pooled AP: observed difference {obs[SELN][0] - obs[B1n][0]:+.4f} | 95% interval [{cp[0]:+.4f}, {cp[1]:+.4f}] | "
    f"share of resamples with selected > B1: {np.mean(dp > 0):.2f}")
log(f"  mean fold AP: observed difference {obs[SELN][1] - obs[B1n][1]:+.4f} | 95% interval [{cm[0]:+.4f}, {cm[1]:+.4f}] | "
    f"share of resamples with selected > B1: {np.mean(dm > 0):.2f}")
log(f"  ({NBOOT} resamples of {nb:,} banks; an interval that contains 0 means the larger model is not clearly better than B1)")

# ---------------------------------------------------------------- D. label noise
log("\n== D. Label noise: Average Precision against the main label y and the sensitivity label y_alt (models trained on y) ==")
rowsd = []
pooled = {}
for j, (_, _, va) in enumerate(folds):
    mf = va & actionable
    r = [j + 1, int(y[mf].sum()), int(ya[mf].sum())]
    for n in names:
        r.append(f"{average_precision_score(y[mf].astype(int), OOF[n][mf]):.3f} / {average_precision_score(ya[mf].astype(int), OOF[n][mf]):.3f}")
    rowsd.append(r)
r = ["pooled", int(y[m_all].sum()), int(ya[m_all].sum())]
for n in names:
    a1 = average_precision_score(y[m_all].astype(int), OOF[n][m_all])
    a2 = average_precision_score(ya[m_all].astype(int), OOF[n][m_all])
    pooled[n] = (a1, a2)
    r.append(f"{a1:.3f} / {a2:.3f}")
rowsd.append(r)
td = pd.DataFrame(rowsd, columns=["fold", "positives y", "positives y_alt", "B0 AP(y) / AP(y_alt)", "B1 AP(y) / AP(y_alt)", "selected AP(y) / AP(y_alt)"])
log(td.to_string(index=False))
order_y = sorted(names, key=lambda n: -pooled[n][0])
order_ya = sorted(names, key=lambda n: -pooled[n][1])
same = order_y == order_ya
log(f"pooled ranking by AP against y: {[n.split(' ')[0] for n in order_y]} | against y_alt: {[n.split(' ')[0] for n in order_ya]} | same order: {same}")
log(f"\nbuilt in {time.time() - t0:.0f} s")
(RES / "part5_dev_extras.txt").write_text("\n".join(out), encoding="utf-8")

# ---------------------------------------------------------------- README block
b1c = tcal[(tcal.model == "B1")]
sc = tcal[(tcal.model == "selected")]
md += [
    "### Part 5.4 development-period robustness analyses (`results/part5_dev_extras.txt`)",
    "",
    "Script `part5_dev_extras.py`; it reads only development rows and refits the committed models, and it stops unless the out-of-fold Average Precision of B1 and of the selected "
    "model reproduces Table 12 to four decimals (it does).",
    "",
    "- **False-alarm cost: implied kappa.** The cost of a false alarm cannot be observed, but the number of banks that supervisors can review can be assumed. "
    "The kappa at which the cost-optimal alarm rate equals a capacity of 1%, 2%, 5% and 10% of bank-quarters is (asset-weighted costs, as in Part 5.3): "
    + "; ".join(f"{r[0]}: B1 {r[1]}, selected {r[3]}" for r in rows) + ". "
    f"The main kappa of 0.02 corresponds to an alarm rate of {100 * res_a[(B1n, 'asset-weighted')][1][2]:.2f}% for B1 and {100 * res_a[(SELN, 'asset-weighted')][1][2]:.2f}% for the selected model. "
    "This replaces one assumption (kappa) by another that is easier to state (capacity); it does not make the cost observable.",
    f"- **Calibration.** B1 uses a balanced class weight, so its probabilities are scores: mean predicted rate {b1c.iloc[0].mean_pred:.3f} against observed {b1c.iloc[0].observed:.4f} in fold 1; "
    f"isotonic recalibration fitted on earlier folds lowers its Brier score from {b1c.iloc[1].brier_raw:.4f} to {b1c.iloc[1].brier_isotonic:.4f} (fold 2), from {b1c.iloc[2].brier_raw:.4f} to {b1c.iloc[2].brier_isotonic:.4f} (fold 3) and from {b1c.iloc[3].brier_raw:.5f} to {b1c.iloc[3].brier_isotonic:.5f} (fold 4). "
    f"The selected model is trained without class weight: its mean predicted rate is {sc.iloc[0].mean_pred:.4f}, {sc.iloc[1].mean_pred:.4f}, {sc.iloc[2].mean_pred:.4f} and {sc.iloc[3].mean_pred:.5f} against observed {sc.iloc[0].observed:.4f}, {sc.iloc[1].observed:.4f}, {sc.iloc[2].observed:.4f} and {sc.iloc[3].observed:.5f}; "
    "the gap between predicted and observed rate changes from fold to fold with the base rate of the training window, which is consistent with the instability of its score threshold (Part 5.3) but does not prove it.",
    f"- **Is the larger model clearly better?** Bank-level bootstrap ({NBOOT} resamples): selected minus B1 pooled AP {obs[SELN][0] - obs[B1n][0]:+.4f}, 95% interval [{cp[0]:+.4f}, {cp[1]:+.4f}]"
    f" ({'contains 0: not clearly better' if cp[0] <= 0 <= cp[1] else 'excludes 0'}); mean fold AP {obs[SELN][1] - obs[B1n][1]:+.4f}, 95% interval [{cm[0]:+.4f}, {cm[1]:+.4f}]"
    f" ({'contains 0' if cm[0] <= 0 <= cm[1] else 'excludes 0'}; the selection rule used this metric).",
    f"- **Label noise.** Against the sensitivity label y_alt (positives {int(ya[m_all].sum()):,} against {int(y[m_all].sum()):,}), the pooled ranking of the three models by AP is {[n.split(' ')[0] for n in order_ya]} "
    f"against {[n.split(' ')[0] for n in order_y]} for y ({'same order' if same else 'the order changes'}): the models were trained on y only, so this measures how much the evaluation depends on the label definition.",
    "",
    "| Fold | Positives y | Positives y_alt | B0 AP(y) / AP(y_alt) | B1 AP(y) / AP(y_alt) | Selected AP(y) / AP(y_alt) |", "|---|---|---|---|---|---|",
]
for r in rowsd:
    md.append("| " + " | ".join(str(c) for c in r) + " |")
(RES / "part5_dev_extras_readme_block.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\nresults/part5_dev_extras.txt and results/part5_dev_extras_readme_block.md written")