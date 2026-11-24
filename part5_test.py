"""Part 5.2: evaluation on the TEST period, used exactly once.

Order of the checks (nothing is read from the test rows before all of them pass):
  1. the lock file must not exist (a second run is refused);
  2. results/part5_selection.json must contain every key and must be COMMITTED in git (no pending change);
  3. only then the lock is written (time, hash of the selection file, git commit) and the test rows are read.
Models are refitted on all development rows; thresholds and alarm rates come from the development period.
Writes: results/part5_test.txt, results/part5_test_readme_block.md
"""
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from bank_common import HORIZON, ql
from bank_models import bank_recall, cost_at, eval_metrics, make_model

RES = Path("results")
SEL = RES / "part5_selection.json"
LOCK = RES / "part5_test_LOCK.json"

if LOCK.exists():
    raise SystemExit("The test period was already used (results/part5_test_LOCK.json exists). "
                     "Running it again would be test reuse: refusing.")
sel = json.loads(SEL.read_text(encoding="utf-8"))
need = ["selected", "baselines", "kappas", "loss_rate", "mean_cfn", "models", "thresholds", "alarm_rates_pct",
        "dev_median_asset"]
missing = [k for k in need if k not in sel]
if missing:
    raise SystemExit(f"results/part5_selection.json lacks {missing}: rerun part5_validation.py "
                     "(the test period has NOT been opened)")


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


try:
    tracked = git("ls-files", "--error-unmatch", SEL.as_posix())
    dirty = git("status", "--porcelain", "--", SEL.as_posix())
    head = git("rev-parse", "HEAD").stdout.strip()
except FileNotFoundError:
    raise SystemExit("git was not found: cannot verify that the selection file is committed (test NOT opened)")
if tracked.returncode != 0 or dirty.stdout.strip():
    raise SystemExit("results/part5_selection.json is not committed (or has pending changes). Commit it first; "
                     "the test period has NOT been opened")

feat = pd.read_parquet("data_storage/part4/features.parquet",
                       filters=[("period", "in", ["dev", "test"])]).reset_index(drop=True)
dev = (feat["period"] == "dev").to_numpy()
tst = (feat["period"] == "test").to_numpy()
y = feat["y"].to_numpy(float)
assert not np.isnan(y[dev | tst]).any()
q = feat["qidx"].to_numpy()
cert = feat["CERT"].to_numpy()
actionable = ~((y == 1) & (feat["y_before_public"].to_numpy() == 1))
asset = np.exp(feat["log_asset"].to_numpy(float))
asset = np.where(np.isnan(asset), sel["dev_median_asset"], asset)
cfn = sel["loss_rate"] * asset / HORIZON
SELN = sel["selected"]
names = sel["baselines"] + [SELN]
B1 = sel["baselines"][1]
out = []


def log(s=""):
    out.append(str(s))


# all checks passed: from here the test period is considered opened
LOCK.write_text(json.dumps({"opened": datetime.now().isoformat(timespec="seconds"),
                            "selection_sha256": hashlib.sha256(SEL.read_bytes()).hexdigest(),
                            "git_head": head}, indent=1))

log(f"test period {ql(q[tst].min())} to {ql(q[tst].max())}: {int(tst.sum()):,} rows, {int((y[tst] == 1).sum()):,} positive rows "
    f"({int((y[tst & actionable] == 1).sum()):,} actionable) from {len(set(cert[tst & (y == 1)])):,} failing banks")
log(f"models refitted on all development rows ({int(dev.sum()):,} rows); selected model: {SELN}; git commit {head[:10]}")

scores = {}
for n in names:
    spec = sel["models"][n]
    X = feat[spec["cols"]].to_numpy(float)
    m = make_model(spec["kind"], **spec["params"]).fit(X[dev], y[dev].astype(int))
    scores[n] = m.predict_proba(X[tst])[:, 1]
    print(f"fitted {n}", flush=True)

yt, ct, ac, cft, qt = y[tst], cert[tst], actionable[tst], cfn[tst], q[tst]
ya = feat["y_alt"].to_numpy(float)[tst]
rows = []
for n in names:
    a = eval_metrics(yt[ac], scores[n][ac])
    b = eval_metrics(yt, scores[n])
    c = eval_metrics(ya, scores[n])
    rows.append((n, round(a["AP"], 4), round(a["AUC"], 4), round(a["recall@1%"], 3), round(a["recall@5%"], 3),
                 round(a["lift@5%"], 1), round(b["AP"], 4), round(c["AP"], 4)))
mt = pd.DataFrame(rows, columns=["model", "AP", "AUC", "recall@1%", "recall@5%", "lift@5%", "AP_all_rows", "AP_vs_y_alt"])
log(f"\n== Test results (actionable rows; chance level of AP = base rate {100 * yt[ac].mean():.3f}%) ==")
log(mt.to_string(index=False))

# ---- rule 1: score threshold fixed on the development period
crow = []
for kappa in sel["kappas"]:
    cfp = kappa * sel["mean_cfn"]
    for n in (B1, SELN):
        v = sel["thresholds"][n][str(kappa)]
        t = np.inf if v is None else v
        s, yy, cc, cr = scores[n][ac], yt[ac], cft[ac], ct[ac]
        cost, fn, fp, alarm = cost_at(s, yy, cc, cfp, t)
        no_alarm = float((yy * cc).sum())
        br, _ = bank_recall(cr, yy, alarm)
        crow.append((kappa, n, round(100 * alarm.mean(), 2), round(float(alarm[yy == 1].mean()), 3), round(br, 3),
                     round(1 - cost / no_alarm, 3), round(cost / 1e6, 2), round(no_alarm / 1e6, 2)))
ctab = pd.DataFrame(crow, columns=["kappa", "model", "alarm_rate_%", "recall_rows", "recall_banks",
                                   "cost_saving_vs_no_alarm", "cost_bn_USD", "no_alarm_cost_bn_USD"])
log("\n== Rule 1, score threshold from the development period: test cost (kappa 0.02 is the main case) ==")
log(ctab.to_string(index=False))

# ---- rule 2: capacity rule, flag the top r% of banks of EACH quarter (r = development cost-optimal alarm rate)
rates = sel["alarm_rates_pct"]
caprows = []
for kappa in sel["kappas"]:
    cfp = kappa * sel["mean_cfn"]
    for n in (B1, SELN):
        r = rates[n][str(kappa)] / 100
        rk = pd.Series(scores[n]).groupby(qt).rank(ascending=False, pct=True, method="first").to_numpy()
        a, yy, cc, cr = (rk <= r)[ac], yt[ac], cft[ac], ct[ac]
        fn = float(cc[(yy == 1) & ~a].sum())
        fp = int(((yy == 0) & a).sum())
        no_alarm = float((yy * cc).sum())
        cost = fn + fp * cfp
        br, _ = bank_recall(cr, yy, a)
        caprows.append((kappa, n, round(100 * a.mean(), 2), round(float(a[yy == 1].mean()), 3), round(br, 3),
                        round(1 - cost / no_alarm, 3), round(cost / 1e6, 2)))
captab = pd.DataFrame(caprows, columns=["kappa", "model", "alarm_rate_%", "recall_rows", "recall_banks",
                                        "cost_saving_vs_no_alarm", "cost_bn_USD"])
log("\n== Rule 2, capacity rule (top r% of banks flagged in each quarter): test cost ==")
log(captab.to_string(index=False))

# ---- Average Precision of the selected model by test year
yr = pd.DataFrame({"year": qt // 4, "y": yt, "s": scores[SELN], "ac": ac})
yrows = []
for yv, g in yr[yr.ac].groupby("year"):
    mm = eval_metrics(g.y.to_numpy(), g.s.to_numpy())
    yrows.append((int(yv), mm["rows"], mm["pos"], round(mm["AP"], 4) if mm["pos"] else np.nan, round(mm["base_rate"] * 100, 3)))
ytab = pd.DataFrame(yrows, columns=["year", "rows", "positive_rows", "AP", "base_rate_%"])
log(f"\n== {SELN}: Average Precision by test year ==")
log(ytab.to_string(index=False))
(RES / "part5_test.txt").write_text("\n".join(out), encoding="utf-8")

md = [
    "### Part 5.2 test evaluation, used once (`results/part5_test.txt`)",
    "",
    f"The test period ({ql(q[tst].min())} to {ql(q[tst].max())}) was opened exactly once, after `results/part5_selection.json` was committed; "
    "`results/part5_test_LOCK.json` records the time, the hash of that file and the git commit, and the script refuses to run a second time "
    "or when the selection file is not committed. All models are refitted on all development rows; thresholds and alarm rates come from the "
    "development period. Evaluation uses the actionable rows (reports that were public before the failure).",
    "",
    "| Model | AP | ROC-AUC | Recall @1% | Recall @5% | Lift @5% | AP, all rows | AP vs `y_alt` |",
    "|---|---|---|---|---|---|---|---|",
]
for r in mt.itertuples():
    md.append(f"| {r.model} | {r.AP} | {r.AUC} | {r._4} | {r._5} | {r._6} | {r.AP_all_rows} | {r.AP_vs_y_alt} |")
md += ["", "Rule 1 (score threshold from the development period):", "",
       "| kappa | Model | Alarm rate % | Recall (bank-quarters) | Recall (failing banks) | Cost saving vs no alarm | Cost (bn USD) | No-alarm cost (bn USD) |",
       "|---|---|---|---|---|---|---|---|"]
for r in ctab.itertuples():
    md.append(f"| {r.kappa} | {r.model} | {r._3} | {r.recall_rows} | {r.recall_banks} | {r.cost_saving_vs_no_alarm} | {r.cost_bn_USD} | {r.no_alarm_cost_bn_USD} |")
md += ["", "Rule 2 (capacity rule: top r% of banks flagged in each quarter, r from the development optimum):", "",
       "| kappa | Model | Alarm rate % | Recall (bank-quarters) | Recall (failing banks) | Cost saving vs no alarm | Cost (bn USD) |",
       "|---|---|---|---|---|---|---|"]
for r in captab.itertuples():
    md.append(f"| {r.kappa} | {r.model} | {r._3} | {r.recall_rows} | {r.recall_banks} | {r.cost_saving_vs_no_alarm} | {r.cost_bn_USD} |")
md += ["", f"Average Precision of {SELN} by test year:", "", "| Year | Rows | Positive rows | AP | Base rate % |", "|---|---|---|---|---|"]
for r in ytab.itertuples():
    md.append(f"| {r.year} | {r.rows:,} | {r.positive_rows:,} | {r.AP} | {r._5} |")
(RES / "part5_test_readme_block.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(out)[-3500:])
print("\nresults/part5_test.txt written; the test period is now used and locked")