"""Part 5 definitions: purged validation folds, models, metrics and the cost-based alarm threshold."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from bank_common import SEED, Q, ql, preprocessing

BASELINE_FEATURES = ["eq_assets", "npl_loans", "roaq"]
KAPPA_MAIN = 0.02                                  # false-alarm cost = kappa x average cost of a missed failure
KAPPAS = [0.005, 0.01, 0.02, 0.05, 0.1]
RATES = (0.01, 0.05)                               # alarm capacity: share of bank-quarters flagged

# (last training quarter, first validation quarter, last validation quarter): 4 quarters purged in between
FOLDS = [
	(Q(1986, 4), Q(1988, 1), Q(1989, 4)),
	(Q(1988, 4), Q(1990, 1), Q(1991, 4)),
	(Q(1990, 4), Q(1992, 1), Q(1995, 4)),
	(Q(1994, 4), Q(1996, 1), Q(2006, 4)),
]


def fold_masks(q, y):
	lab = ~np.isnan(y)
	return [(f"fold{i}", lab & (q <= te), lab & (q >= vs) & (q <= ve)) for i, (te, vs, ve) in enumerate(FOLDS, 1)]


def purge_gap(q, tr, va):
	"""Quarters strictly between the last training quarter and the first validation quarter."""
	return int(q[va].min()) - int(q[tr].max()) - 1


class SingleFeature:
	"""Baseline B0: rank by one feature (higher = riskier); missing values get the training median."""

	def fit(self, X, y=None):
		self.med_ = float(np.nanmedian(X[:, 0]))
		return self

	def predict_proba(self, X):
		s = np.where(np.isnan(X[:, 0]), self.med_, X[:, 0])
		return np.column_stack([-s, s])


def make_model(kind, **p):
	if kind == "single":
		return SingleFeature()
	if kind == "logit":
		clf = LogisticRegression(C=p.get("C", 1.0), class_weight="balanced", max_iter=1000, random_state=SEED)
		return Pipeline([("prep", preprocessing("linear")), ("clf", clf)])
	if kind == "gbm":
		clf = HistGradientBoostingClassifier(
			max_iter=p.get("max_iter", 200), learning_rate=p.get("lr", 0.1), max_leaf_nodes=p.get("leaves", 15),
			min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=SEED)
		return Pipeline([("prep", preprocessing("tree")), ("clf", clf)])
	raise ValueError(kind)


def eval_metrics(y, s):
	"""Average Precision, ROC-AUC and capacity metrics (recall, precision, lift when the top 1% / 5% rows are flagged)."""
	y = np.asarray(y).astype(int)
	out = {"rows": len(y), "pos": int(y.sum()), "base_rate": float(y.mean()) if len(y) else np.nan}
	if y.sum() == 0 or y.sum() == len(y):
		return {**out, "AP": np.nan, "AUC": np.nan, **{f"{m}@{int(r * 100)}%": np.nan for r in RATES for m in ("recall", "precision", "lift")}}
	out["AP"] = float(average_precision_score(y, s))
	out["AUC"] = float(roc_auc_score(y, s))
	order = np.argsort(-s, kind="mergesort")
	for r in RATES:
		k = max(int(round(r * len(y))), 1)
		caught = int(y[order[:k]].sum())
		out[f"recall@{int(r * 100)}%"] = caught / y.sum()
		out[f"precision@{int(r * 100)}%"] = caught / k
		out[f"lift@{int(r * 100)}%"] = (caught / k) / out["base_rate"]
	return out


def bank_recall(cert, y, alarm):
	"""Share of failing banks with at least one alarm during the quarters that precede their failure."""
	d = pd.DataFrame({"c": cert, "pos": np.asarray(y) == 1, "hit": (np.asarray(y) == 1) & np.asarray(alarm)})
	g = d[d.pos].groupby("c")["hit"].any()
	return (float(g.mean()) if len(g) else float("nan")), int(len(g))


def cost_at(s, y, cfn, cfp, t):
	"""Cost of raising an alarm when score >= t: missed-failure cost of unflagged positives + cfp per false alarm."""
	alarm = s >= t
	fn = float(cfn[(y == 1) & ~alarm].sum())
	fp = int(((y == 0) & alarm).sum())
	return fn + fp * cfp, fn, fp, alarm


def best_threshold(s, y, cfn, cfp):
	"""Threshold minimising the total cost over all possible cut-offs (alarm = score >= t). inf = never alarm."""
	order = np.argsort(-s, kind="mergesort")
	ys, cs = y[order], cfn[order]
	total_fn = float((y * cfn).sum())
	cost = np.concatenate([[total_fn], (total_fn - np.cumsum(ys * cs)) + np.cumsum(1 - ys) * cfp])
	k = int(np.argmin(cost))
	return (float(s[order][k - 1]) if k > 0 else float("inf")), float(cost[k])
