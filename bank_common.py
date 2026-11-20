"""Definitions shared by Parts 4 and 5: period boundaries, feature list, leak-free preprocessing pipelines.

Everything that LEARNS from data (clip bounds, medians, scaling) lives inside scikit-learn Pipelines
and is fitted on training rows only. Models and validation folds are added in Part 5.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
HORIZON = 4


def Q(year, quarter):
	"""Quarter index, same formula as the panel (year * 4 + quarter - 1)."""
	return year * 4 + (quarter - 1)


def ql(q):
	return f"{int(q) // 4}Q{int(q) % 4 + 1}"


DEV = (Q(1984, 1), Q(2006, 4))
GAP = (Q(2007, 1), Q(2007, 4))   # four quarters between dev and test: a label looks four quarters ahead
TEST = (Q(2008, 1), Q(2014, 4))

GROUPS = {
	"capital": ["eq_assets", "d_eq_assets_4q", "eq_assets_med_q", "eq_assets_pct_q"],
	"asset quality": ["npl_loans", "reserves_loans", "d_npl_loans_4q", "npl_loans_med_q", "ore_assets"],
	"earnings": ["roaq", "roa_q_mean4", "roa_q_std4", "nimy"],
	"liquidity and funding": ["liquid_assets", "bro_dep", "dep_assets"],
	"loan book": ["loans_assets", "lnre_assets", "lnci_assets", "lncon_assets"],
	"growth": ["asset_yoy", "loans_yoy", "dep_yoy"],
	"size": ["log_asset"],
}
FEATURES_MODEL = [f for g in GROUPS.values() for f in g]      # 24 features used by the models
FEATURES_ALL = FEATURES_MODEL + ["roa_q_n4"]                  # built, but roa_q_n4 is not used
FORBIDDEN = ["y", "y_alt", "y_before_public", "avail_date", "ACTIVE", "qidx", "CERT", "REPDTE", "period"]


def dev_mask(q, y):
	return (q >= DEV[0]) & (q <= DEV[1]) & ~np.isnan(y)


def test_mask(q, y):
	return (q >= TEST[0]) & (q <= TEST[1]) & ~np.isnan(y)


class QuantileClipper(BaseEstimator, TransformerMixin):
	"""Clip every column to [lower, upper] quantiles LEARNED ON THE TRAINING ROWS (NaN stays NaN)."""

	def __init__(self, lower=0.005, upper=0.995):
		self.lower = lower
		self.upper = upper

	def fit(self, X, y=None):
		X = np.asarray(X, dtype=float)
		self.lo_ = np.nanquantile(X, self.lower, axis=0)
		self.hi_ = np.nanquantile(X, self.upper, axis=0)
		return self

	def transform(self, X):
		return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def preprocessing(kind="linear"):
	"""Leak-free preprocessing. linear: clip, median imputation, standardisation. tree: clip only
	(the tree model handles missing values natively)."""
	steps = [("clip", QuantileClipper())]
	if kind == "linear":
		steps += [("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
				  ("scale", StandardScaler())]
	return Pipeline(steps)
