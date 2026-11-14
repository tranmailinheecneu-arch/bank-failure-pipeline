"""Part 2.3: clean, point-in-time panel with the four-quarter-ahead failure label.

Reads  : data_storage/panel/*/*.parquet           (Part 1 output)
         data_storage/exits/bank_exits.parquet    (Part 2.2 output)
Writes : data_storage/panel_clean/YEAR=*/part.parquet   clean panel with labels
         results/part2_clean_panel.txt                  full audit (numbers for the report)
         results/part2_clean_readme_block.md            ready-to-paste README block

Label  : y(i,q) = 1 if bank i FAILS in (q, q + 4 quarters]; only RESTYPE = FAILURE counts.
         Mergers and other exits are y = 0. y is null for quarters after 2025Q2 (incomplete labels).
Extras : y_alt (sensitivity: also open-bank assistance and failure-like change codes),
         avail_date (quarter end + 35 days, assumed publication lag),
         y_before_public (y = 1 but the bank failed before the report was public).
Rules  : rows of failed banks from the failure quarter on are dropped (they would not exist in real time);
         2 banks with FAILDATE before their first report are excluded;
         an exact 0 in RBCRWAJ and RBC1AAJ is a "not reported" code (whole eras, a fixed set of banks):
         set to null in every quarter and flagged (flag_rbc_zero);
         NOTHING is imputed or clipped here: those statistics are learned on training data only (Part 4).
Columns y, y_alt, y_before_public and avail_date are label/metadata columns: never use them as features.
"""
import shutil
import time
from pathlib import Path

import polars as pl

PANEL = "data_storage/panel/*/*.parquet"
EXITS = "data_storage/exits/bank_exits.parquet"
OUT = Path("data_storage/panel_clean")
RES = Path("results")
RES.mkdir(exist_ok=True)

HORIZON = 4
LAST_LABEL_Q = 2025 * 4 + 1  # 2025Q2: last quarter whose 4-quarter-ahead label is complete
LAG_DAYS = 35  # assumption (same as Part 2.2): report public about 35 days after quarter end
ZERO_COLS = ["RBCRWAJ", "RBC1AAJ"]
NEG_COLS = ["ORE", "SC", "LNLSNET", "LNLSGR", "INTAN", "INTINC", "LNATRES", "NCLNLS"]
FAIL_CLASSES = ["failure", "reported_after_failure_date"]
BAD_CLASS = "failure_date_before_first_report"
LEAK_COLS = ["exit_class", "fail_q", "last_q", "exit_bucket", "had_assistance", "FAILDATE_d"]

out = []


def log(s=""):
    out.append(str(s))


def show(df):
    return df.to_pandas().to_string(index=False)


def ql(q):
    return f"{q // 4}Q{q % 4 + 1}"


t0 = time.time()

# ---------------------------------------------------------------------------
# 1. Load panel and exit classes
# ---------------------------------------------------------------------------
raw = pl.read_parquet(PANEL)
if "YEAR" in raw.columns:
    raw = raw.drop("YEAR")
raw = raw.with_columns(
    pl.col("CERT").cast(pl.Int64),
    pl.col("REPDTE").cast(pl.Utf8),
).with_columns(
    (
        pl.col("REPDTE").str.slice(0, 4).cast(pl.Int32) * 4
        + (pl.col("REPDTE").str.slice(4, 2).cast(pl.Int32) - 1) // 3
    ).alias("qidx"),
    pl.col("REPDTE").str.strptime(pl.Date, "%Y%m%d").alias("rep_date"),
)
n_raw = raw.height
log(f"panel rows read: {n_raw:,} | banks: {raw['CERT'].n_unique():,} | columns: {raw.width}")

need = ["CERT", "exit_class", "fail_q", "last_q", "exit_bucket", "had_assistance", "FAILDATE_d"]
ex_all = pl.read_parquet(EXITS)
missing = [c for c in need if c not in ex_all.columns]
if missing:
    raise SystemExit(f"{EXITS} lacks columns {missing}: rerun part2_exits.py first")
ex = ex_all.select(need).with_columns(pl.col("CERT").cast(pl.Int64))

df = raw.join(ex, on="CERT", how="left")
if df.filter(pl.col("exit_class").is_null()).height:
    raise SystemExit("some panel banks have no exit class: rerun part2_exits.py")

is_fail = pl.col("exit_class").is_in(FAIL_CLASSES)

# ---------------------------------------------------------------------------
# 2. Exclusions and cut at the failure quarter
# ---------------------------------------------------------------------------
bad = df.filter(pl.col("exit_class") == BAD_CLASS)
n_bad_rows, n_bad_banks = bad.height, bad["CERT"].n_unique()
df = df.filter(pl.col("exit_class") != BAD_CLASS)

post_mask = is_fail & (pl.col("qidx") >= pl.col("fail_q"))
post = df.filter(post_mask)
n_post_rows, n_post_banks = post.height, post["CERT"].n_unique()
df = df.filter(~post_mask)
log("\n== Exclusions ==")
log(f"banks with FAILDATE before their first report: {n_bad_banks} banks, {n_bad_rows:,} rows excluded")
log(f"rows of failed banks dated in or after the failure quarter: {n_post_rows:,} rows of {n_post_banks:,} banks dropped")

# ---------------------------------------------------------------------------
# 3. Data-quality handling that needs no statistics learned from the data distribution
# ---------------------------------------------------------------------------
log("\n== Exact zeros in risk-based capital ratios (a 'not reported' code) ==")
df = df.with_columns(
    pl.sum_horizontal([(pl.col(c) == 0).fill_null(False).cast(pl.Int8) for c in ZERO_COLS])
    .cast(pl.Int8).alias("flag_rbc_zero")
)
n_flag_rbc = df.filter(pl.col("flag_rbc_zero") > 0).height
zero_rows = []
for c in ZERO_COLS:
    zmask = pl.col(c) == 0
    zsub = df.filter(zmask)
    n_zero = zsub.height
    n_zero_banks = zsub["CERT"].n_unique()
    first_q = ql(int(zsub["qidx"].min())) if n_zero else "-"
    last_q_ = ql(int(zsub["qidx"].max())) if n_zero else "-"
    df = df.with_columns(pl.when(zmask).then(None).otherwise(pl.col(c)).alias(c))
    zero_rows.append((c, n_zero, n_zero_banks, first_q, last_q_))
zero_df = pl.DataFrame(
    zero_rows, orient="row",
    schema=["column", "zeros_set_to_null", "banks_affected", "first_quarter", "last_quarter"],
)
log(show(zero_df))
log(f"rows with at least one such zero (flag_rbc_zero > 0): {n_flag_rbc:,}")
log("rule: an exact 0 in these ratios is treated as 'not reported' in every quarter "
    "(evidence by year in results/part2_zero_pattern.txt)")

neg_cols = [c for c in NEG_COLS if c in df.columns]
neg_counts = {c: df.filter(pl.col(c) < 0).height for c in neg_cols}
df = df.with_columns(
    pl.sum_horizontal([(pl.col(c) < 0).fill_null(False).cast(pl.Int8) for c in neg_cols])
    .cast(pl.Int8).alias("flag_neg_fixed")
).with_columns([pl.when(pl.col(c) < 0).then(None).otherwise(pl.col(c)).alias(c) for c in neg_cols])
n_neg_rows = df.filter(pl.col("flag_neg_fixed") > 0).height
log("\n== Negative values in items that cannot be negative (set to null, flagged) ==")
log(", ".join(f"{c}: {n:,}" for c, n in neg_counts.items()))
log(f"rows with at least one such value: {n_neg_rows:,}")

df = df.sort(["CERT", "qidx"]).with_columns(
    (pl.col("DEP") == 0).fill_null(False).cast(pl.Int8).alias("flag_dep_zero"),
    ((pl.col("qidx") - pl.col("qidx").shift(1).over("CERT")) > 1).fill_null(False).cast(pl.Int8).alias("gap_before"),
)
n_dep0 = int(df["flag_dep_zero"].sum())
n_gap_rows = int(df["gap_before"].sum())
n_gap_banks = df.filter(pl.col("gap_before") == 1)["CERT"].n_unique()
log(f"\nrows with DEP = 0 (flag_dep_zero): {n_dep0:,}")
log(f"rows that follow a missing quarter (gap_before): {n_gap_rows:,} in {n_gap_banks:,} banks")

# ---------------------------------------------------------------------------
# 4. Labels and availability timestamp
# ---------------------------------------------------------------------------
dq = pl.col("fail_q") - pl.col("qidx")
labelled = pl.col("qidx") <= LAST_LABEL_Q
y_raw = (is_fail & dq.is_between(1, HORIZON)).fill_null(False)
alt_q = (
    pl.when(is_fail).then(pl.col("fail_q"))
    .when(pl.col("had_assistance").fill_null(False)).then(pl.col("fail_q"))
    .when((pl.col("exit_class") == "other_exit") & (pl.col("exit_bucket") == "failure_like"))
    .then(pl.col("last_q") + 1)
    .otherwise(None)
)
y_alt_raw = (alt_q - pl.col("qidx")).is_between(1, HORIZON).fill_null(False)
df = df.with_columns(
    pl.when(labelled).then(y_raw.cast(pl.Int8)).otherwise(None).alias("y"),
    pl.when(labelled).then(y_alt_raw.cast(pl.Int8)).otherwise(None).alias("y_alt"),
    pl.col("rep_date").dt.offset_by(f"{LAG_DAYS}d").alias("avail_date"),
).with_columns(
    ((pl.col("y") == 1) & (pl.col("FAILDATE_d") <= pl.col("avail_date")))
    .fill_null(False).cast(pl.Int8).alias("y_before_public")
)

# ---------------------------------------------------------------------------
# 5. Write the clean panel (label helper columns that reveal the future are not written)
# ---------------------------------------------------------------------------
clean = df.drop(LEAK_COLS).with_columns((pl.col("qidx") // 4).alias("_year"))
if OUT.exists():
    shutil.rmtree(OUT)
for (yv,) in clean.select("_year").unique().sort("_year").iter_rows():
    d = OUT / f"YEAR={yv}"
    d.mkdir(parents=True)
    clean.filter(pl.col("_year") == yv).drop("_year").write_parquet(d / "part.parquet", compression="zstd")
size_mb = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file()) / 1e6

# ---------------------------------------------------------------------------
# 6. Verification on the written data (independent of the code that built the labels)
# ---------------------------------------------------------------------------
chk = pl.read_parquet(str(OUT / "*/*.parquet"))
if "YEAR" in chk.columns:
    chk = chk.drop("YEAR")
chk_ex = chk.join(ex.select("CERT", "exit_class", "fail_q"), on="CERT", how="left")

expected_rows = n_raw - n_bad_rows - n_post_rows
expected_pos = (
    raw.select("CERT", "qidx")
    .join(ex.filter(pl.col("exit_class").is_in(FAIL_CLASSES)).select("CERT", "fail_q"), on="CERT")
    .filter(
        (pl.col("qidx") >= pl.col("fail_q") - HORIZON)
        & (pl.col("qidx") < pl.col("fail_q"))
        & (pl.col("qidx") <= LAST_LABEL_Q)
    ).height
)
got_pos = int(chk["y"].sum())
checks = [
    ("row count equals raw rows minus exclusions", chk.height == expected_rows, f"{chk.height:,} vs {expected_rows:,}"),
    ("no duplicate (CERT, REPDTE)", chk.height - chk.select(["CERT", "REPDTE"]).unique().height == 0, ""),
    ("y is null exactly after 2025Q2",
     chk.filter(pl.col("y").is_null() != (pl.col("qidx") > LAST_LABEL_Q)).height == 0, ""),
    ("positives equal an independent recount", got_pos == expected_pos, f"{got_pos:,} vs {expected_pos:,}"),
    ("no positive label for banks that did not fail",
     chk_ex.filter((pl.col("y") == 1) & ~pl.col("exit_class").is_in(FAIL_CLASSES)).height == 0, ""),
    ("no row dated in or after the failure quarter",
     chk_ex.filter(pl.col("exit_class").is_in(FAIL_CLASSES) & (pl.col("qidx") >= pl.col("fail_q"))).height == 0, ""),
    ("no future-revealing columns written", not any(c in chk.columns for c in LEAK_COLS), ""),
    ("no exact zero left in the capital ratios",
     all(chk.filter(pl.col(c) == 0).height == 0 for c in ZERO_COLS), ""),
]
log("\n== Verification checks ==")
all_ok = True
for name, ok, detail in checks:
    all_ok &= bool(ok)
    log(f"{'PASS' if ok else 'FAIL'}  {name} {detail}".rstrip())

# ---------------------------------------------------------------------------
# 7. Label summary
# ---------------------------------------------------------------------------
lab = chk.filter(pl.col("y").is_not_null())
n_lab = lab.height
n_pos = int(lab["y"].sum())
pos = lab.filter(pl.col("y") == 1)
pos_banks = pos["CERT"].n_unique()
n_fail_banks = ex.filter(pl.col("exit_class").is_in(FAIL_CLASSES)).height - 0
n_alt = int(lab["y_alt"].sum())
alt_banks = lab.filter(pl.col("y_alt") == 1)["CERT"].n_unique()
n_pre = int(pos["y_before_public"].sum())
n_unlab = chk.height - n_lab

log("\n== Label summary ==")
log(f"clean panel: {chk.height:,} rows, {chk['CERT'].n_unique():,} banks, {chk.width} columns, {size_mb:.1f} MB (zstd, by year)")
log(f"labelled rows (quarter <= 2025Q2): {n_lab:,} | unlabelled rows (later quarters, y = null): {n_unlab:,}")
log(f"y = 1 rows: {n_pos:,} ({100 * n_pos / n_lab:.3f}% of labelled rows) in {pos_banks:,} failed banks")
log(f"banks with a FAILURE event in the panel: {n_fail_banks:,}; without any y = 1 row: {n_fail_banks - pos_banks:,} "
    "(failure dated after the labelled window: July 2026, after the last panel quarter)")
log(f"y_alt = 1 rows: {n_alt:,} in {alt_banks:,} banks (sensitivity label, assumptions in the script header)")
log(f"y = 1 rows where the bank failed before the report was public ({LAG_DAYS}-day lag): {n_pre:,} "
    f"({100 * n_pre / max(n_pos, 1):.1f}% of positives)")

by_year = (
    lab.with_columns((pl.col("qidx") // 4).alias("year"))
    .group_by("year")
    .agg(pl.len().alias("rows"), pl.col("y").sum().alias("y1_rows"), pl.col("y_alt").sum().alias("y_alt1_rows"))
    .sort("year")
    .with_columns((pl.col("y1_rows") / pl.col("rows") * 100).round(3).alias("y1_pct"))
)
log("\n== Labelled rows and positives by quarter year ==")
log(show(by_year))

fy = (
    pos.select("CERT").unique()
    .join(ex.select("CERT", "fail_q"), on="CERT")
    .with_columns((pl.col("fail_q") // 4).alias("fail_year"))
    .group_by("fail_year").agg(pl.len().alias("failing_banks")).sort("fail_year")
)
log("\n== Failing banks (with at least one y = 1 row) by year of failure ==")
log(show(fy))
log("\nPreview for the train/test design (Part 5 decides the split; the test period needs at least 50 failures):")
for a, b in [(1984, 1992), (1993, 2007), (2008, 2014), (2015, 2026)]:
    n = int(fy.filter((pl.col("fail_year") >= a) & (pl.col("fail_year") <= b))["failing_banks"].sum())
    log(f"  failures in {a}-{b}: {n:,}")

elapsed = time.time() - t0
log(f"\nbuilt in {elapsed:.1f} s")
(RES / "part2_clean_panel.txt").write_text("\n".join(out), encoding="utf-8")

# ---------------------------------------------------------------------------
# 8. README block with the real numbers
# ---------------------------------------------------------------------------
z = {r[0]: r for r in zero_rows}
zr = z.get("RBCRWAJ", ("RBCRWAJ", 0, 0, "-", "-"))
z1 = z.get("RBC1AAJ", ("RBC1AAJ", 0, 0, "-", "-"))
neg_txt = ", ".join(f"{c} {n:,}" for c, n in neg_counts.items())
lines = [
    "### Part 2.3 clean panel and label (`results/part2_clean_panel.txt`)",
    "",
    f"Output `data_storage/panel_clean/` ({chk.height:,} rows, {chk['CERT'].n_unique():,} banks, {chk.width} columns, {size_mb:.0f} MB). "
    "Label: `y = 1` if the bank fails in (q, q + 4 quarters]; only RESTYPE = FAILURE counts and a merger is never a failure. "
    "`y` is null after 2025Q2 because those labels are incomplete.",
    "",
    "| Problem | Handling | Effect |",
    "|---|---|---|",
    f"| Rows of failed banks dated in or after the failure quarter | dropped (they would not exist in real time) | {n_post_rows:,} rows, {n_post_banks:,} banks |",
    f"| FAILDATE before the first report | bank excluded | {n_bad_banks} banks, {n_bad_rows:,} rows |",
    f"| Exact zero used as \"not reported\" in RBCRWAJ and RBC1AAJ (whole eras and a fixed set of banks, see `results/part2_zero_pattern.txt`) | set to null in every quarter and flagged (`flag_rbc_zero`) | RBCRWAJ {zr[1]:,} values ({zr[2]:,} banks), RBC1AAJ {z1[1]:,} values ({z1[2]:,} banks); {n_flag_rbc:,} rows flagged |",
    f"| Negative values in items that cannot be negative | set to null and flagged (`flag_neg_fixed`) | {n_neg_rows:,} rows ({neg_txt}) |",
    f"| Zero denominator (DEP = 0) | kept, flagged (`flag_dep_zero`); safe division in Part 4 | {n_dep0:,} rows |",
    f"| Missing quarter inside a bank's history | flagged (`gap_before`); lags use the calendar quarter index `qidx`, not the row position | {n_gap_rows:,} rows, {n_gap_banks:,} banks |",
    "| Structural nulls (1984-1989 items, P3ASSET before 2001) and extreme ratios | left as they are: imputation and clipping bounds are learned on training data only, inside the Part 4 pipeline | none |",
    f"| Report not yet public when the bank failed | `avail_date` = quarter end + {LAG_DAYS} days (assumed lag); `y_before_public` flags the affected positives | {n_pre:,} of {n_pos:,} positive rows (one per bank) |",
    "",
    f"- Labelled rows: {n_lab:,}; positive rows: {n_pos:,} ({100 * n_pos / n_lab:.3f}%) from {pos_banks:,} failed banks. "
    f"{n_fail_banks - pos_banks:,} of the {n_fail_banks:,} banks with a FAILURE event have no positive row: they failed in July 2026, after the last panel quarter, so their labelled rows correctly have y = 0.",
    f"- Sensitivity label `y_alt` (also counts the open-bank assistance banks and the failure-like change-code exits): {n_alt:,} positive rows from {alt_banks:,} banks.",
    "- `y`, `y_alt`, `y_before_public` and `avail_date` are label or timing columns and must not be used as features.",
    f"- Verification on the written data: {sum(1 for _, ok, _ in checks if ok)} of {len(checks)} checks passed "
    "(row count, no duplicates, null labels only after 2025Q2, independent recount of positives, no label for non-failed banks, no row after the failure quarter, no future-revealing column, no exact zero left in the capital ratios).",
]
(RES / "part2_clean_readme_block.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print("== Verification checks ==")
for name, ok, detail in checks:
    print(f"{'PASS' if ok else 'FAIL'}  {name} {detail}".rstrip())
print(f"\nclean panel: {chk.height:,} rows | y = 1 rows: {n_pos:,} in {pos_banks:,} banks | built in {elapsed:.1f} s")
print("full report: results/part2_clean_panel.txt | README block: results/part2_clean_readme_block.md")
if not all_ok:
    raise SystemExit("verification FAILED: do not use the clean panel before fixing this")