"""Part 2.2: classify every bank that leaves the panel (failure vs other exits).

Reads  : data_storage/panel/*/*.parquet  (Part 1 output)
         data_raw/**/failures.parquet and institutions.parquet (download_fdic.py output)
Writes : results/part2_exits.txt                 full audit (all numbers used in the report)
         results/part2_exits_readme_block.md     ready-to-paste README block with the real numbers
         results/part2_exits_plan.txt            Polars query plan (projection pushdown evidence)
         results/part2_changecodes.json          cache of FDIC change-code descriptions
         results/fig_part2_exits_by_year.png     exits per year
         data_storage/exits/bank_exits.parquet   one row per bank (CERT) with exit class

Rules (point-in-time safe: only the failures table decides what is a failure)
  survivor          : still reporting in the last quarter of the panel
  failure           : a FAILURE event (RESTYPE = FAILURE) in the failures table, FAILDATE not before
                      its first report, and its last report at most one quarter after the failure quarter
  other_exit        : stopped reporting without a FAILURE event (merger, closure, conversion, or a bank
                      that only received open-bank ASSISTANCE)
  anomaly classes   : FAILURE event but still reporting afterwards, or FAILDATE before the first
                      report; kept separate so they can be audited, never silently labelled
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

PANEL = "data_storage/panel/*/*.parquet"
RES = Path("results")
RES.mkdir(exist_ok=True)
OUT = Path("data_storage/exits")
OUT.mkdir(parents=True, exist_ok=True)

LAST_LABEL_Q = 2025 * 4 + 1  # 2025Q2: last quarter whose 4-quarter-ahead label is complete (task rule)
LAG_DAYS = 35  # assumption: a Call Report is public about 30-35 days after quarter end (verify in
#               the FFIEC instructions); the data has no filing date, only the quarter-end REPDTE
CODE_CACHE = RES / "part2_changecodes.json"

out = []


def log(s=""):
    out.append(str(s))


def show(df):
    return df.to_pandas().to_string(index=False)


def find(name):
    hits = [p for p in Path("data_raw").rglob(f"{name}.parquet") if "test" not in str(p).lower()]
    if not hits:
        raise SystemExit(f"{name}.parquet not found under data_raw/ (run download_fdic.py first)")
    return max(hits, key=lambda p: p.stat().st_mtime)


def to_date(col):
    """Parse a date column that may be ISO, US (m/d/Y) or YYYYMMDD text; unparsable -> null."""
    s = pl.col(col).cast(pl.Utf8).str.strip_chars()
    return pl.coalesce(
        s.str.slice(0, 10).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        s.str.strptime(pl.Date, "%m/%d/%Y", strict=False),
        s.str.slice(0, 8).str.strptime(pl.Date, "%Y%m%d", strict=False),
    ).alias(col + "_d")


def qidx_expr(date_col):
    return (pl.col(date_col).dt.year() * 4 + (pl.col(date_col).dt.month() - 1) // 3).alias("fail_q")


# ---------------------------------------------------------------------------
# 1. One row per bank from the panel (lazy: only 3 of 51 columns are read)
# ---------------------------------------------------------------------------
t0 = time.time()
lazy = (
    pl.scan_parquet(PANEL)
    .select(pl.col("CERT").cast(pl.Int64), pl.col("REPDTE").cast(pl.Utf8), pl.col("ASSET"))
    .with_columns(
        (
            pl.col("REPDTE").str.slice(0, 4).cast(pl.Int32) * 4
            + (pl.col("REPDTE").str.slice(4, 2).cast(pl.Int32) - 1) // 3
        ).alias("qidx")
    )
    .group_by("CERT")
    .agg(
        pl.col("qidx").min().alias("first_q"),
        pl.col("qidx").max().alias("last_q"),
        pl.len().alias("n_q"),
        pl.col("REPDTE").min().alias("first_repdte"),
        pl.col("REPDTE").max().alias("last_repdte"),
        pl.col("ASSET").sort_by("REPDTE").last().alias("last_asset"),
    )
)
try:
    (RES / "part2_exits_plan.txt").write_text(lazy.explain(), encoding="utf-8")
except Exception as exc:  # explain() output differs across versions; never block the pipeline
    (RES / "part2_exits_plan.txt").write_text(f"explain() failed: {exc}", encoding="utf-8")
bank = lazy.collect().with_columns(
    pl.col("last_repdte").str.strptime(pl.Date, "%Y%m%d", strict=False).alias("last_rep_d")
)
elapsed = time.time() - t0
LAST_Q = int(bank["last_q"].max())
log(f"banks in panel: {bank.height:,} | last quarter index {LAST_Q} "
    f"(= {LAST_Q // 4}Q{LAST_Q % 4 + 1}) | built in {elapsed:.1f} s (lazy scan, 3 of 51 columns)")

# ---------------------------------------------------------------------------
# 2. Failures table (audit + one row per bank)
# ---------------------------------------------------------------------------
fraw = pl.read_parquet(find("failures"))
fcols = [c for c in ["CERT", "NAME", "FAILDATE", "FAILYR", "COST", "RESTYPE", "QBFASSET"] if c in fraw.columns]
fails = fraw.select(fcols).with_columns(pl.col("CERT").cast(pl.Int64, strict=False), to_date("FAILDATE"))
n_fail_rows = fails.height
n_fail_unparsed = fails.filter(pl.col("FAILDATE_d").is_null()).height
n_fail_nocert = fails.filter(pl.col("CERT").is_null()).height
n_nocert_1984 = fails.filter(pl.col("CERT").is_null() & (pl.col("FAILDATE_d") >= pl.date(1984, 1, 1))).height
fails = fails.filter(pl.col("CERT").is_not_null() & pl.col("FAILDATE_d").is_not_null()).with_columns(
    qidx_expr("FAILDATE_d")
)
if "RESTYPE" in fails.columns:  # FAILURE = bank closed/resolved; ASSISTANCE = open-bank assistance
    fails = fails.with_columns(
        (pl.col("RESTYPE").cast(pl.Utf8).str.to_uppercase() == "FAILURE").fill_null(False).alias("is_closure")
    )
else:
    fails = fails.with_columns(pl.lit(True).alias("is_closure"))
n_fail_dups = fails.height - fails["CERT"].n_unique()
# a CERT can appear several times (e.g. assistance, then failure): keep a closure event if any,
# then the event closest to the bank's last report, ties -> the later event
inside = (
    fails.join(bank.select("CERT", "last_q"), on="CERT", how="inner")
    .with_columns((pl.col("fail_q") - pl.col("last_q")).abs().alias("dist"))
    .sort(["CERT", "is_closure", "dist", "FAILDATE_d"], descending=[False, True, False, True])
    .unique(subset="CERT", keep="first", maintain_order=True)
    .drop("last_q", "dist")
)
outside = (
    fails.join(bank.select("CERT"), on="CERT", how="anti")
    .sort("FAILDATE_d")
    .unique(subset="CERT", keep="first", maintain_order=True)
)
fails1 = pl.concat([inside, outside.select(inside.columns)])
log("\n== Failures table audit ==")
log(f"rows: {n_fail_rows:,} | unparsable FAILDATE: {n_fail_unparsed} | CERT repeated (second event dropped): {n_fail_dups}")
log(f"rows without a usable CERT (cannot be linked to the panel): {n_fail_nocert:,} (dated 1984 or later: {n_nocert_1984:,})")
log(f"FAILDATE range: {fails1['FAILDATE_d'].min()} to {fails1['FAILDATE_d'].max()}")
if "RESTYPE" in fails1.columns:
    log("\nRESTYPE values (resolution type; check which ones are real closures):")
    log(show(fails1.group_by("RESTYPE").agg(pl.len().alias("n")).sort("n", descending=True)))
if "RESTYPE" in fails.columns and n_fail_dups:
    multi = (
        fails.group_by("CERT")
        .agg(pl.len().alias("k"), pl.col("RESTYPE").fill_null("?").sort_by("FAILDATE_d").alias("seq"))
        .filter(pl.col("k") > 1)
        .with_columns(pl.col("seq").list.join(" > ").alias("pattern"))
    )
    log("\nCERTs with several failure/assistance events: event sequence by date")
    log(show(multi.group_by("pattern").agg(pl.len().alias("banks")).sort("banks", descending=True).head(10)))
not_in_panel = fails1.join(bank.select("CERT"), on="CERT", how="anti")
nip_window = not_in_panel.filter(pl.col("FAILDATE_d") >= pl.date(1984, 1, 1))
log(f"\nfailures whose CERT never appears in the panel: {not_in_panel.height:,} "
    f"(of which dated 1984 or later: {nip_window.height:,}; the rest failed before the panel starts)")

# ---------------------------------------------------------------------------
# 3. Institutions table (status flags, end date, change code of the last event)
# ---------------------------------------------------------------------------
iraw = pl.read_parquet(find("institutions"))
icols = [c for c in ["CERT", "NAME", "STNAME", "BKCLASS", "ACTIVE", "CHANGEC1", "ENDEFYMD"] if c in iraw.columns]
inst = iraw.select(icols).with_columns(pl.col("CERT").cast(pl.Int64, strict=False)).unique(subset="CERT", keep="first")
if "ENDEFYMD" in inst.columns:
    inst = inst.with_columns(to_date("ENDEFYMD")).with_columns(
        pl.when(pl.col("ENDEFYMD_d").dt.year() >= 9000)
        .then(pl.lit(None, dtype=pl.Date))
        .otherwise(pl.col("ENDEFYMD_d"))
        .alias("inst_end_d")
    ).drop("ENDEFYMD", "ENDEFYMD_d")
if "CHANGEC1" in inst.columns:
    inst = inst.with_columns(pl.col("CHANGEC1").cast(pl.Int64, strict=False))
n_inst_not_panel = inst.join(bank.select("CERT"), on="CERT", how="anti").height
log(f"\ninstitutions rows: {inst.height:,} | not in panel: {n_inst_not_panel:,}")

# ---------------------------------------------------------------------------
# 4. Classify
# ---------------------------------------------------------------------------
fsel = ["CERT", "FAILDATE_d", "fail_q", "is_closure"] + [c for c in ["RESTYPE", "COST"] if c in fails1.columns]
b = bank.join(fails1.select(fsel), on="CERT", how="left").join(inst, on="CERT", how="left")

in_f = pl.col("FAILDATE_d").is_not_null()
closure = in_f & pl.col("is_closure").fill_null(False)  # a FAILURE event (not open-bank assistance)
surv = pl.col("last_q") == LAST_Q
b = b.with_columns(
    pl.when(surv & in_f).then(pl.lit("survivor_listed_in_failures"))
    .when(surv).then(pl.lit("survivor"))
    .when(closure & (pl.col("fail_q") < pl.col("first_q"))).then(pl.lit("failure_date_before_first_report"))
    .when(closure & (pl.col("last_q") > pl.col("fail_q") + 1)).then(pl.lit("reported_after_failure_date"))
    .when(closure).then(pl.lit("failure"))
    .otherwise(pl.lit("other_exit"))  # includes banks that only received open-bank assistance
    .alias("exit_class"),
    (in_f & ~pl.col("is_closure").fill_null(False)).alias("had_assistance"),
)

# 4a. other exits: describe the FDIC change code of the last structural event, bucket by keywords
codes = sorted(
    c for c in b.filter(pl.col("exit_class") == "other_exit").get_column("CHANGEC1").unique().to_list() if c is not None
) if "CHANGEC1" in b.columns else []
cache = json.loads(CODE_CACHE.read_text(encoding="utf-8")) if CODE_CACHE.exists() else {}


def fetch_desc(code):
    q = urllib.parse.urlencode({"filters": f"CHANGECODE:{code}", "fields": "CHANGECODE,CHANGECODE_DESC",
                                "limit": 1, "format": "json"})
    req = urllib.request.Request(f"https://api.fdic.gov/banks/history?{q}",
                                 headers={"User-Agent": "BFT65B-course-pipeline/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            js = json.load(r)
        time.sleep(0.5)
        return js["data"][0]["data"]["CHANGECODE_DESC"]
    except Exception:
        return None


for c in codes:
    if str(c) not in cache or cache[str(c)] is None:
        cache[str(c)] = fetch_desc(c)
CODE_CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")


def bucket(desc):
    if not desc:
        return "unmapped"
    d = desc.upper()
    if ("FAIL" in d or "PAYOFF" in d or "RECEIV" in d or "CONSERVAT" in d or "BRIDGE" in d
            or ("ASSIST" in d and "WITHOUT" not in d)):
        return "failure_like"  # reviewed below: never used as a label by itself
    if "MERGER" in d or "ACQUI" in d or "CONSOLIDAT" in d or "ABSORB" in d:
        return "merger"
    if "LIQUID" in d or "CLOS" in d or "CEASE" in d or "TERMINAT" in d:
        return "closure_or_liquidation"
    if "CONVERSION" in d or "CHARTER" in d or "INSUR" in d:
        return "charter_or_insurance_change"
    return "other"


if codes:
    cmap = pl.DataFrame(
        {"CHANGEC1": codes, "code_desc": [cache.get(str(c)) for c in codes],
         "exit_bucket": [bucket(cache.get(str(c))) for c in codes]},
        schema={"CHANGEC1": pl.Int64, "code_desc": pl.Utf8, "exit_bucket": pl.Utf8},
    )
    b = b.join(cmap, on="CHANGEC1", how="left")
    b = b.with_columns(
        pl.when(pl.col("exit_class") == "other_exit").then(pl.col("exit_bucket").fill_null("unmapped"))
        .otherwise(pl.lit(None, dtype=pl.Utf8)).alias("exit_bucket")
    )
else:
    b = b.with_columns(pl.lit(None, dtype=pl.Utf8).alias("exit_bucket"))

b.write_parquet(OUT / "bank_exits.parquet", compression="zstd")

# ---------------------------------------------------------------------------
# 5. Report
# ---------------------------------------------------------------------------
cls_tab = b.group_by("exit_class").agg(pl.len().alias("banks")).sort("banks", descending=True)
cls_tab = cls_tab.with_columns((pl.col("banks") / pl.col("banks").sum() * 100).round(2).alias("pct"))
cls = dict(cls_tab.select("exit_class", "banks").iter_rows())
n_f, n_o, n_s = cls.get("failure", 0), cls.get("other_exit", 0), cls.get("survivor", 0)
n_anom = sum(v for k, v in cls.items() if k not in ("failure", "other_exit", "survivor"))
log("\n== Exit class per bank (CERT) ==")
log(show(cls_tab))
n_assist = b.filter(pl.col("had_assistance")).height
log(f"\nbanks with an ASSISTANCE-only event in the failures table (never labelled as failures): {n_assist}")
if "RESTYPE" in b.columns:
    log("\n== Exit class x RESTYPE (banks listed in the failures table) ==")
    log(show(b.filter(pl.col("FAILDATE_d").is_not_null()).group_by("exit_class", "RESTYPE")
             .agg(pl.len().alias("banks")).sort("exit_class", "RESTYPE")))

fr = b.filter(pl.col("exit_class") == "failure").with_columns((pl.col("fail_q") - pl.col("last_q")).alias("gap_q"))
log("\n== Failures: quarters between last report and failure quarter (0 = failed in the quarter of its last report) ==")
log(show(fr.group_by("gap_q").agg(pl.len().alias("banks")).sort("gap_q")))
n_gap_gt4 = fr.filter(pl.col("gap_q") > 4).height
log(f"failures whose last report is more than 4 quarters before FAILDATE (can never receive y = 1): {n_gap_gt4}")

log("\n== Point-in-time exposure: days from the last Call Report date to FAILDATE ==")
lag = fr.with_columns((pl.col("FAILDATE_d") - pl.col("last_rep_d")).dt.total_days().alias("days"))
lag = lag.with_columns(
    pl.when(pl.col("days") <= 0).then(pl.lit("a) report dated on/after failure"))
    .when(pl.col("days") <= LAG_DAYS).then(pl.lit(f"b) 1-{LAG_DAYS} days (report not yet public)"))
    .when(pl.col("days") <= 90).then(pl.lit(f"c) {LAG_DAYS + 1}-90 days"))
    .otherwise(pl.lit("d) more than 90 days")).alias("lag_bucket")
)
lag_tab = lag.group_by("lag_bucket").agg(pl.len().alias("failed_banks")).sort("lag_bucket")
log(show(lag_tab))
n_lag_exposed = lag.filter(pl.col("days") <= LAG_DAYS).height
log(f"failed banks whose final report was probably not public before failure: {n_lag_exposed} of {n_f}")

other = b.filter(pl.col("exit_class") == "other_exit")
log("\n== Other exits by FDIC change code of the last structural event (code, description, bucket) ==")
if "CHANGEC1" in other.columns:
    log(show(other.group_by("CHANGEC1", "code_desc", "exit_bucket").agg(pl.len().alias("banks"))
             .sort("banks", descending=True).head(30)))
    log("\n== Other exits by bucket ==")
    log(show(other.group_by("exit_bucket").agg(pl.len().alias("banks")).sort("banks", descending=True)))
n_fl = other.filter(pl.col("exit_bucket") == "failure_like").height
log(f"\nother exits whose change code reads like a failure/assistance but that are NOT in the failures table: {n_fl}")

anom = b.filter(~pl.col("exit_class").is_in(["failure", "other_exit", "survivor"]))
log("\n== Anomalies (first 15 rows of each class, for manual inspection) ==")
if anom.height:
    cols = [c for c in ["CERT", "exit_class", "first_repdte", "last_repdte", "FAILDATE_d", "RESTYPE", "ACTIVE"] if c in anom.columns]
    log(show(anom.select(cols).sort("exit_class").head(15)))
else:
    log("none")

if "ACTIVE" in b.columns:
    log("\n== Consistency: institutions.ACTIVE vs still reporting in the last quarter ==")
    log(show(b.with_columns((pl.col("last_q") == LAST_Q).alias("still_reporting"))
             .group_by("ACTIVE", "still_reporting").agg(pl.len().alias("banks")).sort("ACTIVE", "still_reporting")))

yr = (
    b.filter(pl.col("exit_class").is_in(["failure", "other_exit"]))
    .with_columns((pl.col("last_q") // 4).alias("year"))
    .group_by("year", "exit_class").agg(pl.len().alias("n")).to_pandas()
    .pivot_table(index="year", columns="exit_class", values="n", fill_value=0)
)
yr = yr.reindex(columns=["failure", "other_exit"], fill_value=0)
yr = yr.reindex(range(int(yr.index.min()), int(yr.index.max()) + 1), fill_value=0)
log("\n== Exits per year (year of last Call Report) ==")
log(yr.to_string())

# peak years (numbers for the report text)
pk_f, pk_o = int(yr["failure"].idxmax()), int(yr["other_exit"].idxmax())
log(f"\npeak failure year: {pk_f} ({int(yr['failure'].max())}) | peak other-exit year: {pk_o} ({int(yr['other_exit'].max())})")

(RES / "part2_exits.txt").write_text("\n".join(out), encoding="utf-8")

# ---------------------------------------------------------------------------
# 6. Figure: small multiples (different scales), colour-blind-safe palette, direct peak labels
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
spec = [("failure", "Failures", "#D55E00"), ("other_exit", "Other exits (mergers, closures, conversions)", "#0072B2")]
for ax, (col, label, color) in zip(axes, spec):
    ax.bar(yr.index, yr[col], color=color, width=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(label, loc="left", fontsize=10, fontweight="bold", color=color)
    pk = int(yr[col].idxmax())
    ax.annotate(f"{pk}: {int(yr[col].max()):,}", xy=(pk, yr[col].max()), xytext=(5, 2),
                textcoords="offset points", fontsize=8)
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
axes[1].set_xlabel("Year of last Call Report")
fig.suptitle(f"Banks leaving the panel: {n_f:,} failures vs {n_o:,} other exits", x=0.01, ha="left",
             fontsize=12, fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(RES / "fig_part2_exits_by_year.png", dpi=200)

# ---------------------------------------------------------------------------
# 7. README block with the real numbers
# ---------------------------------------------------------------------------
lines = [
    "### Part 2.2 exit classification (`results/part2_exits.txt`)",
    "",
    f"Every one of the {bank.height:,} banks in the panel is assigned one exit class "
    f"(last quarter of the panel: {LAST_Q // 4}Q{LAST_Q % 4 + 1}). "
    "Only the failures table decides what is a failure; a merger is never a failure.",
    "",
    "| Exit class | Banks | Share |",
    "|---|---|---|",
]
for r in cls_tab.iter_rows(named=True):
    lines.append(f"| {r['exit_class']} | {r['banks']:,} | {r['pct']:.2f}% |")
lines += [
    "",
    f"- Failures table: {n_fail_rows:,} rows; {n_fail_nocert:,} have no usable CERT ({n_nocert_1984:,} dated 1984 or later) "
    f"and {not_in_panel.height:,} linked banks never appear in the panel ({nip_window.height:,} of them dated 1984 or later; "
    "the rest failed before the panel starts).",
    f"- {n_assist:,} banks have only an open-bank ASSISTANCE event (RESTYPE = ASSISTANCE): they kept operating, "
    "so they are not labelled as failures; only RESTYPE = FAILURE counts.",
    f"- Failures by year of last report peak in {pk_f} ({int(yr['failure'].max())} banks); other exits peak in "
    f"{pk_o} ({int(yr['other_exit'].max())} banks) (`results/fig_part2_exits_by_year.png`).",
    f"- {n_gap_gt4} failures have their last report more than four quarters before FAILDATE; these cannot receive a positive four-quarter-ahead label.",
    f"- {n_lag_exposed} of {n_f:,} classified failures may have a final report that was not public before failure "
    f"under the {LAG_DAYS}-day publication-lag assumption.",
    f"- {n_fl} other exits carry a receivership, bridge-bank, payoff or assisted-merger FDIC change code but are not "
    "linked to a FAILURE row (candidates for failures missing from the failures table; reviewed in Part 2.3).",
    f"- {n_anom} banks fall in anomaly classes (FAILURE event but still reporting, or FAILDATE before the first report).",
    f"- Built with a lazy Polars scan that reads 3 of 51 columns in {elapsed:.1f} s (`results/part2_exits_plan.txt`).",
]
(RES / "part2_exits_readme_block.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print("\n".join(out)[:4000])
print("\n... full report: results/part2_exits.txt | README block: results/part2_exits_readme_block.md")