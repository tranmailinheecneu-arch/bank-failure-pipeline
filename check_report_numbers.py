"""Check that key numbers quoted in the report and README appear in results.

Run from the repository root after the pipeline:
	python check_report_numbers.py

Each check specifies a result file, description, required regex, and minimum
match count. Exit with 0 if all checks pass, otherwise 1. Only results/*.txt
and results/*.json are read.
"""
import re
import sys
from pathlib import Path

RES = Path("results")

# (file, description, regex, minimum number of matches)
CHECKS = [
	("part1_completeness.txt", "170 quarters", r"quarters: 170", 1),
	("part1_completeness.txt", "1,682,616 rows", r"total rows: 1,682,616", 1),
	("part1_completeness.txt", "rows equal the API total", r"rows == API total in every quarter: True", 1),
	("part1_storage.txt", "CSV 615.6 MB", r"615\.589", 1),
	("part1_storage.txt", "final layout 201.8 MB", r"201\.844", 1),
	("part1_fair_benchmark.txt", "Polars CSV 17.5 s", r"17\.509", 1),
	("part1_polars_csv_check.txt", "inference over all rows 17.3 s", r"infer schema from all rows: read all 17\.339", 1),
	("part1_polars_csv_check.txt", "explicit schema 0.648 s", r"explicit schema \(no inference\): read all 0\.648", 1),
	("part2_dq_report.txt", "24,064 banks", r"rows: 1,682,616 \| banks: 24,064", 1),
	("part2_zero_pattern.txt", "RBCRWAJ zeros since 2020", r"RBCRWAJ == 0 since 2020: 45,865 rows, 2,621 distinct banks", 1),
	("part2_exits.txt", "other_exit 16,910", r"other_exit\s+16910", 1),
	("part2_exits.txt", "failure 2,662", r"\bfailure\s+2662", 1),
	("part2_exits.txt", "841 of 2,662 not public", r"841 of 2662", 1),
	("part2_exits.txt", "865 failure-like exits", r"NOT in the failures table: 865", 1),
	("part2_clean_panel.txt", "10,860 positive rows, 2,837 banks", r"y = 1 rows: 10,860 .* in 2,837 failed banks", 1),
	("part2_clean_panel.txt", "798 rows not public", r"798 \(7\.3% of positives\)", 1),
	("part2_clean_panel.txt", "1,681,431 clean rows", r"clean panel: 1,681,431 rows", 1),
	("part3_polars_pandas.txt", "pandas 5.46 / 11.02 s", r"best 5\.46 s \| median 11\.02 s", 1),
	("part3_polars_pandas.txt", "Polars eager 2.01 s", r"best 2\.01 s \| median 2\.51 s", 1),
	("part3_polars_pandas.txt", "Polars lazy 2.23 s", r"best 2\.23 s \| median 2\.59 s", 1),
	("part3_std_check.txt", "5 rows above 1e-9", r"rows with \|diff\| > 1e-09: 5", 1),
	("part3_spark_results.txt", "Polars lazy 2.89 s", r"polars lazy : 2\.89 s / 3\.02 s", 1),
	("part3_spark_results.txt", "Spark 52.51 s", r"spark local : 52\.51 s / 55\.37 s", 1),
	("part3_spark_results.txt", "18.2 times", r"18\.2x faster", 1),
	("part3_scaling_v2.txt", "6x in memory: 3,130 MB", r"memory k = 6: 10,088,586 rows \| best 13\.96 s \| peak 3,130 MB", 1),
	("part3_scaling_v2.txt", "6x streaming: 3,100 MB", r"streaming k = 6: 10,088,586 rows \| best 18\.91 s \| peak 3,100 MB", 1),
	("part4_features.txt", "development period counts", r"dev\s+1193702\s+1193702\s+8678\s+2290\s+8140", 1),
	("part4_features.txt", "test period counts", r"test\s+212645\s+212645\s+2005\s+513\s+1764", 1),
	("part4_features.txt", "nine leakage tests pass", r"^PASS", 9),
	("part5_validation.txt", "loss rate 0.1959", r"= 0\.1959", 1),
	("part5_validation.txt", "selected model mean fold AP 0.2992", r"31 leaves\s+0\.2992", 1),
	("part5_test.txt", "selected model test AP 0.5250", r"31 leaves\s+0\.5250", 1),
	("part5_test.txt", "B1 test AP 0.4946", r"logistic, 3 features\s+0\.4946", 1),
	("part5_test.txt", "B0 test AP 0.1748", r"npl_loans only\s+0\.1748", 1),
	("part5_test.txt", "chance level 0.830%", r"0\.830%", 1),
	("part5_test_LOCK.json", "single test run 2026-11-21 22:39", r"2026-11-21T22:39:49", 1),
	("part5_test_LOCK.json", "git commit f33db5b0b1", r"f33db5b0b1", 1),
	("part5_dev_extras.txt", "development models reproduce Table 12", r"reproduction OK", 1),
	("part5_dev_extras.txt", "bootstrap interval of the pooled AP difference", r"\[-0\.0070, \+0\.0248\]", 1),
	("part5_mllib_results.txt", "MLlib trees area under PR 0.0826", r"area under PR 0\.0826", 1),
]


def main():
	bad = 0
	for fname, what, pattern, need in CHECKS:
		path = RES / fname
		if not path.exists():
			print(f"MISSING FILE  {fname:28s} {what}")
			bad += 1
			continue
		text = path.read_text(encoding="utf-8", errors="replace")
		n = len(re.findall(pattern, text, flags=re.MULTILINE))
		if n >= need:
			print(f"ok            {fname:28s} {what}")
		else:
			print(f"NOT FOUND     {fname:28s} {what}   (pattern: {pattern})")
			bad += 1
	print(f"\n{len(CHECKS) - bad} of {len(CHECKS)} checks passed")
	sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
	main()
