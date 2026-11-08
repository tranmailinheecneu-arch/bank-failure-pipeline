 # Bank Failure Early-Warning Pipeline (BFT65B)

Individual final report, National Economics University, BFT65B Big Data Analytics in Finance.

Data source: [US FDIC BankFind API](https://api.fdic.gov/banks). Data files are not committed.

## Setup

```bash
python -m venv .venv
```

```powershell
.venv\Scripts\activate
pip install -r requirements.txt
```

## Part 1: Acquisition and storage

1. Download the data (about one hour; resumable): `python fdic_download_starter.py`
2. Run the CSV vs. Parquet storage benchmark: `python part1_storage_benchmark.py`
   (output: `results/part1_storage.txt`)
3. Build the final storage layout (Parquet + zstd, partitioned by year): `python build_storage.py`
4. Fair benchmark and float32 check: `python part1_extra_checks.py`
   (outputs in `results/`: schema table, completeness evidence, benchmark tables)

## Data download record

- Downloaded on: 2026-11-07
- Quarters: 170 (1984Q1 to 2026Q2)
- Panel rows: 1,682,616
- Distinct banks: 24,064
- Failures table: 4,117 rows
- Institutions table: 27,835 rows

## Sources

- `fdic_download_starter.py`: starter script provided by the course instructor.
- AI assistance: Claude (Anthropic) was used for guidance on setup and code.
