"""
Script 01: Data Acquisition — Merge Batched DAEN Exports

Scans data/raw/ for all Excel (.xlsx) and CSV (.csv) files downloaded from
the TGA DAEN web interface. Merges them into a single dataset, removes
duplicate cases, and saves to data/processed/daen_merged.csv.

If no files are found, prints a step-by-step download guide.

Usage:
    python scripts/01_data_acquisition.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

# ── Column name mapping ─────────────────────────────────────────────────────
# Maps various DAEN export column names to our internal standard names.
# The DAEN interface has changed over time, so we handle multiple variants.

COLUMN_MAP = {
    "case number": "case_number",
    "case no": "case_number",
    "case no.": "case_number",
    "case_no": "case_number",
    "case id": "case_number",
    "report entry date": "report_date",
    "report date": "report_date",
    "date of report": "report_date",
    "date": "report_date",
    "age": "age",
    "age (years)": "age",
    "age group": "age",
    "patient age": "age",
    "sex": "sex",
    "gender": "sex",
    "patient sex": "sex",
    "patient gender": "sex",
    "medicines": "medicines",
    "medicine": "medicines",
    "medicines reported as being taken": "medicines",
    "drug": "medicines",
    "drugs": "medicines",
    "drug name": "medicines",
    "product name": "medicines",
    "active ingredient": "active_ingredient",
    "active ingredients": "active_ingredient",
    "ingredient": "active_ingredient",
    "reactions": "reactions",
    "reaction": "reactions",
    "reaction term": "reactions",
    "meddra reaction terms": "reactions",
    "meddra reaction term": "reactions",
    "adverse event": "reactions",
    "adverse events": "reactions",
    "adverse reaction": "reactions",
    "preferred term": "reactions",
    "meddra preferred term": "reactions",
    "reporter type": "reporter_type",
    "reporter": "reporter_type",
    "source": "reporter_type",
    "report source": "reporter_type",
    "outcome": "outcome",
    "seriousness": "outcome",
}


def print_download_guide():
    """Print instructions for downloading DAEN data in batches."""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                     DAEN DOWNLOAD GUIDE                            ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║  No data files found in data/raw/                                  ║
║  Follow these steps to download DAEN data in batches:              ║
║                                                                    ║
╚══════════════════════════════════════════════════════════════════════╝

STEPS:

  1. Go to: https://daen.tga.gov.au/medicines-search/
  2. Accept the terms and conditions
  3. Leave the product name field BLANK (to get all reports)
  4. Set a date range (see batches below)
  5. Click "Search"
  6. Click the "List of Reports" tab
  7. Click the three-dot menu icon (⋮) → "Data with current layout"
  8. Save the Excel file to: data/raw/
  9. Repeat for each date range batch

SUGGESTED DATE RANGE BATCHES:

  Each batch must stay under 150,000 rows. Use these ranges as a
  starting point — if a batch exceeds the limit, split it further.

  Batch 01:  01/01/1971  →  31/12/2005   (early sparse years)
  Batch 02:  01/01/2006  →  31/12/2010
  Batch 03:  01/01/2011  →  31/12/2013
  Batch 04:  01/01/2014  →  31/12/2016
  Batch 05:  01/01/2017  →  31/12/2018
  Batch 06:  01/01/2019  →  31/12/2019
  Batch 07:  01/01/2020  →  31/12/2020
  Batch 08:  01/01/2021  →  31/12/2021
  Batch 09:  01/01/2022  →  31/12/2022
  Batch 10:  01/01/2023  →  31/12/2023
  Batch 11:  01/01/2024  →  31/12/2024
  Batch 12:  01/01/2025  →  today

  Note: Date format on the DAEN site is DD/MM/YYYY.
  Recent years (2019+) have higher volume and may need narrower windows.

NAMING CONVENTION (recommended):

  daen_batch_01_1971_2005.xlsx
  daen_batch_02_2006_2010.xlsx
  ... etc.

After downloading, re-run this script to merge all batches.
""")


def find_raw_files():
    """Find all Excel and CSV files in data/raw/."""
    if not RAW_DIR.exists():
        return []

    files = []
    for ext in ["*.xlsx", "*.xls", "*.csv"]:
        files.extend(RAW_DIR.glob(ext))

    # Exclude hidden/temp files
    files = [f for f in files if not f.name.startswith(("~", "."))]
    return sorted(files)


def read_file(filepath):
    """Read a single DAEN export file."""
    suffix = filepath.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(filepath, low_memory=False)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(filepath, engine="openpyxl" if suffix == ".xlsx" else None)
        else:
            print(f"  ⚠ Skipping unsupported file: {filepath.name}")
            return None
    except Exception as e:
        print(f"  ✗ Error reading {filepath.name}: {e}")
        return None

    return df


def standardise_columns(df):
    """Map DAEN export column names to standard internal names."""
    # Strip whitespace and lowercase
    df.columns = df.columns.str.strip().str.lower()

    rename = {}
    for col in df.columns:
        if col in COLUMN_MAP:
            rename[col] = COLUMN_MAP[col]

    df = df.rename(columns=rename)
    return df


def main():
    print("=" * 70)
    print("  TGA DAEN Data Acquisition — Merge Batched Exports")
    print("=" * 70)

    # ── Find raw files ───────────────────────────────────────────────────
    files = find_raw_files()
    if not files:
        print_download_guide()
        sys.exit(1)

    print(f"\nFound {len(files)} file(s) in {RAW_DIR}/\n")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}  ({size_mb:.1f} MB)")

    # ── Read each file ───────────────────────────────────────────────────
    dfs = []
    for f in files:
        print(f"\nReading {f.name} ...")
        df = read_file(f)
        if df is None:
            continue

        df = standardise_columns(df)
        print(f"  Rows: {len(df):,}  |  Columns: {list(df.columns)}")
        df["_source_file"] = f.name
        dfs.append(df)

    if not dfs:
        print("\nNo valid data could be read. Check your files and try again.")
        sys.exit(1)

    # ── Check column consistency ─────────────────────────────────────────
    col_sets = [set(df.columns) - {"_source_file"} for df in dfs]
    common_cols = set.intersection(*col_sets)
    all_cols = set.union(*col_sets)

    if common_cols != all_cols:
        only_in_some = all_cols - common_cols
        print(f"\n  Warning: Some columns appear only in certain files: {sorted(only_in_some)}")
        print(f"  Common columns across all files: {sorted(common_cols)}")

    # ── Merge ────────────────────────────────────────────────────────────
    merged = pd.concat(dfs, ignore_index=True)
    print(f"\nConcatenated total: {len(merged):,} rows")

    # ── Deduplicate ──────────────────────────────────────────────────────
    if "case_number" in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset="case_number", keep="first")
        dupes = before - len(merged)
        print(f"Deduplicated on case_number: removed {dupes:,} duplicates")
    else:
        before = len(merged)
        # Fall back to full-row deduplication (excluding source file tag)
        dedup_cols = [c for c in merged.columns if c != "_source_file"]
        merged = merged.drop_duplicates(subset=dedup_cols, keep="first")
        dupes = before - len(merged)
        print(f"Warning: No 'case_number' column found.")
        print(f"  Performed full-row deduplication: removed {dupes:,} exact duplicates")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  MERGED DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  Total unique records: {len(merged):,}")
    print(f"  Columns: {[c for c in merged.columns if c != '_source_file']}")

    print(f"\n  Column completeness:")
    for col in merged.columns:
        if col == "_source_file":
            continue
        non_null = merged[col].notna().sum()
        pct = 100 * non_null / len(merged)
        n_unique = merged[col].nunique()
        print(f"    {col:25s}  {non_null:>10,} non-null ({pct:5.1f}%)  |  {n_unique:,} unique")

    if "report_date" in merged.columns:
        dates = pd.to_datetime(merged["report_date"], errors="coerce", dayfirst=True)
        valid = dates.notna().sum()
        if valid > 0:
            print(f"\n  Date range: {dates.min().date()} → {dates.max().date()}")
            print(f"  Valid dates: {valid:,} / {len(merged):,}")

    # ── Per-file breakdown ───────────────────────────────────────────────
    print(f"\n  Records per source file:")
    for fname, count in merged["_source_file"].value_counts().sort_index().items():
        print(f"    {fname:40s}  {count:>10,} rows")

    # ── Save ─────────────────────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "daen_merged.csv"
    merged.to_csv(out_path, index=False)

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"\n  Saved: {out_path}")
    print(f"  Size:  {size_mb:.1f} MB")
    print(f"\n{'=' * 70}")
    print(f"  Next step: python scripts/02_data_cleaning.py")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
