"""
Script 02: Data Cleaning — Standardise and Structure DAEN Data

Reads the merged DAEN dataset from 01_data_acquisition.py, then:
  1. Parses dates into standard format
  2. Detects and splits multi-value medicines/reactions fields
  3. Standardises drug names and extracts active ingredients
  4. Creates normalised output tables for downstream analysis
  5. Generates descriptive statistics

Outputs (in data/processed/):
  - daen_cases.csv           One row per case (case_number, report_date, age, sex)
  - daen_case_drugs.csv      One row per case–drug combination
  - daen_case_reactions.csv   One row per case–reaction combination
  - daen_drug_reaction_pairs.csv  All drug–reaction pairs (for disproportionality)
  - descriptive_stats.txt    Summary statistics report

Usage:
    python scripts/02_data_cleaning.py
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import Counter
import sys

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "tables"
MERGED_PATH = PROCESSED_DIR / "daen_merged.csv"

# ── Configuration ────────────────────────────────────────────────────────────

# Minimum reports for a drug to be included in analysis
MIN_DRUG_REPORTS = 3

# Common delimiters in DAEN multi-value fields
CANDIDATE_DELIMITERS = [";", "|", "\n", ","]


# ── Helper functions ─────────────────────────────────────────────────────────


def detect_delimiter(series, candidates=None):
    """
    Auto-detect the delimiter used in a multi-value text column.

    Samples non-null values and picks the candidate delimiter that appears
    most frequently and consistently across cells.
    """
    if candidates is None:
        candidates = CANDIDATE_DELIMITERS

    sample = series.dropna().head(500).astype(str)
    if len(sample) == 0:
        return None

    best_delim = None
    best_score = 0

    for delim in candidates:
        # Count how many cells contain this delimiter
        cells_with = sample.str.contains(re.escape(delim), regex=True).sum()
        # Average number of items per cell when split
        avg_parts = sample.str.split(re.escape(delim)).apply(len).mean()

        # Score: fraction of cells that split × average parts
        score = (cells_with / len(sample)) * avg_parts
        if score > best_score:
            best_score = score
            best_delim = delim

    # Only return a delimiter if it appears in at least 5% of cells
    if best_delim:
        frac = sample.str.contains(re.escape(best_delim), regex=True).mean()
        if frac < 0.05:
            return None

    return best_delim


def split_multivalue(series, delimiter):
    """Split a multi-value column and return an exploded Series."""
    if delimiter is None:
        # No delimiter detected — treat each cell as a single value
        return series.str.strip()
    return series.str.split(re.escape(delimiter)).explode().str.strip()


def parse_suspected_flag(name):
    """
    Extract the '- Suspected' / '- Not suspected' flag from a DAEN medicine string.

    Returns (name_without_flag, suspected_flag) where suspected_flag is one of:
      'suspected', 'not_suspected', or None if no flag found.
    """
    if pd.isna(name) or not isinstance(name, str):
        return name, None

    # Match " - Suspected" or " - Not suspected" (case-insensitive) at end of string
    match = re.search(r"\s*-\s*(Not\s+suspected|Suspected)\s*$", name, re.IGNORECASE)
    if match:
        flag_text = match.group(1).strip().lower()
        clean_name = name[: match.start()].strip()
        flag = "not_suspected" if "not" in flag_text else "suspected"
        return clean_name, flag

    return name.strip(), None


def standardise_drug_name(name):
    """
    Standardise a drug/medicine name string (after suspected flag is removed).

    Handles patterns like:
      - "Trade Name (active ingredient)"
      - "Trade name not specified [active ingredient]"
      - Plain active ingredient names

    Returns (drug_name_clean, active_ingredient).
    """
    if pd.isna(name) or not isinstance(name, str):
        return name, None

    name = name.strip()

    # Pattern: "Trade name not specified [active ingredient]"
    match = re.match(r"^[Tt]rade\s+name\s+not\s+specified\s*\[(.+)\]$", name)
    if match:
        ingredient = match.group(1).strip().lower()
        return ingredient, ingredient

    # Pattern: "Trade name not specified (active ingredient)"
    match = re.match(r"^[Tt]rade\s+name\s+not\s+specified\s*\((.+)\)$", name)
    if match:
        ingredient = match.group(1).strip().lower()
        return ingredient, ingredient

    # Pattern: "Trade Name (active ingredient)"
    match = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", name)
    if match:
        trade = match.group(1).strip()
        ingredient = match.group(2).strip().lower()
        return trade, ingredient

    # Pattern: "Trade Name [active ingredient]"
    match = re.match(r"^(.+?)\s*\[([^\]]+)\]\s*$", name)
    if match:
        trade = match.group(1).strip()
        ingredient = match.group(2).strip().lower()
        return trade, ingredient

    # Plain name — assume it's an active ingredient
    return name.strip(), name.strip().lower()


def parse_age(age_str):
    """
    Parse age values which may be numeric, ranges, or descriptive groups.
    Returns a cleaned string suitable for analysis.
    """
    if pd.isna(age_str):
        return None

    age_str = str(age_str).strip()

    # Dash or empty means missing
    if age_str in ("-", "", "nan", "none", "not specified", "unknown"):
        return None

    # Already numeric
    try:
        val = float(age_str)
        if 0 <= val <= 120:
            return str(int(val))
        return None
    except ValueError:
        pass

    # Return as-is for descriptive groups (e.g., "65-74 years", "neonate", "<1")
    return age_str


def generate_descriptive_stats(cases_df, drugs_df, reactions_df, pairs_df):
    """Generate a descriptive statistics report string."""
    lines = []
    lines.append("=" * 70)
    lines.append("  TGA DAEN — DESCRIPTIVE STATISTICS")
    lines.append("=" * 70)

    # Overall counts
    lines.append(f"\n  DATASET OVERVIEW")
    lines.append(f"  {'─' * 40}")
    lines.append(f"  Total cases:                {len(cases_df):>10,}")
    lines.append(f"  Total case-drug records:    {len(drugs_df):>10,}")
    lines.append(f"  Total case-reaction records: {len(reactions_df):>10,}")
    lines.append(f"  Total drug-reaction pairs:  {len(pairs_df):>10,}")
    lines.append(f"  Unique drugs:               {drugs_df['drug_name_clean'].nunique():>10,}")
    lines.append(f"  Unique reactions:            {reactions_df['reaction'].nunique():>10,}")

    # Drugs per case
    drugs_per_case = drugs_df.groupby("case_number").size()
    lines.append(f"\n  DRUGS PER CASE")
    lines.append(f"  {'─' * 40}")
    lines.append(f"  Mean:   {drugs_per_case.mean():.1f}")
    lines.append(f"  Median: {drugs_per_case.median():.0f}")
    lines.append(f"  Max:    {drugs_per_case.max()}")

    # Reactions per case
    rxns_per_case = reactions_df.groupby("case_number").size()
    lines.append(f"\n  REACTIONS PER CASE")
    lines.append(f"  {'─' * 40}")
    lines.append(f"  Mean:   {rxns_per_case.mean():.1f}")
    lines.append(f"  Median: {rxns_per_case.median():.0f}")
    lines.append(f"  Max:    {rxns_per_case.max()}")

    # Sex distribution
    if "sex" in cases_df.columns:
        lines.append(f"\n  SEX DISTRIBUTION")
        lines.append(f"  {'─' * 40}")
        sex_counts = cases_df["sex"].value_counts(dropna=False)
        for val, count in sex_counts.items():
            pct = 100 * count / len(cases_df)
            label = val if pd.notna(val) else "Missing"
            lines.append(f"  {str(label):20s}  {count:>10,}  ({pct:5.1f}%)")

    # Age distribution
    if "age" in cases_df.columns:
        lines.append(f"\n  AGE DISTRIBUTION")
        lines.append(f"  {'─' * 40}")
        age_counts = cases_df["age"].value_counts(dropna=False).head(15)
        for val, count in age_counts.items():
            pct = 100 * count / len(cases_df)
            label = val if pd.notna(val) else "Missing"
            lines.append(f"  {str(label):20s}  {count:>10,}  ({pct:5.1f}%)")

    # Temporal distribution
    if "report_date" in cases_df.columns:
        lines.append(f"\n  REPORTS BY YEAR")
        lines.append(f"  {'─' * 40}")
        dates = pd.to_datetime(cases_df["report_date"], errors="coerce")
        valid_dates = dates.dropna()
        if len(valid_dates) > 0:
            year_counts = valid_dates.dt.year.value_counts().sort_index()
            for year, count in year_counts.items():
                bar = "█" * max(1, int(50 * count / year_counts.max()))
                lines.append(f"  {int(year)}  {count:>8,}  {bar}")
        else:
            lines.append("  No valid dates found.")

    # Top 20 drugs
    lines.append(f"\n  TOP 20 MOST REPORTED DRUGS")
    lines.append(f"  {'─' * 40}")
    top_drugs = drugs_df["drug_name_clean"].value_counts().head(20)
    for drug, count in top_drugs.items():
        lines.append(f"  {str(drug):40s}  {count:>8,}")

    # Top 20 reactions
    lines.append(f"\n  TOP 20 MOST REPORTED REACTIONS")
    lines.append(f"  {'─' * 40}")
    top_rxns = reactions_df["reaction"].value_counts().head(20)
    for rxn, count in top_rxns.items():
        lines.append(f"  {str(rxn):40s}  {count:>8,}")

    lines.append(f"\n{'=' * 70}\n")
    return "\n".join(lines)


def main():
    print("=" * 70)
    print("  TGA DAEN Data Cleaning — Standardise and Structure")
    print("=" * 70)

    # ── Load merged data ─────────────────────────────────────────────────
    if not MERGED_PATH.exists():
        print(f"\n  Merged data not found at: {MERGED_PATH}")
        print(f"  Run script 01 first: python scripts/01_data_acquisition.py")
        sys.exit(1)

    print(f"\nLoading {MERGED_PATH.name} ...")
    df = pd.read_csv(MERGED_PATH, low_memory=False)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # ── Parse dates ──────────────────────────────────────────────────────
    if "report_date" in df.columns:
        print("\nParsing dates ...")
        df["report_date"] = pd.to_datetime(
            df["report_date"], errors="coerce", format="mixed", dayfirst=True
        )
        valid = df["report_date"].notna().sum()
        print(f"  Valid dates: {valid:,} / {len(df):,}")

    # ── Parse age ────────────────────────────────────────────────────────
    if "age" in df.columns:
        print("Standardising age values ...")
        df["age"] = df["age"].apply(parse_age)

    # ── Parse sex ────────────────────────────────────────────────────────
    if "sex" in df.columns:
        print("Standardising sex values ...")
        df["sex"] = df["sex"].astype(str).str.strip().str.title()
        df.loc[df["sex"].isin(["Nan", "None", "", "Unknown", "Not Specified"]), "sex"] = None

    # ── Build cases table ────────────────────────────────────────────────
    case_cols = ["case_number"]
    for col in ["report_date", "age", "sex", "reporter_type", "outcome"]:
        if col in df.columns:
            case_cols.append(col)

    if "case_number" not in df.columns:
        print("\n  Warning: No 'case_number' column. Creating synthetic case IDs.")
        df["case_number"] = range(1, len(df) + 1)

    cases = df[case_cols].drop_duplicates(subset="case_number", keep="first").copy()
    cases = cases.reset_index(drop=True)
    print(f"\nCases table: {len(cases):,} unique cases")

    # ── Parse medicines ──────────────────────────────────────────────────
    print("\nParsing medicines column ...")

    if "medicines" not in df.columns and "active_ingredient" in df.columns:
        # Some exports have active_ingredient instead of medicines
        df["medicines"] = df["active_ingredient"]
        print("  Using 'active_ingredient' column as medicines source")

    if "medicines" in df.columns:
        # Detect delimiter
        med_delim = detect_delimiter(df["medicines"])
        if med_delim:
            delim_repr = repr(med_delim)
            print(f"  Detected delimiter: {delim_repr}")
        else:
            print("  No multi-value delimiter detected (single medicine per cell)")

        # Build case-drug table by exploding multi-value cells
        med_df = df[["case_number", "medicines"]].dropna(subset=["medicines"]).copy()
        med_df["medicines"] = med_df["medicines"].astype(str)

        if med_delim:
            med_df = med_df.assign(
                medicines=med_df["medicines"].str.split(re.escape(med_delim))
            ).explode("medicines")

        med_df["medicines"] = med_df["medicines"].str.strip()
        med_df = med_df[med_df["medicines"].str.len() > 0]

        # Extract suspected/not-suspected flag
        print("  Extracting suspected/not-suspected flags ...")
        flag_parsed = med_df["medicines"].apply(parse_suspected_flag)
        med_df["medicine_clean"] = flag_parsed.apply(lambda x: x[0])
        med_df["suspected"] = flag_parsed.apply(lambda x: x[1])

        suspected_counts = med_df["suspected"].value_counts(dropna=False)
        for val, count in suspected_counts.items():
            label = val if pd.notna(val) else "no flag"
            print(f"    {label}: {count:,}")

        # Standardise drug names
        print("  Standardising drug names ...")
        parsed = med_df["medicine_clean"].apply(standardise_drug_name)
        med_df["drug_name_clean"] = parsed.apply(lambda x: x[0])
        med_df["active_ingredient"] = parsed.apply(lambda x: x[1])

        # Drop duplicates within same case
        drugs = med_df[["case_number", "drug_name_clean", "active_ingredient", "suspected"]].drop_duplicates()
        drugs = drugs.reset_index(drop=True)

        print(f"  Case-drug records: {len(drugs):,}")
        print(f"  Unique drug names: {drugs['drug_name_clean'].nunique():,}")
        print(f"  Unique active ingredients: {drugs['active_ingredient'].nunique():,}")

        # Summary of suspected vs not
        n_suspected = (drugs["suspected"] == "suspected").sum()
        n_not = (drugs["suspected"] == "not_suspected").sum()
        print(f"  Suspected drug records: {n_suspected:,}  |  Not suspected: {n_not:,}")
    else:
        print("  Warning: No 'medicines' column found. Drug analysis will be limited.")
        drugs = pd.DataFrame(columns=["case_number", "drug_name_clean", "active_ingredient"])

    # ── Parse reactions ──────────────────────────────────────────────────
    print("\nParsing reactions column ...")

    if "reactions" in df.columns:
        rxn_delim = detect_delimiter(df["reactions"])
        if rxn_delim:
            delim_repr = repr(rxn_delim)
            print(f"  Detected delimiter: {delim_repr}")
        else:
            print("  No multi-value delimiter detected (single reaction per cell)")

        rxn_df = df[["case_number", "reactions"]].dropna(subset=["reactions"]).copy()
        rxn_df["reactions"] = rxn_df["reactions"].astype(str)

        if rxn_delim:
            rxn_df = rxn_df.assign(
                reactions=rxn_df["reactions"].str.split(re.escape(rxn_delim))
            ).explode("reactions")

        rxn_df["reactions"] = rxn_df["reactions"].str.strip()
        rxn_df = rxn_df[rxn_df["reactions"].str.len() > 0]

        # Standardise reaction terms (title case for MedDRA PTs)
        rxn_df["reaction"] = rxn_df["reactions"].str.title()

        # Drop duplicates within same case
        reactions = rxn_df[["case_number", "reaction"]].drop_duplicates()
        reactions = reactions.reset_index(drop=True)

        print(f"  Case-reaction records: {len(reactions):,}")
        print(f"  Unique reaction terms: {reactions['reaction'].nunique():,}")
    else:
        print("  Warning: No 'reactions' column found. Reaction analysis will be limited.")
        reactions = pd.DataFrame(columns=["case_number", "reaction"])

    # ── Build drug–reaction pair table ───────────────────────────────────
    print("\nBuilding drug–reaction pair table ...")

    if len(drugs) > 0 and len(reactions) > 0:
        # Merge drugs and reactions on case_number to get all pairs
        # Include suspected flag so we can filter downstream
        pairs = drugs[["case_number", "drug_name_clean", "active_ingredient", "suspected"]].merge(
            reactions[["case_number", "reaction"]],
            on="case_number",
            how="inner",
        )
        print(f"  Total drug–reaction pair records: {len(pairs):,}")

        # Count occurrences of each unique pair (suspected drugs only)
        suspected_pairs = pairs[pairs["suspected"] == "suspected"]
        print(f"  Suspected-drug pair records: {len(suspected_pairs):,}")

        pair_counts = (
            suspected_pairs.groupby(["drug_name_clean", "active_ingredient", "reaction"])
            .agg(n_cases=("case_number", "nunique"))
            .reset_index()
            .sort_values("n_cases", ascending=False)
        )
        print(f"  Unique suspected drug–reaction pairs: {len(pair_counts):,}")
        print(f"  Pairs with >= {MIN_DRUG_REPORTS} reports: "
              f"{(pair_counts['n_cases'] >= MIN_DRUG_REPORTS).sum():,}")
    else:
        pairs = pd.DataFrame(columns=["case_number", "drug_name_clean", "active_ingredient", "suspected", "reaction"])
        pair_counts = pd.DataFrame(columns=["drug_name_clean", "active_ingredient", "reaction", "n_cases"])

    # ── Save outputs ─────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("Saving cleaned datasets ...")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cases_path = PROCESSED_DIR / "daen_cases.csv"
    drugs_path = PROCESSED_DIR / "daen_case_drugs.csv"
    rxns_path = PROCESSED_DIR / "daen_case_reactions.csv"
    pairs_path = PROCESSED_DIR / "daen_drug_reaction_pairs.csv"

    cases.to_csv(cases_path, index=False)
    drugs.to_csv(drugs_path, index=False)
    reactions.to_csv(rxns_path, index=False)
    pair_counts.to_csv(pairs_path, index=False)

    for path in [cases_path, drugs_path, rxns_path, pairs_path]:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.name:40s}  {size_mb:>6.1f} MB")

    # ── Descriptive statistics ───────────────────────────────────────────
    print(f"\n{'─' * 70}")
    stats_report = generate_descriptive_stats(cases, drugs, reactions, pair_counts)
    print(stats_report)

    stats_path = OUTPUT_DIR / "descriptive_stats.txt"
    with open(stats_path, "w") as f:
        f.write(stats_report)
    print(f"  Stats saved to: {stats_path}")

    # ── Final summary ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  CLEANING COMPLETE")
    print(f"  Output files in: {PROCESSED_DIR}/")
    print(f"{'=' * 70}")
    print(f"\n  Next step: python scripts/03_disproportionality.py\n")


if __name__ == "__main__":
    main()
