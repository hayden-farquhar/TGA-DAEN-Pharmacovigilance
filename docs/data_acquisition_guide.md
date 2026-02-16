# Data Acquisition Guide

## Overview

The TGA Database of Adverse Event Notifications (DAEN) is publicly accessible at:

https://www.tga.gov.au/safety/safety-data/database-adverse-event-notifications-daen

The DAEN web interface imposes a **150,000-row export cap** per query, meaning the full dataset cannot be downloaded in a single export. This guide describes how to download the data in batches and merge them using the provided scripts.

## Step 1: Access the DAEN Search Interface

1. Navigate to the DAEN search page: https://apps.tga.gov.au/PROD/DAEN/daen-entry.aspx
2. Select **"Medicines"** as the product type
3. Leave the medicine name field blank (to retrieve all medicines)
4. Set the date range for each batch (see below)

## Step 2: Download in Batches

Due to the 150,000-row cap, download the data in date-range batches. The following batches were used in the original study:

| Batch | Date Range | Approximate Rows |
|-------|-----------|-----------------|
| 1 | 1 Jan 1971 -- 31 Dec 2005 | ~80,000 |
| 2 | 1 Jan 2006 -- 31 Dec 2015 | ~120,000 |
| 3 | 1 Jan 2016 -- 31 Dec 2019 | ~100,000 |
| 4 | 1 Jan 2020 -- 31 Dec 2022 | ~150,000 |
| 5 | 1 Jan 2023 -- present | ~100,000 |

**Note:** These row counts are approximate and will change as new reports are added. Adjust date ranges if a single batch exceeds 150,000 rows.

### Export settings

- Select **"Export to Excel"** (produces .xlsx files) or **"Export to CSV"**
- Include all available fields
- The export includes: Case Number, Report Entry Date, Age (Years), Sex, Medicines Reported as Being Taken, MedDRA Reaction Terms

## Step 3: Place Files in data/raw/

Save all downloaded files (Excel or CSV) into the `data/raw/` directory:

```
data/
└── raw/
    ├── daen_batch_1971_2005.xlsx
    ├── daen_batch_2006_2015.xlsx
    ├── daen_batch_2016_2019.xlsx
    ├── daen_batch_2020_2022.xlsx
    └── daen_batch_2023_present.xlsx
```

File names do not matter -- the merge script scans for all `.xlsx` and `.csv` files in `data/raw/`.

## Step 4: Run the Merge Script

```bash
python scripts/01_data_acquisition.py
```

This script:
1. Reads all Excel/CSV files from `data/raw/`
2. Standardises column names across different export formats
3. Concatenates all batches
4. Deduplicates on case number
5. Saves the merged dataset to `data/processed/daen_merged.csv`

## Data Fields

| Field | Description |
|-------|-------------|
| Case Number | Unique identifier for each adverse event report |
| Report Entry Date | Date the report was entered into the DAEN |
| Age (Years) | Patient age at time of report (may be missing) |
| Sex | Patient sex (Female, Male, Not stated) |
| Medicines Reported as Being Taken | Newline-delimited list of medicines with "Suspected" or "Not suspected" flags |
| MedDRA Reaction Terms | Newline-delimited list of adverse event terms coded to MedDRA Preferred Terms |

## Important Notes

- **Data currency:** The DAEN is updated regularly. Results may differ depending on when data are downloaded.
- **No denominator data:** The DAEN contains only numerator data (reports). Population exposure data are not available, precluding incidence rate calculations.
- **Missing fields:** The public DAEN export does not include reporter type (healthcare professional vs consumer), outcome severity, dechallenge/rechallenge information, or narrative text.
- **Drug name format:** Medicines are listed as "Trade Name (active ingredient) - Suspected" or "Trade name not specified [active ingredient] - Not suspected". Script 02 handles parsing of these formats.

## Verification

After merging, `01_data_acquisition.py` prints summary statistics. For the dataset used in the study (downloaded February 2026):
- Total unique cases: 664,747
- Date range: 1971--2026
- Total rows before deduplication: ~700,000+
