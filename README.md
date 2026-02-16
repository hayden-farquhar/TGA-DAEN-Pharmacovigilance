# Computational Pharmacovigilance of Australia's Database of Adverse Event Notifications (DAEN)

Analysis code and reference data for the study: *"Computational Pharmacovigilance of Australia's Database of Adverse Event Notifications: A Multi-Method Signal Detection Study Using Disproportionality Analysis, Machine Learning, and Network Analysis"* by Hayden Farquhar.

## Overview

This repository contains the complete analysis pipeline for the first independent computational pharmacovigilance analysis of Australia's DAEN. The study applies four disproportionality analysis (DPA) methods, two machine learning classifiers, and network analysis to 664,747 adverse event reports spanning 1971--2026.

### Methods implemented

- **Proportional Reporting Ratio (PRR)** with Yates-corrected chi-squared
- **Reporting Odds Ratio (ROR)** with 95% confidence intervals
- **Empirical Bayesian Geometric Mean (EBGM)** via the Multi-Item Gamma Poisson Shrinker (MGPS) with maximum likelihood prior fitting
- **Information Component (IC)** via the Bayesian Confidence Propagation Neural Network (BCPNN) with Noren variance approximation
- **XGBoost and Random Forest** classifiers trained on a literature-verified reference set
- **Bipartite drug--adverse event network** with Louvain community detection
- **Nine sensitivity analyses** (FDR correction, COVID-19 masking, temporal/sex/age stratification, threshold sensitivity, signal detection latency)

## Data Acquisition

The DAEN data are publicly available from the TGA website but require batched downloads due to a 150,000-row export cap. See [`docs/data_acquisition_guide.md`](docs/data_acquisition_guide.md) for step-by-step instructions.

**Important:** Raw DAEN data are not included in this repository. You must download them yourself from the TGA website.

## Repository Structure

```
TGA-DAEN-Pharmacovigilance/
├── README.md
├── LICENSE
├── requirements.txt
├── scripts/
│   ├── 01_data_acquisition.py       # Merge batched DAEN exports
│   ├── 02_data_cleaning.py          # Standardise drug names, deduplicate, parse fields
│   ├── 03_disproportionality.py     # PRR, ROR, EBGM (MGPS), BCPNN (IC)
│   ├── 04_ml_signal_detection.py    # XGBoost/RF training, evaluation, scoring
│   ├── 05_network_analysis.py       # Bipartite network, centrality, Louvain communities
│   ├── 06_visualisation.py          # Publication-ready figures (Figs 1-7)
│   ├── 07_sensitivity_validation.py # FDR, COVID masking, stratification, latency
│   ├── 07_additional_validation.py  # DPA reference validation, ML CIs, label concordance
│   ├── 08_manuscript_revisions.py   # MGPS validation, SHAP, repeated CV, border-zone analysis
│   ├── 09_revision_figures.py       # SHAP plots, prior distribution, calibration
│   ├── 10_external_reference_validation.py  # OMOP/EU-ADR/Harpaz crosswalk
│   └── 11_expanded_reference_ml.py  # Expanded 170-pair reference set ML validation
├── data/
│   └── reference/
│       ├── ml_reference_set.csv             # 76-pair internal reference set (50 pos, 26 neg)
│       ├── ml_reference_set_expanded.csv    # 170-pair expanded set (142 pos, 28 neg)
│       ├── drug_name_crosswalk.csv          # 40-entry USAN-to-INN drug name mapping
│       ├── negative_control_rubric.md       # Four-criterion negative control selection rubric
│       ├── omop_reference_set.csv           # OMOP 399-pair reference set (Ryan et al. 2013)
│       ├── euadr_reference_set.csv          # EU-ADR reference set (Coloma et al. 2013)
│       ├── harpaz_2014_reference_set.csv    # Harpaz time-indexed reference standard
│       ├── harpaz_2014_drugs.csv            # Harpaz drug list
│       └── harpaz_2014_event_definitions.csv # Harpaz event definitions
└── docs/
    └── data_acquisition_guide.md    # Step-by-step DAEN download instructions
```

## Installation

```bash
# Clone the repository
git clone https://github.com/hayden-farquhar/TGA-DAEN-Pharmacovigilance.git
cd TGA-DAEN-Pharmacovigilance

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Analysis

Scripts are numbered sequentially and must be run in order. Each script reads from and writes to directories relative to the project root.

```bash
# Step 1: Download DAEN data (see docs/data_acquisition_guide.md)
# Place Excel/CSV exports in data/raw/

# Step 2: Merge batched exports
python scripts/01_data_acquisition.py

# Step 3: Clean and standardise
python scripts/02_data_cleaning.py

# Step 4: Disproportionality analysis (PRR, ROR, EBGM, BCPNN)
python scripts/03_disproportionality.py

# Step 5: Machine learning signal detection
python scripts/04_ml_signal_detection.py

# Step 6: Network analysis
python scripts/05_network_analysis.py

# Step 7: Generate figures
python scripts/06_visualisation.py

# Step 8: Sensitivity analyses
python scripts/07_sensitivity_validation.py

# Step 9: Additional validation (reference set, label concordance, TGA actions)
python scripts/07_additional_validation.py

# Step 10: Manuscript revision analyses (MGPS validation, SHAP, repeated CV, etc.)
python scripts/08_manuscript_revisions.py

# Step 11: Revision figures
python scripts/09_revision_figures.py

# Step 12: External reference set validation (OMOP, EU-ADR, Harpaz crosswalk)
python scripts/10_external_reference_validation.py

# Step 13: Expanded reference set ML validation
python scripts/11_expanded_reference_ml.py
```

### Expected runtime

On a modern machine (Apple M-series or equivalent):
- Scripts 01--07: ~5 minutes total
- Script 08 (manuscript revisions): ~10 minutes
- Scripts 10--11 (external validation): ~3 minutes

### Expected outputs

Scripts generate outputs in the following directories (created automatically):
- `outputs/tables/` -- CSV files with DPA scores, signal lists, validation results
- `outputs/figures/` -- PNG and PDF figures (300 DPI)
- `outputs/supplementary/` -- Full signal lists, sensitivity reports
- `outputs/revisions/` -- Revision analysis outputs

## Reference Sets

### Internal reference set (76 pairs)

- **50 positive controls**: Well-established drug--adverse event pairs from published pharmacovigilance literature and product information (e.g., clozapine--agranulocytosis, warfarin--haemorrhage, metformin--lactic acidosis)
- **26 negative controls**: Curated via a four-criterion rubric requiring absence from product labels, no published case reports, pharmacological implausibility, and avoidance of idiosyncratic reactions. See `data/reference/negative_control_rubric.md` for the full decision framework.

### Expanded reference set (170 pairs)

Incorporates controls from published reference sets (OMOP, EU-ADR, Harpaz) mapped to DAEN drug names via a 40-entry USAN-to-INN crosswalk.

## Key References

- DuMouchel W. Bayesian data mining in large frequency tables. *Am Stat*. 1999;53(3):177-190.
- Bate A, et al. A Bayesian neural network method for ADR signal generation. *Eur J Clin Pharmacol*. 1998;54(4):315-321.
- Evans SJW, et al. Use of PRRs for signal generation. *Pharmacoepidemiol Drug Saf*. 2001;10(6):483-486.
- Ryan PB, et al. Defining a reference set for drug safety. *Drug Saf*. 2013;36(Suppl 1):S33-S47.
- Wisniewski AFZ, et al. Good signal detection practices: IMI PROTECT. *Drug Saf*. 2016;39(6):469-490.
- Cutroneo PM, et al. READUS-PV reporting guideline. *Drug Saf*. 2024;47(6):553-570.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code, please cite:

> Farquhar H. Computational Pharmacovigilance of Australia's Database of Adverse Event Notifications: A Multi-Method Signal Detection Study Using Disproportionality Analysis, Machine Learning, and Network Analysis. *[Manuscript submitted for publication]*. 2026.

## Contact

Hayden Farquhar -- hayden.farquhar@icloud.com

ORCID: [0009-0002-6226-440X](https://orcid.org/0009-0002-6226-440X)
