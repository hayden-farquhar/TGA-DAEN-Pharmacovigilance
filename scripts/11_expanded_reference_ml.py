"""
Script 11: Expanded Reference Set ML Validation
================================================

Expands the ML reference set using OMOP/EU-ADR/Harpaz positive and negative
controls identified via the drug name crosswalk (script 10). Re-runs ML
validation with stratified CV and class weights to handle imbalance.

Steps:
  1. Load expansion candidates from script 10 output
  2. Audit negative candidates against 4-criterion rubric
  3. Deduplicate positives against existing reference set
  4. Create expanded reference set
  5. Engineer features from full DPA data
  6. Run 10×5-fold stratified CV with class weights
  7. Run LOOCV
  8. Compute DeLong's tests (expanded vs original)
  9. DPA reference validation on expanded set

Depends on outputs from scripts 03, 07, 10.

Outputs:
    data/reference/ml_reference_set_expanded.csv
    outputs/revisions/revision_expanded_ml_cv.csv
    outputs/revisions/revision_expanded_loocv.csv
    outputs/revisions/revision_expanded_delong.csv
    outputs/revisions/revision_expanded_dpa_performance.csv
    outputs/revisions/revision_expanded_summary_report.txt

Usage:
    python scripts/11_expanded_reference_ml.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, RepeatedStratifiedKFold, LeaveOneOut,
    cross_val_predict,
)
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from pathlib import Path
import time
import warnings
import io

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
REFERENCE_DIR = PROJECT_DIR / "data" / "reference"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "tables"
REVISION_DIR = PROJECT_DIR / "outputs" / "revisions"

# ── Feature definitions (same as scripts 04/07/08) ───────────────────────────

FEATURES_DPA = [
    "prr", "prr_lower95", "prr_upper95", "prr_chi2",
    "ror", "ror_lower95", "ror_upper95",
    "ebgm", "eb05",
    "ic", "ic025", "ic975",
]

FEATURES_NON_DPA = [
    "log_a", "log_expected", "log_n_drug", "log_n_reaction",
    "year_mean", "year_std", "year_range", "recent_fraction",
    "age_mean", "age_std", "pct_female",
    "mean_drugs_per_case", "mean_rxns_per_case",
]

FEATURES_ALL = FEATURES_DPA + FEATURES_NON_DPA


# ══════════════════════════════════════════════════════════════════════════════
#  NEGATIVE CANDIDATE AUDIT
# ══════════════════════════════════════════════════════════════════════════════

# Pairs that pass the 4-criterion rubric (manually audited):
#
# Criterion 1 (label check): Not on product label
# Criterion 2 (case reports): No published case reports
# Criterion 3 (pharmacological plausibility): No plausible pathway
# Criterion 4 (idiosyncratic avoidance): Not a broad idiosyncratic reaction
#
# Hepatotoxicity and nephrotoxicity outcomes are excluded entirely (too broad,
# like rhabdomyolysis/agranulocytosis — almost any drug can cause liver/kidney
# injury under certain circumstances).

APPROVED_NEGATIVES = [
    # Harpaz reference: no respiratory mechanism for antiepileptic
    ("levetiracetam", "• Dyspnoea"),
    # Harpaz reference: no respiratory mechanism for PPI
    ("pantoprazole sodium sesquihydrate", "• Dyspnoea"),
]

# Negatives rejected with reasoning:
REJECTED_NEGATIVES = {
    # --- EBGM signal (likely real associations) ---
    ("phenoxymethylpenicillin", "• Hepatitis Cholestatic"):
        "EBGM signal (6.35); penicillins cause cholestatic hepatitis",
    ("darbepoetin alfa", "• Renal Impairment"):
        "EBGM signal (14.15); confounding by indication (CKD treatment)",
    ("entecavir monohydrate", "• Renal Impairment"):
        "EBGM signal (7.2); renal adverse effects documented on label",
    ("lactulose", "• Acute Kidney Injury"):
        "EBGM signal (6.34); confounding (hepatorenal syndrome in cirrhosis)",
    ("prochlorperazine", "• Acute Kidney Injury"):
        "EBGM signal (12.6)",
    ("timolol maleate", "• Aplastic Anaemia"):
        "EBGM signal (16.4)",
    ("ferrous sulfate; folic acid", "• Acute Kidney Injury"):
        "EBGM signal (5.98); combination product complicates attribution",
    ("benserazide; levodopa", "• Acute Kidney Injury"):
        "EBGM signal (24.97)",
    # --- Broad organ-toxicity outcomes (same problem as rhabdomyolysis) ---
    ("griseofulvin", "• Hepatitis"):
        "Griseofulvin hepatotoxicity is well-documented",
    ("tinidazole", "• Hepatitis"):
        "Nitroimidazoles can cause hepatitis",
    ("clozapine", "• Blood Creatinine Increased"):
        "Clozapine nephritis documented; interstitial nephritis pathway",
    ("infliximab", "• Renal Failure"):
        "Lupus-like syndrome with nephritis documented for anti-TNFs",
    ("nortriptyline hydrochloride", "• Renal Impairment"):
        "Anticholinergic urinary retention → prerenal AKI pathway",
    ("hyoscine butylbromide", "• Acute Kidney Injury"):
        "Anticholinergic → urinary retention → AKI pathway",
    ("aluminium hydroxide; magnesium hydroxide; simethicone", "• Renal Impairment"):
        "Combination product with aluminium (nephrotoxic in renal failure)",
    ("temazepam", "• Acute Kidney Injury"):
        "CNS depression → hypotension → prerenal AKI in overdose",
    ("carbidopa monohydrate; levodopa", "• Hepatitis Cholestatic"):
        "Levodopa hepatotoxicity documented (rare)",
    ("glyceryl trinitrate", "• Hepatitis"):
        "Only 6 reports but broad hepatotoxicity outcome; conservative exclusion",
    ("levothyroxine sodium", "• Acute Kidney Injury"):
        "Confounding by indication (hypothyroidism → CKD pathway)",
    # --- Documented associations ---
    ("irbesartan", "• Pancytopenia"):
        "ARB-induced pancytopenia documented in case reports",
    ("prochlorperazine maleate", "• Haematemesis"):
        "Phenothiazines can cause blood dyscrasias → GI bleeding",
    ("temazepam", "• Haematemesis"):
        "Only 5 reports; benzodiazepine GI effects marginal but conservative",
    ("simvastatin", "• Gastrointestinal Haemorrhage"):
        "Confounding by aspirin co-prescription in cardiovascular patients",
    ("methylphenidate hydrochloride", "• Nightmare"):
        "Stimulant-induced sleep disruption and nightmares well-documented",
    ("codeine phosphate hemihydrate; doxylamine succinate; paracetamol",
     "• Blood Creatinine Increased"):
        "Paracetamol nephrotoxicity in overdose; combination product",
}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════


def _clopper_pearson(k, n, alpha=0.05):
    """Clopper-Pearson exact binomial 95% CI."""
    if n == 0:
        return (0.0, 1.0)
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (lo, hi)


def _delong_test(y_true, scores1, scores2):
    """DeLong's test for comparing two AUC-ROC values."""
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)

    pos_scores1 = scores1[y_true == 1]
    neg_scores1 = scores1[y_true == 0]
    pos_scores2 = scores2[y_true == 1]
    neg_scores2 = scores2[y_true == 0]

    # Placement values
    V10_1 = np.array([np.mean(s > neg_scores1) + 0.5 * np.mean(s == neg_scores1)
                       for s in pos_scores1])
    V01_1 = np.array([np.mean(pos_scores1 > s) + 0.5 * np.mean(pos_scores1 == s)
                       for s in neg_scores1])
    V10_2 = np.array([np.mean(s > neg_scores2) + 0.5 * np.mean(s == neg_scores2)
                       for s in pos_scores2])
    V01_2 = np.array([np.mean(pos_scores2 > s) + 0.5 * np.mean(pos_scores2 == s)
                       for s in neg_scores2])

    auc1 = np.mean(V10_1)
    auc2 = np.mean(V10_2)

    # Covariance
    S10 = np.cov(np.vstack([V10_1, V10_2]))
    S01 = np.cov(np.vstack([V01_1, V01_2]))
    S = S10 / n1 + S01 / n0

    diff = auc1 - auc2
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return auc1, auc2, 0.0, 1.0
    z = diff / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc1, auc2, z, p


def engineer_features(disp_df, cases, drugs, reactions):
    """Engineer per-pair features from case-level data."""
    cases["case_number"] = cases["case_number"].astype(str)
    drugs["case_number"] = drugs["case_number"].astype(str)
    reactions["case_number"] = reactions["case_number"].astype(str)

    susp = drugs[drugs["suspected"] == "suspected"][
        ["case_number", "active_ingredient"]].copy()
    susp = susp.dropna(subset=["active_ingredient"])
    susp["active_ingredient"] = susp["active_ingredient"].str.strip().str.lower()
    susp = susp.drop_duplicates()
    rxn = reactions[["case_number", "reaction"]].drop_duplicates()

    case_meta = cases[["case_number"]].copy()
    case_meta["report_year"] = pd.to_datetime(
        cases["report_date"], errors="coerce", format="mixed",
        dayfirst=True).dt.year
    case_meta["age_numeric"] = pd.to_numeric(cases["age"], errors="coerce")
    if "sex" in cases.columns:
        case_meta["is_female"] = (
            cases["sex"].str.lower().eq("female").astype(float)
        )

    drugs_per_case = susp.groupby("case_number").size().rename("n_drugs_case")
    rxns_per_case = rxn.groupby("case_number").size().rename("n_rxns_case")

    triples = susp.merge(rxn, on="case_number", how="inner")
    triples = triples.merge(case_meta, on="case_number", how="left")
    triples = triples.merge(drugs_per_case, on="case_number", how="left")
    triples = triples.merge(rxns_per_case, on="case_number", how="left")

    agg = triples.groupby(["active_ingredient", "reaction"]).agg(
        year_mean=("report_year", "mean"),
        year_std=("report_year", "std"),
        year_min=("report_year", "min"),
        year_max=("report_year", "max"),
        age_mean=("age_numeric", "mean"),
        age_std=("age_numeric", "std"),
        pct_female=("is_female", "mean"),
        mean_drugs_per_case=("n_drugs_case", "mean"),
        mean_rxns_per_case=("n_rxns_case", "mean"),
    ).reset_index()
    agg["year_range"] = agg["year_max"] - agg["year_min"]

    n_recent = triples[triples["report_year"] >= 2024].groupby(
        ["active_ingredient", "reaction"]).size().rename("n_recent")
    n_total = triples.groupby(
        ["active_ingredient", "reaction"]).size().rename("n_total_t")
    recent = pd.concat([n_recent, n_total], axis=1).fillna(0)
    recent["recent_fraction"] = recent["n_recent"] / recent["n_total_t"]
    agg = agg.merge(recent[["recent_fraction"]].reset_index(),
                    on=["active_ingredient", "reaction"], how="left")
    agg["recent_fraction"] = agg["recent_fraction"].fillna(0)

    disp_df = disp_df.merge(agg, on=["active_ingredient", "reaction"],
                            how="left")
    disp_df["log_a"] = np.log1p(disp_df["a"])
    disp_df["log_expected"] = np.log1p(disp_df["expected"])
    disp_df["log_n_drug"] = np.log1p(disp_df["n_drug"])
    disp_df["log_n_reaction"] = np.log1p(disp_df["n_reaction"])

    return disp_df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1: BUILD EXPANDED REFERENCE SET
# ══════════════════════════════════════════════════════════════════════════════


def build_expanded_reference_set(out):
    """Load expansion candidates, audit, deduplicate, create expanded set."""
    out.write("=" * 70 + "\n")
    out.write("  STEP 1: Build Expanded Reference Set\n")
    out.write("=" * 70 + "\n\n")

    # Load existing reference set
    ref_orig = pd.read_csv(REFERENCE_DIR / "ml_reference_set.csv")
    out.write(f"  Original reference set: {len(ref_orig)} pairs "
              f"({(ref_orig['label'] == 1).sum()} pos, "
              f"{(ref_orig['label'] == 0).sum()} neg)\n")

    # Create lookup of existing pairs (lowercase for matching)
    existing_pairs = set()
    for _, row in ref_orig.iterrows():
        key = (row["active_ingredient"].strip().lower(),
               row["reaction"].strip().lower())
        existing_pairs.add(key)

    # Load expansion candidates
    cand = pd.read_csv(REVISION_DIR / "revision_reference_expansion_candidates.csv")
    out.write(f"  Expansion candidates: {len(cand)} pairs "
              f"({(cand['ground_truth'] == 1).sum()} pos, "
              f"{(cand['ground_truth'] == 0).sum()} neg)\n\n")

    # ── Process negatives ─────────────────────────────────────────────────
    neg_cand = cand[cand["ground_truth"] == 0].copy()
    out.write(f"  Negative candidates: {len(neg_cand)}\n")

    approved_keys = set(
        (d.strip().lower(), r.strip().lower()) for d, r in APPROVED_NEGATIVES
    )

    new_negatives = []
    for _, row in neg_cand.iterrows():
        key = (row["daen_drug"].strip().lower(),
               row["daen_reaction"].strip().lower())
        if key in existing_pairs:
            out.write(f"    SKIP (already in set): {key[0]} → {key[1]}\n")
            continue
        if key in approved_keys:
            new_negatives.append(row)
            out.write(f"    APPROVED: {key[0]} → {key[1]}\n")
        else:
            reason = REJECTED_NEGATIVES.get(
                (row["daen_drug"], row["daen_reaction"]), "Not in approved list"
            )
            out.write(f"    REJECTED: {key[0]} → {key[1]} — {reason}\n")

    out.write(f"\n  Approved new negatives: {len(new_negatives)}\n\n")

    # ── Process positives ─────────────────────────────────────────────────
    pos_cand = cand[cand["ground_truth"] == 1].copy()
    out.write(f"  Positive candidates: {len(pos_cand)}\n")

    new_positives = []
    seen_pairs = set(existing_pairs)  # track for dedup within expansion too
    for _, row in pos_cand.iterrows():
        key = (row["daen_drug"].strip().lower(),
               row["daen_reaction"].strip().lower())
        if key in seen_pairs:
            out.write(f"    SKIP (duplicate): {key[0]} → {key[1]}\n")
            continue
        seen_pairs.add(key)
        new_positives.append(row)
        out.write(f"    ADDED: {key[0]} → {key[1]} "
                  f"(n={row['n_reports']}, EBGM={row['ebgm']:.1f})\n")

    out.write(f"\n  New unique positives: {len(new_positives)}\n")

    # ── Build expanded reference set ──────────────────────────────────────

    # Load full DPA data for the new pairs
    disp_full = pd.read_csv(
        PROJECT_DIR / "outputs" / "tables" / "disproportionality_full.csv"
    )

    # Original set already has DPA columns — keep as-is
    expanded_rows = []
    for _, row in ref_orig.iterrows():
        expanded_rows.append({
            "active_ingredient": row["active_ingredient"],
            "reaction": row["reaction"],
            "label": row["label"],
            "control_type": row["control_type"],
            "source": "original",
            "a": row["a"],
            "expected": row["expected"],
            "prr": row["prr"],
            "ror": row["ror"],
            "ebgm": row["ebgm"],
            "ic": row["ic"],
        })

    # Add new positives
    for row in new_positives:
        drug = row["daen_drug"]
        rxn = row["daen_reaction"]
        # Look up full DPA data
        match = disp_full[
            (disp_full["active_ingredient"] == drug) &
            (disp_full["reaction"] == rxn)
        ]
        if len(match) == 0:
            out.write(f"  WARNING: No DPA match for {drug} → {rxn}\n")
            continue
        m = match.iloc[0]
        expanded_rows.append({
            "active_ingredient": drug,
            "reaction": rxn,
            "label": 1,
            "control_type": "positive",
            "source": row["source"],
            "a": m["a"],
            "expected": m["expected"],
            "prr": m["prr"],
            "ror": m["ror"],
            "ebgm": m["ebgm"],
            "ic": m["ic"],
        })

    # Add new negatives
    for row in new_negatives:
        drug = row["daen_drug"]
        rxn = row["daen_reaction"]
        match = disp_full[
            (disp_full["active_ingredient"] == drug) &
            (disp_full["reaction"] == rxn)
        ]
        if len(match) == 0:
            out.write(f"  WARNING: No DPA match for {drug} → {rxn}\n")
            continue
        m = match.iloc[0]
        expanded_rows.append({
            "active_ingredient": drug,
            "reaction": rxn,
            "label": 0,
            "control_type": "negative",
            "source": row["source"],
            "a": m["a"],
            "expected": m["expected"],
            "prr": m["prr"],
            "ror": m["ror"],
            "ebgm": m["ebgm"],
            "ic": m["ic"],
        })

    expanded_df = pd.DataFrame(expanded_rows)
    n_pos = (expanded_df["label"] == 1).sum()
    n_neg = (expanded_df["label"] == 0).sum()
    out.write(f"\n  Expanded reference set: {len(expanded_df)} pairs "
              f"({n_pos} pos, {n_neg} neg, ratio {n_pos/n_neg:.1f}:1)\n")
    out.write(f"  By source: {expanded_df['source'].value_counts().to_dict()}\n")

    # Save
    expanded_df.to_csv(REFERENCE_DIR / "ml_reference_set_expanded.csv",
                       index=False)
    out.write(f"  Saved: data/reference/ml_reference_set_expanded.csv\n\n")

    return expanded_df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2: DPA REFERENCE VALIDATION ON EXPANDED SET
# ══════════════════════════════════════════════════════════════════════════════


def dpa_validation(expanded_df, disp_featured, out):
    """Evaluate DPA binary thresholds against expanded reference set."""
    out.write("=" * 70 + "\n")
    out.write("  STEP 2: DPA Validation on Expanded Set\n")
    out.write("=" * 70 + "\n\n")

    ref_merged = expanded_df[["active_ingredient", "reaction", "label"]].merge(
        disp_featured[["active_ingredient", "reaction",
                        "signal_prr", "signal_ror", "signal_ebgm",
                        "signal_bcpnn", "n_methods_signal"]],
        on=["active_ingredient", "reaction"], how="left"
    )

    y = ref_merged["label"].values
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()

    methods = [
        ("PRR", "signal_prr"),
        ("ROR", "signal_ror"),
        ("EBGM", "signal_ebgm"),
        ("BCPNN", "signal_bcpnn"),
        ("Consensus (4/4)", None),
        ("Any method (>=1)", None),
    ]

    rows = []
    for name, col in methods:
        if col:
            pred = ref_merged[col].fillna(False).astype(int).values
        elif "4/4" in name:
            pred = (ref_merged["n_methods_signal"].fillna(0) >= 4).astype(int).values
        else:
            pred = (ref_merged["n_methods_signal"].fillna(0) >= 1).astype(int).values

        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        sens = tp / n_pos if n_pos > 0 else 0
        spec = tn / n_neg if n_neg > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) > 0 else 0

        sens_ci = _clopper_pearson(tp, n_pos)
        spec_ci = _clopper_pearson(tn, n_neg)

        rows.append({
            "method": name,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "sensitivity": round(sens, 4),
            "sens_lower95": round(sens_ci[0], 4),
            "sens_upper95": round(sens_ci[1], 4),
            "specificity": round(spec, 4),
            "spec_lower95": round(spec_ci[0], 4),
            "spec_upper95": round(spec_ci[1], 4),
            "PPV": round(ppv, 4), "NPV": round(npv, 4),
            "F1": round(f1, 4),
        })

        out.write(f"  {name:<25s} Sens={sens:.3f} [{sens_ci[0]:.3f}-{sens_ci[1]:.3f}]  "
                  f"Spec={spec:.3f} [{spec_ci[0]:.3f}-{spec_ci[1]:.3f}]  "
                  f"F1={f1:.3f}\n")

    dpa_df = pd.DataFrame(rows)
    dpa_df.to_csv(REVISION_DIR / "revision_expanded_dpa_performance.csv",
                  index=False)
    out.write(f"\n  Saved: revision_expanded_dpa_performance.csv\n\n")
    return dpa_df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3: ML REPEATED STRATIFIED CV WITH CLASS WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════


def ml_repeated_cv(expanded_df, disp_featured, out):
    """10×5-fold stratified CV with class weights for imbalance handling."""
    out.write("=" * 70 + "\n")
    out.write("  STEP 3: ML Repeated Stratified CV (10×5-fold)\n")
    out.write("=" * 70 + "\n\n")

    ref_merged = expanded_df[["active_ingredient", "reaction", "label"]].merge(
        disp_featured, on=["active_ingredient", "reaction"], how="left"
    )

    # Drop pairs that didn't match feature data
    n_before = len(ref_merged)
    ref_merged = ref_merged.dropna(subset=["prr"])
    n_after = len(ref_merged)
    if n_before != n_after:
        out.write(f"  WARNING: {n_before - n_after} pairs dropped "
                  f"(no feature match)\n")

    y = ref_merged["label"].values
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    out.write(f"  Dataset: {len(y)} pairs ({n_pos} pos, {n_neg} neg)\n")

    # Class weight ratio for XGBoost
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    out.write(f"  scale_pos_weight: {scale_pos_weight:.3f}\n")
    out.write(f"  RF class_weight: 'balanced'\n\n")

    model_configs = [
        ("XGBoost All", FEATURES_ALL, lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=42, verbosity=0)),
        ("XGBoost DPA", FEATURES_DPA, lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=42, verbosity=0)),
        ("XGBoost NonDPA", FEATURES_NON_DPA, lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=42, verbosity=0)),
        ("RF All", FEATURES_ALL, lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1)),
        ("RF DPA", FEATURES_DPA, lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ]

    # Also run without class weights for comparison
    model_configs_noweight = [
        ("XGBoost All (no weight)", FEATURES_ALL, lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42, verbosity=0)),
        ("RF All (no weight)", FEATURES_ALL, lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
    ]

    all_configs = model_configs + model_configs_noweight

    cv_rows = []
    summary_rows = []

    for model_name, features, factory in all_configs:
        X = ref_merged[features].values
        fold_aucs = []
        fold_pr_aucs = []

        for rep in range(10):
            cv = StratifiedKFold(n_splits=5, shuffle=True,
                                 random_state=42 + rep)
            for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", factory()),
                ])
                pipe.fit(X[train_idx], y[train_idx])
                y_prob = pipe.predict_proba(X[test_idx])[:, 1]

                try:
                    auc = roc_auc_score(y[test_idx], y_prob)
                except ValueError:
                    auc = np.nan

                try:
                    pr_auc = average_precision_score(y[test_idx], y_prob)
                except ValueError:
                    pr_auc = np.nan

                fold_aucs.append(auc)
                fold_pr_aucs.append(pr_auc)

                cv_rows.append({
                    "model": model_name, "repeat": rep + 1,
                    "fold": fold_i + 1,
                    "auc_roc": round(auc, 4) if not np.isnan(auc) else np.nan,
                    "pr_auc": round(pr_auc, 4) if not np.isnan(pr_auc) else np.nan,
                })

        aucs = [a for a in fold_aucs if not np.isnan(a)]
        pr_aucs = [a for a in fold_pr_aucs if not np.isnan(a)]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs, ddof=1)
        ci_lo = mean_auc - 1.96 * std_auc / np.sqrt(len(aucs))
        ci_hi = mean_auc + 1.96 * std_auc / np.sqrt(len(aucs))
        mean_pr = np.mean(pr_aucs) if pr_aucs else np.nan

        summary_rows.append({
            "model": model_name,
            "n_folds": len(aucs),
            "mean_auc_roc": round(mean_auc, 4),
            "std_auc_roc": round(std_auc, 4),
            "ci_lower": round(max(ci_lo, 0), 4),
            "ci_upper": round(min(ci_hi, 1), 4),
            "mean_pr_auc": round(mean_pr, 4) if not np.isnan(mean_pr) else np.nan,
        })

        out.write(f"  {model_name:<30s} AUC={mean_auc:.4f} ± {std_auc:.4f}  "
                  f"[{max(ci_lo, 0):.4f}-{min(ci_hi, 1):.4f}]  "
                  f"PR-AUC={mean_pr:.4f}\n")

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(REVISION_DIR / "revision_expanded_ml_cv.csv", index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(REVISION_DIR / "revision_expanded_ml_cv_summary.csv",
                      index=False)
    out.write(f"\n  Saved: revision_expanded_ml_cv.csv, "
              f"revision_expanded_ml_cv_summary.csv\n\n")

    return ref_merged, summary_df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4: LOOCV
# ══════════════════════════════════════════════════════════════════════════════


def ml_loocv(ref_merged, out):
    """Leave-one-out CV on expanded set."""
    out.write("=" * 70 + "\n")
    out.write("  STEP 4: Leave-One-Out CV\n")
    out.write("=" * 70 + "\n\n")

    y = ref_merged["label"].values
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    configs = [
        ("XGBoost All (weighted)", FEATURES_ALL, lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=42, verbosity=0)),
        ("RF All (balanced)", FEATURES_ALL, lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ]

    loocv_rows = []
    loo = LeaveOneOut()

    for model_name, features, factory in configs:
        X = ref_merged[features].values
        y_pred_proba = np.zeros(len(y))

        for train_idx, test_idx in loo.split(X):
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", factory()),
            ])
            pipe.fit(X[train_idx], y[train_idx])
            y_pred_proba[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

        auc = roc_auc_score(y, y_pred_proba)
        pr_auc = average_precision_score(y, y_pred_proba)
        brier = brier_score_loss(y, y_pred_proba)

        loocv_rows.append({
            "model": model_name,
            "n_samples": len(y),
            "n_pos": int(n_pos),
            "n_neg": int(n_neg),
            "auc_roc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "brier_score": round(brier, 4),
        })

        out.write(f"  {model_name:<30s} AUC={auc:.4f}  "
                  f"PR-AUC={pr_auc:.4f}  Brier={brier:.4f}\n")

    loocv_df = pd.DataFrame(loocv_rows)
    loocv_df.to_csv(REVISION_DIR / "revision_expanded_loocv.csv", index=False)
    out.write(f"\n  Saved: revision_expanded_loocv.csv\n\n")
    return loocv_df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5: DELONG COMPARISONS (ORIGINAL vs EXPANDED)
# ══════════════════════════════════════════════════════════════════════════════


def delong_comparisons(ref_merged, out):
    """DeLong's tests comparing models on expanded set."""
    out.write("=" * 70 + "\n")
    out.write("  STEP 5: DeLong's Tests\n")
    out.write("=" * 70 + "\n\n")

    y = ref_merged["label"].values
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Get OOF predictions for each model
    models = {}

    # XGBoost all (weighted)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=42, verbosity=0)),
    ])
    X_all = ref_merged[FEATURES_ALL].values
    models["XGB_all_w"] = cross_val_predict(pipe, X_all, y, cv=cv,
                                             method="predict_proba")[:, 1]

    # XGBoost DPA (weighted)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=42, verbosity=0)),
    ])
    X_dpa = ref_merged[FEATURES_DPA].values
    models["XGB_dpa_w"] = cross_val_predict(pipe, X_dpa, y, cv=cv,
                                             method="predict_proba")[:, 1]

    # XGBoost non-DPA (weighted)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=42, verbosity=0)),
    ])
    X_nondpa = ref_merged[FEATURES_NON_DPA].values
    models["XGB_nondpa_w"] = cross_val_predict(pipe, X_nondpa, y, cv=cv,
                                                method="predict_proba")[:, 1]

    # Random Forest (balanced)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced",
            random_state=42, n_jobs=-1)),
    ])
    models["RF_all_b"] = cross_val_predict(pipe, X_all, y, cv=cv,
                                            method="predict_proba")[:, 1]

    # Single DPA scores
    for dpa_name, col in [("EBGM", "ebgm"), ("PRR", "prr"),
                           ("ROR", "ror"), ("IC", "ic")]:
        scores = ref_merged[col].values.copy()
        scores = np.nan_to_num(scores, nan=0.0,
                               posinf=np.nanmax(scores[np.isfinite(scores)]),
                               neginf=0.0)
        models[dpa_name] = scores

    comparisons = [
        ("XGB_all_w", "XGB_nondpa_w", "XGBoost all vs non-DPA"),
        ("XGB_all_w", "RF_all_b", "XGBoost vs Random Forest"),
        ("XGB_all_w", "EBGM", "XGBoost vs EBGM standalone"),
        ("XGB_all_w", "PRR", "XGBoost vs PRR standalone"),
        ("XGB_dpa_w", "EBGM", "XGBoost DPA vs EBGM standalone"),
        ("RF_all_b", "EBGM", "Random Forest vs EBGM standalone"),
        ("XGB_all_w", "XGB_dpa_w", "XGBoost all vs DPA-only"),
    ]

    delong_rows = []
    for key1, key2, desc in comparisons:
        try:
            auc1, auc2, z, p = _delong_test(y, models[key1], models[key2])
            delong_rows.append({
                "comparison": desc,
                "model1": key1, "auc1": round(auc1, 4),
                "model2": key2, "auc2": round(auc2, 4),
                "z_statistic": round(z, 4),
                "p_value": round(p, 6),
                "significant_005": p < 0.05,
            })
            sig = " *" if p < 0.05 else ""
            out.write(f"  {desc}: AUC {auc1:.3f} vs {auc2:.3f}, "
                      f"z={z:.3f}, p={p:.4f}{sig}\n")
        except Exception as e:
            out.write(f"  {desc}: FAILED ({e})\n")

    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(REVISION_DIR / "revision_expanded_delong.csv", index=False)
    out.write(f"\n  Saved: revision_expanded_delong.csv\n\n")
    return delong_df


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 6: SENSITIVITY BY SIGNAL STRENGTH TIER
# ══════════════════════════════════════════════════════════════════════════════


def tiered_evaluation(expanded_df, disp_featured, out):
    """Stratify positives by EBGM strength and evaluate per tier."""
    out.write("=" * 70 + "\n")
    out.write("  STEP 6: Tiered Sensitivity Evaluation\n")
    out.write("=" * 70 + "\n\n")

    ref_merged = expanded_df[["active_ingredient", "reaction", "label"]].merge(
        disp_featured[["active_ingredient", "reaction", "ebgm", "eb05",
                        "signal_ebgm", "signal_prr", "signal_ror",
                        "signal_bcpnn", "n_methods_signal"]],
        on=["active_ingredient", "reaction"], how="left"
    )

    positives = ref_merged[ref_merged["label"] == 1].copy()

    # Tier by EBGM strength
    positives["tier"] = pd.cut(
        positives["ebgm"],
        bins=[-np.inf, 2, 5, 50, np.inf],
        labels=["Sub-threshold (<2)", "Weak (2-5)", "Moderate (5-50)",
                "Strong (>50)"]
    )

    rows = []
    for tier_name, group in positives.groupby("tier", observed=True):
        n = len(group)
        for method, col in [("EBGM", "signal_ebgm"), ("PRR", "signal_prr"),
                             ("ROR", "signal_ror"), ("BCPNN", "signal_bcpnn"),
                             ("Consensus", None)]:
            if col:
                detected = group[col].fillna(False).sum()
            else:
                detected = (group["n_methods_signal"].fillna(0) >= 4).sum()

            sens = detected / n if n > 0 else 0
            ci = _clopper_pearson(int(detected), n)

            rows.append({
                "tier": tier_name,
                "n_pairs": n,
                "method": method,
                "detected": int(detected),
                "sensitivity": round(sens, 4),
                "ci_lower": round(ci[0], 4),
                "ci_upper": round(ci[1], 4),
            })

        out.write(f"  {tier_name}: n={n}\n")

    tier_df = pd.DataFrame(rows)
    tier_df.to_csv(REVISION_DIR / "revision_expanded_tiered.csv", index=False)
    out.write(f"\n  Saved: revision_expanded_tiered.csv\n\n")

    # Print summary
    for method in ["EBGM", "PRR", "Consensus"]:
        out.write(f"  {method} sensitivity by tier:\n")
        subset = tier_df[tier_df["method"] == method]
        for _, r in subset.iterrows():
            out.write(f"    {r['tier']}: {r['detected']}/{r['n_pairs']} "
                      f"= {r['sensitivity']:.3f} "
                      f"[{r['ci_lower']:.3f}-{r['ci_upper']:.3f}]\n")
        out.write("\n")

    return tier_df


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    out = io.StringIO()

    out.write("=" * 70 + "\n")
    out.write("  Script 11: Expanded Reference Set ML Validation\n")
    out.write("  " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    out.write("=" * 70 + "\n\n")

    REVISION_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Build expanded set ────────────────────────────────────────
    expanded_df = build_expanded_reference_set(out)

    # ── Load and engineer features ────────────────────────────────────────
    out.write("Loading DPA data and engineering features ...\n")
    disp_full = pd.read_csv(
        PROJECT_DIR / "outputs" / "tables" / "disproportionality_full.csv"
    )
    cases = pd.read_csv(PROCESSED_DIR / "daen_cases.csv")
    drugs = pd.read_csv(PROCESSED_DIR / "daen_case_drugs.csv")
    reactions = pd.read_csv(PROCESSED_DIR / "daen_case_reactions.csv")

    disp_featured = engineer_features(disp_full, cases, drugs, reactions)
    out.write(f"  Feature-enriched pairs: {len(disp_featured):,}\n\n")

    # ── Step 2: DPA validation ────────────────────────────────────────────
    dpa_df = dpa_validation(expanded_df, disp_featured, out)

    # ── Step 3: ML repeated CV ────────────────────────────────────────────
    ref_merged, cv_summary = ml_repeated_cv(expanded_df, disp_featured, out)

    # ── Step 4: LOOCV ─────────────────────────────────────────────────────
    loocv_df = ml_loocv(ref_merged, out)

    # ── Step 5: DeLong comparisons ────────────────────────────────────────
    delong_df = delong_comparisons(ref_merged, out)

    # ── Step 6: Tiered evaluation ─────────────────────────────────────────
    tier_df = tiered_evaluation(expanded_df, disp_featured, out)

    # ── Final summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t0
    out.write(f"\n{'=' * 70}\n")
    out.write(f"  Completed in {elapsed:.1f} seconds\n")
    out.write(f"{'=' * 70}\n")

    report = out.getvalue()
    print(report)

    with open(REVISION_DIR / "revision_expanded_summary_report.txt", "w") as f:
        f.write(report)

    print(f"\nAll outputs saved to {REVISION_DIR}/")


if __name__ == "__main__":
    main()
