"""
Script 04: Machine Learning Signal Detection

Trains XGBoost and random forest classifiers on labelled positive/negative
control pairs to benchmark ML signal detection against traditional
disproportionality methods (PRR, ROR, EBGM, BCPNN).

Features are engineered from:
  - Disproportionality scores (PRR, ROR, EBGM, IC)
  - Reporting volume (observed, expected, marginals)
  - Temporal patterns (reporting trend, recency)
  - Demographics (age, sex distributions)
  - Polypharmacy (co-reported drugs per case)

Evaluation:
  - 5-fold stratified cross-validation
  - AUC-ROC comparison: individual DPA methods vs ML models
  - Feature importance analysis

Outputs (in outputs/tables/):
  - ml_reference_set.csv          Matched control pairs with labels
  - ml_evaluation_results.csv     AUC-ROC for all methods
  - ml_feature_importance.csv     Feature importance rankings
  - ml_all_pairs_scored.csv       ML probability scores for all 103K pairs

Usage:
    python scripts/04_ml_signal_detection.py
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "tables"
REFERENCE_DIR = PROJECT_DIR / "data" / "reference"

# ═══════════════════════════════════════════════════════════════════════════════
#  REFERENCE SET: POSITIVE AND NEGATIVE CONTROL PAIRS
# ═══════════════════════════════════════════════════════════════════════════════
#
# Positive controls: established drug–AE associations from product labels,
# regulatory warnings, and pharmacovigilance literature.
# Negative controls: pharmacologically implausible drug–AE combinations
# using the same drugs/reactions as positives.
#
# Each tuple: (active_ingredient_substring, MedDRA_reaction_term)

POSITIVE_CONTROLS = [
    # Antipsychotics — haematological, metabolic, cardiac
    ("clozapine", "Agranulocytosis"),
    ("clozapine", "Neutropenia"),
    ("clozapine", "Myocarditis"),
    ("clozapine", "Weight Increased"),
    ("olanzapine", "Weight Increased"),
    ("olanzapine", "Diabetes Mellitus"),
    ("haloperidol", "Neuroleptic Malignant Syndrome"),
    # Statins — musculoskeletal
    ("atorvastatin", "Rhabdomyolysis"),
    ("simvastatin", "Rhabdomyolysis"),
    ("rosuvastatin", "Rhabdomyolysis"),
    ("atorvastatin", "Myalgia"),
    ("simvastatin", "Myalgia"),
    # NSAIDs — GI and renal
    ("celecoxib", "Gastrointestinal Haemorrhage"),
    ("diclofenac", "Gastrointestinal Haemorrhage"),
    ("ibuprofen", "Gastrointestinal Haemorrhage"),
    ("naproxen", "Gastrointestinal Haemorrhage"),
    ("celecoxib", "Renal Impairment"),
    ("diclofenac", "Renal Impairment"),
    # Anticoagulants — bleeding, HIT
    ("warfarin", "Haemorrhage"),
    ("heparin", "Thrombocytopenia"),
    ("rivaroxaban", "Haemorrhage"),
    ("enoxaparin", "Thrombocytopenia"),
    # Antiepileptics — dermatological, hepatic
    ("carbamazepine", "Stevens-Johnson Syndrome"),
    ("lamotrigine", "Stevens-Johnson Syndrome"),
    ("phenytoin", "Stevens-Johnson Syndrome"),
    ("carbamazepine", "Hyponatraemia"),
    ("valproate", "Pancreatitis"),
    ("valproate", "Thrombocytopenia"),
    # Antibiotics — tendon, allergic
    ("ciprofloxacin", "Tendon Rupture"),
    ("ciprofloxacin", "Tendon Disorder"),
    ("amoxicillin", "Anaphylactic Reaction"),
    ("flucloxacillin", "Hepatitis"),
    # Antidepressants — serotonergic
    ("sertraline", "Serotonin Syndrome"),
    ("venlafaxine", "Serotonin Syndrome"),
    ("fluoxetine", "Serotonin Syndrome"),
    # Immunosuppressants — infection, renal, hepatic
    ("methotrexate", "Pancytopenia"),
    ("ciclosporin", "Renal Impairment"),
    ("tacrolimus", "Renal Impairment"),
    ("infliximab", "Tuberculosis"),
    ("adalimumab", "Tuberculosis"),
    # Cardiac — thyroid, pulmonary
    ("amiodarone", "Hypothyroidism"),
    ("amiodarone", "Pulmonary Fibrosis"),
    # Metabolic / endocrine
    ("metformin", "Lactic Acidosis"),
    ("lithium", "Hypothyroidism"),
    # Dermatology / rheumatology
    ("allopurinol", "Stevens-Johnson Syndrome"),
    # Oncology
    ("doxorubicin", "Cardiomyopathy"),
    ("paclitaxel", "Peripheral Neuropathy"),
    # Other well-known pairs
    ("prednisolone", "Osteoporosis"),
    ("oxycodone", "Drug Dependence"),
    ("isotretinoin", "Depression"),
    ("sulfamethoxazole", "Stevens-Johnson Syndrome"),
]

NEGATIVE_CONTROLS = [
    # Literature-verified pharmacologically implausible drug–AE combinations.
    # Curated Feb 2026 after systematic review found that rhabdomyolysis,
    # agranulocytosis, and SJS are poor negative reactions (documented across
    # many drug classes as idiosyncratic effects). This revised set uses
    # mechanism-specific reactions paired with drugs lacking the relevant
    # pharmacological activity, plus common reactions paired with drugs that
    # have no pathway to produce them — maximising both implausibility and
    # data-matching probability (≥3 reports required).
    #
    # ── Tendon Rupture (fluoroquinolone collagen disruption–specific) ──
    ("clozapine", "Tendon Rupture"),
    ("olanzapine", "Tendon Rupture"),
    ("haloperidol", "Tendon Rupture"),
    ("lithium", "Tendon Rupture"),
    ("amiodarone", "Tendon Rupture"),
    ("metformin", "Tendon Rupture"),
    ("methotrexate", "Tendon Rupture"),
    ("doxorubicin", "Tendon Rupture"),
    ("venlafaxine", "Tendon Rupture"),
    ("warfarin", "Tendon Rupture"),
    ("carbamazepine", "Tendon Rupture"),
    ("paclitaxel", "Tendon Rupture"),
    ("oxycodone", "Tendon Rupture"),
    #
    # ── Serotonin Syndrome (requires 5-HT reuptake inhibition / agonism) ──
    ("prednisolone", "Serotonin Syndrome"),
    ("warfarin", "Serotonin Syndrome"),
    ("metformin", "Serotonin Syndrome"),
    ("atorvastatin", "Serotonin Syndrome"),
    ("amiodarone", "Serotonin Syndrome"),
    ("allopurinol", "Serotonin Syndrome"),
    ("diclofenac", "Serotonin Syndrome"),
    #
    # ── Hypothyroidism (amiodarone / lithium / checkpoint inhibitor–specific) ──
    ("atorvastatin", "Hypothyroidism"),
    ("simvastatin", "Hypothyroidism"),
    ("ciprofloxacin", "Hypothyroidism"),
    ("diclofenac", "Hypothyroidism"),
    ("celecoxib", "Hypothyroidism"),
    ("oxycodone", "Hypothyroidism"),
    #
    # ── Osteoporosis (corticosteroid / aromatase inhibitor–specific) ──
    ("clozapine", "Osteoporosis"),          # actually protective (literature)
    ("metformin", "Osteoporosis"),           # bone-protective (literature)
    ("atorvastatin", "Osteoporosis"),        # statins may be bone-protective
    ("ciprofloxacin", "Osteoporosis"),
    ("allopurinol", "Osteoporosis"),
    ("diclofenac", "Osteoporosis"),
    #
    # ── Lactic Acidosis (biguanide / NRTI / linezolid–specific) ──
    ("allopurinol", "Lactic Acidosis"),
    ("carbamazepine", "Lactic Acidosis"),
    ("warfarin", "Lactic Acidosis"),
    ("ciprofloxacin", "Lactic Acidosis"),
    #
    # ── Drug Dependence (requires mu-opioid or GABAergic agonism) ──
    ("metformin", "Drug Dependence"),
    ("warfarin", "Drug Dependence"),
    ("atorvastatin", "Drug Dependence"),
    ("allopurinol", "Drug Dependence"),
    ("ciprofloxacin", "Drug Dependence"),
    ("clozapine", "Drug Dependence"),
    ("diclofenac", "Drug Dependence"),
    #
    # ── Cardiomyopathy (anthracycline / trastuzumab–specific) ──
    ("allopurinol", "Cardiomyopathy"),       # possibly protective (literature)
    ("metformin", "Cardiomyopathy"),          # cardioprotective (literature)
    ("ciprofloxacin", "Cardiomyopathy"),
    #
    # ── NMS (requires dopamine D2 receptor blockade) ──
    ("metformin", "Neuroleptic Malignant Syndrome"),
    ("warfarin", "Neuroleptic Malignant Syndrome"),
    ("atorvastatin", "Neuroleptic Malignant Syndrome"),
    ("ciprofloxacin", "Neuroleptic Malignant Syndrome"),
    #
    # ── Haemorrhage (anticoagulant / antiplatelet / NSAID–specific) ──
    ("clozapine", "Haemorrhage"),
    ("metformin", "Haemorrhage"),
    ("allopurinol", "Haemorrhage"),
    ("lithium", "Haemorrhage"),
    #
    # ── GI Haemorrhage (NSAID / anticoagulant GI erosion–specific) ──
    ("lithium", "Gastrointestinal Haemorrhage"),
    ("olanzapine", "Gastrointestinal Haemorrhage"),
    ("metformin", "Gastrointestinal Haemorrhage"),
    #
    # ── Weight Increased (antipsychotic / corticosteroid / insulin–specific) ──
    ("warfarin", "Weight Increased"),
    ("ciprofloxacin", "Weight Increased"),
    ("allopurinol", "Weight Increased"),
    ("methotrexate", "Weight Increased"),
    #
    # ── Myalgia (statin / bisphosphonate–specific muscle toxicity) ──
    ("warfarin", "Myalgia"),
    ("metformin", "Myalgia"),
    ("clozapine", "Myalgia"),
    #
    # ── Myocarditis (clozapine / checkpoint inhibitor–specific) ──
    # NOTE: ciprofloxacin excluded — fluoroquinolone-induced eosinophilic
    # myocarditis is a documented association.
    ("metformin", "Myocarditis"),
    ("warfarin", "Myocarditis"),
    #
    # ── Diabetes Mellitus (antipsychotic / corticosteroid–specific) ──
    ("warfarin", "Diabetes Mellitus"),
    ("allopurinol", "Diabetes Mellitus"),
    #
    # ── Hyponatraemia (SSRI / AED / thiazide–specific SIADH pathway) ──
    ("atorvastatin", "Hyponatraemia"),
    ("allopurinol", "Hyponatraemia"),
    ("metformin", "Hyponatraemia"),
    #
    # ── Agranulocytosis (kept ONLY for drugs with no documented association) ──
    ("simvastatin", "Agranulocytosis"),      # confirmed safe (literature review)
    ("oxycodone", "Agranulocytosis"),        # opioids have no myelotoxicity
]


# ═══════════════════════════════════════════════════════════════════════════════
#  REFERENCE SET MATCHING
# ═══════════════════════════════════════════════════════════════════════════════


def match_controls(disp_df, controls, label):
    """
    Match control pairs to actual data using substring ingredient matching
    and case-insensitive reaction matching.

    Returns (matched_list, unmatched_list).
    """
    data_ingredients = disp_df["active_ingredient"].unique()
    data_reactions_lower = {r.lower(): r for r in disp_df["reaction"].unique()}

    matched = []
    unmatched = []

    for drug_pat, rxn_pat in controls:
        drug_lower = drug_pat.lower()
        rxn_lower = rxn_pat.lower()

        # Substring match on ingredient
        hits_drug = [d for d in data_ingredients if drug_lower in d]

        # Case-insensitive exact match on reaction, then substring
        if rxn_lower in data_reactions_lower:
            hits_rxn = [data_reactions_lower[rxn_lower]]
        else:
            hits_rxn = [data_reactions_lower[r] for r in data_reactions_lower if rxn_lower in r]

        found = False
        for d in hits_drug:
            for r in hits_rxn:
                mask = (disp_df["active_ingredient"] == d) & (disp_df["reaction"] == r)
                if mask.any():
                    idx = disp_df[mask].index[0]
                    matched.append({"idx": idx, "label": label,
                                    "drug_query": drug_pat, "rxn_query": rxn_pat})
                    found = True
                    break
            if found:
                break

        if not found:
            reason = []
            if not hits_drug:
                reason.append(f"drug not found")
            if not hits_rxn:
                reason.append(f"reaction not found")
            if hits_drug and hits_rxn:
                reason.append("pair has < 3 reports")
            unmatched.append((drug_pat, rxn_pat, "; ".join(reason)))

    return matched, unmatched


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════


def engineer_features(disp_df, cases, drugs, reactions):
    """
    Engineer additional per-pair features from case-level data.

    Adds temporal, demographic, and polypharmacy features to the
    disproportionality results DataFrame.
    """
    print("  Preparing case-level merge ...")

    # Ensure consistent case_number dtype across all DataFrames
    cases["case_number"] = cases["case_number"].astype(str)
    drugs["case_number"] = drugs["case_number"].astype(str)
    reactions["case_number"] = reactions["case_number"].astype(str)

    # Suspected drugs, deduplicated by (case, ingredient)
    susp = drugs[drugs["suspected"] == "suspected"][["case_number", "active_ingredient"]].copy()
    susp = susp.dropna(subset=["active_ingredient"])
    susp["active_ingredient"] = susp["active_ingredient"].str.strip().str.lower()
    susp = susp.drop_duplicates()

    rxn = reactions[["case_number", "reaction"]].drop_duplicates()

    # Case metadata
    case_meta = cases[["case_number"]].copy()
    case_meta["report_year"] = pd.to_datetime(cases["report_date"], errors="coerce", format="mixed", dayfirst=True).dt.year
    case_meta["age_numeric"] = pd.to_numeric(cases["age"], errors="coerce")
    if "sex" in cases.columns:
        case_meta["is_female"] = cases["sex"].str.lower().eq("female").astype(float)

    # Polypharmacy: drugs per case, reactions per case
    drugs_per_case = susp.groupby("case_number").size().rename("n_drugs_case")
    rxns_per_case = rxn.groupby("case_number").size().rename("n_rxns_case")

    print("  Merging drug–reaction triples with case metadata ...")
    triples = susp.merge(rxn, on="case_number", how="inner")
    triples = triples.merge(case_meta, on="case_number", how="left")
    triples = triples.merge(drugs_per_case, on="case_number", how="left")
    triples = triples.merge(rxns_per_case, on="case_number", how="left")
    print(f"  Triples: {len(triples):,}")

    print("  Aggregating per-pair features ...")
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

    # Recent fraction (2024+)
    n_recent = triples[triples["report_year"] >= 2024].groupby(
        ["active_ingredient", "reaction"]
    ).size().rename("n_recent")
    n_total = triples.groupby(["active_ingredient", "reaction"]).size().rename("n_total_t")
    recent = pd.concat([n_recent, n_total], axis=1).fillna(0)
    recent["recent_fraction"] = recent["n_recent"] / recent["n_total_t"]
    agg = agg.merge(
        recent[["recent_fraction"]].reset_index(),
        on=["active_ingredient", "reaction"], how="left"
    )
    agg["recent_fraction"] = agg["recent_fraction"].fillna(0)

    # Merge into disp_df
    disp_df = disp_df.merge(agg, on=["active_ingredient", "reaction"], how="left")

    # Log-transformed volume features
    disp_df["log_a"] = np.log1p(disp_df["a"])
    disp_df["log_expected"] = np.log1p(disp_df["expected"])
    disp_df["log_n_drug"] = np.log1p(disp_df["n_drug"])
    disp_df["log_n_reaction"] = np.log1p(disp_df["n_reaction"])

    return disp_df


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE SETS
# ═══════════════════════════════════════════════════════════════════════════════

# Individual DPA scores (for single-feature AUC comparison)
DPA_SINGLE = {
    "PRR": "prr",
    "ROR": "ror",
    "EBGM": "ebgm",
    "IC": "ic",
}

# Feature set A: DPA features only
FEATURES_DPA = [
    "prr", "prr_lower95", "prr_upper95", "prr_chi2",
    "ror", "ror_lower95", "ror_upper95",
    "ebgm", "eb05",
    "ic", "ic025", "ic975",
]

# Feature set B: non-DPA features (volume + temporal + demographic + polypharmacy)
FEATURES_NON_DPA = [
    "log_a", "log_expected", "log_n_drug", "log_n_reaction",
    "year_mean", "year_std", "year_range", "recent_fraction",
    "age_mean", "age_std", "pct_female",
    "mean_drugs_per_case", "mean_rxns_per_case",
]

# Feature set C: all features combined
FEATURES_ALL = FEATURES_DPA + FEATURES_NON_DPA


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING AND EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_model(X, y, model, model_name, n_splits=5):
    """
    Evaluate a model using stratified k-fold CV.
    Returns dict with AUC-ROC, AUC-PR, and out-of-fold predictions.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Pipeline with imputation (handles NaN from missing demographics)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])

    # Out-of-fold probability predictions
    y_prob = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]

    auc_roc = roc_auc_score(y, y_prob)
    auc_pr = average_precision_score(y, y_prob)

    return {"model": model_name, "auc_roc": auc_roc, "auc_pr": auc_pr, "y_prob": y_prob}


def get_feature_importance(X, y, feature_names):
    """Train on full data and return feature importance from XGBoost."""
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    model.fit(X_imp, y)

    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return importance, model, imputer


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    print("=" * 70)
    print("  TGA DAEN ML Signal Detection")
    print("  XGBoost · Random Forest vs PRR · ROR · EBGM · BCPNN")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    print("\nLoading data ...")
    disp_df = pd.read_csv(OUTPUT_DIR / "disproportionality_full.csv")
    cases = pd.read_csv(PROCESSED_DIR / "daen_cases.csv", low_memory=False)
    drugs = pd.read_csv(PROCESSED_DIR / "daen_case_drugs.csv", low_memory=False)
    reactions = pd.read_csv(PROCESSED_DIR / "daen_case_reactions.csv")
    print(f"  Pairs:     {len(disp_df):>10,}")
    print(f"  Cases:     {len(cases):>10,}")

    # ── Feature engineering ──────────────────────────────────────────────
    print("\nEngineering features from case-level data ...")
    disp_df = engineer_features(disp_df, cases, drugs, reactions)
    print(f"  Total features available: {len(FEATURES_ALL)}")

    # ── Match reference set ──────────────────────────────────────────────
    print("\nMatching reference control pairs to dataset ...")

    pos_matched, pos_unmatched = match_controls(disp_df, POSITIVE_CONTROLS, label=1)
    neg_matched, neg_unmatched = match_controls(disp_df, NEGATIVE_CONTROLS, label=0)

    print(f"\n  POSITIVE CONTROLS: {len(pos_matched)} matched / "
          f"{len(POSITIVE_CONTROLS)} candidates ({100*len(pos_matched)/len(POSITIVE_CONTROLS):.0f}%)")
    print(f"  NEGATIVE CONTROLS: {len(neg_matched)} matched / "
          f"{len(NEGATIVE_CONTROLS)} candidates ({100*len(neg_matched)/len(NEGATIVE_CONTROLS):.0f}%)")

    if pos_unmatched:
        print(f"\n  Unmatched positives ({len(pos_unmatched)}):")
        for drug, rxn, reason in pos_unmatched[:10]:
            print(f"    {drug} + {rxn}: {reason}")
        if len(pos_unmatched) > 10:
            print(f"    ... and {len(pos_unmatched) - 10} more")

    if neg_unmatched:
        print(f"\n  Unmatched negatives ({len(neg_unmatched)}):")
        for drug, rxn, reason in neg_unmatched[:10]:
            print(f"    {drug} + {rxn}: {reason}")
        if len(neg_unmatched) > 10:
            print(f"    ... and {len(neg_unmatched) - 10} more")

    # Build reference DataFrame
    all_matched = pos_matched + neg_matched
    if len(pos_matched) < 10 or len(neg_matched) < 10:
        print("\n  ERROR: Too few matched controls for ML training. Need ≥10 of each.")
        print("  Check that ingredient/reaction names match the cleaned data.")
        return

    ref_indices = [m["idx"] for m in all_matched]
    ref_labels = np.array([m["label"] for m in all_matched])
    ref_df = disp_df.loc[ref_indices].copy()
    ref_df["label"] = ref_labels
    ref_df["control_type"] = ["positive" if l == 1 else "negative" for l in ref_labels]

    # Save reference set
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    ref_save_cols = ["active_ingredient", "reaction", "label", "control_type",
                     "a", "expected", "prr", "ror", "ebgm", "ic"]
    ref_df[ref_save_cols].to_csv(REFERENCE_DIR / "ml_reference_set.csv", index=False)

    print(f"\n  Reference set: {len(ref_df)} pairs "
          f"({(ref_labels == 1).sum()} positive, {(ref_labels == 0).sum()} negative)")

    # ── Single-feature DPA AUC ───────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  DPA SINGLE-FEATURE EVALUATION (AUC-ROC on reference set)")
    print(f"{'=' * 70}")

    results = []
    y = ref_labels

    for name, col in DPA_SINGLE.items():
        scores = ref_df[col].values
        # Handle any NaN/inf
        valid = np.isfinite(scores)
        if valid.sum() < len(scores):
            scores = np.nan_to_num(scores, nan=0, posinf=scores[valid].max(), neginf=0)
        auc = roc_auc_score(y, scores)
        ap = average_precision_score(y, scores)
        results.append({"model": f"{name} (single)", "features": col,
                        "auc_roc": auc, "auc_pr": ap})
        print(f"  {name:>8s}:  AUC-ROC = {auc:.3f}  |  AUC-PR = {ap:.3f}")

    # ── ML model evaluation ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  ML MODEL EVALUATION (5-fold stratified CV)")
    print(f"{'=' * 70}")

    feature_sets = {
        "DPA features only": FEATURES_DPA,
        "Non-DPA features only": FEATURES_NON_DPA,
        "All features": FEATURES_ALL,
    }

    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42, verbosity=0,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1,
        ),
    }

    for fs_name, features in feature_sets.items():
        print(f"\n  Feature set: {fs_name} ({len(features)} features)")
        X = ref_df[features].values

        for model_name, model in models.items():
            res = evaluate_model(X, y, model, f"{model_name} — {fs_name}")
            results.append({
                "model": res["model"],
                "features": fs_name,
                "auc_roc": res["auc_roc"],
                "auc_pr": res["auc_pr"],
            })
            print(f"    {model_name:>15s}:  AUC-ROC = {res['auc_roc']:.3f}  |  AUC-PR = {res['auc_pr']:.3f}")

    # ── Results comparison table ─────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values("auc_roc", ascending=False)

    print(f"\n{'=' * 70}")
    print("  COMPLETE RESULTS RANKING (by AUC-ROC)")
    print(f"{'=' * 70}")
    print(f"\n  {'Model':<45s} {'AUC-ROC':>8s} {'AUC-PR':>8s}")
    print(f"  {'─' * 63}")
    for _, row in results_df.iterrows():
        print(f"  {row['model']:<45s} {row['auc_roc']:>8.3f} {row['auc_pr']:>8.3f}")

    # ── Feature importance ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  FEATURE IMPORTANCE (XGBoost, all features)")
    print(f"{'=' * 70}")

    X_all = ref_df[FEATURES_ALL].values
    importance, trained_model, trained_imputer = get_feature_importance(X_all, y, FEATURES_ALL)

    print(f"\n  {'Feature':<30s} {'Importance':>10s}")
    print(f"  {'─' * 42}")
    for _, row in importance.head(15).iterrows():
        bar = "█" * max(1, int(30 * row["importance"] / importance["importance"].max()))
        print(f"  {row['feature']:<30s} {row['importance']:>10.4f}  {bar}")

    # ── Score all pairs ──────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("Scoring all pairs with trained XGBoost (all features) ...")

    X_full = disp_df[FEATURES_ALL].values
    X_full_imp = trained_imputer.transform(X_full)
    disp_df["ml_probability"] = trained_model.predict_proba(X_full_imp)[:, 1]

    # Distribution of ML scores
    print(f"  ML probability distribution:")
    for threshold in [0.9, 0.8, 0.7, 0.5, 0.3]:
        n = (disp_df["ml_probability"] >= threshold).sum()
        print(f"    P >= {threshold}: {n:>8,} pairs")

    # ── Save outputs ─────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("Saving results ...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluation results
    eval_path = OUTPUT_DIR / "ml_evaluation_results.csv"
    results_df.to_csv(eval_path, index=False, float_format="%.4f")

    # Feature importance
    imp_path = OUTPUT_DIR / "ml_feature_importance.csv"
    importance.to_csv(imp_path, index=False, float_format="%.4f")

    # All pairs scored
    score_cols = [
        "active_ingredient", "reaction", "a", "expected",
        "prr", "ror", "ebgm", "eb05", "ic", "ic025",
        "signal_prr", "signal_ror", "signal_ebgm", "signal_bcpnn",
        "n_methods_signal", "ml_probability",
    ]
    scored_path = OUTPUT_DIR / "ml_all_pairs_scored.csv"
    disp_df.sort_values("ml_probability", ascending=False)[score_cols].to_csv(
        scored_path, index=False, float_format="%.4f"
    )

    # Top ML signals not caught by all 4 DPA methods
    ml_novel = disp_df[
        (disp_df["ml_probability"] >= 0.7) & (disp_df["n_methods_signal"] < 4)
    ].sort_values("ml_probability", ascending=False)

    if len(ml_novel) > 0:
        print(f"\n  TOP ML-FLAGGED PAIRS NOT CAUGHT BY ALL 4 DPA METHODS ({len(ml_novel):,} total)")
        print(f"  {'─' * 75}")
        print(f"  {'Drug':<30s} {'Reaction':<25s} {'P(ML)':>6s} {'EBGM':>6s} {'n_DPA':>5s}")
        print(f"  {'─' * 75}")
        for _, row in ml_novel.head(15).iterrows():
            drug = str(row["active_ingredient"])[:29]
            rxn = str(row["reaction"])[:24]
            print(f"  {drug:<30s} {rxn:<25s} {row['ml_probability']:>6.3f}"
                  f" {row['ebgm']:>6.1f} {int(row['n_methods_signal']):>5d}")

    for path in [eval_path, imp_path, scored_path]:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.name:45s}  {size_mb:>6.1f} MB")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  ML SIGNAL DETECTION COMPLETE  ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"\n  Next step: python scripts/05_network_analysis.py\n")


if __name__ == "__main__":
    main()
