"""
Script 07: Additional Statistical Validation

Performs five supplementary validation analyses for manuscript preparation:

  1. Reference Set DPA Validation
     Sensitivity, specificity, PPV, NPV, F1 of each DPA method's binary
     signal threshold against 50 positive + 20 negative control pairs.

  2. ML Cross-Validation Confidence Intervals
     Per-fold AUC-ROC with mean +/- SD and 95% CI.

  3. ML-DPA Concordance Analysis
     Cross-tabulation of ML probability thresholds with DPA consensus signals.

  4. MGPS Prior Goodness-of-Fit
     Probability Integral Transform tested for Uniform(0,1) via KS test.

  5. Signal-to-Label Concordance
     Fraction of top consensus signals matching well-known drug-AE
     associations from published product information.

Depends on outputs from scripts 03 and 04. Run after all prior scripts.

Outputs:
  outputs/tables/validation_dpa_reference_performance.csv
  outputs/tables/validation_ml_cv_folds.csv
  outputs/tables/validation_ml_dpa_concordance.csv
  outputs/tables/validation_mgps_gof.csv
  outputs/tables/validation_label_concordance.csv
  outputs/supplementary/additional_validation_report.txt

Usage:
    python scripts/07_additional_validation.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
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
SUPP_DIR = PROJECT_DIR / "outputs" / "supplementary"

# ── Feature definitions (same as script 04) ──────────────────────────────────

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


# ═══════════════════════════════════════════════════════════════════════════════
#  KNOWN LABELLED ASSOCIATIONS (Analysis 5)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Well-established drug-AE pairs from product information and pharmacovigilance
# literature. Used for clinical face-validity assessment.
# Matching uses case-insensitive substring on both drug and reaction.

KNOWN_ASSOCIATIONS = [
    # ── Antipsychotics ──
    ("clozapine", "Agranulocytosis"),
    ("clozapine", "Neutropenia"),
    ("clozapine", "Myocarditis"),
    ("clozapine", "Weight Increased"),
    ("clozapine", "Diabetes Mellitus"),
    ("clozapine", "Seizure"),
    ("clozapine", "Constipation"),
    ("olanzapine", "Weight Increased"),
    ("olanzapine", "Diabetes Mellitus"),
    ("quetiapine", "Weight Increased"),
    ("quetiapine", "Somnolence"),
    ("risperidone", "Weight Increased"),
    ("risperidone", "Hyperprolactinaemia"),
    ("haloperidol", "Neuroleptic Malignant Syndrome"),
    ("haloperidol", "Tardive Dyskinesia"),
    # ── Statins ──
    ("atorvastatin", "Myalgia"),
    ("atorvastatin", "Rhabdomyolysis"),
    ("atorvastatin", "Hepatitis"),
    ("simvastatin", "Myalgia"),
    ("simvastatin", "Rhabdomyolysis"),
    ("rosuvastatin", "Myalgia"),
    ("rosuvastatin", "Rhabdomyolysis"),
    ("pravastatin", "Myalgia"),
    # ── NSAIDs ──
    ("celecoxib", "Gastrointestinal Haemorrhage"),
    ("celecoxib", "Renal Impairment"),
    ("celecoxib", "Myocardial Infarction"),
    ("diclofenac", "Gastrointestinal Haemorrhage"),
    ("diclofenac", "Renal Impairment"),
    ("ibuprofen", "Gastrointestinal Haemorrhage"),
    ("ibuprofen", "Renal Impairment"),
    ("naproxen", "Gastrointestinal Haemorrhage"),
    ("meloxicam", "Gastrointestinal Haemorrhage"),
    ("indomethacin", "Gastrointestinal Haemorrhage"),
    # ── Anticoagulants / Antithrombotics ──
    ("warfarin", "Haemorrhage"),
    ("warfarin", "Skin Necrosis"),
    ("heparin", "Thrombocytopenia"),
    ("enoxaparin", "Thrombocytopenia"),
    ("enoxaparin", "Haemorrhage"),
    ("rivaroxaban", "Haemorrhage"),
    ("apixaban", "Haemorrhage"),
    ("dabigatran", "Haemorrhage"),
    ("clopidogrel", "Haemorrhage"),
    # ── Antidepressants ──
    ("sertraline", "Serotonin Syndrome"),
    ("sertraline", "Hyponatraemia"),
    ("fluoxetine", "Serotonin Syndrome"),
    ("fluoxetine", "Hyponatraemia"),
    ("venlafaxine", "Serotonin Syndrome"),
    ("venlafaxine", "Hyponatraemia"),
    ("citalopram", "Serotonin Syndrome"),
    ("escitalopram", "Serotonin Syndrome"),
    ("mirtazapine", "Weight Increased"),
    ("mirtazapine", "Somnolence"),
    # ── Antibiotics ──
    ("ciprofloxacin", "Tendon Rupture"),
    ("ciprofloxacin", "Tendon Disorder"),
    ("ciprofloxacin", "Peripheral Neuropathy"),
    ("amoxicillin", "Anaphylactic Reaction"),
    ("amoxicillin", "Diarrhoea"),
    ("flucloxacillin", "Hepatitis"),
    ("flucloxacillin", "Jaundice"),
    ("vancomycin", "Thrombocytopenia"),
    ("vancomycin", "Renal Impairment"),
    ("gentamicin", "Deafness"),
    ("nitrofurantoin", "Pulmonary Fibrosis"),
    ("sulfamethoxazole", "Stevens-Johnson Syndrome"),
    ("sulfamethoxazole", "Agranulocytosis"),
    # ── Antiepileptics ──
    ("carbamazepine", "Stevens-Johnson Syndrome"),
    ("carbamazepine", "Hyponatraemia"),
    ("carbamazepine", "Aplastic Anaemia"),
    ("lamotrigine", "Stevens-Johnson Syndrome"),
    ("phenytoin", "Stevens-Johnson Syndrome"),
    ("valproate", "Pancreatitis"),
    ("valproate", "Thrombocytopenia"),
    ("valproate", "Hepatitis"),
    ("valproate", "Weight Increased"),
    ("levetiracetam", "Aggression"),
    ("topiramate", "Weight Decreased"),
    ("topiramate", "Paraesthesia"),
    # ── Cardiovascular ──
    ("amiodarone", "Hypothyroidism"),
    ("amiodarone", "Hyperthyroidism"),
    ("amiodarone", "Pulmonary Fibrosis"),
    ("amiodarone", "Hepatitis"),
    ("digoxin", "Nausea"),
    ("digoxin", "Arrhythmia"),
    ("enalapril", "Cough"),
    ("ramipril", "Cough"),
    ("perindopril", "Cough"),
    ("lisinopril", "Cough"),
    ("enalapril", "Angioedema"),
    ("ramipril", "Angioedema"),
    ("perindopril", "Angioedema"),
    ("amlodipine", "Oedema Peripheral"),
    ("amlodipine", "Oedema"),
    ("atenolol", "Bradycardia"),
    ("metoprolol", "Bradycardia"),
    ("diltiazem", "Bradycardia"),
    ("verapamil", "Constipation"),
    ("spironolactone", "Hyperkalaemia"),
    ("spironolactone", "Gynaecomastia"),
    ("furosemide", "Hypokalaemia"),
    ("hydrochlorothiazide", "Hyponatraemia"),
    ("hydrochlorothiazide", "Hypokalaemia"),
    # ── Metabolic / Endocrine ──
    ("metformin", "Lactic Acidosis"),
    ("metformin", "Diarrhoea"),
    ("metformin", "Nausea"),
    ("insulin", "Hypoglycaemia"),
    ("gliclazide", "Hypoglycaemia"),
    ("lithium", "Hypothyroidism"),
    ("lithium", "Tremor"),
    ("levothyroxine", "Palpitations"),
    ("prednisolone", "Osteoporosis"),
    ("prednisolone", "Diabetes Mellitus"),
    ("prednisolone", "Weight Increased"),
    ("prednisolone", "Insomnia"),
    ("dexamethasone", "Hyperglycaemia"),
    ("prednisone", "Osteoporosis"),
    ("prednisone", "Weight Increased"),
    # ── Immunosuppressants ──
    ("methotrexate", "Pancytopenia"),
    ("methotrexate", "Hepatitis"),
    ("methotrexate", "Pneumonitis"),
    ("methotrexate", "Stomatitis"),
    ("ciclosporin", "Renal Impairment"),
    ("ciclosporin", "Hypertension"),
    ("tacrolimus", "Renal Impairment"),
    ("tacrolimus", "Diabetes Mellitus"),
    ("azathioprine", "Pancytopenia"),
    ("azathioprine", "Hepatitis"),
    ("mycophenolate", "Diarrhoea"),
    ("mycophenolate", "Leucopenia"),
    # ── Biologics / Monoclonal Antibodies ──
    ("infliximab", "Tuberculosis"),
    ("infliximab", "Infusion Related Reaction"),
    ("adalimumab", "Tuberculosis"),
    ("adalimumab", "Injection Site Reaction"),
    ("etanercept", "Injection Site Reaction"),
    ("rituximab", "Infusion Related Reaction"),
    ("nivolumab", "Colitis"),
    ("nivolumab", "Hepatitis"),
    ("nivolumab", "Pneumonitis"),
    ("nivolumab", "Hypothyroidism"),
    ("pembrolizumab", "Colitis"),
    ("pembrolizumab", "Hepatitis"),
    ("pembrolizumab", "Pneumonitis"),
    ("pembrolizumab", "Hypothyroidism"),
    ("trastuzumab", "Cardiomyopathy"),
    ("bevacizumab", "Hypertension"),
    ("bevacizumab", "Proteinuria"),
    # ── Chemotherapy ──
    ("doxorubicin", "Cardiomyopathy"),
    ("doxorubicin", "Neutropenia"),
    ("paclitaxel", "Peripheral Neuropathy"),
    ("paclitaxel", "Neutropenia"),
    ("cisplatin", "Renal Impairment"),
    ("cisplatin", "Deafness"),
    ("cisplatin", "Nausea"),
    ("cyclophosphamide", "Neutropenia"),
    ("cyclophosphamide", "Haemorrhagic Cystitis"),
    ("vincristine", "Peripheral Neuropathy"),
    ("imatinib", "Oedema"),
    ("imatinib", "Nausea"),
    ("capecitabine", "Diarrhoea"),
    ("fluorouracil", "Stomatitis"),
    ("fluorouracil", "Diarrhoea"),
    # ── PPI ──
    ("omeprazole", "Hypomagnesaemia"),
    ("esomeprazole", "Hypomagnesaemia"),
    ("pantoprazole", "Hypomagnesaemia"),
    # ── Opioids ──
    ("oxycodone", "Drug Dependence"),
    ("oxycodone", "Constipation"),
    ("oxycodone", "Respiratory Depression"),
    ("morphine", "Constipation"),
    ("morphine", "Respiratory Depression"),
    ("fentanyl", "Respiratory Depression"),
    ("codeine", "Constipation"),
    ("tramadol", "Seizure"),
    ("tramadol", "Serotonin Syndrome"),
    # ── Dermatology ──
    ("isotretinoin", "Depression"),
    ("isotretinoin", "Hepatitis"),
    ("allopurinol", "Stevens-Johnson Syndrome"),
    # ── Vaccines ──
    ("tozinameran", "Myocarditis"),
    ("tozinameran", "Pericarditis"),
    ("tozinameran", "Anaphylactic Reaction"),
    ("influenza", "Guillain-Barre Syndrome"),
    # ── Other ──
    ("montelukast", "Depression"),
    ("montelukast", "Suicidal Ideation"),
    ("sildenafil", "Headache"),
    ("tamoxifen", "Endometrial Cancer"),
    ("tamoxifen", "Thromboembolism"),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: MGPS Prior Fitting (from script 03)
# ═══════════════════════════════════════════════════════════════════════════════


def fit_mgps_prior(n_obs, E, max_iter=5000):
    """Fit two-component gamma mixture prior via MLE (DuMouchel 1999)."""

    def neg_log_lik(params):
        alpha = 1.0 / (1.0 + np.exp(-params[0]))
        a1 = np.exp(params[1])
        b1 = np.exp(params[2])
        a2 = np.exp(params[3])
        b2 = np.exp(params[4])

        p1 = b1 / (b1 + E)
        p2 = b2 / (b2 + E)
        log_nb1 = stats.nbinom.logpmf(n_obs, a1, p1)
        log_nb2 = stats.nbinom.logpmf(n_obs, a2, p2)
        log_mix = np.logaddexp(np.log(alpha) + log_nb1,
                               np.log(1 - alpha) + log_nb2)
        nll = -np.sum(log_mix)
        return nll if np.isfinite(nll) else 1e15

    x0 = np.array([np.log(0.2 / 0.8), np.log(0.2), np.log(0.1),
                    np.log(2.0), np.log(2.0)])
    result = minimize(neg_log_lik, x0, method="Nelder-Mead",
                      options={"maxiter": max_iter, "xatol": 1e-8,
                               "fatol": 1e-8})

    alpha = 1.0 / (1.0 + np.exp(-result.x[0]))
    a1, b1 = np.exp(result.x[1]), np.exp(result.x[2])
    a2, b2 = np.exp(result.x[3]), np.exp(result.x[4])
    return alpha, a1, b1, a2, b2, result


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER: Feature Engineering (from script 04)
# ═══════════════════════════════════════════════════════════════════════════════


def engineer_features(disp_df, cases, drugs, reactions):
    """Engineer per-pair temporal, demographic, and polypharmacy features."""
    print("  Engineering features from case-level data ...")

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
        case_meta["is_female"] = cases["sex"].str.lower().eq("female").astype(
            float)

    drugs_per_case = susp.groupby("case_number").size().rename("n_drugs_case")
    rxns_per_case = rxn.groupby("case_number").size().rename("n_rxns_case")

    print("  Building triples ...")
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


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 1: Reference Set DPA Validation
# ═══════════════════════════════════════════════════════════════════════════════


def _clopper_pearson(k, n, alpha=0.05):
    """Clopper-Pearson exact binomial 95% CI for proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (lo, hi)


def analysis_1(disp_df, out):
    out.write("=" * 70 + "\n")
    out.write("  Analysis 1: Reference Set DPA Validation\n")
    out.write("=" * 70 + "\n\n")

    ref = pd.read_csv(REFERENCE_DIR / "ml_reference_set.csv")
    ref = ref.merge(
        disp_df[["active_ingredient", "reaction", "signal_prr", "signal_ror",
                 "signal_ebgm", "signal_bcpnn", "n_methods_signal"]],
        on=["active_ingredient", "reaction"], how="left")

    y = ref["label"].values
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    out.write(f"  Reference set: {n_pos} positive, {n_neg} negative controls\n\n")

    methods = [
        ("PRR (>=2, chi2>=4)", "signal_prr"),
        ("ROR (lower CI>1)", "signal_ror"),
        ("EBGM (EB05>=2)", "signal_ebgm"),
        ("BCPNN (IC025>0)", "signal_bcpnn"),
        ("Consensus (all 4)", None),
    ]

    rows = []
    out.write(f"  {'Method':<22s} {'Sens':>6s} {'Spec':>6s} {'PPV':>6s} "
              f"{'NPV':>6s} {'F1':>6s}  TP  FP  TN  FN\n")
    out.write(f"  {'─' * 76}\n")

    for name, col in methods:
        if col:
            pred = ref[col].astype(int).values
        else:
            pred = (ref["n_methods_signal"] == 4).astype(int).values

        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0

        # Clopper-Pearson exact 95% CIs
        sens_ci = _clopper_pearson(tp, tp + fn)
        spec_ci = _clopper_pearson(tn, tn + fp)

        rows.append({"Method": name, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                     "Sensitivity": round(sens, 4),
                     "Sens_CI_lo": round(sens_ci[0], 4),
                     "Sens_CI_hi": round(sens_ci[1], 4),
                     "Specificity": round(spec, 4),
                     "Spec_CI_lo": round(spec_ci[0], 4),
                     "Spec_CI_hi": round(spec_ci[1], 4),
                     "PPV": round(ppv, 4), "NPV": round(npv, 4),
                     "F1": round(f1, 4)})

        out.write(f"  {name:<22s} {sens:>6.3f} {spec:>6.3f} {ppv:>6.3f} "
                  f"{npv:>6.3f} {f1:>6.3f}  {tp:>2d}  {fp:>2d}  {tn:>2d}  "
                  f"{fn:>2d}\n")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_DIR / "validation_dpa_reference_performance.csv",
                      index=False)
    out.write(f"\n  Saved: validation_dpa_reference_performance.csv\n")

    # Report exact binomial CIs
    out.write(f"\n  Clopper-Pearson exact 95% CIs:\n")
    out.write(f"  {'Method':<22s} {'Sensitivity 95% CI':>22s} "
              f"{'Specificity 95% CI':>22s}\n")
    out.write(f"  {'─' * 70}\n")
    for r in rows:
        out.write(f"  {r['Method']:<22s} "
                  f"[{r['Sens_CI_lo']:.3f}, {r['Sens_CI_hi']:.3f}]"
                  f"{'':>8s}"
                  f"[{r['Spec_CI_lo']:.3f}, {r['Spec_CI_hi']:.3f}]\n")
    out.write(f"\n  NOTE: With n_pos={n_pos} and n_neg={n_neg}, exact CIs\n"
              f"  reflect the finite-sample uncertainty of these estimates.\n")

    # Flag negative controls with elevated DPA scores
    neg = ref[ref["label"] == 0].copy()
    flagged = neg[neg["n_methods_signal"] >= 1].sort_values(
        "n_methods_signal", ascending=False)

    if len(flagged) > 0:
        out.write(f"\n  FLAGGED NEGATIVE CONTROLS ({len(flagged)} with "
                  f">=1 DPA signal):\n")
        out.write(f"  {'─' * 70}\n")
        out.write(f"  {'Drug':<30s} {'Reaction':<25s} {'EBGM':>6s} "
                  f"{'IC':>6s} {'n_DPA':>5s}\n")
        out.write(f"  {'─' * 70}\n")
        for _, r in flagged.iterrows():
            drug = str(r["active_ingredient"])[:29]
            rxn = str(r["reaction"]).replace("• ", "")[:24]
            out.write(f"  {drug:<30s} {rxn:<25s} {r['ebgm']:>6.1f} "
                      f"{r['ic']:>6.2f} {int(r['n_methods_signal']):>5d}\n")
        out.write("\n  NOTE: These may represent genuine associations "
                  "misclassified as\n  negative controls, or low-level "
                  "confounding in SRS data.\n")

    out.write("\n")
    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 2: ML Cross-Validation Confidence Intervals
# ═══════════════════════════════════════════════════════════════════════════════


def analysis_2(disp_df, out):
    out.write("=" * 70 + "\n")
    out.write("  Analysis 2: ML Cross-Validation Confidence Intervals\n")
    out.write("=" * 70 + "\n\n")

    ref = pd.read_csv(REFERENCE_DIR / "ml_reference_set.csv")
    ref_merged = ref[["active_ingredient", "reaction", "label"]].merge(
        disp_df, on=["active_ingredient", "reaction"], how="left")

    y = ref_merged["label"].values
    out.write(f"  Reference set: {(y == 1).sum()} positive, "
              f"{(y == 0).sum()} negative\n\n")

    feature_sets = {
        "DPA only": FEATURES_DPA,
        "Non-DPA only": FEATURES_NON_DPA,
        "All features": FEATURES_ALL,
    }

    model_factories = {
        "XGBoost": lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42, verbosity=0),
        "Random Forest": lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1),
    }

    all_rows = []

    out.write(f"  {'Model':<16s} {'Features':<16s} {'Mean AUC':>9s} "
              f"{'SD':>7s} {'95% CI':>17s}  Folds\n")
    out.write(f"  {'─' * 76}\n")

    for fs_name, features in feature_sets.items():
        X = ref_merged[features].values

        for model_name, factory in model_factories.items():
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_aucs = []
            fold_aps = []

            for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                pipe = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", factory()),
                ])
                pipe.fit(X[train_idx], y[train_idx])
                y_prob = pipe.predict_proba(X[test_idx])[:, 1]

                auc = roc_auc_score(y[test_idx], y_prob)
                ap = average_precision_score(y[test_idx], y_prob)
                fold_aucs.append(auc)
                fold_aps.append(ap)

                all_rows.append({
                    "model": model_name, "features": fs_name,
                    "fold": fold_i + 1,
                    "auc_roc": round(auc, 4), "auc_pr": round(ap, 4),
                })

            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs, ddof=1)
            se = std_auc / np.sqrt(5)
            ci_lo = max(0, mean_auc - 1.96 * se)
            ci_hi = min(1, mean_auc + 1.96 * se)

            folds_str = ", ".join(f"{a:.3f}" for a in fold_aucs)
            out.write(f"  {model_name:<16s} {fs_name:<16s} {mean_auc:>9.4f} "
                      f"{std_auc:>7.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  "
                      f"{folds_str}\n")

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUTPUT_DIR / "validation_ml_cv_folds.csv", index=False)
    out.write(f"\n  Saved: validation_ml_cv_folds.csv\n\n")
    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 3: ML-DPA Concordance
# ═══════════════════════════════════════════════════════════════════════════════


def analysis_3(out):
    out.write("=" * 70 + "\n")
    out.write("  Analysis 3: ML-DPA Concordance Analysis\n")
    out.write("=" * 70 + "\n\n")

    scored = pd.read_csv(OUTPUT_DIR / "ml_all_pairs_scored.csv")
    consensus = scored["n_methods_signal"] == 4
    n_total = len(scored)
    n_consensus = consensus.sum()

    out.write(f"  Total pairs:      {n_total:>10,}\n")
    out.write(f"  DPA consensus:    {n_consensus:>10,}\n\n")

    thresholds = [0.3, 0.5, 0.7, 0.9]
    rows = []

    out.write(f"  {'ML Thresh':>10s} {'ML+DPA+':>8s} {'ML only':>8s} "
              f"{'DPA only':>9s} {'Neither':>8s} {'Concord%':>9s} "
              f"{'Jaccard':>8s}\n")
    out.write(f"  {'─' * 64}\n")

    for t in thresholds:
        ml_pos = scored["ml_probability"] >= t

        both = int((ml_pos & consensus).sum())
        ml_only = int((ml_pos & ~consensus).sum())
        dpa_only = int((~ml_pos & consensus).sum())
        neither = int((~ml_pos & ~consensus).sum())

        concordance = 100 * (both + neither) / n_total
        union = both + ml_only + dpa_only
        jaccard = both / union if union > 0 else 0

        rows.append({
            "ml_threshold": t,
            "ml_pos_dpa_pos": both,
            "ml_pos_dpa_neg": ml_only,
            "ml_neg_dpa_pos": dpa_only,
            "ml_neg_dpa_neg": neither,
            "ml_total": int(ml_pos.sum()),
            "dpa_total": int(n_consensus),
            "concordance_pct": round(concordance, 2),
            "jaccard": round(jaccard, 4),
        })

        out.write(f"  P >= {t:<4.1f}   {both:>8,} {ml_only:>8,} "
                  f"{dpa_only:>9,} {neither:>8,} {concordance:>8.1f}% "
                  f"{jaccard:>8.3f}\n")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_DIR / "validation_ml_dpa_concordance.csv",
                      index=False)

    # Novel ML signals (high ML prob, not DPA consensus)
    ml_novel = scored[(scored["ml_probability"] >= 0.7) & (~consensus)]
    ml_novel = ml_novel.sort_values("ml_probability", ascending=False)

    if len(ml_novel) > 0:
        out.write(f"\n  ML-ONLY SIGNALS (P>=0.7, not DPA consensus): "
                  f"{len(ml_novel):,}\n")
        out.write(f"  {'─' * 70}\n")
        out.write(f"  {'Drug':<30s} {'Reaction':<22s} {'P(ML)':>6s} "
                  f"{'EBGM':>6s} {'n':>5s} {'n_DPA':>5s}\n")
        out.write(f"  {'─' * 70}\n")
        for _, r in ml_novel.head(15).iterrows():
            drug = str(r["active_ingredient"])[:29]
            rxn = str(r["reaction"]).replace("• ", "")[:21]
            out.write(f"  {drug:<30s} {rxn:<22s} {r['ml_probability']:>6.3f} "
                      f"{r['ebgm']:>6.1f} {int(r['a']):>5d} "
                      f"{int(r['n_methods_signal']):>5d}\n")

    out.write(f"\n  Saved: validation_ml_dpa_concordance.csv\n\n")
    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 4: MGPS Prior Goodness-of-Fit
# ═══════════════════════════════════════════════════════════════════════════════


def analysis_4(disp_df, out):
    out.write("=" * 70 + "\n")
    out.write("  Analysis 4: MGPS Prior Goodness-of-Fit (PIT)\n")
    out.write("=" * 70 + "\n\n")

    n_obs = disp_df["a"].values.astype(int)
    E = disp_df["expected"].values

    # Attempt to refit; fall back to known-good parameters from script 03
    out.write("  Fitting MGPS two-gamma mixture prior ...\n")
    alpha, a1, b1, a2, b2, opt = fit_mgps_prior(n_obs, E)

    if opt.fun >= 1e14:
        out.write("  Refitting returned degenerate solution; using "
                  "parameters from original script 03 fit.\n")
        alpha, a1, b1, a2, b2 = 0.2444, 0.5200, 0.0063, 1.1700, 0.4100

    out.write(f"  Prior parameters:\n")
    out.write(f"    alpha = {alpha:.4f}\n")
    out.write(f"    Component 1: Gamma({a1:.4f}, {b1:.4f})  "
              f"mean = {a1/b1:.2f}\n")
    out.write(f"    Component 2: Gamma({a2:.4f}, {b2:.4f})  "
              f"mean = {a2/b2:.2f}\n")
    if opt.fun < 1e14:
        out.write(f"    Converged: {opt.success}  |  -LL: {opt.fun:,.0f}\n\n")
    else:
        out.write(f"    (Parameters from original MGPS fitting in "
                  f"script 03)\n\n")

    # Probability Integral Transform
    out.write("  Computing randomised PIT values ...\n")

    p1 = b1 / (b1 + E)
    p2 = b2 / (b2 + E)

    cdf_at_n = (alpha * stats.nbinom.cdf(n_obs, a1, p1) +
                (1 - alpha) * stats.nbinom.cdf(n_obs, a2, p2))
    cdf_at_nm1 = (alpha * stats.nbinom.cdf(n_obs - 1, a1, p1) +
                  (1 - alpha) * stats.nbinom.cdf(n_obs - 1, a2, p2))
    # n_obs >= 3 always, so n_obs - 1 >= 2, no clamping needed

    rng = np.random.default_rng(42)
    pit = rng.uniform(cdf_at_nm1, cdf_at_n)

    # KS test against Uniform(0,1)
    ks_stat, ks_p = stats.kstest(pit, "uniform")

    out.write(f"\n  Kolmogorov-Smirnov test (H0: PIT ~ Uniform):\n")
    out.write(f"    KS statistic: {ks_stat:.6f}\n")
    out.write(f"    p-value:      {ks_p:.6e}\n")

    if ks_p < 0.05:
        out.write(f"    Result: REJECTED at alpha=0.05\n")
        out.write(f"    NOTE: With N={len(n_obs):,} observations, the KS "
                  f"test is very powerful.\n")
        out.write(f"    The KS statistic ({ks_stat:.4f}) indicates the "
                  f"magnitude of deviation.\n")
        out.write(f"    Values < 0.05 suggest adequate practical fit.\n")
    else:
        out.write(f"    Result: Not rejected (adequate fit)\n")

    # Decile histogram
    decile_counts, bin_edges = np.histogram(pit, bins=10, range=(0, 1))
    expected_per = len(pit) / 10
    chi2_val = np.sum((decile_counts - expected_per) ** 2 / expected_per)
    chi2_p = 1 - stats.chi2.cdf(chi2_val, df=9)

    out.write(f"\n  Decile uniformity (chi-squared, df=9):\n")
    out.write(f"    chi2 = {chi2_val:.2f},  p = {chi2_p:.6e}\n\n")
    out.write(f"  PIT decile histogram (expected: {expected_per:.0f} each):\n")
    for i in range(10):
        bar = "█" * max(1, int(40 * decile_counts[i] / decile_counts.max()))
        pct_dev = 100 * (decile_counts[i] - expected_per) / expected_per
        out.write(f"    [{i/10:.1f}, {(i+1)/10:.1f})  {decile_counts[i]:>7,}  "
                  f"({pct_dev:>+5.1f}%)  {bar}\n")

    # Save results
    gof_df = pd.DataFrame({
        "metric": ["alpha", "a1", "b1", "a2", "b2",
                   "ks_statistic", "ks_p_value",
                   "chi2_decile", "chi2_p_value"],
        "value": [alpha, a1, b1, a2, b2, ks_stat, ks_p, chi2_val, chi2_p],
    })
    gof_df.to_csv(OUTPUT_DIR / "validation_mgps_gof.csv", index=False)

    out.write(f"\n  Saved: validation_mgps_gof.csv\n\n")
    return gof_df


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 5: Signal-to-Label Concordance
# ═══════════════════════════════════════════════════════════════════════════════


def _is_known_association(drug, reaction):
    """Check if a drug-reaction pair matches any known labelled association."""
    drug_lower = str(drug).lower()
    rxn_lower = str(reaction).lower().replace("• ", "").strip()
    for d_pat, r_pat in KNOWN_ASSOCIATIONS:
        if d_pat.lower() in drug_lower and r_pat.lower() in rxn_lower:
            return True
    return False


def analysis_5(disp_df, out):
    out.write("=" * 70 + "\n")
    out.write("  Analysis 5: Signal-to-Label Concordance\n")
    out.write("=" * 70 + "\n\n")

    out.write(f"  Known association reference list: "
              f"{len(KNOWN_ASSOCIATIONS)} pairs\n\n")

    consensus = disp_df[disp_df["n_methods_signal"] == 4].sort_values(
        "ebgm", ascending=False).copy()
    out.write(f"  Consensus signals to check: {len(consensus):,}\n\n")

    # Vectorised matching
    consensus["known_labelled"] = [
        _is_known_association(r["active_ingredient"], r["reaction"])
        for _, r in consensus.iterrows()
    ]

    # Part A: What fraction of consensus signals are known?
    cutoffs = [50, 100, 200, 500, 1000, len(consensus)]
    rows = []

    out.write(f"  A) What fraction of consensus signals are known labelled ADRs?\n\n")
    out.write(f"  {'Top N (by EBGM)':>18s} {'N known':>8s} {'% known':>8s}\n")
    out.write(f"  {'─' * 38}\n")

    for n in cutoffs:
        subset = consensus.head(n)
        n_known = int(subset["known_labelled"].sum())
        pct = 100 * n_known / len(subset) if len(subset) > 0 else 0
        label = f"Top {n}" if n < len(consensus) else "All"
        rows.append({"top_n": label, "n_signals": len(subset),
                     "n_known": n_known, "pct_known": round(pct, 1)})
        out.write(f"  {label:>18s} {n_known:>8,} {pct:>7.1f}%\n")

    out.write(f"\n  NOTE: Top signals by EBGM are dominated by rare/specific\n")
    out.write(f"  drug-AE pairs. Well-known associations involve common drugs\n")
    out.write(f"  with moderate EBGM values (typically 2-200).\n")

    # Part B: What fraction of known associations are recovered?
    out.write(f"\n  B) What fraction of known associations are recovered as "
              f"consensus signals?\n\n")

    # Build lookup structures for fast matching
    cons_pairs = list(zip(
        consensus["active_ingredient"].str.lower(),
        consensus["reaction"].str.lower().str.replace("• ", "", regex=False)
            .str.strip()))

    all_pairs = list(zip(
        disp_df["active_ingredient"].str.lower(),
        disp_df["reaction"].str.lower().str.replace("• ", "", regex=False)
            .str.strip()))

    any_sig_mask = disp_df["n_methods_signal"] >= 1
    any_pairs = list(zip(
        disp_df.loc[any_sig_mask, "active_ingredient"].str.lower(),
        disp_df.loc[any_sig_mask, "reaction"].str.lower()
            .str.replace("• ", "", regex=False).str.strip()))

    def pattern_in_list(d_pat, r_pat, pair_list):
        d_low, r_low = d_pat.lower(), r_pat.lower()
        return any(d_low in d and r_low in r for d, r in pair_list)

    n_total_known = len(KNOWN_ASSOCIATIONS)
    n_in_data = 0
    n_recovered_consensus = 0
    n_recovered_any = 0

    for d_pat, r_pat in KNOWN_ASSOCIATIONS:
        if not pattern_in_list(d_pat, r_pat, all_pairs):
            continue
        n_in_data += 1
        if pattern_in_list(d_pat, r_pat, cons_pairs):
            n_recovered_consensus += 1
        if pattern_in_list(d_pat, r_pat, any_pairs):
            n_recovered_any += 1

    pct_cons = 100 * n_recovered_consensus / n_in_data if n_in_data else 0
    pct_any = 100 * n_recovered_any / n_in_data if n_in_data else 0

    out.write(f"  Known associations defined:           {n_total_known:>5d}\n")
    out.write(f"  Matchable in DAEN data:               {n_in_data:>5d}\n")
    out.write(f"  Recovered by consensus (all 4 DPA):   {n_recovered_consensus:>5d}"
              f"  ({pct_cons:.1f}%)\n")
    out.write(f"  Recovered by any method (>=1 DPA):    {n_recovered_any:>5d}"
              f"  ({pct_any:.1f}%)\n")

    rows.append({"top_n": "Recovery (consensus)", "n_signals": n_in_data,
                 "n_known": n_recovered_consensus,
                 "pct_known": round(pct_cons, 1)})
    rows.append({"top_n": "Recovery (any method)", "n_signals": n_in_data,
                 "n_known": n_recovered_any,
                 "pct_known": round(pct_any, 1)})

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_DIR / "validation_label_concordance.csv",
                      index=False)

    # Show top known signals
    known_hits = consensus[consensus["known_labelled"]].head(20)
    if len(known_hits) > 0:
        out.write(f"\n  TOP 20 KNOWN SIGNALS RECOVERED (ranked by EBGM):\n")
        out.write(f"  {'─' * 70}\n")
        out.write(f"  {'Rank':>5s} {'Drug':<30s} {'Reaction':<22s} "
                  f"{'n':>5s} {'EBGM':>7s}\n")
        out.write(f"  {'─' * 70}\n")
        for rank, (_, r) in enumerate(known_hits.iterrows(), 1):
            drug = str(r["active_ingredient"])[:29]
            rxn = str(r["reaction"]).replace("• ", "")[:21]
            out.write(f"  {rank:>5d} {drug:<30s} {rxn:<22s} "
                      f"{int(r['a']):>5d} {r['ebgm']:>7.1f}\n")

    # Show unmatched known associations (not in consensus signals)
    unmatched = []
    for d_pat, r_pat in KNOWN_ASSOCIATIONS:
        if not pattern_in_list(d_pat, r_pat, all_pairs):
            continue
        if not pattern_in_list(d_pat, r_pat, cons_pairs):
            unmatched.append((d_pat, r_pat))

    if unmatched:
        out.write(f"\n  KNOWN ASSOCIATIONS IN DATA BUT NOT IN CONSENSUS "
                  f"({len(unmatched)}):\n")
        for d, r in unmatched[:20]:
            out.write(f"    {d} + {r}\n")
        if len(unmatched) > 20:
            out.write(f"    ... and {len(unmatched) - 20} more\n")

    out.write(f"\n  Saved: validation_label_concordance.csv\n\n")
    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
#  Analysis 6: TGA Regulatory Action Concordance
# ═══════════════════════════════════════════════════════════════════════════════

# Drug-AE pairs that triggered formal TGA regulatory action (withdrawals,
# restrictions, boxed warnings, PI updates, safety alerts). Compiled from
# TGA safety communications, market actions, and Medicines Safety Updates.
# Each entry: (drug_pattern, reaction_pattern, action_year, tier, action_desc)
#   tier 1 = market withdrawal/cancellation
#   tier 2 = restriction / boxed warning / contraindication
#   tier 3 = PI label update / safety alert

TGA_REGULATORY_ACTIONS = [
    # ── Tier 1: Withdrawals / Cancellations ──
    ("rofecoxib", "Myocardial Infarction", 2004, 1,
     "Voluntary withdrawal (cardiovascular risk)"),
    ("lumiracoxib", "Hepatic Failure", 2007, 1,
     "TGA cancellation (severe hepatotoxicity)"),
    ("cerivastatin", "Rhabdomyolysis", 2001, 1,
     "Withdrawal from sale"),
    ("sibutramine", "Myocardial Infarction", 2010, 1,
     "Market withdrawal (cardiovascular events)"),
    ("dextropropoxyphene", "Cardiac Arrest", 2012, 1,
     "TGA cancellation (cardiac deaths)"),
    ("pholcodine", "Anaphylactic Reaction", 2023, 1,
     "TGA cancellation + recall (NMBA anaphylaxis risk)"),
    ("cisapride", "Electrocardiogram QT Prolonged", 2000, 1,
     "Withdrawal from sale (QT prolongation / cardiac arrest)"),
    # ── Tier 2: Restrictions / Boxed Warnings ──
    ("rosiglitazone", "Myocardial Infarction", 2008, 2,
     "Contraindicated in ischaemic heart disease"),
    ("pioglitazone", "Bladder Cancer", 2012, 2,
     "TGA safety alert + PI update"),
    ("strontium", "Myocardial Infarction", 2014, 2,
     "Contraindicated in cardiovascular disease"),
    ("domperidone", "Ventricular Arrhythmia", 2014, 2,
     "Dose restriction + cardiac risk warning"),
    ("metoclopramide", "Tardive Dyskinesia", 2015, 2,
     "Duration restriction (max 5 days) + dose cap"),
    ("codeine", "Drug Dependence", 2018, 2,
     "Rescheduled to prescription-only (OTC removed)"),
    ("pregabalin", "Drug Dependence", 2021, 2,
     "Boxed warning added to PI/CMI"),
    ("gabapentin", "Drug Dependence", 2021, 2,
     "Boxed warning added to PI/CMI"),
    ("febuxostat", "Cardiac Death", 2019, 2,
     "Boxed warning + contraindication in CVD"),
    ("dabigatran", "Gastrointestinal Haemorrhage", 2013, 2,
     "TGA safety alert + PI update (major bleeding risk)"),
    # ── Tier 3: PI Updates / Safety Alerts ──
    ("ciprofloxacin", "Tendon Rupture", 2019, 3,
     "Strengthened PI warnings (fluoroquinolone class)"),
    ("ciprofloxacin", "Aortic Aneurysm", 2019, 3,
     "PI update (aortic aneurysm/dissection risk)"),
    ("ciprofloxacin", "Peripheral Neuropathy", 2019, 3,
     "Strengthened PI warnings"),
    ("dapagliflozin", "Diabetic Ketoacidosis", 2018, 3,
     "TGA safety alert (SGLT2 inhibitor class)"),
    ("empagliflozin", "Diabetic Ketoacidosis", 2018, 3,
     "TGA safety alert (SGLT2 inhibitor class)"),
    ("valproate", "Foetal Malformation", 2019, 3,
     "TGA safety alert (teratogenicity)"),
    ("montelukast", "Depression", 2018, 3,
     "Strengthened neuropsychiatric warnings"),
    ("montelukast", "Suicidal Ideation", 2024, 3,
     "More prominent safety warnings"),
    ("isotretinoin", "Depression", 2022, 3,
     "PI update (mental health assessment required)"),
    ("finasteride", "Depression", 2021, 3,
     "PI update (uncommon ADR added)"),
    ("alendronate", "Osteonecrosis Of Jaw", 2007, 3,
     "TGA safety alert (bisphosphonate class)"),
    ("zoledronic acid", "Osteonecrosis Of Jaw", 2007, 3,
     "TGA safety alert (bisphosphonate class)"),
    ("omeprazole", "Hypomagnesaemia", 2011, 3,
     "TGA safety alert (PPI class)"),
    ("pantoprazole", "Hypomagnesaemia", 2011, 3,
     "TGA safety alert (PPI class)"),
    ("natalizumab", "Progressive Multifocal Leukoencephalopathy", 2017, 3,
     "TGA MSU (restricted distribution)"),
    ("lenalidomide", "Second Primary Malignancy", 2012, 3,
     "TGA MSU"),
    ("clozapine", "Myocarditis", None, 3,
     "TGA mandatory monitoring (ongoing)"),
    ("semaglutide", "Suicidal Ideation", 2024, 3,
     "PI warnings aligned across GLP-1 RA class"),
    ("tramadol", "Seizure", None, 3,
     "PI warnings"),
    ("tramadol", "Hypoglycaemia", 2018, 3,
     "PI update"),
    ("bupropion", "Serotonin Syndrome", None, 3,
     "TGA MSU"),
]


def analysis_6(disp_df, out):
    """TGA Regulatory Action Concordance — face-validity assessment."""
    out.write("=" * 70 + "\n")
    out.write("  Analysis 6: TGA Regulatory Action Concordance\n")
    out.write("=" * 70 + "\n\n")

    out.write(f"  TGA regulatory actions to check: {len(TGA_REGULATORY_ACTIONS)}\n\n")

    # Build lookup structures
    all_pairs = list(zip(
        disp_df["active_ingredient"].str.lower(),
        disp_df["reaction"].str.lower().str.replace("• ", "", regex=False).str.strip()
    ))

    def find_match(d_pat, r_pat):
        """Find matching row in disp_df using substring matching."""
        d_low, r_low = d_pat.lower(), r_pat.lower()
        mask = (
            disp_df["active_ingredient"].str.lower().str.contains(d_low, na=False)
            & disp_df["reaction"].str.lower().str.replace("• ", "", regex=False)
            .str.strip().str.contains(r_low, na=False)
        )
        if mask.any():
            return disp_df[mask].iloc[0]
        return None

    # Match each TGA action against DAEN signals
    results = []
    for drug, rxn, year, tier, desc in TGA_REGULATORY_ACTIONS:
        row = find_match(drug, rxn)
        matched = row is not None
        if matched:
            n_reports = int(row["a"])
            ebgm = float(row["ebgm"])
            ic = float(row["ic"])
            n_methods = int(row["n_methods_signal"])
            consensus = n_methods == 4
            any_signal = n_methods >= 1
        else:
            n_reports = 0
            ebgm = 0.0
            ic = 0.0
            n_methods = 0
            consensus = False
            any_signal = False

        results.append({
            "drug": drug,
            "reaction": rxn,
            "action_year": year,
            "tier": tier,
            "action_description": desc,
            "in_daen": matched,
            "n_reports": n_reports,
            "ebgm": round(ebgm, 1),
            "ic": round(ic, 2),
            "n_methods_signal": n_methods,
            "consensus_signal": consensus,
            "any_signal": any_signal,
        })

    results_df = pd.DataFrame(results)

    # Summary statistics
    n_total = len(results_df)
    n_in_daen = int(results_df["in_daen"].sum())
    n_consensus = int(results_df["consensus_signal"].sum())
    n_any = int(results_df["any_signal"].sum())

    out.write(f"  TGA actions defined:               {n_total:>5d}\n")
    out.write(f"  Matchable in DAEN data:            {n_in_daen:>5d}\n")
    out.write(f"  Recovered by consensus (all 4):    {n_consensus:>5d}"
              f"  ({100*n_consensus/n_in_daen:.1f}%)\n" if n_in_daen > 0 else "\n")
    out.write(f"  Recovered by any method (>=1):     {n_any:>5d}"
              f"  ({100*n_any/n_in_daen:.1f}%)\n\n" if n_in_daen > 0 else "\n\n")

    # By tier
    out.write("  Recovery by regulatory action tier:\n")
    out.write(f"  {'Tier':>6s}  {'Description':<40s}  {'N':>3s}  "
              f"{'In DAEN':>7s}  {'Consensus':>9s}  {'Any DPA':>7s}\n")
    out.write(f"  {'─' * 80}\n")
    tier_labels = {1: "Withdrawal / Cancellation",
                   2: "Restriction / Boxed Warning",
                   3: "PI Update / Safety Alert"}
    for t in [1, 2, 3]:
        subset = results_df[results_df["tier"] == t]
        n_t = len(subset)
        n_d = int(subset["in_daen"].sum())
        n_c = int(subset["consensus_signal"].sum())
        n_a = int(subset["any_signal"].sum())
        pct_c = f"{100*n_c/n_d:.0f}%" if n_d > 0 else "N/A"
        pct_a = f"{100*n_a/n_d:.0f}%" if n_d > 0 else "N/A"
        out.write(f"  {t:>6d}  {tier_labels[t]:<40s}  {n_t:>3d}  "
                  f"{n_d:>7d}  {pct_c:>9s}  {pct_a:>7s}\n")
    out.write("\n")

    # Detailed table of matched pairs
    matched_df = results_df[results_df["in_daen"]].sort_values(
        ["tier", "ebgm"], ascending=[True, False])
    out.write("  MATCHED TGA REGULATORY ACTIONS:\n")
    out.write(f"  {'─' * 95}\n")
    out.write(f"  {'Tier':>4s} {'Drug':<28s} {'Reaction':<28s} "
              f"{'Year':>5s} {'n':>5s} {'EBGM':>7s} {'DPA':>4s} "
              f"{'Consensus':>9s}\n")
    out.write(f"  {'─' * 95}\n")
    for _, r in matched_df.iterrows():
        yr = str(int(r["action_year"])) if pd.notna(r["action_year"]) else "  --"
        cons_str = "YES" if r["consensus_signal"] else "no"
        out.write(f"  {int(r['tier']):>4d} {r['drug']:<28s} "
                  f"{r['reaction'][:28]:<28s} {yr:>5s} "
                  f"{int(r['n_reports']):>5d} {r['ebgm']:>7.1f} "
                  f"{int(r['n_methods_signal']):>4d}/4 "
                  f"{cons_str:>9s}\n")

    # Show unmatched pairs
    unmatched = results_df[~results_df["in_daen"]]
    if len(unmatched) > 0:
        out.write(f"\n  NOT MATCHED IN DAEN ({len(unmatched)}):\n")
        for _, r in unmatched.iterrows():
            out.write(f"    T{int(r['tier'])}: {r['drug']} + {r['reaction']}\n")

    # Save
    results_df.to_csv(OUTPUT_DIR / "validation_tga_regulatory_concordance.csv",
                      index=False)
    out.write(f"\n  Saved: validation_tga_regulatory_concordance.csv\n\n")
    return results_df


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    print("=" * 70)
    print("  TGA DAEN — Additional Statistical Validation")
    print("  6 analyses for manuscript preparation")
    print("=" * 70)

    # Capture output for report file
    report = io.StringIO()

    class TeeWriter:
        """Write to both stdout and a StringIO buffer."""
        def __init__(self, buf):
            self.buf = buf
        def write(self, s):
            print(s, end="")
            self.buf.write(s)

    out = TeeWriter(report)

    # ── Load data ─────────────────────────────────────────────────────────
    out.write("\nLoading data ...\n")
    disp_df = pd.read_csv(OUTPUT_DIR / "disproportionality_full.csv")
    out.write(f"  DPA pairs: {len(disp_df):,}\n")

    # ── Analysis 1 (no feature engineering needed) ────────────────────────
    out.write("\n")
    analysis_1(disp_df, out)

    # ── Feature engineering for Analysis 2 ────────────────────────────────
    out.write("Loading case-level data for feature engineering ...\n")
    cases = pd.read_csv(PROCESSED_DIR / "daen_cases.csv", low_memory=False)
    drugs = pd.read_csv(PROCESSED_DIR / "daen_case_drugs.csv",
                        low_memory=False)
    reactions = pd.read_csv(PROCESSED_DIR / "daen_case_reactions.csv")
    out.write(f"  Cases: {len(cases):,}  Drugs: {len(drugs):,}  "
              f"Reactions: {len(reactions):,}\n\n")

    disp_featured = engineer_features(disp_df.copy(), cases, drugs, reactions)

    # ── Analysis 2 ────────────────────────────────────────────────────────
    out.write("\n")
    analysis_2(disp_featured, out)

    # Free memory
    del cases, drugs, reactions, disp_featured

    # ── Analysis 3 ────────────────────────────────────────────────────────
    analysis_3(out)

    # ── Analysis 4 ────────────────────────────────────────────────────────
    analysis_4(disp_df, out)

    # ── Analysis 5 ────────────────────────────────────────────────────────
    analysis_5(disp_df, out)

    # ── Analysis 6 ────────────────────────────────────────────────────────
    analysis_6(disp_df, out)

    # ── Save report ───────────────────────────────────────────────────────
    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    report_path = SUPP_DIR / "additional_validation_report.txt"
    report_path.write_text(report.getvalue())
    out.write(f"{'─' * 70}\n")
    out.write(f"Full report saved: {report_path.name}\n")

    elapsed = time.time() - t0
    out.write(f"\n{'=' * 70}\n")
    out.write(f"  ALL ADDITIONAL VALIDATION ANALYSES COMPLETE  ({elapsed:.0f}s)\n")
    out.write(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
