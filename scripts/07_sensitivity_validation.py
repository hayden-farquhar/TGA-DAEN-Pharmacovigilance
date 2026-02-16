"""
Script 07: Sensitivity Analyses and Statistical Validation

Nine analyses to strengthen the manuscript for Drug Safety review:

  1. Multiple testing correction (Benjamini-Hochberg FDR + Bonferroni)
  2. Inter-method agreement (pairwise Cohen's kappa, Fleiss' kappa)
  3. COVID-19 vaccine impact (DPA with/without COVID vaccine cases)
  4. Minimum report threshold sensitivity (n≥3 vs n≥5 vs n≥10)
  5. Time-period stratification (pre-COVID ≤2020 vs COVID-era 2021+)
  6. Sex-stratified DPA (female vs male)
  7. Age-stratified DPA (pediatric <18, adult 18–64, elderly ≥65)
  8. Masking/competition bias (unmasked signals after COVID exclusion)
  9. Temporal signal detection (year of first detectability, cumulative curves)

Stratified analyses compute PRR, ROR, and IC (EBGM omitted for speed as it
requires per-stratum prior fitting; noted in manuscript as a limitation of
the sensitivity analyses only).

Outputs (in outputs/tables/ and outputs/supplementary/):
  - sensitivity_fdr_correction.csv
  - sensitivity_method_agreement.csv
  - sensitivity_stratified_summary.csv
  - sensitivity_masking_unmasked.csv
  - sensitivity_temporal_detection.csv
  - sensitivity_temporal_cumulative.csv
  - sensitivity_validation_report.txt

Usage:
    python scripts/07_sensitivity_validation.py
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2 as chi2_dist, spearmanr
from sklearn.metrics import cohen_kappa_score
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "tables"
SUPP_DIR = PROJECT_DIR / "outputs" / "supplementary"

# ── Configuration ────────────────────────────────────────────────────────────

MIN_REPORTS = 3
PRR_THRESHOLD = 2.0
PRR_CHI2_THRESHOLD = 4.0
ROR_LOWER_CI_THRESHOLD = 1.0
IC025_THRESHOLD = 0.0

EXCLUDE_INGREDIENTS = {
    "trade name not specified", "product not coded",
    "not specified", "unknown", "",
}

COVID_VACCINE_PATTERNS = ["tozinameran", "elasomeran", "chadox"]


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

report_lines = []


def log(msg=""):
    """Print and buffer for report file."""
    print(msg)
    report_lines.append(msg)


def compute_dpa_for_subset(susp_sub, rxn_sub, N_sub, min_reports=3):
    """
    Lightweight DPA (PRR, ROR, IC) for a case subset.

    Parameters
    ----------
    susp_sub : DataFrame with [case_number, active_ingredient]
    rxn_sub  : DataFrame with [case_number, reaction]
    N_sub    : int — total cases in subset
    min_reports : int

    Returns
    -------
    DataFrame with DPA results, or None if no valid pairs.
    """
    if len(susp_sub) == 0 or len(rxn_sub) == 0 or N_sub == 0:
        return None

    drug_counts = susp_sub.groupby("active_ingredient")["case_number"].nunique()
    rxn_counts = rxn_sub.groupby("reaction")["case_number"].nunique()

    pairs = susp_sub.merge(rxn_sub, on="case_number", how="inner")
    pc = pairs.groupby(["active_ingredient", "reaction"]).size().reset_index(name="a")
    pc = pc[pc["a"] >= min_reports].copy()

    if len(pc) == 0:
        return None

    pc["n_drug"] = pc["active_ingredient"].map(drug_counts).fillna(0).astype(int)
    pc["n_reaction"] = pc["reaction"].map(rxn_counts).fillna(0).astype(int)
    pc["N"] = N_sub

    a = pc["a"].values.astype(float)
    nd = pc["n_drug"].values.astype(float)
    nr = pc["n_reaction"].values.astype(float)
    N = float(N_sub)

    b = nd - a
    c = nr - a
    d = N - a - b - c

    pc["expected"] = nd * nr / N

    # PRR
    ac, bc, cc, dc = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    prr = (ac / (ac + bc)) / (cc / (cc + dc))
    ln_prr = np.log(prr)
    se_prr = np.sqrt(1 / ac - 1 / (ac + bc) + 1 / cc - 1 / (cc + dc))
    pc["prr"] = prr
    pc["prr_lower95"] = np.exp(ln_prr - 1.96 * se_prr)
    n_total = a + b + c + d
    pc["prr_chi2"] = (
        n_total * (np.abs(a * d - b * c) - n_total / 2) ** 2
    ) / ((a + b) * (c + d) * (a + c) * (b + d))

    # ROR
    ror = (ac * dc) / (bc * cc)
    ln_ror = np.log(ror)
    se_ror = np.sqrt(1 / ac + 1 / bc + 1 / cc + 1 / dc)
    pc["ror"] = ror
    pc["ror_lower95"] = np.exp(ln_ror - 1.96 * se_ror)

    # IC (BCPNN)
    ic = np.log2(((a + 0.5) * (N + 0.5)) / ((nd + 0.5) * (nr + 0.5)))
    ic_var = (1.0 / np.log(2) ** 2) * (1.0 / (a + 0.5) - 1.0 / (N + 0.5))
    ic_se = np.sqrt(np.maximum(ic_var, 0))
    pc["ic"] = ic
    pc["ic025"] = ic - 1.96 * ic_se

    # Signal flags
    pc["signal_prr"] = (
        (pc["prr"] >= PRR_THRESHOLD)
        & (pc["prr_chi2"] >= PRR_CHI2_THRESHOLD)
        & (pc["a"] >= min_reports)
    )
    pc["signal_ror"] = pc["ror_lower95"] > ROR_LOWER_CI_THRESHOLD
    pc["signal_ic"] = pc["ic025"] > IC025_THRESHOLD
    pc["n_methods_signal"] = (
        pc["signal_prr"].astype(int)
        + pc["signal_ror"].astype(int)
        + pc["signal_ic"].astype(int)
    )

    return pc.reset_index(drop=True)


def fleiss_kappa(ratings_matrix):
    """
    Fleiss' kappa for multi-rater agreement.

    ratings_matrix : ndarray (N_subjects × N_categories)
        Each row sums to n (number of raters).
    """
    N, k = ratings_matrix.shape
    n = int(ratings_matrix.sum(axis=1).mean())
    if n <= 1:
        return float("nan")

    P_i = (1.0 / (n * (n - 1))) * (np.sum(ratings_matrix ** 2, axis=1) - n)
    P_bar = np.mean(P_i)

    p_j = np.sum(ratings_matrix, axis=0) / (N * n)
    P_e = np.sum(p_j ** 2)

    if P_e >= 1.0:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


def benjamini_hochberg(p_values, alpha=0.05):
    """BH FDR correction.  Returns (adjusted_p, significant_bool)."""
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = p_values[order]

    adjusted = np.empty(n)
    adjusted[order[-1]] = min(sorted_p[-1], 1.0)
    for i in range(n - 2, -1, -1):
        adjusted[order[i]] = min(sorted_p[i] * n / (i + 1), adjusted[order[i + 1]])
    adjusted = np.minimum(adjusted, 1.0)

    return adjusted, adjusted <= alpha


def jaccard(set_a, set_b):
    """Jaccard similarity coefficient."""
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def stratum_summary(name, dpa, N_cases):
    """One-row summary dict for a DPA stratum."""
    if dpa is None:
        return dict(
            stratum=name, N_cases=N_cases, N_pairs=0,
            signals_prr=0, signals_ror=0, signals_ic=0,
            signals_all3=0, signals_any=0,
        )
    return dict(
        stratum=name,
        N_cases=N_cases,
        N_pairs=len(dpa),
        signals_prr=int(dpa["signal_prr"].sum()),
        signals_ror=int(dpa["signal_ror"].sum()),
        signals_ic=int(dpa["signal_ic"].sum()),
        signals_all3=int((dpa["n_methods_signal"] == 3).sum()),
        signals_any=int((dpa["n_methods_signal"] >= 1).sum()),
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════


def load_all_data():
    """Load all required datasets and prepare base structures."""
    log("Loading data ...")

    cases = pd.read_csv(PROCESSED_DIR / "daen_cases.csv", low_memory=False)
    drugs = pd.read_csv(PROCESSED_DIR / "daen_case_drugs.csv", low_memory=False)
    reactions = pd.read_csv(PROCESSED_DIR / "daen_case_reactions.csv")
    full_dpa = pd.read_csv(OUTPUT_DIR / "disproportionality_full.csv")

    # Normalise case_number types
    for df in [cases, drugs, reactions]:
        df["case_number"] = df["case_number"].astype(str)

    log(f"  Cases:     {len(cases):>10,}")
    log(f"  Drugs:     {len(drugs):>10,}")
    log(f"  Reactions: {len(reactions):>10,}")
    log(f"  DPA pairs: {len(full_dpa):>10,}")

    # Suspected drugs — deduplicated (case, ingredient)
    susp = drugs[drugs["suspected"] == "suspected"][["case_number", "active_ingredient"]].copy()
    susp = susp.dropna(subset=["active_ingredient"])
    susp["active_ingredient"] = susp["active_ingredient"].str.strip().str.lower()
    susp = susp[~susp["active_ingredient"].isin(EXCLUDE_INGREDIENTS)]
    susp = susp.drop_duplicates()

    # Reactions — deduplicated (case, reaction)
    rxn = reactions[["case_number", "reaction"]].drop_duplicates()

    # Case metadata
    case_meta = cases[["case_number"]].copy()
    case_meta["report_year"] = pd.to_datetime(
        cases["report_date"], errors="coerce", format="mixed", dayfirst=True
    ).dt.year
    case_meta["age_numeric"] = pd.to_numeric(cases["age"], errors="coerce")
    case_meta["sex"] = cases["sex"].fillna("Unknown")

    return cases, drugs, reactions, full_dpa, susp, rxn, case_meta


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 1 : MULTIPLE TESTING CORRECTION (FDR)
# ═══════════════════════════════════════════════════════════════════════════════


def analysis_1_fdr(full_dpa):
    log(f"\n{'=' * 70}")
    log("  Analysis 1: Multiple Testing Correction")
    log(f"{'=' * 70}")

    n_pairs = len(full_dpa)
    chi2_values = full_dpa["prr_chi2"].values.copy()
    chi2_values = np.nan_to_num(chi2_values, nan=0, posinf=chi2_values[np.isfinite(chi2_values)].max())

    # Raw p-values from chi-squared (1 df)
    p_values = chi2_dist.sf(chi2_values, df=1)
    p_values = np.clip(p_values, 1e-300, 1.0)  # avoid exact zeros

    n_raw_sig = np.sum(p_values < 0.05)
    log(f"\n  Total pairs tested:               {n_pairs:>10,}")
    log(f"  Uncorrected significant (p<0.05): {n_raw_sig:>10,}")

    # Benjamini-Hochberg at multiple thresholds
    log(f"\n  Benjamini-Hochberg FDR:")
    fdr_results = {}
    for q in [0.01, 0.05, 0.10]:
        adj_p, sig = benjamini_hochberg(p_values, alpha=q)
        n_sig = sig.sum()
        pct = 100 * n_sig / n_raw_sig if n_raw_sig > 0 else 0
        fdr_results[q] = n_sig
        log(f"    q = {q:.2f}:  {n_sig:>10,}  ({pct:.1f}% of uncorrected)")
        if q == 0.05:
            full_dpa["p_value"] = p_values
            full_dpa["p_adjusted_bh"] = adj_p
            full_dpa["significant_fdr05"] = sig

    # Bonferroni
    bonf_threshold = 0.05 / n_pairs
    n_bonf = np.sum(p_values < bonf_threshold)
    pct_bonf = 100 * n_bonf / n_raw_sig if n_raw_sig > 0 else 0
    log(f"\n  Bonferroni (α = 0.05):")
    log(f"    Adjusted threshold:             {bonf_threshold:.2e}")
    log(f"    Surviving signals:              {n_bonf:>10,}  ({pct_bonf:.1f}% of uncorrected)")
    full_dpa["significant_bonf"] = p_values < bonf_threshold

    # Impact on consensus signals
    consensus_mask = full_dpa["n_methods_signal"] == 4
    n_consensus = consensus_mask.sum()
    n_consensus_fdr = (consensus_mask & full_dpa["significant_fdr05"]).sum()
    pct_retained = 100 * n_consensus_fdr / n_consensus if n_consensus > 0 else 0
    log(f"\n  Impact on original consensus signals (all 4 methods):")
    log(f"    Original:                       {n_consensus:>10,}")
    log(f"    After FDR (q=0.05) on PRR χ²:   {n_consensus_fdr:>10,}  ({pct_retained:.1f}% retained)")

    # Save
    fdr_path = OUTPUT_DIR / "sensitivity_fdr_correction.csv"
    fdr_cols = [
        "active_ingredient", "reaction", "a", "expected",
        "prr", "prr_chi2", "p_value", "p_adjusted_bh",
        "significant_fdr05", "significant_bonf",
        "signal_prr", "signal_ror", "signal_ebgm", "signal_bcpnn",
        "n_methods_signal",
    ]
    full_dpa[fdr_cols].to_csv(fdr_path, index=False, float_format="%.6f")
    log(f"\n  Saved: {fdr_path.name}")

    return full_dpa


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 2 : INTER-METHOD AGREEMENT
# ═══════════════════════════════════════════════════════════════════════════════


def analysis_2_agreement(full_dpa):
    log(f"\n{'=' * 70}")
    log("  Analysis 2: Inter-Method Agreement")
    log(f"{'=' * 70}")

    methods = ["signal_prr", "signal_ror", "signal_ebgm", "signal_bcpnn"]
    labels = ["PRR", "ROR", "EBGM", "BCPNN"]

    # Pairwise Cohen's kappa
    log(f"\n  Pairwise Cohen's kappa:")
    header = f"  {'':>8s}" + "".join(f"  {l:>7s}" for l in labels)
    log(header)

    kappa_rows = []
    for i, (mi, li) in enumerate(zip(methods, labels)):
        line = f"  {li:>8s}"
        for j, (mj, lj) in enumerate(zip(methods, labels)):
            if i == j:
                k = 1.0
            else:
                k = cohen_kappa_score(
                    full_dpa[mi].astype(int).values,
                    full_dpa[mj].astype(int).values,
                )
            line += f"  {k:>7.3f}"
            if i < j:
                kappa_rows.append({"method_1": li, "method_2": lj, "cohens_kappa": round(k, 4)})
        log(line)

    # Fleiss' kappa (4 binary raters)
    n_signal = np.column_stack([full_dpa[m].astype(int).values for m in methods])
    n_no_signal = 4 - n_signal.sum(axis=1, keepdims=True)
    ratings = np.hstack([n_signal.sum(axis=1, keepdims=True), n_no_signal])  # (N, 2)
    fk = fleiss_kappa(ratings)
    log(f"\n  Fleiss' kappa (all 4 methods):     {fk:.4f}")

    # Interpretation
    if fk >= 0.81:
        interp = "almost perfect"
    elif fk >= 0.61:
        interp = "substantial"
    elif fk >= 0.41:
        interp = "moderate"
    elif fk >= 0.21:
        interp = "fair"
    else:
        interp = "slight"
    log(f"  Interpretation (Landis & Koch):    {interp}")

    # Agreement distribution
    log(f"\n  Agreement distribution:")
    for n_agree in range(5):
        count = (full_dpa["n_methods_signal"] == n_agree).sum()
        pct = 100 * count / len(full_dpa)
        bar = "█" * max(1, int(40 * pct / 100))
        log(f"    {n_agree}/4 methods agree: {count:>8,} ({pct:>5.1f}%)  {bar}")

    # Save
    kappa_df = pd.DataFrame(kappa_rows)
    kappa_df.loc[len(kappa_df)] = {"method_1": "ALL", "method_2": "Fleiss", "cohens_kappa": round(fk, 4)}
    agree_path = OUTPUT_DIR / "sensitivity_method_agreement.csv"
    kappa_df.to_csv(agree_path, index=False)
    log(f"\n  Saved: {agree_path.name}")

    return kappa_df


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 3 : COVID-19 VACCINE IMPACT
# ═══════════════════════════════════════════════════════════════════════════════


def analysis_3_covid(full_dpa, susp, rxn, case_meta):
    log(f"\n{'=' * 70}")
    log("  Analysis 3: COVID-19 Vaccine Impact")
    log(f"{'=' * 70}")

    # Identify COVID vaccine cases
    covid_mask = susp["active_ingredient"].str.contains(
        "|".join(COVID_VACCINE_PATTERNS), case=False, na=False
    )
    covid_cases = set(susp[covid_mask]["case_number"])
    all_cases = set(case_meta["case_number"])
    non_covid_cases = all_cases - covid_cases

    log(f"\n  COVID vaccine ingredients matched:   {susp[covid_mask]['active_ingredient'].nunique()}")
    log(f"  Cases with COVID vaccine suspected:  {len(covid_cases):>10,}")
    log(f"  Cases without COVID vaccine:         {len(non_covid_cases):>10,}")

    # Re-compute DPA on non-COVID cases
    log(f"\n  Re-computing DPA on non-COVID cases ...")
    susp_nc = susp[susp["case_number"].isin(non_covid_cases)]
    rxn_nc = rxn[rxn["case_number"].isin(non_covid_cases)]
    dpa_nc = compute_dpa_for_subset(susp_nc, rxn_nc, len(non_covid_cases), min_reports=MIN_REPORTS)

    if dpa_nc is not None:
        log(f"  Non-COVID pairs (≥{MIN_REPORTS} reports):    {len(dpa_nc):>10,}")
        log(f"  Signals (PRR):                      {dpa_nc['signal_prr'].sum():>10,}")
        log(f"  Signals (ROR):                      {dpa_nc['signal_ror'].sum():>10,}")
        log(f"  Signals (IC):                       {dpa_nc['signal_ic'].sum():>10,}")
        log(f"  Signals (all 3):                    {(dpa_nc['n_methods_signal'] == 3).sum():>10,}")
    else:
        log("  ERROR: No valid pairs in non-COVID subset.")

    # Compare with full dataset (non-COVID drugs only)
    full_non_covid = full_dpa[~full_dpa["active_ingredient"].str.contains(
        "|".join(COVID_VACCINE_PATTERNS), case=False, na=False
    )]
    log(f"\n  Full-dataset non-COVID drug pairs:   {len(full_non_covid):>10,}")

    # Correlation of scores between full and reduced datasets
    if dpa_nc is not None:
        shared = full_non_covid.merge(
            dpa_nc[["active_ingredient", "reaction", "prr", "ic"]],
            on=["active_ingredient", "reaction"],
            how="inner",
            suffixes=("_full", "_reduced"),
        )
        if len(shared) > 10:
            rho_prr, _ = spearmanr(shared["prr_full"], shared["prr_reduced"])
            rho_ic, _ = spearmanr(shared["ic_full"], shared["ic_reduced"])
            log(f"\n  Spearman correlation (shared pairs, n={len(shared):,}):")
            log(f"    PRR:  ρ = {rho_prr:.4f}")
            log(f"    IC:   ρ = {rho_ic:.4f}")

    return dpa_nc, non_covid_cases


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 4 : MINIMUM REPORT THRESHOLD SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════════


def analysis_4_threshold(full_dpa):
    log(f"\n{'=' * 70}")
    log("  Analysis 4: Minimum Report Threshold Sensitivity")
    log(f"{'=' * 70}")

    thresholds = [3, 5, 10, 25, 50]
    rows = []

    for t in thresholds:
        sub = full_dpa[full_dpa["a"] >= t].copy()
        n_pairs = len(sub)
        # Re-apply signal thresholds (with updated min_reports for PRR)
        sig_prr = ((sub["prr"] >= PRR_THRESHOLD) & (sub["prr_chi2"] >= PRR_CHI2_THRESHOLD)).sum()
        sig_ror = (sub["ror_lower95"] > ROR_LOWER_CI_THRESHOLD).sum()
        sig_ebgm = (sub["eb05"] >= 2.0).sum()
        sig_bcpnn = (sub["ic025"] > IC025_THRESHOLD).sum()
        sig_all4 = (sub["n_methods_signal"] == 4).sum()

        rows.append(dict(
            min_reports=t, n_pairs=n_pairs,
            signals_prr=int(sig_prr), signals_ror=int(sig_ror),
            signals_ebgm=int(sig_ebgm), signals_bcpnn=int(sig_bcpnn),
            signals_all4=int(sig_all4),
        ))

    log(f"\n  {'Threshold':>10s} {'Pairs':>8s} {'PRR':>8s} {'ROR':>8s} {'EBGM':>8s} {'BCPNN':>8s} {'All 4':>8s}")
    log(f"  {'─' * 60}")
    for r in rows:
        log(
            f"  n ≥ {r['min_reports']:>4d}  {r['n_pairs']:>8,}  {r['signals_prr']:>8,}"
            f"  {r['signals_ror']:>8,}  {r['signals_ebgm']:>8,}"
            f"  {r['signals_bcpnn']:>8,}  {r['signals_all4']:>8,}"
        )

    threshold_df = pd.DataFrame(rows)
    return threshold_df


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSES 5–7 : STRATIFIED DPA (TIME, SEX, AGE)
# ═══════════════════════════════════════════════════════════════════════════════


def run_stratified_analyses(susp, rxn, case_meta):
    log(f"\n{'=' * 70}")
    log("  Analyses 5–7: Stratified Disproportionality (PRR, ROR, IC)")
    log(f"{'=' * 70}")

    # Attach metadata to case_number for filtering
    case_set = set(case_meta["case_number"])
    susp_m = susp[susp["case_number"].isin(case_set)].copy()
    rxn_m = rxn[rxn["case_number"].isin(case_set)].copy()

    # Build lookup: case_number → (report_year, age, sex)
    meta_dict = case_meta.set_index("case_number")

    summaries = []
    strata_results = {}  # name → DPA DataFrame

    # ── Analysis 5: Time-period stratification ─────────────────────────
    log(f"\n  Analysis 5: Time-Period Stratification")
    log(f"  {'─' * 50}")

    year_map = meta_dict["report_year"]
    for stratum_name, year_filter in [
        ("Pre-COVID (≤2020)", lambda y: y <= 2020),
        ("COVID-era (2021+)", lambda y: y >= 2021),
    ]:
        cases_in = set(year_map[year_map.apply(year_filter)].index)
        N_s = len(cases_in)
        susp_s = susp_m[susp_m["case_number"].isin(cases_in)]
        rxn_s = rxn_m[rxn_m["case_number"].isin(cases_in)]
        log(f"    {stratum_name}: {N_s:,} cases, computing DPA ...")
        dpa_s = compute_dpa_for_subset(susp_s, rxn_s, N_s, MIN_REPORTS)
        summ = stratum_summary(stratum_name, dpa_s, N_s)
        summ["stratification"] = "time_period"
        summaries.append(summ)
        strata_results[stratum_name] = dpa_s
        log(f"      Pairs: {summ['N_pairs']:,}  |  Signals (all 3): {summ['signals_all3']:,}")

    # Correlation between time periods
    if strata_results.get("Pre-COVID (≤2020)") is not None and strata_results.get("COVID-era (2021+)") is not None:
        pre = strata_results["Pre-COVID (≤2020)"]
        post = strata_results["COVID-era (2021+)"]
        shared = pre.merge(
            post[["active_ingredient", "reaction", "prr", "ic"]],
            on=["active_ingredient", "reaction"], how="inner", suffixes=("_pre", "_post"),
        )
        if len(shared) > 10:
            rho, _ = spearmanr(shared["prr_pre"], shared["prr_post"])
            log(f"    Spearman ρ (PRR, shared pairs n={len(shared):,}): {rho:.4f}")

    # ── Analysis 6: Sex stratification ─────────────────────────────────
    log(f"\n  Analysis 6: Sex Stratification")
    log(f"  {'─' * 50}")

    sex_map = meta_dict["sex"]
    for stratum_name, sex_val in [("Female", "Female"), ("Male", "Male")]:
        cases_in = set(sex_map[sex_map == sex_val].index)
        N_s = len(cases_in)
        susp_s = susp_m[susp_m["case_number"].isin(cases_in)]
        rxn_s = rxn_m[rxn_m["case_number"].isin(cases_in)]
        log(f"    {stratum_name}: {N_s:,} cases, computing DPA ...")
        dpa_s = compute_dpa_for_subset(susp_s, rxn_s, N_s, MIN_REPORTS)
        summ = stratum_summary(stratum_name, dpa_s, N_s)
        summ["stratification"] = "sex"
        summaries.append(summ)
        strata_results[stratum_name] = dpa_s
        log(f"      Pairs: {summ['N_pairs']:,}  |  Signals (all 3): {summ['signals_all3']:,}")

    # Sex correlation
    if strata_results.get("Female") is not None and strata_results.get("Male") is not None:
        fem = strata_results["Female"]
        mal = strata_results["Male"]
        shared = fem.merge(
            mal[["active_ingredient", "reaction", "prr", "ic"]],
            on=["active_ingredient", "reaction"], how="inner", suffixes=("_f", "_m"),
        )
        if len(shared) > 10:
            rho, _ = spearmanr(shared["prr_f"], shared["prr_m"])
            log(f"    Spearman ρ (PRR, shared pairs n={len(shared):,}): {rho:.4f}")

    # ── Analysis 7: Age stratification ─────────────────────────────────
    log(f"\n  Analysis 7: Age Stratification")
    log(f"  {'─' * 50}")

    age_map = meta_dict["age_numeric"]
    for stratum_name, age_filter in [
        ("Paediatric (<18)", lambda a: a < 18),
        ("Adult (18–64)", lambda a: (a >= 18) & (a < 65)),
        ("Elderly (≥65)", lambda a: a >= 65),
    ]:
        valid_ages = age_map.dropna()
        cases_in = set(valid_ages[valid_ages.apply(age_filter)].index)
        N_s = len(cases_in)
        susp_s = susp_m[susp_m["case_number"].isin(cases_in)]
        rxn_s = rxn_m[rxn_m["case_number"].isin(cases_in)]
        log(f"    {stratum_name}: {N_s:,} cases, computing DPA ...")
        dpa_s = compute_dpa_for_subset(susp_s, rxn_s, N_s, MIN_REPORTS)
        summ = stratum_summary(stratum_name, dpa_s, N_s)
        summ["stratification"] = "age"
        summaries.append(summ)
        strata_results[stratum_name] = dpa_s
        log(f"      Pairs: {summ['N_pairs']:,}  |  Signals (all 3): {summ['signals_all3']:,}")

    # Summary table
    strat_df = pd.DataFrame(summaries)
    log(f"\n  {'─' * 75}")
    log(f"  {'Stratum':<25s} {'N cases':>10s} {'Pairs':>8s} {'PRR':>7s} {'ROR':>7s} {'IC':>7s} {'All 3':>7s}")
    log(f"  {'─' * 75}")
    for _, r in strat_df.iterrows():
        log(
            f"  {r['stratum']:<25s} {r['N_cases']:>10,} {r['N_pairs']:>8,}"
            f" {r['signals_prr']:>7,} {r['signals_ror']:>7,}"
            f" {r['signals_ic']:>7,} {r['signals_all3']:>7,}"
        )

    strat_path = OUTPUT_DIR / "sensitivity_stratified_summary.csv"
    strat_df.to_csv(strat_path, index=False)
    log(f"\n  Saved: {strat_path.name}")

    return strat_df, strata_results


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 8 : MASKING / COMPETITION BIAS
# ═══════════════════════════════════════════════════════════════════════════════


def analysis_8_masking(full_dpa, dpa_no_covid, susp):
    log(f"\n{'=' * 70}")
    log("  Analysis 8: Masking / Competition Bias (COVID Vaccine Removal)")
    log(f"{'=' * 70}")

    if dpa_no_covid is None:
        log("  SKIPPED: no COVID-excluded DPA available.")
        return None

    # Focus on non-COVID drugs in both datasets
    covid_pattern = "|".join(COVID_VACCINE_PATTERNS)
    full_non_covid = full_dpa[
        ~full_dpa["active_ingredient"].str.contains(covid_pattern, case=False, na=False)
    ].copy()
    reduced_non_covid = dpa_no_covid[
        ~dpa_no_covid["active_ingredient"].str.contains(covid_pattern, case=False, na=False)
    ].copy()

    # Shared pairs
    full_keys = set(
        full_non_covid["active_ingredient"] + "|||" + full_non_covid["reaction"]
    )
    reduced_keys = set(
        reduced_non_covid["active_ingredient"] + "|||" + reduced_non_covid["reaction"]
    )

    # Signals in full dataset vs reduced (using 3-method consensus: PRR+ROR+IC)
    full_non_covid["key"] = full_non_covid["active_ingredient"] + "|||" + full_non_covid["reaction"]
    reduced_non_covid["key"] = reduced_non_covid["active_ingredient"] + "|||" + reduced_non_covid["reaction"]

    # Full dataset signals (using original 4-method consensus from EBGM)
    full_signals = set(full_non_covid[full_non_covid["n_methods_signal"] == 4]["key"])

    # Reduced dataset signals (3-method consensus: PRR+ROR+IC)
    reduced_signals = set(reduced_non_covid[reduced_non_covid["n_methods_signal"] == 3]["key"])

    # Newly unmasked: signal in reduced but NOT in full
    unmasked = reduced_signals - full_signals
    # Disappeared: signal in full but NOT in reduced
    disappeared = full_signals - reduced_signals

    log(f"\n  Non-COVID pairs in full dataset:     {len(full_keys):>8,}")
    log(f"  Non-COVID pairs in reduced dataset:  {len(reduced_keys):>8,}")
    log(f"  Shared pairs:                        {len(full_keys & reduced_keys):>8,}")
    log(f"\n  Full-dataset signals (4 methods):    {len(full_signals):>8,}")
    log(f"  Reduced-dataset signals (3 methods): {len(reduced_signals):>8,}")
    log(f"  Newly unmasked signals:              {len(unmasked):>8,}")
    log(f"  Disappeared signals:                 {len(disappeared):>8,}")

    # Detail on unmasked signals
    if len(unmasked) > 0:
        unmasked_df = reduced_non_covid[reduced_non_covid["key"].isin(unmasked)].copy()
        unmasked_df = unmasked_df.sort_values("a", ascending=False)
        log(f"\n  TOP 20 UNMASKED SIGNALS (sorted by report count)")
        log(f"  {'─' * 75}")
        log(f"  {'Drug':<30s} {'Reaction':<25s} {'n':>5s} {'PRR':>7s} {'IC':>6s}")
        log(f"  {'─' * 75}")
        for _, row in unmasked_df.head(20).iterrows():
            drug = str(row["active_ingredient"])[:29]
            rxn = str(row["reaction"])[:24]
            log(f"  {drug:<30s} {rxn:<25s} {row['a']:>5.0f} {row['prr']:>7.1f} {row['ic']:>6.2f}")

        # Save
        save_cols = ["active_ingredient", "reaction", "a", "expected", "prr", "ror", "ic", "ic025"]
        mask_path = OUTPUT_DIR / "sensitivity_masking_unmasked.csv"
        unmasked_df[save_cols].to_csv(mask_path, index=False, float_format="%.4f")
        log(f"\n  Saved: {mask_path.name}  ({len(unmasked_df):,} signals)")
        return unmasked_df

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS 9 : TEMPORAL SIGNAL DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


def analysis_9_temporal(full_dpa, susp, rxn, case_meta):
    log(f"\n{'=' * 70}")
    log("  Analysis 9: Temporal Signal Detection")
    log(f"{'=' * 70}")

    # Get consensus signals (all 4 methods)
    consensus = full_dpa[full_dpa["n_methods_signal"] == 4][
        ["active_ingredient", "reaction", "a", "expected"]
    ].copy()
    consensus["pair_key"] = consensus["active_ingredient"] + "|||" + consensus["reaction"]
    consensus_keys = set(consensus["pair_key"])
    log(f"\n  Consensus signals to track: {len(consensus_keys):,}")

    # Build triples with report year
    log("  Building temporal triples ...")
    year_map = case_meta.set_index("case_number")["report_year"]

    susp_y = susp.copy()
    susp_y["report_year"] = susp_y["case_number"].map(year_map)
    susp_y = susp_y.dropna(subset=["report_year"])
    susp_y["report_year"] = susp_y["report_year"].astype(int)

    rxn_y = rxn.copy()
    rxn_y["report_year"] = rxn_y["case_number"].map(year_map)
    rxn_y = rxn_y.dropna(subset=["report_year"])
    rxn_y["report_year"] = rxn_y["report_year"].astype(int)

    triples = susp_y.merge(rxn_y[["case_number", "reaction", "report_year"]],
                           on=["case_number", "report_year"], how="inner")
    triples["pair_key"] = triples["active_ingredient"] + "|||" + triples["reaction"]

    # Filter to consensus pairs only
    cons_triples = triples[triples["pair_key"].isin(consensus_keys)]
    log(f"  Consensus triples: {len(cons_triples):,}")

    # Count reports per pair per year
    yearly = (
        cons_triples.groupby(["pair_key", "report_year"])
        .size()
        .reset_index(name="n_reports")
        .sort_values(["pair_key", "report_year"])
    )

    # Cumulative reports
    yearly["cumulative"] = yearly.groupby("pair_key")["n_reports"].cumsum()

    # First year with ≥3 cumulative reports (minimum detectable)
    detectable = yearly[yearly["cumulative"] >= 3].copy()
    first_detect = detectable.groupby("pair_key")["report_year"].first().reset_index()
    first_detect.columns = ["pair_key", "year_first_detectable"]

    # First year with ≥10 cumulative reports (robust signal)
    robust = yearly[yearly["cumulative"] >= 10].copy()
    first_robust = robust.groupby("pair_key")["report_year"].first().reset_index()
    first_robust.columns = ["pair_key", "year_robust"]

    # Merge back
    temporal = consensus.merge(first_detect, on="pair_key", how="left")
    temporal = temporal.merge(first_robust, on="pair_key", how="left")

    # First year of any report
    first_any = yearly.groupby("pair_key")["report_year"].first().reset_index()
    first_any.columns = ["pair_key", "year_first_report"]
    temporal = temporal.merge(first_any, on="pair_key", how="left")

    # Latency: years from first report to detectable
    temporal["latency_to_detectable"] = (
        temporal["year_first_detectable"] - temporal["year_first_report"]
    )

    log(f"\n  Temporal summary:")
    log(f"    Signals with year data:           {temporal['year_first_detectable'].notna().sum():>8,}")
    log(f"    Median first report year:         {temporal['year_first_report'].median():>8.0f}")
    log(f"    Median year first detectable:     {temporal['year_first_detectable'].median():>8.0f}")
    log(f"    Median year robust (≥10 reports): {temporal['year_robust'].median():>8.0f}")
    log(f"    Median latency (first→detectable):{temporal['latency_to_detectable'].median():>7.0f} years")

    # Cumulative detection curve: by year, fraction of consensus signals detectable
    all_years = sorted(yearly["report_year"].unique())
    cum_curve = []
    for yr in all_years:
        detectable_by_yr = first_detect[first_detect["year_first_detectable"] <= yr]
        frac = len(detectable_by_yr) / len(consensus) if len(consensus) > 0 else 0
        cum_curve.append({"year": yr, "n_detectable": len(detectable_by_yr),
                          "fraction_detectable": round(frac, 4)})

    cum_df = pd.DataFrame(cum_curve)

    # Milestones
    log(f"\n  Cumulative detection milestones:")
    for yr in [2000, 2005, 2010, 2015, 2020, 2025]:
        row = cum_df[cum_df["year"] <= yr]
        if len(row) > 0:
            last = row.iloc[-1]
            log(
                f"    By {yr}: {last['n_detectable']:>6,.0f} / {len(consensus):,}"
                f" ({100 * last['fraction_detectable']:.1f}%)"
            )

    # Early detection: signals first detectable ≥5 years before most recent data
    early_detect = temporal[temporal["year_first_detectable"] <= 2020]
    log(f"\n  Signals detectable by 2020 (pre-COVID): {len(early_detect):,} / {len(consensus):,}")

    # Save
    save_cols = [
        "active_ingredient", "reaction", "a",
        "year_first_report", "year_first_detectable", "year_robust",
        "latency_to_detectable",
    ]
    temp_path = OUTPUT_DIR / "sensitivity_temporal_detection.csv"
    temporal[save_cols].to_csv(temp_path, index=False)

    cum_path = SUPP_DIR / "sensitivity_temporal_cumulative.csv"
    cum_df.to_csv(cum_path, index=False)

    log(f"\n  Saved: {temp_path.name}")
    log(f"  Saved: {cum_path.name}")

    return temporal, cum_df


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    log("=" * 70)
    log("  TGA DAEN — Sensitivity Analyses & Statistical Validation")
    log("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────
    cases, drugs, reactions, full_dpa, susp, rxn, case_meta = load_all_data()

    # ── Analysis 1: FDR Correction ────────────────────────────────────────
    full_dpa = analysis_1_fdr(full_dpa)

    # ── Analysis 2: Method Agreement ──────────────────────────────────────
    analysis_2_agreement(full_dpa)

    # ── Analysis 3: COVID-19 Vaccine Impact ───────────────────────────────
    dpa_no_covid, non_covid_cases = analysis_3_covid(full_dpa, susp, rxn, case_meta)

    # ── Analysis 4: Threshold Sensitivity ─────────────────────────────────
    threshold_df = analysis_4_threshold(full_dpa)

    # ── Analyses 5–7: Stratified DPA ──────────────────────────────────────
    strat_df, strata_results = run_stratified_analyses(susp, rxn, case_meta)

    # ── Analysis 8: Masking / Competition Bias ────────────────────────────
    analysis_8_masking(full_dpa, dpa_no_covid, susp)

    # ── Analysis 9: Temporal Signal Detection ─────────────────────────────
    analysis_9_temporal(full_dpa, susp, rxn, case_meta)

    # ── Save comprehensive report ─────────────────────────────────────────
    elapsed = time.time() - t0

    log(f"\n{'=' * 70}")
    log(f"  ALL SENSITIVITY ANALYSES COMPLETE  ({elapsed:.0f}s)")
    log(f"{'=' * 70}")

    # Save text report
    report_path = SUPP_DIR / "sensitivity_validation_report.txt"
    SUPP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    # Also save threshold sensitivity
    threshold_df.to_csv(OUTPUT_DIR / "sensitivity_threshold.csv", index=False)

    log(f"\n  Report saved: {report_path.name}")

    # List all outputs
    log(f"\n  Output files:")
    for p in sorted(OUTPUT_DIR.glob("sensitivity_*")):
        size_kb = p.stat().st_size / 1024
        log(f"    {p.name:50s} {size_kb:>8.1f} KB")
    for p in sorted(SUPP_DIR.glob("sensitivity_*")):
        size_kb = p.stat().st_size / 1024
        log(f"    {p.name:50s} {size_kb:>8.1f} KB")

    log(f"\n  Next step: Review results, then start manuscript preparation.\n")


if __name__ == "__main__":
    main()
