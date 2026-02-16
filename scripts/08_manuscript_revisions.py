"""
Script 08: Manuscript Revision Analyses

Implements 25 analytical revisions from manuscript_revision_recommendations.md:

  Section A: MGPS Validation (multi-start, EM comparison, prior sensitivity)
  Section B: Reference Set Enhancements (tiered evaluation, bootstrap CIs)
  Section C: ML Enhancements (SHAP, feature ablation, DeLong, repeated CV,
             LOOCV, calibration)
  Section D: Consensus and Method Analysis (border-zone, threshold comparison,
             weighted consensus, three-tier, ROR-BCPNN disagreement)
  Section E: Network Enhancements (cross-community edges, link prediction,
             network vs DPA, temporal network)

Depends on outputs from scripts 03, 04, 05, 07.

Outputs:
    outputs/revisions/revision_*.csv  — All analytical outputs

Usage:
    python scripts/08_manuscript_revisions.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import digamma, polygamma
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import networkx as nx
from networkx.algorithms import bipartite
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
FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"

# ── Feature definitions (same as scripts 04/07) ─────────────────────────────

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
#  HELPERS (reused from scripts 03/04/07)
# ═══════════════════════════════════════════════════════════════════════════════


def _clopper_pearson(k, n, alpha=0.05):
    """Clopper-Pearson exact binomial 95% CI for proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (lo, hi)


def fit_mgps_prior(n_obs, E, x0=None, max_iter=5000):
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

    if x0 is None:
        x0 = np.array([np.log(0.2 / 0.8), np.log(0.2), np.log(0.1),
                        np.log(2.0), np.log(2.0)])

    result = minimize(neg_log_lik, x0, method="Nelder-Mead",
                      options={"maxiter": max_iter, "xatol": 1e-8,
                               "fatol": 1e-8})

    alpha = 1.0 / (1.0 + np.exp(-result.x[0]))
    a1, b1 = np.exp(result.x[1]), np.exp(result.x[2])
    a2, b2 = np.exp(result.x[3]), np.exp(result.x[4])
    return alpha, a1, b1, a2, b2, result


def compute_ebgm_with_prior(df, alpha, a1, b1, a2, b2):
    """Compute EBGM and EB05 for all pairs given prior parameters."""
    n = df["a"].values.round().astype(float)
    E = df["expected"].values.astype(np.float64)

    p1 = b1 / (b1 + E)
    p2 = b2 / (b2 + E)
    log_nb1 = stats.nbinom.logpmf(n.astype(int), a1, p1)
    log_nb2 = stats.nbinom.logpmf(n.astype(int), a2, p2)
    log_num = np.log(alpha) + log_nb1
    log_den = np.logaddexp(log_num, np.log(1 - alpha) + log_nb2)
    Q = np.exp(log_num - log_den)

    ebgm = np.exp(
        Q * (digamma(a1 + n) - np.log(b1 + E))
        + (1 - Q) * (digamma(a2 + n) - np.log(b2 + E))
    )

    # EB05 via vectorised bisection
    shape1 = a1 + n
    scale1 = 1.0 / (b1 + E)
    shape2 = a2 + n
    scale2 = 1.0 / (b2 + E)
    m = len(n)
    lower = np.zeros(m)
    upper = np.maximum(n / np.maximum(E, 1e-10), 10.0) * 10.0
    for _ in range(60):
        mid = (lower + upper) / 2.0
        cdf = (Q * stats.gamma.cdf(mid, shape1, scale=scale1) +
               (1 - Q) * stats.gamma.cdf(mid, shape2, scale=scale2))
        lower = np.where(cdf < 0.05, mid, lower)
        upper = np.where(cdf >= 0.05, mid, upper)
        if np.max(upper - lower) < 1e-4:
            break
    eb05 = (lower + upper) / 2.0

    return ebgm, eb05


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
        case_meta["is_female"] = cases["sex"].str.lower().eq("female").astype(float)

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

    disp_df = disp_df.merge(agg, on=["active_ingredient", "reaction"], how="left")
    disp_df["log_a"] = np.log1p(disp_df["a"])
    disp_df["log_expected"] = np.log1p(disp_df["expected"])
    disp_df["log_n_drug"] = np.log1p(disp_df["n_drug"])
    disp_df["log_n_reaction"] = np.log1p(disp_df["n_reaction"])

    return disp_df, triples


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION A: MGPS VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════


def section_a_mgps_validation(disp_df, out):
    """MGPS multi-start, EM comparison, and prior sensitivity."""
    out.write("\n" + "=" * 70 + "\n")
    out.write("  SECTION A: MGPS Validation\n")
    out.write("=" * 70 + "\n")

    # Ensure clean integer counts and positive expected values
    # (CSV loading can introduce NaN or float rounding issues)
    mask = disp_df["a"].notna() & disp_df["expected"].notna() & (disp_df["expected"] > 0)
    n_obs = disp_df.loc[mask, "a"].values.round().astype(int)
    E = disp_df.loc[mask, "expected"].values.astype(np.float64)
    out.write(f"  MGPS fitting: {mask.sum():,} valid pairs "
              f"(dropped {(~mask).sum()} with NaN/zero expected)\n")

    # ── A1: Multi-start optimisation ─────────────────────────────────────
    out.write("\n  A1: Multi-start MGPS optimisation\n")
    out.write("  " + "─" * 60 + "\n")

    alpha_grid = [0.1, 0.3, 0.5, 0.7, 0.9]
    a1_grid = [0.2, 0.5, 1.0, 2.0]
    b1_grid = [0.01, 0.1, 0.5]
    a2_grid = [0.5, 1.0, 2.0]
    b2_grid = [0.1, 0.5, 2.0]

    starts = []
    for al in alpha_grid:
        for a1v in a1_grid:
            for b1v in b1_grid:
                for a2v in a2_grid:
                    for b2v in b2_grid:
                        starts.append((al, a1v, b1v, a2v, b2v))

    # Subsample to ~25 diverse starts for speed
    rng = np.random.default_rng(42)
    if len(starts) > 25:
        idx = rng.choice(len(starts), 25, replace=False)
        starts = [starts[i] for i in idx]

    results = []
    for i, (al, a1v, b1v, a2v, b2v) in enumerate(starts):
        x0 = np.array([
            np.log(al / (1 - al)),
            np.log(a1v), np.log(b1v),
            np.log(a2v), np.log(b2v),
        ])
        try:
            alpha, a1, b1, a2, b2, opt = fit_mgps_prior(n_obs, E, x0=x0,
                                                          max_iter=3000)
            results.append({
                "start": i + 1,
                "init_alpha": al, "init_a1": a1v, "init_b1": b1v,
                "init_a2": a2v, "init_b2": b2v,
                "final_alpha": round(alpha, 4),
                "final_a1": round(a1, 4), "final_b1": round(b1, 4),
                "final_a2": round(a2, 4), "final_b2": round(b2, 4),
                "neg_log_lik": round(opt.fun, 2),
                "converged": opt.success,
            })
        except Exception:
            pass

    ms_df = pd.DataFrame(results).sort_values("neg_log_lik")
    ms_df.to_csv(REVISION_DIR / "revision_mgps_multistart.csv", index=False)

    best = ms_df.iloc[0]
    worst = ms_df.iloc[-1]
    nll_range = worst["neg_log_lik"] - best["neg_log_lik"]
    n_converged = ms_df["converged"].sum()

    out.write(f"  Starting points tested: {len(ms_df)}\n")
    out.write(f"  Converged: {n_converged}/{len(ms_df)}\n")
    out.write(f"  Best -LL: {best['neg_log_lik']:.2f}\n")
    out.write(f"  Worst -LL: {worst['neg_log_lik']:.2f}\n")
    out.write(f"  Range: {nll_range:.2f}\n")
    out.write(f"  Best params: alpha={best['final_alpha']}, "
              f"a1={best['final_a1']}, b1={best['final_b1']}, "
              f"a2={best['final_a2']}, b2={best['final_b2']}\n")

    # Check convergence to similar solutions
    nll_vals = ms_df["neg_log_lik"].values
    within_10 = np.sum(np.abs(nll_vals - nll_vals[0]) < 10)
    out.write(f"  Solutions within 10 of best -LL: {within_10}/{len(ms_df)}\n")
    out.write(f"  Saved: revision_mgps_multistart.csv\n")

    # ── A2: EM algorithm comparison ──────────────────────────────────────
    out.write("\n  A2: EM algorithm for gamma mixture fitting\n")
    out.write("  " + "─" * 60 + "\n")

    em_results = _em_gamma_mixture(n_obs, E, max_iter=200, out=out)
    em_df = pd.DataFrame([em_results])
    em_df.to_csv(REVISION_DIR / "revision_mgps_em_comparison.csv", index=False)
    out.write(f"  Saved: revision_mgps_em_comparison.csv\n")

    # ── A4: Prior sensitivity analysis ───────────────────────────────────
    out.write("\n  A4: Prior sensitivity analysis\n")
    out.write("  " + "─" * 60 + "\n")

    priors = {
        "Fitted (DAEN)": (0.2444, 0.52, 0.006, 1.17, 0.41),
        "DuMouchel FAERS": (0.2, 0.25, 0.5, 1.5, 2.0),
        "Vague/Uninformative": (0.5, 1.0, 1.0, 1.0, 1.0),
    }

    consensus_mask = disp_df["n_methods_signal"] == 4
    sens_rows = []

    for name, (al, a1, b1, a2, b2) in priors.items():
        ebgm_vals, eb05_vals = compute_ebgm_with_prior(disp_df, al, a1, b1, a2, b2)
        n_eb05_ge2 = (eb05_vals >= 2).sum()
        # Of consensus signals, how many still have EB05 >= 2?
        consensus_eb05 = eb05_vals[consensus_mask]
        n_stable = (consensus_eb05 >= 2).sum()
        pct_stable = 100 * n_stable / consensus_mask.sum()

        sens_rows.append({
            "prior": name,
            "alpha": al, "a1": a1, "b1": b1, "a2": a2, "b2": b2,
            "total_eb05_ge2": int(n_eb05_ge2),
            "consensus_stable": int(n_stable),
            "consensus_total": int(consensus_mask.sum()),
            "pct_stable": round(pct_stable, 2),
        })
        out.write(f"  {name}: EB05≥2 signals={n_eb05_ge2:,}, "
                  f"consensus stable={pct_stable:.1f}%\n")

    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(REVISION_DIR / "revision_mgps_prior_sensitivity.csv",
                   index=False)
    out.write(f"  Saved: revision_mgps_prior_sensitivity.csv\n")

    return ms_df, em_results, sens_df


def _em_gamma_mixture(n_obs, E, max_iter=200, tol=1e-6, out=None):
    """EM algorithm for two-component gamma mixture prior."""
    # Filter out any remaining NaN/inf
    valid = np.isfinite(n_obs.astype(float)) & np.isfinite(E) & (E > 0)
    n_obs = n_obs[valid]
    E = E[valid]

    # Initialise from known-good parameters
    alpha = 0.3
    a1, b1 = 0.5, 0.01
    a2, b2 = 1.2, 0.4

    N = len(n_obs)
    prev_ll = -np.inf
    ll = -np.inf

    for iteration in range(max_iter):
        # E-step: compute posterior membership probabilities
        p1 = b1 / (b1 + E)
        p2 = b2 / (b2 + E)
        log_nb1 = stats.nbinom.logpmf(n_obs, a1, p1)
        log_nb2 = stats.nbinom.logpmf(n_obs, a2, p2)

        # Guard against -inf in logpmf
        log_nb1 = np.where(np.isfinite(log_nb1), log_nb1, -700)
        log_nb2 = np.where(np.isfinite(log_nb2), log_nb2, -700)

        log_w1 = np.log(alpha) + log_nb1
        log_w2 = np.log(1 - alpha) + log_nb2
        log_total = np.logaddexp(log_w1, log_w2)

        Q = np.exp(log_w1 - log_total)  # posterior prob of component 1
        Q = np.clip(Q, 1e-10, 1 - 1e-10)  # prevent degenerate Q

        ll = np.sum(log_total[np.isfinite(log_total)])
        if not np.isfinite(ll):
            if out:
                out.write(f"  EM: non-finite LL at iteration {iteration + 1}\n")
            break
        if np.abs(ll - prev_ll) < tol:
            if out:
                out.write(f"  EM converged at iteration {iteration + 1}, "
                          f"LL={ll:.2f}\n")
            break
        prev_ll = ll

        # M-step: update parameters
        alpha = np.clip(np.nanmean(Q), 0.01, 0.99)

        # Update gamma parameters via moment matching
        ratios = n_obs / np.maximum(E, 1e-10)

        # Component 1 (weighted by Q)
        w1_sum = np.nansum(Q)
        if w1_sum > 1:
            mean1 = np.nansum(Q * ratios) / w1_sum
            var1 = np.nansum(Q * (ratios - mean1) ** 2) / w1_sum
            if var1 > 0 and mean1 > 0 and np.isfinite(mean1) and np.isfinite(var1):
                b1_new = mean1 / max(var1, 1e-10)
                a1_new = mean1 * b1_new
                a1 = np.clip(a1_new, 0.01, 100)
                b1 = np.clip(b1_new, 1e-6, 100)

        # Component 2 (weighted by 1-Q)
        w2_sum = np.nansum(1 - Q)
        if w2_sum > 1:
            mean2 = np.nansum((1 - Q) * ratios) / w2_sum
            var2 = np.nansum((1 - Q) * (ratios - mean2) ** 2) / w2_sum
            if var2 > 0 and mean2 > 0 and np.isfinite(mean2) and np.isfinite(var2):
                b2_new = mean2 / max(var2, 1e-10)
                a2_new = mean2 * b2_new
                a2 = np.clip(a2_new, 0.01, 100)
                b2 = np.clip(b2_new, 1e-6, 100)

    if out:
        out.write(f"  EM result: alpha={alpha:.4f}, "
                  f"a1={a1:.4f}, b1={b1:.4f}, a2={a2:.4f}, b2={b2:.4f}\n")
        out.write(f"  EM final LL: {ll:.2f}\n")

    return {
        "method": "EM",
        "alpha": round(alpha, 4), "a1": round(a1, 4), "b1": round(b1, 4),
        "a2": round(a2, 4), "b2": round(b2, 4),
        "log_likelihood": round(float(ll), 2), "iterations": iteration + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION B: REFERENCE SET ENHANCEMENTS
# ═══════════════════════════════════════════════════════════════════════════════


def section_b_reference_set(disp_df, out):
    """Tiered evaluation and bootstrap CIs."""
    out.write("\n" + "=" * 70 + "\n")
    out.write("  SECTION B: Reference Set Enhancements\n")
    out.write("=" * 70 + "\n")

    ref = pd.read_csv(REFERENCE_DIR / "ml_reference_set.csv")
    ref = ref.merge(
        disp_df[["active_ingredient", "reaction", "signal_prr", "signal_ror",
                 "signal_ebgm", "signal_bcpnn", "n_methods_signal",
                 "ebgm", "eb05"]],
        on=["active_ingredient", "reaction"], how="left",
        suffixes=("_ref", ""))

    y = ref["label"].values
    pos = ref[ref["label"] == 1].copy()
    neg = ref[ref["label"] == 0].copy()

    # ── B1: Tiered evaluation ────────────────────────────────────────────
    out.write("\n  B1: Tiered evaluation (positive controls by EBGM)\n")
    out.write("  " + "─" * 60 + "\n")

    # Use EBGM column (not _ref suffix)
    ebgm_col = "ebgm" if "ebgm" in pos.columns else "ebgm_ref"
    pos_ebgm = pos[ebgm_col].values

    tiers = []
    for tier_name, lo, hi in [("Strong (EBGM>50)", 50, np.inf),
                               ("Moderate (5-50)", 5, 50),
                               ("Weak (EBGM<5)", 0, 5)]:
        mask = (pos_ebgm >= lo) & (pos_ebgm < hi)
        n_tier = mask.sum()

        # Sensitivity for this tier: how many are detected by consensus?
        n_meth_col = "n_methods_signal" if "n_methods_signal" in pos.columns else "n_methods_signal_ref"
        detected = (pos.loc[mask.values if hasattr(mask, 'values') else mask,
                            n_meth_col] == 4).sum()
        sens = detected / n_tier if n_tier > 0 else 0
        ci = _clopper_pearson(int(detected), int(n_tier))

        tiers.append({
            "tier": tier_name,
            "n_positives": int(n_tier),
            "detected_consensus": int(detected),
            "sensitivity": round(sens, 4),
            "sens_ci_lo": round(ci[0], 4),
            "sens_ci_hi": round(ci[1], 4),
        })
        out.write(f"  {tier_name}: {detected}/{n_tier} detected, "
                  f"sens={sens:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]\n")

    tier_df = pd.DataFrame(tiers)
    tier_df.to_csv(REVISION_DIR / "revision_tiered_evaluation.csv", index=False)
    out.write(f"  Saved: revision_tiered_evaluation.csv\n")

    # ── B2: Bootstrap CIs ────────────────────────────────────────────────
    out.write("\n  B2: Bootstrap CIs for DPA performance\n")
    out.write("  " + "─" * 60 + "\n")

    n_boot = 10000
    rng = np.random.default_rng(42)
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()

    methods = [
        ("PRR", "signal_prr"),
        ("ROR", "signal_ror"),
        ("EBGM", "signal_ebgm"),
        ("BCPNN", "signal_bcpnn"),
        ("Consensus", None),
    ]

    boot_rows = []
    for name, col in methods:
        if col:
            # Handle potential suffix
            actual_col = col if col in ref.columns else col + "_ref"
            pred = ref[actual_col].astype(int).values
        else:
            n_meth_col = "n_methods_signal" if "n_methods_signal" in ref.columns else "n_methods_signal_ref"
            pred = (ref[n_meth_col] == 4).astype(int).values

        boot_sens = []
        boot_spec = []

        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]

        for _ in range(n_boot):
            # Stratified bootstrap: resample pos and neg separately
            bp = rng.choice(pos_idx, size=n_pos, replace=True)
            bn = rng.choice(neg_idx, size=n_neg, replace=True)
            bi = np.concatenate([bp, bn])

            y_b = y[bi]
            p_b = pred[bi]

            tp = ((y_b == 1) & (p_b == 1)).sum()
            fn = ((y_b == 1) & (p_b == 0)).sum()
            tn = ((y_b == 0) & (p_b == 0)).sum()
            fp = ((y_b == 0) & (p_b == 1)).sum()

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            boot_sens.append(sens)
            boot_spec.append(spec)

        boot_sens = np.array(boot_sens)
        boot_spec = np.array(boot_spec)

        sens_lo, sens_hi = np.percentile(boot_sens, [2.5, 97.5])
        spec_lo, spec_hi = np.percentile(boot_spec, [2.5, 97.5])

        # Also compute Clopper-Pearson for comparison
        tp_orig = ((y == 1) & (pred == 1)).sum()
        fn_orig = ((y == 1) & (pred == 0)).sum()
        tn_orig = ((y == 0) & (pred == 0)).sum()
        fp_orig = ((y == 0) & (pred == 1)).sum()
        sens_orig = tp_orig / (tp_orig + fn_orig) if (tp_orig + fn_orig) > 0 else 0
        spec_orig = tn_orig / (tn_orig + fp_orig) if (tn_orig + fp_orig) > 0 else 0
        cp_sens = _clopper_pearson(int(tp_orig), int(tp_orig + fn_orig))
        cp_spec = _clopper_pearson(int(tn_orig), int(tn_orig + fp_orig))

        boot_rows.append({
            "method": name,
            "sensitivity": round(sens_orig, 4),
            "boot_sens_lo": round(sens_lo, 4),
            "boot_sens_hi": round(sens_hi, 4),
            "cp_sens_lo": round(cp_sens[0], 4),
            "cp_sens_hi": round(cp_sens[1], 4),
            "specificity": round(spec_orig, 4),
            "boot_spec_lo": round(spec_lo, 4),
            "boot_spec_hi": round(spec_hi, 4),
            "cp_spec_lo": round(cp_spec[0], 4),
            "cp_spec_hi": round(cp_spec[1], 4),
        })
        out.write(f"  {name}: Sens={sens_orig:.3f} boot[{sens_lo:.3f},{sens_hi:.3f}] "
                  f"CP[{cp_sens[0]:.3f},{cp_sens[1]:.3f}] | "
                  f"Spec={spec_orig:.3f} boot[{spec_lo:.3f},{spec_hi:.3f}]\n")

    boot_df = pd.DataFrame(boot_rows)
    boot_df.to_csv(REVISION_DIR / "revision_bootstrap_cis.csv", index=False)
    out.write(f"  Saved: revision_bootstrap_cis.csv\n")

    return tier_df, boot_df


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION C: ML ENHANCEMENTS
# ═══════════════════════════════════════════════════════════════════════════════


def _delong_test(y_true, y_score1, y_score2):
    """
    DeLong's test for comparing two AUC values.
    Returns (auc1, auc2, z_stat, p_value).
    """
    n1 = (y_true == 1).sum()
    n0 = (y_true == 0).sum()

    auc1 = roc_auc_score(y_true, y_score1)
    auc2 = roc_auc_score(y_true, y_score2)

    # Placement values
    pos1 = y_score1[y_true == 1]
    neg1 = y_score1[y_true == 0]
    pos2 = y_score2[y_true == 1]
    neg2 = y_score2[y_true == 0]

    # V10 and V01 for each score
    v10_1 = np.array([np.mean(p > neg1) + 0.5 * np.mean(p == neg1) for p in pos1])
    v01_1 = np.array([np.mean(n < pos1) + 0.5 * np.mean(n == pos1) for n in neg1])
    v10_2 = np.array([np.mean(p > neg2) + 0.5 * np.mean(p == neg2) for p in pos2])
    v01_2 = np.array([np.mean(n < pos2) + 0.5 * np.mean(n == pos2) for n in neg2])

    # Covariance matrix of (AUC1, AUC2)
    s10 = np.cov(v10_1, v10_2)
    s01 = np.cov(v01_1, v01_2)

    S = s10 / n1 + s01 / n0

    diff = auc1 - auc2
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]

    if var_diff <= 0:
        return auc1, auc2, 0.0, 1.0

    z = diff / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))

    return auc1, auc2, z, p


def section_c_ml_enhancements(disp_featured, out):
    """SHAP, feature ablation, DeLong, repeated CV, LOOCV, calibration."""
    out.write("\n" + "=" * 70 + "\n")
    out.write("  SECTION C: ML Enhancements\n")
    out.write("=" * 70 + "\n")

    ref = pd.read_csv(REFERENCE_DIR / "ml_reference_set.csv")
    ref_merged = ref[["active_ingredient", "reaction", "label"]].merge(
        disp_featured, on=["active_ingredient", "reaction"], how="left")

    y = ref_merged["label"].values
    X_all = ref_merged[FEATURES_ALL].values
    X_dpa = ref_merged[FEATURES_DPA].values
    X_nondpa = ref_merged[FEATURES_NON_DPA].values
    feature_names = FEATURES_ALL

    # ── C1: SHAP analysis (via XGBoost native pred_contribs) ────────────
    out.write("\n  C1: SHAP analysis (XGBoost native)\n")
    out.write("  " + "─" * 60 + "\n")

    try:
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X_all)

        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42, verbosity=0)
        xgb_model.fit(X_imp, y)

        # Use XGBoost's built-in SHAP via pred_contribs (no shap package needed)
        booster = xgb_model.get_booster()
        dmat = xgb.DMatrix(X_imp, feature_names=feature_names)
        contribs = booster.predict(dmat, pred_contribs=True)
        # contribs shape: (n_samples, n_features + 1), last column is bias
        shap_values = contribs[:, :-1]

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        shap_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
            "gain_importance": xgb_model.feature_importances_,
        }).sort_values("mean_abs_shap", ascending=False)

        shap_df.to_csv(REVISION_DIR / "revision_shap_importance.csv", index=False)

        # Save raw SHAP values for figure generation
        shap_raw = pd.DataFrame(shap_values, columns=feature_names)
        shap_raw.to_csv(REVISION_DIR / "revision_shap_raw_values.csv", index=False)

        # Save feature matrix for SHAP plots
        pd.DataFrame(X_imp, columns=feature_names).to_csv(
            REVISION_DIR / "revision_shap_feature_matrix.csv", index=False)

        out.write(f"  Top 5 features by mean |SHAP|:\n")
        for _, r in shap_df.head(5).iterrows():
            out.write(f"    {r['feature']:<25s} {r['mean_abs_shap']:.4f}\n")
        out.write(f"  Saved: revision_shap_importance.csv\n")

    except Exception as e:
        out.write(f"  SHAP failed ({type(e).__name__}: {e}) — skipping\n")
        shap_df = pd.DataFrame()

    # ── C2: Feature ablation ─────────────────────────────────────────────
    out.write("\n  C2: Feature ablation study\n")
    out.write("  " + "─" * 60 + "\n")

    imp = SimpleImputer(strategy="median")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline AUC with all features
    pipe_base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42, verbosity=0)),
    ])
    from sklearn.model_selection import cross_val_predict
    y_prob_base = cross_val_predict(pipe_base, X_all, y, cv=cv,
                                    method="predict_proba")[:, 1]
    auc_base = roc_auc_score(y, y_prob_base)

    ablation_rows = []
    for i, feat in enumerate(feature_names):
        # Remove feature i
        X_ablated = np.delete(X_all, i, axis=1)
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                eval_metric="logloss", random_state=42, verbosity=0)),
        ])
        y_prob = cross_val_predict(pipe, X_ablated, y, cv=cv,
                                   method="predict_proba")[:, 1]
        auc_without = roc_auc_score(y, y_prob)
        delta = auc_base - auc_without

        ablation_rows.append({
            "feature": feat,
            "auc_with_all": round(auc_base, 4),
            "auc_without": round(auc_without, 4),
            "delta_auc": round(delta, 4),
        })

    ablation_df = pd.DataFrame(ablation_rows).sort_values("delta_auc",
                                                           ascending=False)
    ablation_df.to_csv(REVISION_DIR / "revision_feature_ablation.csv",
                       index=False)
    out.write(f"  Baseline AUC: {auc_base:.4f}\n")
    out.write(f"  Top 5 features by ablation impact:\n")
    for _, r in ablation_df.head(5).iterrows():
        out.write(f"    {r['feature']:<25s} delta={r['delta_auc']:+.4f}\n")
    out.write(f"  Saved: revision_feature_ablation.csv\n")

    # ── C3: DeLong's tests ───────────────────────────────────────────────
    out.write("\n  C3: DeLong's tests for AUC comparison\n")
    out.write("  " + "─" * 60 + "\n")

    # Get OOF predictions for each model
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models_scores = {}

    # XGBoost all features
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42, verbosity=0)),
    ])
    models_scores["XGB_all"] = cross_val_predict(
        pipe, X_all, y, cv=cv, method="predict_proba")[:, 1]

    # XGBoost DPA only
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42, verbosity=0)),
    ])
    models_scores["XGB_dpa"] = cross_val_predict(
        pipe, X_dpa, y, cv=cv, method="predict_proba")[:, 1]

    # XGBoost non-DPA
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42, verbosity=0)),
    ])
    models_scores["XGB_nondpa"] = cross_val_predict(
        pipe, X_nondpa, y, cv=cv, method="predict_proba")[:, 1]

    # Random Forest all features
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
    ])
    models_scores["RF_all"] = cross_val_predict(
        pipe, X_all, y, cv=cv, method="predict_proba")[:, 1]

    # Single DPA scores (for DeLong)
    for col_name, col in [("EBGM", "ebgm"), ("PRR", "prr"),
                           ("ROR", "ror"), ("IC", "ic")]:
        scores = ref_merged[col].values.copy()
        scores = np.nan_to_num(scores, nan=0.0,
                               posinf=np.nanmax(scores[np.isfinite(scores)]),
                               neginf=0.0)
        models_scores[col_name] = scores

    # Run DeLong comparisons
    comparisons = [
        ("XGB_all", "XGB_nondpa", "XGBoost all vs non-DPA"),
        ("XGB_all", "RF_all", "XGBoost vs Random Forest"),
        ("XGB_all", "EBGM", "XGBoost vs EBGM standalone"),
        ("XGB_all", "PRR", "XGBoost vs PRR standalone"),
        ("XGB_all", "ROR", "XGBoost vs ROR standalone"),
        ("XGB_all", "IC", "XGBoost vs IC standalone"),
        ("XGB_dpa", "XGB_nondpa", "XGB DPA-only vs non-DPA"),
        ("RF_all", "EBGM", "Random Forest vs EBGM standalone"),
    ]

    delong_rows = []
    for key1, key2, desc in comparisons:
        try:
            auc1, auc2, z, p = _delong_test(y, models_scores[key1],
                                             models_scores[key2])
            delong_rows.append({
                "comparison": desc,
                "model1": key1, "auc1": round(auc1, 4),
                "model2": key2, "auc2": round(auc2, 4),
                "z_statistic": round(z, 4),
                "p_value": round(p, 6),
                "significant_005": p < 0.05,
            })
            sig = "*" if p < 0.05 else ""
            out.write(f"  {desc}: AUC {auc1:.3f} vs {auc2:.3f}, "
                      f"z={z:.3f}, p={p:.4f} {sig}\n")
        except Exception as e:
            out.write(f"  {desc}: FAILED ({e})\n")
            delong_rows.append({
                "comparison": desc,
                "model1": key1, "auc1": 0,
                "model2": key2, "auc2": 0,
                "z_statistic": 0, "p_value": 1.0,
                "significant_005": False,
            })

    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(REVISION_DIR / "revision_delong_tests.csv", index=False)
    out.write(f"  Saved: revision_delong_tests.csv\n")

    # ── C4: Repeated stratified CV (10 × 5-fold) ────────────────────────
    out.write("\n  C4: Repeated stratified CV (10 × 5-fold)\n")
    out.write("  " + "─" * 60 + "\n")

    rep_rows = []
    model_configs = [
        ("XGBoost All", FEATURES_ALL, lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42, verbosity=0)),
        ("XGBoost DPA", FEATURES_DPA, lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42, verbosity=0)),
        ("RF All", FEATURES_ALL, lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
    ]

    for model_name, features, factory in model_configs:
        X = ref_merged[features].values
        fold_aucs = []

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
                fold_aucs.append(auc)
                rep_rows.append({
                    "model": model_name, "repeat": rep + 1,
                    "fold": fold_i + 1, "auc_roc": round(auc, 4),
                })

        fold_aucs = np.array([a for a in fold_aucs if not np.isnan(a)])
        mean_auc = np.mean(fold_aucs)
        sd_auc = np.std(fold_aucs, ddof=1)
        se = sd_auc / np.sqrt(len(fold_aucs))
        ci_lo = max(0, mean_auc - 1.96 * se)
        ci_hi = min(1, mean_auc + 1.96 * se)

        out.write(f"  {model_name}: {mean_auc:.4f} ± {sd_auc:.4f} "
                  f"[{ci_lo:.4f}, {ci_hi:.4f}] ({len(fold_aucs)} folds)\n")

    rep_df = pd.DataFrame(rep_rows)
    rep_df.to_csv(REVISION_DIR / "revision_repeated_cv.csv", index=False)
    out.write(f"  Saved: revision_repeated_cv.csv\n")

    # Wilcoxon signed-rank test between XGBoost All and RF All
    xgb_aucs = rep_df[rep_df["model"] == "XGBoost All"]["auc_roc"].values
    rf_aucs = rep_df[rep_df["model"] == "RF All"]["auc_roc"].values
    if len(xgb_aucs) == len(rf_aucs) and len(xgb_aucs) > 0:
        try:
            w_stat, w_p = stats.wilcoxon(xgb_aucs, rf_aucs)
            out.write(f"  Wilcoxon XGB vs RF: W={w_stat:.1f}, p={w_p:.4f}\n")
        except Exception:
            out.write(f"  Wilcoxon test: could not compute\n")

    # ── C5: LOOCV ────────────────────────────────────────────────────────
    out.write("\n  C5: Leave-one-out CV\n")
    out.write("  " + "─" * 60 + "\n")

    loo = LeaveOneOut()
    X = ref_merged[FEATURES_ALL].values
    loo_probs = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                eval_metric="logloss", random_state=42, verbosity=0)),
        ])
        pipe.fit(X[train_idx], y[train_idx])
        loo_probs[test_idx] = pipe.predict_proba(X[test_idx])[:, 1]

    loo_auc = roc_auc_score(y, loo_probs)
    loo_ap = average_precision_score(y, loo_probs)

    loo_df = pd.DataFrame({
        "active_ingredient": ref_merged["active_ingredient"].values,
        "reaction": ref_merged["reaction"].values,
        "label": y,
        "loo_probability": loo_probs,
    })
    loo_df.to_csv(REVISION_DIR / "revision_loocv_results.csv", index=False)

    out.write(f"  LOOCV AUC-ROC: {loo_auc:.4f}\n")
    out.write(f"  LOOCV AUC-PR:  {loo_ap:.4f}\n")
    out.write(f"  Saved: revision_loocv_results.csv\n")

    # ── C6: Calibration and Brier score ──────────────────────────────────
    out.write("\n  C6: Calibration and Brier score\n")
    out.write("  " + "─" * 60 + "\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cal_rows = []

    for model_name, features, factory in [
        ("XGBoost", FEATURES_ALL, lambda: xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=42, verbosity=0)),
        ("Random Forest", FEATURES_ALL, lambda: RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
    ]:
        X = ref_merged[features].values
        y_prob = cross_val_predict(
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", factory()),
            ]), X, y, cv=cv, method="predict_proba")[:, 1]

        brier = brier_score_loss(y, y_prob)

        # Calibration curve
        try:
            prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=5,
                                                     strategy="uniform")
            cal_str = "; ".join(
                f"{pt:.2f}vs{pp:.2f}" for pt, pp in zip(prob_true, prob_pred))
        except Exception:
            prob_true, prob_pred = [], []
            cal_str = "N/A"

        cal_rows.append({
            "model": model_name,
            "brier_score": round(brier, 4),
            "n_bins": 5,
            "calibration_bins": cal_str,
        })
        out.write(f"  {model_name}: Brier={brier:.4f}\n")

    # Save calibration data for figure
    # Use XGBoost OOF predictions
    X = ref_merged[FEATURES_ALL].values
    y_prob_xgb = cross_val_predict(
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                eval_metric="logloss", random_state=42, verbosity=0)),
        ]), X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method="predict_proba")[:, 1]

    cal_detail = pd.DataFrame({
        "label": y,
        "xgb_probability": y_prob_xgb,
    })
    cal_detail.to_csv(REVISION_DIR / "revision_calibration_data.csv", index=False)

    cal_df = pd.DataFrame(cal_rows)
    cal_df.to_csv(REVISION_DIR / "revision_calibration.csv", index=False)
    out.write(f"  Saved: revision_calibration.csv\n")

    return shap_df, delong_df, rep_df


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION D: CONSENSUS AND METHOD ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════


def section_d_consensus(disp_df, out):
    """Border-zone analysis, threshold comparison, weighted consensus,
    three-tier, ROR-BCPNN disagreement."""
    out.write("\n" + "=" * 70 + "\n")
    out.write("  SECTION D: Consensus and Method Analysis\n")
    out.write("=" * 70 + "\n")

    # Load validation sets
    label_conc = pd.read_csv(OUTPUT_DIR / "validation_label_concordance.csv")
    tga_conc = pd.read_csv(OUTPUT_DIR / "validation_tga_regulatory_concordance.csv")

    ref = pd.read_csv(REFERENCE_DIR / "ml_reference_set.csv")
    ref = ref.merge(
        disp_df[["active_ingredient", "reaction", "signal_prr", "signal_ror",
                 "signal_ebgm", "signal_bcpnn", "n_methods_signal"]],
        on=["active_ingredient", "reaction"], how="left",
        suffixes=("_ref", ""))
    y_ref = ref["label"].values

    # ── D1: Border-zone analysis ─────────────────────────────────────────
    out.write("\n  D1: Border-zone analysis (1-3 methods, not all 4)\n")
    out.write("  " + "─" * 60 + "\n")

    border = disp_df[
        (disp_df["n_methods_signal"] >= 1) & (disp_df["n_methods_signal"] < 4)
    ].copy()

    out.write(f"  Border-zone pairs: {len(border):,}\n")
    out.write(f"  Mean report count: {border['a'].mean():.1f}\n")
    out.write(f"  Median report count: {border['a'].median():.0f}\n")

    # Method distribution in border zone
    for n_meth in [1, 2, 3]:
        n = (border["n_methods_signal"] == n_meth).sum()
        out.write(f"  Flagged by {n_meth}/4: {n:,}\n")

    # Save border zone details
    border_save = border[["active_ingredient", "reaction", "a", "expected",
                          "prr", "ror", "ebgm", "ic", "n_methods_signal",
                          "signal_prr", "signal_ror", "signal_ebgm",
                          "signal_bcpnn"]].copy()
    border_save.to_csv(REVISION_DIR / "revision_border_zone.csv", index=False)
    out.write(f"  Saved: revision_border_zone.csv\n")

    # ── D2: Threshold comparison ─────────────────────────────────────────
    out.write("\n  D2: Threshold comparison (4/4, ≥3/4, ≥1/4)\n")
    out.write("  " + "─" * 60 + "\n")

    thresh_rows = []
    for thresh_name, thresh in [("Consensus (4/4)", 4), ("Majority (≥3/4)", 3),
                                 ("Any (≥1/4)", 1)]:
        n_meth_col = "n_methods_signal" if "n_methods_signal" in ref.columns else "n_methods_signal_ref"
        pred = (ref[n_meth_col] >= thresh).astype(int).values

        tp = ((y_ref == 1) & (pred == 1)).sum()
        fn = ((y_ref == 1) & (pred == 0)).sum()
        tn = ((y_ref == 0) & (pred == 0)).sum()
        fp = ((y_ref == 0) & (pred == 1)).sum()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0

        thresh_rows.append({
            "threshold": thresh_name,
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        })
        out.write(f"  {thresh_name}: Sens={sens:.3f}, Spec={spec:.3f}\n")

    thresh_df = pd.DataFrame(thresh_rows)
    thresh_df.to_csv(REVISION_DIR / "revision_threshold_comparison.csv",
                     index=False)
    out.write(f"  Saved: revision_threshold_comparison.csv\n")

    # ── D3: Weighted consensus score ─────────────────────────────────────
    out.write("\n  D3: Weighted consensus score\n")
    out.write("  " + "─" * 60 + "\n")

    # Compute specificity for each method to use as weights
    methods_spec = {}
    for name, col in [("PRR", "signal_prr"), ("ROR", "signal_ror"),
                       ("EBGM", "signal_ebgm"), ("BCPNN", "signal_bcpnn")]:
        actual_col = col if col in ref.columns else col + "_ref"
        pred = ref[actual_col].astype(int).values
        tn = ((y_ref == 0) & (pred == 0)).sum()
        fp = ((y_ref == 0) & (pred == 1)).sum()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        methods_spec[name] = spec

    total_spec = sum(methods_spec.values())
    weights = {k: v / total_spec for k, v in methods_spec.items()}

    out.write(f"  Specificity weights:\n")
    for k, v in weights.items():
        out.write(f"    {k}: spec={methods_spec[k]:.3f}, weight={v:.3f}\n")

    # Compute weighted score for all pairs
    disp_df["weighted_score"] = (
        weights["PRR"] * disp_df["signal_prr"].astype(float) +
        weights["ROR"] * disp_df["signal_ror"].astype(float) +
        weights["EBGM"] * disp_df["signal_ebgm"].astype(float) +
        weights["BCPNN"] * disp_df["signal_bcpnn"].astype(float)
    )

    # Grid search optimal threshold on reference set
    ref_weighted = ref.copy()
    for name, col in [("PRR", "signal_prr"), ("ROR", "signal_ror"),
                       ("EBGM", "signal_ebgm"), ("BCPNN", "signal_bcpnn")]:
        actual_col = col if col in ref_weighted.columns else col + "_ref"
        ref_weighted[col + "_float"] = ref_weighted[actual_col].astype(float)

    ref_weighted["weighted_score"] = (
        weights["PRR"] * ref_weighted["signal_prr_float"] +
        weights["ROR"] * ref_weighted["signal_ror_float"] +
        weights["EBGM"] * ref_weighted["signal_ebgm_float"] +
        weights["BCPNN"] * ref_weighted["signal_bcpnn_float"]
    )

    best_f1 = 0
    best_thresh = 0.5
    wc_rows = []
    for t in np.arange(0.1, 1.01, 0.05):
        pred = (ref_weighted["weighted_score"] >= t).astype(int).values
        tp = ((y_ref == 1) & (pred == 1)).sum()
        fp = ((y_ref == 0) & (pred == 1)).sum()
        fn = ((y_ref == 1) & (pred == 0)).sum()
        tn = ((y_ref == 0) & (pred == 0)).sum()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        wc_rows.append({
            "threshold": round(t, 2),
            "sensitivity": round(sens, 4),
            "specificity": round(spec, 4),
            "f1": round(f1, 4),
        })
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    wc_df = pd.DataFrame(wc_rows)
    wc_df.to_csv(REVISION_DIR / "revision_weighted_consensus.csv", index=False)
    out.write(f"  Optimal threshold: {best_thresh:.2f} (F1={best_f1:.3f})\n")
    out.write(f"  Saved: revision_weighted_consensus.csv\n")

    # ── D4: Three-tier restructuring ─────────────────────────────────────
    out.write("\n  D4: Three-tier signal restructuring\n")
    out.write("  " + "─" * 60 + "\n")

    tier1 = disp_df[disp_df["n_methods_signal"] == 4]
    tier2 = disp_df[(disp_df["n_methods_signal"] >= 1) &
                     (disp_df["n_methods_signal"] < 4)]
    tier3 = disp_df[disp_df["n_methods_signal"] == 0]

    tier_summary = pd.DataFrame([
        {"tier": "Tier 1 (Consensus/EBGM)", "n_pairs": len(tier1),
         "mean_a": round(tier1["a"].mean(), 1),
         "median_a": round(tier1["a"].median(), 0),
         "mean_ebgm": round(tier1["ebgm"].mean(), 1)},
        {"tier": "Tier 2 (Border zone)", "n_pairs": len(tier2),
         "mean_a": round(tier2["a"].mean(), 1),
         "median_a": round(tier2["a"].median(), 0),
         "mean_ebgm": round(tier2["ebgm"].mean(), 1)},
        {"tier": "Tier 3 (Non-signal)", "n_pairs": len(tier3),
         "mean_a": round(tier3["a"].mean(), 1),
         "median_a": round(tier3["a"].median(), 0),
         "mean_ebgm": round(tier3["ebgm"].mean(), 1)},
    ])
    tier_summary.to_csv(REVISION_DIR / "revision_three_tier.csv", index=False)

    for _, r in tier_summary.iterrows():
        out.write(f"  {r['tier']}: {r['n_pairs']:,} pairs, "
                  f"mean a={r['mean_a']}, mean EBGM={r['mean_ebgm']}\n")
    out.write(f"  Saved: revision_three_tier.csv\n")

    # ── D5: ROR-BCPNN disagreement ───────────────────────────────────────
    out.write("\n  D5: ROR–BCPNN disagreement\n")
    out.write("  " + "─" * 60 + "\n")

    disagree = disp_df[disp_df["signal_ror"] != disp_df["signal_bcpnn"]].copy()
    out.write(f"  Pairs where ROR and BCPNN disagree: {len(disagree):,}\n")

    if len(disagree) > 0:
        disagree_save = disagree[["active_ingredient", "reaction", "a",
                                   "expected", "ror", "ror_lower95",
                                   "ic", "ic025", "signal_ror",
                                   "signal_bcpnn"]].copy()
        disagree_save.to_csv(REVISION_DIR / "revision_ror_bcpnn_disagreement.csv",
                            index=False)
        out.write(f"  ROR-only signals: {(disagree['signal_ror'] & ~disagree['signal_bcpnn']).sum()}\n")
        out.write(f"  BCPNN-only signals: {(~disagree['signal_ror'] & disagree['signal_bcpnn']).sum()}\n")
    else:
        pd.DataFrame().to_csv(REVISION_DIR / "revision_ror_bcpnn_disagreement.csv",
                             index=False)
    out.write(f"  Saved: revision_ror_bcpnn_disagreement.csv\n")

    return border, thresh_df, wc_df, tier_summary


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION E: NETWORK ENHANCEMENTS
# ═══════════════════════════════════════════════════════════════════════════════


def build_bipartite_network(disp_df, signal_only=False):
    """Build bipartite graph from disproportionality results."""
    if signal_only:
        df = disp_df[disp_df["n_methods_signal"] == 4].copy()
    else:
        df = disp_df.copy()
    G = nx.Graph()
    for d in df["active_ingredient"].unique():
        G.add_node(f"D:{d}", bipartite=0, node_type="drug", label=d)
    for r in df["reaction"].unique():
        G.add_node(f"R:{r}", bipartite=1, node_type="reaction", label=r)
    for _, row in df.iterrows():
        G.add_edge(f"D:{row['active_ingredient']}", f"R:{row['reaction']}",
                   weight=row["a"], ebgm=row.get("ebgm", 1.0))
    return G


def section_e_network(disp_df, triples_df, out):
    """Cross-community edges, link prediction, network vs DPA, temporal."""
    out.write("\n" + "=" * 70 + "\n")
    out.write("  SECTION E: Network Enhancements\n")
    out.write("=" * 70 + "\n")

    # Build networks
    G_signal = build_bipartite_network(disp_df, signal_only=True)
    drug_nodes = {n for n, d in G_signal.nodes(data=True) if d.get("bipartite") == 0}

    # Drug projection
    drug_proj = bipartite.weighted_projected_graph(G_signal, drug_nodes)

    # Load communities
    comm_df = pd.read_csv(OUTPUT_DIR / "network_communities.csv")
    drug_to_comm = dict(zip(comm_df["drug"], comm_df["community"]))

    # ── E1: Cross-community edge analysis ────────────────────────────────
    out.write("\n  E1: Cross-community edge analysis\n")
    out.write("  " + "─" * 60 + "\n")

    cross_edges = []
    for u, v, data in drug_proj.edges(data=True):
        u_label = G_signal.nodes[u].get("label", u) if u in G_signal else u
        v_label = G_signal.nodes[v].get("label", v) if v in G_signal else v

        u_comm = drug_to_comm.get(u_label, drug_to_comm.get(u, -1))
        v_comm = drug_to_comm.get(v_label, drug_to_comm.get(v, -1))

        if u_comm != v_comm and u_comm >= 0 and v_comm >= 0:
            cross_edges.append({
                "drug1": u_label, "community1": u_comm,
                "drug2": v_label, "community2": v_comm,
                "shared_aes": data.get("weight", 0),
            })

    cross_df = pd.DataFrame(cross_edges)
    if len(cross_df) > 0:
        cross_df = cross_df.sort_values("shared_aes", ascending=False)
        cross_df.head(200).to_csv(
            REVISION_DIR / "revision_cross_community_edges.csv", index=False)
        out.write(f"  Total cross-community edges: {len(cross_df):,}\n")
        out.write(f"  Top 5 by shared AEs:\n")
        for _, r in cross_df.head(5).iterrows():
            out.write(f"    {r['drug1']} (C{r['community1']}) — "
                      f"{r['drug2']} (C{r['community2']}): "
                      f"{int(r['shared_aes'])} shared AEs\n")
    else:
        pd.DataFrame().to_csv(
            REVISION_DIR / "revision_cross_community_edges.csv", index=False)
    out.write(f"  Saved: revision_cross_community_edges.csv\n")

    # ── E2: Link prediction ──────────────────────────────────────────────
    out.write("\n  E2: Link prediction\n")
    out.write("  " + "─" * 60 + "\n")

    # Get non-edges in drug projection (sample for computational feasibility)
    drug_list = list(drug_proj.nodes())
    existing_edges = set(drug_proj.edges())
    rng = np.random.default_rng(42)

    # Sample pairs for prediction
    n_sample = min(10000, len(drug_list) * (len(drug_list) - 1) // 2)
    sample_pairs = []
    attempts = 0
    while len(sample_pairs) < n_sample and attempts < n_sample * 10:
        i, j = rng.choice(len(drug_list), 2, replace=False)
        pair = (drug_list[i], drug_list[j])
        if pair not in existing_edges and (pair[1], pair[0]) not in existing_edges:
            sample_pairs.append(pair)
        attempts += 1

    # Compute link prediction scores on existing edges (for ranking)
    # Use Jaccard coefficient on the bipartite graph (shared AE neighbors)
    out.write(f"  Computing Jaccard, Adamic-Adar, Pref. Attachment...\n")

    # Jaccard on drug projection
    link_rows = []
    try:
        jc = list(nx.jaccard_coefficient(drug_proj, sample_pairs[:5000]))
        aa = list(nx.adamic_adar_index(drug_proj, sample_pairs[:5000]))
        pa = list(nx.preferential_attachment(drug_proj, sample_pairs[:5000]))

        for (u1, v1, jc_score), (u2, v2, aa_score), (u3, v3, pa_score) in zip(jc, aa, pa):
            u_label = G_signal.nodes[u1].get("label", u1) if u1 in G_signal else u1
            v_label = G_signal.nodes[v1].get("label", v1) if v1 in G_signal else v1
            link_rows.append({
                "drug1": u_label, "drug2": v_label,
                "jaccard": round(jc_score, 6),
                "adamic_adar": round(aa_score, 4),
                "pref_attachment": int(pa_score),
            })
    except Exception as e:
        out.write(f"  Link prediction error: {e}\n")

    link_df = pd.DataFrame(link_rows)
    if len(link_df) > 0:
        link_df = link_df.sort_values("jaccard", ascending=False)
        link_df.head(200).to_csv(REVISION_DIR / "revision_link_prediction.csv",
                                  index=False)
        out.write(f"  Non-edge pairs scored: {len(link_df):,}\n")
        out.write(f"  Top 5 predicted links (Jaccard):\n")
        for _, r in link_df.head(5).iterrows():
            out.write(f"    {r['drug1']} — {r['drug2']}: J={r['jaccard']:.4f}\n")
    else:
        pd.DataFrame().to_csv(REVISION_DIR / "revision_link_prediction.csv",
                             index=False)
    out.write(f"  Saved: revision_link_prediction.csv\n")

    # ── E3: Network vs DPA comparison ────────────────────────────────────
    out.write("\n  E3: Network vs DPA comparison\n")
    out.write("  " + "─" * 60 + "\n")

    # Find pairs NOT in consensus but with high network prominence
    non_consensus = disp_df[disp_df["n_methods_signal"] < 4].copy()

    # High edge weight in bipartite graph = high co-report count
    # These are the non-consensus pairs with highest observed counts
    prominent = non_consensus.nlargest(100, "a")
    prominent_save = prominent[["active_ingredient", "reaction", "a",
                                 "expected", "prr", "ror", "ebgm", "ic",
                                 "n_methods_signal"]].copy()
    prominent_save.to_csv(REVISION_DIR / "revision_network_vs_dpa.csv",
                          index=False)
    out.write(f"  Non-consensus pairs with highest report counts: 100 saved\n")
    out.write(f"  Saved: revision_network_vs_dpa.csv\n")

    # ── E4: Temporal network analysis ────────────────────────────────────
    out.write("\n  E4: Temporal network analysis\n")
    out.write("  " + "─" * 60 + "\n")

    if triples_df is not None and len(triples_df) > 0:
        time_points = [2000, 2005, 2010, 2015, 2020, 2025]
        temp_rows = []

        for cutoff in time_points:
            # Filter triples to reports up to cutoff year
            subset = triples_df[triples_df["report_year"] <= cutoff]
            if len(subset) == 0:
                temp_rows.append({
                    "cutoff_year": cutoff, "n_cases": 0, "n_pairs": 0,
                    "n_drug_nodes": 0, "n_reaction_nodes": 0,
                    "n_edges": 0, "density": 0, "n_communities": 0,
                })
                continue

            # Count pairs with >= 3 reports
            pair_counts = subset.groupby(
                ["active_ingredient", "reaction"]).size().reset_index(name="n")
            pairs_ge3 = pair_counts[pair_counts["n"] >= 3]
            n_pairs = len(pairs_ge3)

            if n_pairs < 10:
                temp_rows.append({
                    "cutoff_year": cutoff,
                    "n_cases": subset["case_number"].nunique(),
                    "n_pairs": n_pairs,
                    "n_drug_nodes": pairs_ge3["active_ingredient"].nunique() if n_pairs > 0 else 0,
                    "n_reaction_nodes": pairs_ge3["reaction"].nunique() if n_pairs > 0 else 0,
                    "n_edges": n_pairs, "density": 0, "n_communities": 0,
                })
                continue

            # Build small bipartite network
            G_t = nx.Graph()
            for d in pairs_ge3["active_ingredient"].unique():
                G_t.add_node(f"D:{d}", bipartite=0)
            for r in pairs_ge3["reaction"].unique():
                G_t.add_node(f"R:{r}", bipartite=1)
            for _, row in pairs_ge3.iterrows():
                G_t.add_edge(f"D:{row['active_ingredient']}",
                             f"R:{row['reaction']}",
                             weight=row["n"])

            n_drugs = len([n for n, d in G_t.nodes(data=True)
                          if d.get("bipartite") == 0])
            n_rxns = len([n for n, d in G_t.nodes(data=True)
                         if d.get("bipartite") == 1])

            # Community detection on drug projection
            drug_nodes_t = {n for n, d in G_t.nodes(data=True)
                           if d.get("bipartite") == 0}
            n_comms = 0
            modularity = 0.0
            if len(drug_nodes_t) > 2:
                try:
                    proj_t = bipartite.weighted_projected_graph(G_t, drug_nodes_t)
                    if proj_t.number_of_edges() > 0:
                        from networkx.algorithms import community as nx_comm
                        comms = nx_comm.louvain_communities(
                            proj_t, weight="weight", resolution=1.0, seed=42)
                        n_comms = len(comms)
                        modularity = nx_comm.modularity(
                            proj_t, comms, weight="weight")
                except Exception:
                    pass

            temp_rows.append({
                "cutoff_year": cutoff,
                "n_cases": int(subset["case_number"].nunique()),
                "n_pairs": n_pairs,
                "n_drug_nodes": n_drugs,
                "n_reaction_nodes": n_rxns,
                "n_edges": n_pairs,
                "density": round(nx.density(G_t), 6),
                "n_communities": n_comms,
                "modularity": round(modularity, 4) if modularity else 0,
            })

            out.write(f"  Year ≤{cutoff}: {n_pairs:,} pairs, "
                      f"{n_drugs} drugs, {n_rxns} reactions, "
                      f"{n_comms} communities\n")

        temp_df = pd.DataFrame(temp_rows)
        temp_df.to_csv(REVISION_DIR / "revision_temporal_network.csv",
                       index=False)
        out.write(f"  Saved: revision_temporal_network.csv\n")
    else:
        out.write("  Skipping temporal network (no triples data)\n")
        pd.DataFrame().to_csv(REVISION_DIR / "revision_temporal_network.csv",
                             index=False)

    return cross_df if len(cross_edges) > 0 else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    print("=" * 70)
    print("  TGA DAEN — Manuscript Revision Analyses")
    print("  Sections A–E (25 analyses)")
    print("=" * 70)

    REVISION_DIR.mkdir(parents=True, exist_ok=True)

    report = io.StringIO()

    class TeeWriter:
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

    # ── Section A: MGPS Validation ────────────────────────────────────────
    section_a_mgps_validation(disp_df, out)

    # ── Section B: Reference Set Enhancements ─────────────────────────────
    section_b_reference_set(disp_df, out)

    # ── Load case-level data for feature engineering ──────────────────────
    out.write("\nLoading case-level data for ML analyses ...\n")
    cases = pd.read_csv(PROCESSED_DIR / "daen_cases.csv", low_memory=False)
    drugs = pd.read_csv(PROCESSED_DIR / "daen_case_drugs.csv", low_memory=False)
    reactions = pd.read_csv(PROCESSED_DIR / "daen_case_reactions.csv")
    out.write(f"  Cases: {len(cases):,}  Drugs: {len(drugs):,}  "
              f"Reactions: {len(reactions):,}\n")

    disp_featured, triples = engineer_features(
        disp_df.copy(), cases, drugs, reactions)

    # ── Section C: ML Enhancements ────────────────────────────────────────
    section_c_ml_enhancements(disp_featured, out)

    # Free large objects
    del cases, drugs, reactions, disp_featured

    # ── Section D: Consensus and Method Analysis ──────────────────────────
    section_d_consensus(disp_df, out)

    # ── Section E: Network Enhancements ───────────────────────────────────
    section_e_network(disp_df, triples, out)

    # ── Save report ───────────────────────────────────────────────────────
    report_path = REVISION_DIR / "revision_analysis_report.txt"
    report_path.write_text(report.getvalue())

    elapsed = time.time() - t0
    out.write(f"\n{'=' * 70}\n")
    out.write(f"  ALL REVISION ANALYSES COMPLETE  ({elapsed:.0f}s)\n")
    out.write(f"{'=' * 70}\n")

    # List outputs
    out.write(f"\n  Output files in {REVISION_DIR.name}/:\n")
    for p in sorted(REVISION_DIR.glob("*.csv")):
        size_kb = p.stat().st_size / 1024
        out.write(f"    {p.name:50s} {size_kb:>8.1f} KB\n")


if __name__ == "__main__":
    main()
