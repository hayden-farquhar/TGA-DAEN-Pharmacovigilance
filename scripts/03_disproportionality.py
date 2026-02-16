"""
Script 03: Disproportionality Analysis — PRR, ROR, EBGM, BCPNN

Computes four standard pharmacovigilance signal detection metrics for all
drug–adverse event pairs with ≥3 reports (suspected drugs only), aggregated
by active ingredient.

Methods:
  1. PRR  — Proportional Reporting Ratio (Evans et al. 2001)
  2. ROR  — Reporting Odds Ratio
  3. EBGM — Empirical Bayesian Geometric Mean via MGPS (DuMouchel 1999)
  4. IC   — Information Component / BCPNN (Bate et al. 1998)

Signal thresholds:
  PRR:  PRR ≥ 2  AND  χ² ≥ 4  AND  N ≥ 3
  ROR:  lower 95% CI > 1
  EBGM: EB05 ≥ 2
  IC:   IC025 > 0

Outputs (in outputs/tables/):
  - disproportionality_full.csv     All pairs with all metrics
  - signals_consensus.csv           Pairs flagged by all 4 methods
  - signals_any_method.csv          Pairs flagged by ≥1 method

Usage:
    python scripts/03_disproportionality.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import digamma
from scipy.optimize import minimize
from pathlib import Path
import sys
import time

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "tables"

# ── Configuration ────────────────────────────────────────────────────────────

MIN_REPORTS = 3  # minimum case count per pair

# Signal thresholds
PRR_THRESHOLD = 2.0
PRR_CHI2_THRESHOLD = 4.0
ROR_LOWER_CI_THRESHOLD = 1.0
EBGM_EB05_THRESHOLD = 2.0
IC025_THRESHOLD = 0.0

# Active ingredients to exclude (uninformative)
EXCLUDE_INGREDIENTS = {
    "trade name not specified",
    "product not coded",
    "not specified",
    "unknown",
    "",
}


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING AND CONTINGENCY TABLE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════


def load_data():
    """Load cleaned DAEN data files."""
    cases = pd.read_csv(PROCESSED_DIR / "daen_cases.csv")
    drugs = pd.read_csv(PROCESSED_DIR / "daen_case_drugs.csv", low_memory=False)
    reactions = pd.read_csv(PROCESSED_DIR / "daen_case_reactions.csv")
    return cases, drugs, reactions


def compute_contingency_tables(cases, drugs, reactions):
    """
    Compute 2×2 contingency table values for all drug–AE pairs.

    Aggregates by active_ingredient so that different trade names
    of the same drug are combined. Uses only suspected drugs.

    The 2×2 table for drug D and adverse event E:

                   | E present | E absent |
        D present  |     a     |    b     |  n_drug
        D absent   |     c     |    d     |
                   | n_reaction|          |  N

    Returns DataFrame with columns: active_ingredient, reaction, a, b, c, d,
    n_drug, n_reaction, N, expected.
    """
    N = len(cases)

    # ── Prepare suspected drug records ───────────────────────────────────
    susp = drugs[drugs["suspected"] == "suspected"][["case_number", "active_ingredient"]].copy()
    susp = susp.dropna(subset=["active_ingredient"])
    susp["active_ingredient"] = susp["active_ingredient"].str.strip().str.lower()
    susp = susp[~susp["active_ingredient"].isin(EXCLUDE_INGREDIENTS)]
    susp = susp.drop_duplicates()  # unique (case, ingredient) pairs

    # ── Drug marginals ───────────────────────────────────────────────────
    drug_counts = susp.groupby("active_ingredient")["case_number"].nunique()
    print(f"  Active ingredients (after exclusions): {len(drug_counts):,}")

    # ── Reaction marginals ───────────────────────────────────────────────
    rxn = reactions[["case_number", "reaction"]].drop_duplicates()
    rxn_counts = rxn.groupby("reaction")["case_number"].nunique()
    print(f"  Unique reactions: {len(rxn_counts):,}")

    # ── Pair counts via merge ────────────────────────────────────────────
    # Each row in the merge = one (case, ingredient, reaction) triple
    print("  Computing pair counts (this may take a moment) ...")
    pairs = susp.merge(rxn, on="case_number", how="inner")
    pair_counts = pairs.groupby(["active_ingredient", "reaction"]).size().reset_index(name="a")

    # Filter to minimum reports
    pair_counts = pair_counts[pair_counts["a"] >= MIN_REPORTS].copy()
    print(f"  Pairs with >= {MIN_REPORTS} reports: {len(pair_counts):,}")

    # ── Join marginals and compute 2×2 table ─────────────────────────────
    pair_counts["n_drug"] = pair_counts["active_ingredient"].map(drug_counts).astype(int)
    pair_counts["n_reaction"] = pair_counts["reaction"].map(rxn_counts).astype(int)
    pair_counts["N"] = N

    pair_counts["b"] = pair_counts["n_drug"] - pair_counts["a"]
    pair_counts["c"] = pair_counts["n_reaction"] - pair_counts["a"]
    pair_counts["d"] = N - pair_counts["a"] - pair_counts["b"] - pair_counts["c"]

    # Expected count under independence: E = n_drug * n_reaction / N
    pair_counts["expected"] = (
        pair_counts["n_drug"].astype(float) * pair_counts["n_reaction"].astype(float) / N
    )

    return pair_counts.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  FREQUENTIST METHODS: PRR AND ROR
# ═══════════════════════════════════════════════════════════════════════════════


def compute_prr(df):
    """
    Proportional Reporting Ratio with 95% CI and chi-squared.

    PRR = (a/(a+b)) / (c/(c+d))
    Signal: PRR >= 2, chi² >= 4, a >= 3  (Evans et al. 2001)
    """
    a = df["a"].values.astype(float)
    b = df["b"].values.astype(float)
    c = df["c"].values.astype(float)
    d = df["d"].values.astype(float)

    # Continuity correction for zero cells
    ac, bc, cc, dc = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    prr = (ac / (ac + bc)) / (cc / (cc + dc))
    ln_prr = np.log(prr)
    se = np.sqrt(1 / ac - 1 / (ac + bc) + 1 / cc - 1 / (cc + dc))

    df["prr"] = prr
    df["prr_lower95"] = np.exp(ln_prr - 1.96 * se)
    df["prr_upper95"] = np.exp(ln_prr + 1.96 * se)

    # Yates-corrected chi-squared
    N = a + b + c + d
    df["prr_chi2"] = (N * (np.abs(a * d - b * c) - N / 2) ** 2) / (
        (a + b) * (c + d) * (a + c) * (b + d)
    )

    return df


def compute_ror(df):
    """
    Reporting Odds Ratio with 95% CI.

    ROR = (a*d) / (b*c)
    Signal: lower 95% CI > 1
    """
    a = df["a"].values.astype(float)
    b = df["b"].values.astype(float)
    c = df["c"].values.astype(float)
    d = df["d"].values.astype(float)

    ac, bc, cc, dc = a + 0.5, b + 0.5, c + 0.5, d + 0.5

    ror = (ac * dc) / (bc * cc)
    ln_ror = np.log(ror)
    se = np.sqrt(1 / ac + 1 / bc + 1 / cc + 1 / dc)

    df["ror"] = ror
    df["ror_lower95"] = np.exp(ln_ror - 1.96 * se)
    df["ror_upper95"] = np.exp(ln_ror + 1.96 * se)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  EMPIRICAL BAYESIAN GEOMETRIC MEAN (MGPS / DuMouchel 1999)
# ═══════════════════════════════════════════════════════════════════════════════


def fit_mgps_prior(n_obs, E, max_iter=5000):
    """
    Fit the two-component gamma mixture prior via MLE.

    The marginal distribution of observed count n given expected count E
    under a Gamma(a,b) prior is Negative Binomial:

        P(n|E) = NB(n; a, b/(b+E))

    The mixture model has five parameters:
        alpha  — mixing weight for component 1
        a1, b1 — shape/rate for component 1
        a2, b2 — shape/rate for component 2

    Parameters are estimated by maximising the log-likelihood of the
    observed (n, E) pairs using Nelder-Mead in transformed space.

    Returns (alpha, a1, b1, a2, b2, OptimizeResult).
    """

    def neg_log_lik(params):
        # Transform from unconstrained to constrained space
        alpha = 1.0 / (1.0 + np.exp(-params[0]))
        a1 = np.exp(params[1])
        b1 = np.exp(params[2])
        a2 = np.exp(params[3])
        b2 = np.exp(params[4])

        p1 = b1 / (b1 + E)
        p2 = b2 / (b2 + E)

        log_nb1 = stats.nbinom.logpmf(n_obs, a1, p1)
        log_nb2 = stats.nbinom.logpmf(n_obs, a2, p2)

        log_mix = np.logaddexp(np.log(alpha) + log_nb1, np.log(1 - alpha) + log_nb2)

        nll = -np.sum(log_mix)
        if not np.isfinite(nll):
            return 1e15
        return nll

    # Initial guesses (transformed): alpha=0.2, a1=0.2, b1=0.1, a2=2, b2=2
    x0 = np.array([
        np.log(0.2 / 0.8),  # logit(alpha)
        np.log(0.2),         # log(a1)
        np.log(0.1),         # log(b1)
        np.log(2.0),         # log(a2)
        np.log(2.0),         # log(b2)
    ])

    result = minimize(
        neg_log_lik, x0, method="Nelder-Mead",
        options={"maxiter": max_iter, "xatol": 1e-8, "fatol": 1e-8},
    )

    alpha = 1.0 / (1.0 + np.exp(-result.x[0]))
    a1 = np.exp(result.x[1])
    b1 = np.exp(result.x[2])
    a2 = np.exp(result.x[3])
    b2 = np.exp(result.x[4])

    return alpha, a1, b1, a2, b2, result


def compute_ebgm(df, alpha, a1, b1, a2, b2):
    """
    Compute EBGM and EB05 for all drug–AE pairs.

    The posterior for each pair (given observed n and expected E) is:
        Q * Gamma(a1+n, b1+E) + (1-Q) * Gamma(a2+n, b2+E)

    where Q is the posterior mixing weight for component 1.

    EBGM = geometric mean of posterior = exp(E[log λ])
    EB05  = 5th percentile of posterior mixture
    """
    n = df["a"].values.astype(float)
    E = df["expected"].values

    # Posterior mixing weight Q
    p1 = b1 / (b1 + E)
    p2 = b2 / (b2 + E)

    log_nb1 = stats.nbinom.logpmf(n.astype(int), a1, p1)
    log_nb2 = stats.nbinom.logpmf(n.astype(int), a2, p2)

    log_num = np.log(alpha) + log_nb1
    log_den = np.logaddexp(log_num, np.log(1 - alpha) + log_nb2)
    Q = np.exp(log_num - log_den)

    # EBGM = exp(E[log λ])
    ebgm = np.exp(
        Q * (digamma(a1 + n) - np.log(b1 + E))
        + (1 - Q) * (digamma(a2 + n) - np.log(b2 + E))
    )

    # EB05 via vectorised bisection
    print("  Computing EB05 (vectorised bisection) ...")
    eb05 = _eb05_bisection(n, E, a1, b1, a2, b2, Q)

    df["ebgm"] = ebgm
    df["eb05"] = eb05
    df["ebgm_Q"] = Q  # useful for diagnostics

    return df


def _eb05_bisection(n_obs, E, a1, b1, a2, b2, Q, target=0.05, tol=1e-4, max_iter=60):
    """Compute the 5th percentile of the posterior mixture via vectorised bisection."""
    m = len(n_obs)

    # Posterior gamma parameters: Gamma(shape, rate) → scipy: gamma(shape, scale=1/rate)
    shape1 = a1 + n_obs
    scale1 = 1.0 / (b1 + E)
    shape2 = a2 + n_obs
    scale2 = 1.0 / (b2 + E)

    lower = np.zeros(m)
    upper = np.maximum(n_obs / np.maximum(E, 1e-10), 10.0) * 10.0

    for _ in range(max_iter):
        mid = (lower + upper) / 2.0
        cdf = Q * stats.gamma.cdf(mid, shape1, scale=scale1) + (1 - Q) * stats.gamma.cdf(
            mid, shape2, scale=scale2
        )
        lower = np.where(cdf < target, mid, lower)
        upper = np.where(cdf >= target, mid, upper)
        if np.max(upper - lower) < tol:
            break

    return (lower + upper) / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
#  BCPNN INFORMATION COMPONENT (Bate et al. 1998)
# ═══════════════════════════════════════════════════════════════════════════════


def compute_bcpnn(df):
    """
    Compute the Information Component (IC) with credibility intervals.

    IC = log2(P(x,y) / (P(x) * P(y)))

    With additive (Bayesian) smoothing:
        IC = log2( (a + 0.5)(N + 0.5) / ((n_drug + 0.5)(n_reaction + 0.5)) )

    Approximate variance (Norén et al. 2006):
        Var(IC) ≈ 1/ln(2)² × (1/(a + 0.5) - 1/(N + 0.5))

    Signal: IC025 > 0
    """
    a = df["a"].values.astype(float)
    n_drug = df["n_drug"].values.astype(float)
    n_rxn = df["n_reaction"].values.astype(float)
    N = df["N"].values.astype(float)

    ic = np.log2(((a + 0.5) * (N + 0.5)) / ((n_drug + 0.5) * (n_rxn + 0.5)))

    ic_var = (1.0 / np.log(2) ** 2) * (1.0 / (a + 0.5) - 1.0 / (N + 0.5))
    ic_se = np.sqrt(np.maximum(ic_var, 0))

    df["ic"] = ic
    df["ic025"] = ic - 1.96 * ic_se
    df["ic975"] = ic + 1.96 * ic_se

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════════════════


def classify_signals(df):
    """Apply signal detection criteria for each method."""
    df["signal_prr"] = (
        (df["prr"] >= PRR_THRESHOLD)
        & (df["prr_chi2"] >= PRR_CHI2_THRESHOLD)
        & (df["a"] >= MIN_REPORTS)
    )
    df["signal_ror"] = df["ror_lower95"] > ROR_LOWER_CI_THRESHOLD
    df["signal_ebgm"] = df["eb05"] >= EBGM_EB05_THRESHOLD
    df["signal_bcpnn"] = df["ic025"] > IC025_THRESHOLD

    df["n_methods_signal"] = (
        df["signal_prr"].astype(int)
        + df["signal_ror"].astype(int)
        + df["signal_ebgm"].astype(int)
        + df["signal_bcpnn"].astype(int)
    )

    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    print("=" * 70)
    print("  TGA DAEN Disproportionality Analysis")
    print("  PRR · ROR · EBGM (MGPS) · BCPNN (IC)")
    print("=" * 70)

    # ── Load ─────────────────────────────────────────────────────────────
    print("\nLoading cleaned data ...")
    cases, drugs, reactions = load_data()
    print(f"  Cases:          {len(cases):>10,}")
    print(f"  Drug records:   {len(drugs):>10,}")
    print(f"  Reaction records: {len(reactions):>10,}")

    # ── Contingency tables ───────────────────────────────────────────────
    print("\nBuilding contingency tables (aggregated by active ingredient) ...")
    df = compute_contingency_tables(cases, drugs, reactions)
    print(f"  Unique ingredients in analysis: {df['active_ingredient'].nunique():,}")
    print(f"  Unique reactions in analysis:   {df['reaction'].nunique():,}")

    # ── PRR ──────────────────────────────────────────────────────────────
    print("\nComputing PRR ...")
    df = compute_prr(df)

    # ── ROR ──────────────────────────────────────────────────────────────
    print("Computing ROR ...")
    df = compute_ror(df)

    # ── EBGM ─────────────────────────────────────────────────────────────
    print("\nFitting MGPS two-gamma mixture prior ...")
    n_obs = df["a"].values.astype(int)
    E = df["expected"].values
    alpha, a1, b1, a2, b2, opt = fit_mgps_prior(n_obs, E)

    print(f"  Mixing weight (alpha): {alpha:.4f}")
    print(f"  Component 1: Gamma(a={a1:.4f}, b={b1:.4f})  mean={a1 / b1:.3f}")
    print(f"  Component 2: Gamma(a={a2:.4f}, b={b2:.4f})  mean={a2 / b2:.3f}")
    print(f"  Converged: {opt.success}  |  Final -LL: {opt.fun:,.0f}")

    print("Computing EBGM and EB05 ...")
    df = compute_ebgm(df, alpha, a1, b1, a2, b2)

    # ── BCPNN ────────────────────────────────────────────────────────────
    print("\nComputing IC (BCPNN) ...")
    df = compute_bcpnn(df)

    # ── Classify signals ─────────────────────────────────────────────────
    print("Applying signal thresholds ...")
    df = classify_signals(df)

    # ── Summary ──────────────────────────────────────────────────────────
    n_prr = df["signal_prr"].sum()
    n_ror = df["signal_ror"].sum()
    n_ebgm = df["signal_ebgm"].sum()
    n_bcpnn = df["signal_bcpnn"].sum()
    n_all4 = (df["n_methods_signal"] == 4).sum()
    n_any = (df["n_methods_signal"] >= 1).sum()

    print(f"\n{'=' * 70}")
    print(f"  SIGNAL DETECTION SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  Total pairs analysed:             {len(df):>10,}")
    print(f"\n  PRR signals  (PRR≥2, χ²≥4, n≥3): {n_prr:>10,}")
    print(f"  ROR signals  (lower 95% CI > 1):  {n_ror:>10,}")
    print(f"  EBGM signals (EB05 ≥ 2):          {n_ebgm:>10,}")
    print(f"  BCPNN signals (IC025 > 0):         {n_bcpnn:>10,}")
    print(f"  {'─' * 40}")
    print(f"  Flagged by ALL 4 methods:          {n_all4:>10,}")
    print(f"  Flagged by ≥ 1 method:             {n_any:>10,}")

    # Method agreement matrix
    methods = ["signal_prr", "signal_ror", "signal_ebgm", "signal_bcpnn"]
    labels = ["PRR", "ROR", "EBGM", "BCPNN"]
    print(f"\n  PAIRWISE AGREEMENT (% overlap)")
    print(f"  {'':>8s}", end="")
    for l in labels:
        print(f"  {l:>7s}", end="")
    print()
    for i, (mi, li) in enumerate(zip(methods, labels)):
        print(f"  {li:>8s}", end="")
        for j, (mj, lj) in enumerate(zip(methods, labels)):
            overlap = (df[mi] & df[mj]).sum()
            total = df[mi].sum()
            pct = 100 * overlap / total if total > 0 else 0
            print(f"  {pct:>6.1f}%", end="")
        print()

    # Top consensus signals
    consensus = df[df["n_methods_signal"] == 4].sort_values("ebgm", ascending=False)
    print(f"\n  TOP 25 CONSENSUS SIGNALS (all 4 methods, ranked by EBGM)")
    print(f"  {'─' * 85}")
    header = f"  {'Drug':<35s} {'Reaction':<25s} {'n':>5s} {'E':>7s} {'PRR':>6s} {'EBGM':>6s} {'EB05':>5s} {'IC':>6s}"
    print(header)
    print(f"  {'─' * 85}")
    for _, row in consensus.head(25).iterrows():
        drug = str(row["active_ingredient"])[:34]
        rxn = str(row["reaction"])[:24]
        print(
            f"  {drug:<35s} {rxn:<25s} {row['a']:>5.0f} {row['expected']:>7.1f}"
            f" {row['prr']:>6.1f} {row['ebgm']:>6.1f} {row['eb05']:>5.1f} {row['ic']:>6.2f}"
        )

    # ── Save ─────────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("Saving results ...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    out_cols = [
        "active_ingredient", "reaction",
        "a", "expected", "n_drug", "n_reaction", "N",
        "prr", "prr_lower95", "prr_upper95", "prr_chi2",
        "ror", "ror_lower95", "ror_upper95",
        "ebgm", "eb05",
        "ic", "ic025", "ic975",
        "signal_prr", "signal_ror", "signal_ebgm", "signal_bcpnn",
        "n_methods_signal",
    ]

    full_path = OUTPUT_DIR / "disproportionality_full.csv"
    df[out_cols].to_csv(full_path, index=False, float_format="%.4f")

    consensus_path = OUTPUT_DIR / "signals_consensus.csv"
    consensus[out_cols].to_csv(consensus_path, index=False, float_format="%.4f")

    any_sig = df[df["n_methods_signal"] >= 1].sort_values(
        ["n_methods_signal", "ebgm"], ascending=[False, False]
    )
    any_path = OUTPUT_DIR / "signals_any_method.csv"
    any_sig[out_cols].to_csv(any_path, index=False, float_format="%.4f")

    for path in [full_path, consensus_path, any_path]:
        size_mb = path.stat().st_size / (1024 * 1024)
        rows = len(pd.read_csv(path, nrows=0).columns)  # just for the print
        n_rows = {"disproportionality_full.csv": len(df),
                   "signals_consensus.csv": len(consensus),
                   "signals_any_method.csv": len(any_sig)}
        print(f"  {path.name:45s}  {n_rows.get(path.name, 0):>8,} rows  {size_mb:>6.1f} MB")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  DISPROPORTIONALITY ANALYSIS COMPLETE  ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"\n  Next step: python scripts/04_ml_signal_detection.py\n")


if __name__ == "__main__":
    main()
