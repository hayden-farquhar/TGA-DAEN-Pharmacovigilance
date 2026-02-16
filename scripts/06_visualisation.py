"""
Script 06: Visualisation — Publication-Ready Figures

Generates all figures and formatted tables for the manuscript:

  1. Temporal distribution of DAEN reports by year
  2. Volcano plot (IC vs reporting volume, coloured by signal status)
  3. Method comparison scatter plots (PRR vs ROR, EBGM vs IC)
  4. Signal heatmap (top drugs × top reactions, EBGM scores)
  5. ML comparison bar chart (AUC-ROC across all methods)
  6. Feature importance bar chart
  7. Top 50 consensus signals table (formatted CSV)

Outputs in outputs/figures/ (PNG + PDF) and outputs/tables/.

Usage:
    python scripts/06_visualisation.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
TABLE_DIR = PROJECT_DIR / "outputs" / "tables"
FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size": 10,
    "font.family": "sans-serif",
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

SIGNAL_COLORS = {
    0: "#CCCCCC",
    1: "#90CAF9",
    2: "#42A5F5",
    3: "#1565C0",
    4: "#B71C1C",
}


def clean_name(name):
    """Strip leading bullet characters and whitespace from DAEN names."""
    if pd.isna(name):
        return name
    return str(name).lstrip("•·● ").strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1: TEMPORAL DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════


def fig_temporal(cases_path):
    """Bar chart of DAEN reports by year."""
    cases = pd.read_csv(cases_path, low_memory=False)
    dates = pd.to_datetime(cases["report_date"], errors="coerce", format="mixed", dayfirst=True)
    years = dates.dt.year.dropna().astype(int)
    year_counts = years.value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(12, 4.5))

    colors = ["#B71C1C" if y == 2021 else "#1565C0" for y in year_counts.index]
    ax.bar(year_counts.index, year_counts.values, color=colors, edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Year of Report")
    ax.set_ylabel("Number of Reports")
    ax.set_title("Temporal Distribution of DAEN Adverse Event Reports (1971–2026)")

    # Annotate 2021 spike
    if 2021 in year_counts.index:
        y2021 = year_counts[2021]
        ax.annotate(
            f"COVID-19 vaccine\nrollout ({y2021:,})",
            xy=(2021, y2021), xytext=(2013, y2021 * 0.85),
            fontsize=8, ha="center",
            arrowprops=dict(arrowstyle="->", color="#B71C1C", lw=1.2),
            color="#B71C1C", fontweight="bold",
        )

    ax.set_xlim(1970, 2027)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    save(fig, "fig1_temporal_distribution")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2: VOLCANO PLOT
# ═══════════════════════════════════════════════════════════════════════════════


def fig_volcano(disp_df):
    """Volcano plot: IC (effect size) vs log10(observed count) coloured by signal status."""
    df = disp_df.copy()
    df["log10_a"] = np.log10(df["a"].clip(lower=1))

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot non-signals first, then signals on top
    for n_methods in [0, 1, 2, 3, 4]:
        subset = df[df["n_methods_signal"] == n_methods]
        label = f"{n_methods} methods" if n_methods < 4 else "All 4 methods"
        ax.scatter(
            subset["ic"], subset["log10_a"],
            c=SIGNAL_COLORS[n_methods], s=3, alpha=0.4, label=label,
            edgecolors="none", rasterized=True,
        )

    # Threshold lines
    ax.axvline(x=0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(y=np.log10(3), color="grey", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("Information Component (IC, log₂ scale)")
    ax.set_ylabel("log₁₀(Observed Count)")
    ax.set_title("Volcano Plot: Disproportionality vs Reporting Volume")
    ax.legend(loc="upper left", fontsize=8, markerscale=3, framealpha=0.8)

    save(fig, "fig2_volcano_plot")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3: METHOD COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════


def fig_method_comparison(disp_df):
    """2×2 scatter plots comparing DPA methods pairwise."""
    df = disp_df.copy()
    is_signal = df["n_methods_signal"] == 4

    pairs = [
        ("prr", "ror", "PRR", "ROR"),
        ("ebgm", "ic", "EBGM", "IC"),
        ("prr", "ebgm", "PRR", "EBGM"),
        ("ror", "ic", "ROR", "IC"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for ax, (x_col, y_col, x_label, y_label) in zip(axes.flat, pairs):
        # Clip extreme values for readability
        x = np.log2(df[x_col].clip(lower=0.01))
        y_vals = df[y_col]
        if y_col in ("prr", "ror", "ebgm"):
            y = np.log2(y_vals.clip(lower=0.01))
            y_lab = f"log₂({y_label})"
        else:
            y = y_vals
            y_lab = y_label

        x_lab = f"log₂({x_label})" if x_col in ("prr", "ror", "ebgm") else x_label

        # Non-signals
        ax.scatter(x[~is_signal], y[~is_signal], c="#CCCCCC", s=1, alpha=0.2,
                   edgecolors="none", rasterized=True)
        # Signals
        ax.scatter(x[is_signal], y[is_signal], c="#B71C1C", s=2, alpha=0.3,
                   edgecolors="none", rasterized=True)

        # Correlation
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() > 10:
            r = np.corrcoef(x[valid], y[valid])[0, 1]
            ax.text(0.05, 0.95, f"r = {r:.3f}", transform=ax.transAxes,
                    fontsize=9, va="top", fontweight="bold")

        ax.set_xlabel(x_lab)
        ax.set_ylabel(y_lab)

    fig.suptitle("Pairwise Comparison of Disproportionality Methods", fontsize=13, y=1.01)
    plt.tight_layout()
    save(fig, "fig3_method_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4: SIGNAL HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════


def fig_heatmap(disp_df, top_n=25):
    """Heatmap of log₂(EBGM) for top drugs × top reactions."""
    df = disp_df[disp_df["n_methods_signal"] == 4].copy()
    df["drug_clean"] = df["active_ingredient"].apply(clean_name)
    df["rxn_clean"] = df["reaction"].apply(clean_name)

    # Top drugs by number of consensus signals
    top_drugs = df["drug_clean"].value_counts().head(top_n).index.tolist()
    # Top reactions by number of consensus signals
    top_rxns = df["rxn_clean"].value_counts().head(top_n).index.tolist()

    # Build matrix
    sub = df[df["drug_clean"].isin(top_drugs) & df["rxn_clean"].isin(top_rxns)]
    matrix = sub.pivot_table(index="drug_clean", columns="rxn_clean",
                             values="ebgm", aggfunc="max")

    # Reorder by total signal strength
    drug_order = matrix.sum(axis=1).sort_values(ascending=False).index
    rxn_order = matrix.sum(axis=0).sort_values(ascending=False).index
    matrix = matrix.loc[drug_order, rxn_order]

    # Log transform for better colour scale
    matrix_log = np.log2(matrix.clip(lower=0.5))

    # Truncate labels
    matrix_log.index = [n[:30] for n in matrix_log.index]
    matrix_log.columns = [n[:22] for n in matrix_log.columns]

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        matrix_log, ax=ax, cmap="YlOrRd", linewidths=0.3, linecolor="white",
        cbar_kws={"label": "log₂(EBGM)", "shrink": 0.6},
        mask=matrix_log.isna(),
    )
    ax.set_xlabel("MedDRA Reaction Term")
    ax.set_ylabel("Active Ingredient")
    ax.set_title(f"Signal Heatmap: Top {top_n} Drugs × Top {top_n} Reactions (Consensus Signals)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    save(fig, "fig4_signal_heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5: ML COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════


def fig_ml_comparison(eval_path):
    """Bar chart comparing AUC-ROC across all methods."""
    results = pd.read_csv(eval_path)
    results = results.sort_values("auc_roc", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = []
    for model in results["model"]:
        if "XGBoost" in model:
            colors.append("#1565C0")
        elif "Random Forest" in model:
            colors.append("#42A5F5")
        else:
            colors.append("#FF8A65")

    bars = ax.barh(range(len(results)), results["auc_roc"], color=colors,
                   edgecolor="white", height=0.7)

    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(results["model"], fontsize=8)
    ax.set_xlabel("AUC-ROC")
    ax.set_title("Signal Detection Performance: ML vs Traditional Disproportionality")
    ax.set_xlim(0.85, 1.01)

    # Value labels
    for bar, val in zip(bars, results["auc_roc"]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1565C0", label="XGBoost"),
        Patch(facecolor="#42A5F5", label="Random Forest"),
        Patch(facecolor="#FF8A65", label="Single DPA Method"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    save(fig, "fig5_ml_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════


def fig_feature_importance(imp_path):
    """Horizontal bar chart of XGBoost feature importances."""
    imp = pd.read_csv(imp_path)
    imp = imp[imp["importance"] > 0].sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 5))

    colors = []
    dpa_features = {"prr", "prr_lower95", "prr_upper95", "prr_chi2",
                    "ror", "ror_lower95", "ror_upper95", "ebgm", "eb05",
                    "ic", "ic025", "ic975"}
    for feat in imp["feature"]:
        colors.append("#FF8A65" if feat in dpa_features else "#1565C0")

    ax.barh(range(len(imp)), imp["importance"], color=colors,
            edgecolor="white", height=0.7)
    ax.set_yticks(range(len(imp)))
    ax.set_yticklabels(imp["feature"], fontsize=8)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("XGBoost Feature Importance (All Features Model)")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FF8A65", label="DPA features"),
        Patch(facecolor="#1565C0", label="Non-DPA features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    save(fig, "fig6_feature_importance")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7: METHOD AGREEMENT VENN-STYLE
# ═══════════════════════════════════════════════════════════════════════════════


def fig_method_agreement(disp_df):
    """Stacked bar chart showing the distribution of signal agreement levels."""
    counts = disp_df["n_methods_signal"].value_counts().sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: bar chart of agreement levels
    bars = ax1.bar(counts.index, counts.values,
                   color=[SIGNAL_COLORS.get(i, "#999") for i in counts.index],
                   edgecolor="white")
    ax1.set_xlabel("Number of Methods Flagging as Signal")
    ax1.set_ylabel("Number of Drug–AE Pairs")
    ax1.set_title("Signal Agreement Across Methods")
    ax1.set_xticks(range(5))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for bar, val in zip(bars, counts.values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                 f"{val:,}", ha="center", fontsize=8)

    # Right: method-level signal counts
    methods = {
        "PRR\n(PRR≥2, χ²≥4)": disp_df["signal_prr"].sum(),
        "ROR\n(lower CI>1)": disp_df["signal_ror"].sum(),
        "BCPNN\n(IC025>0)": disp_df["signal_bcpnn"].sum(),
        "EBGM\n(EB05≥2)": disp_df["signal_ebgm"].sum(),
        "All 4\nmethods": (disp_df["n_methods_signal"] == 4).sum(),
    }

    colors2 = ["#42A5F5", "#42A5F5", "#42A5F5", "#42A5F5", "#B71C1C"]
    ax2.bar(methods.keys(), methods.values(), color=colors2, edgecolor="white")
    ax2.set_ylabel("Number of Signals Detected")
    ax2.set_title("Signals by Method")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for i, (_, val) in enumerate(methods.items()):
        ax2.text(i, val + 500, f"{val:,}", ha="center", fontsize=8)

    plt.tight_layout()
    save(fig, "fig7_method_agreement")


# ═══════════════════════════════════════════════════════════════════════════════
#  SUPPLEMENTARY TABLE: TOP CONSENSUS SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════


def table_top_signals(disp_df, n=50):
    """Save a formatted CSV of the top N consensus signals."""
    consensus = disp_df[disp_df["n_methods_signal"] == 4].sort_values("ebgm", ascending=False)

    table = consensus.head(n)[
        ["active_ingredient", "reaction", "a", "expected",
         "prr", "prr_lower95", "prr_upper95",
         "ror", "ror_lower95", "ror_upper95",
         "ebgm", "eb05", "ic", "ic025"]
    ].copy()

    table["active_ingredient"] = table["active_ingredient"].apply(clean_name)
    table["reaction"] = table["reaction"].apply(clean_name)

    table = table.rename(columns={
        "active_ingredient": "Active Ingredient",
        "reaction": "MedDRA Preferred Term",
        "a": "N (observed)",
        "expected": "E (expected)",
        "prr": "PRR",
        "prr_lower95": "PRR Lower 95% CI",
        "prr_upper95": "PRR Upper 95% CI",
        "ror": "ROR",
        "ror_lower95": "ROR Lower 95% CI",
        "ror_upper95": "ROR Upper 95% CI",
        "ebgm": "EBGM",
        "eb05": "EB05",
        "ic": "IC",
        "ic025": "IC025",
    })

    path = TABLE_DIR / "top_50_consensus_signals.csv"
    table.to_csv(path, index=False, float_format="%.2f")
    print(f"  {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════


def save(fig, name):
    """Save figure as both PNG and PDF."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        path = FIGURE_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}.png / .pdf")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  TGA DAEN Visualisation — Publication-Ready Figures")
    print("=" * 70)

    # Load data
    print("\nLoading data ...")
    disp_df = pd.read_csv(TABLE_DIR / "disproportionality_full.csv")
    print(f"  Disproportionality pairs: {len(disp_df):,}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Temporal distribution ──────────────────────────────────
    print("\nGenerating figures ...")
    print("\n  Figure 1: Temporal distribution")
    fig_temporal(PROCESSED_DIR / "daen_cases.csv")

    # ── Figure 2: Volcano plot ───────────────────────────────────────────
    print("  Figure 2: Volcano plot")
    fig_volcano(disp_df)

    # ── Figure 3: Method comparison ──────────────────────────────────────
    print("  Figure 3: Method comparison scatter plots")
    fig_method_comparison(disp_df)

    # ── Figure 4: Signal heatmap ─────────────────────────────────────────
    print("  Figure 4: Signal heatmap")
    fig_heatmap(disp_df, top_n=25)

    # ── Figure 5: ML comparison ──────────────────────────────────────────
    eval_path = TABLE_DIR / "ml_evaluation_results.csv"
    if eval_path.exists():
        print("  Figure 5: ML comparison")
        fig_ml_comparison(eval_path)
    else:
        print("  Figure 5: SKIPPED (ml_evaluation_results.csv not found)")

    # ── Figure 6: Feature importance ─────────────────────────────────────
    imp_path = TABLE_DIR / "ml_feature_importance.csv"
    if imp_path.exists():
        print("  Figure 6: Feature importance")
        fig_feature_importance(imp_path)
    else:
        print("  Figure 6: SKIPPED (ml_feature_importance.csv not found)")

    # ── Figure 7: Method agreement ───────────────────────────────────────
    print("  Figure 7: Method agreement")
    fig_method_agreement(disp_df)

    # ── Supplementary table ──────────────────────────────────────────────
    print("\nGenerating supplementary tables ...")
    table_top_signals(disp_df, n=50)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    figs = list(FIGURE_DIR.glob("fig*.*"))
    print(f"  Generated {len(figs)} files in {FIGURE_DIR}/")
    total_mb = sum(f.stat().st_size for f in figs) / (1024 * 1024)
    print(f"  Total size: {total_mb:.1f} MB")
    print(f"{'=' * 70}")
    print(f"\n  All analysis complete. Ready for manuscript preparation.\n")


if __name__ == "__main__":
    main()
