"""
Script 09: Revision Figures

Generates new and updated figures for manuscript revisions:

  1. SHAP beeswarm plot (global feature importance)
  2. SHAP bar plot (mean |SHAP| per feature)
  3. Prior distribution plot (empirical vs fitted vs alternative priors)
  4. Calibration plot (predicted probability vs observed frequency)
  5. Temporal network evolution (panel plot)
  6. Border-zone characterisation

All figures saved to outputs/figures/revision_*.png (300 DPI + PDF copies).

Depends on outputs from script 08.

Usage:
    python scripts/09_revision_figures.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import digamma
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
import warnings

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "tables"
REVISION_DIR = PROJECT_DIR / "outputs" / "revisions"
FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"

# ── Plotting style ───────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def save_fig(fig, name):
    """Save figure as PNG and PDF."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURE_DIR / f"{name}.png"
    pdf_path = FIGURE_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {png_path.name}, {pdf_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1: SHAP BEESWARM PLOT
# ═══════════════════════════════════════════════════════════════════════════════


def plot_shap_beeswarm():
    """SHAP beeswarm plot for XGBoost feature importance."""
    print("\n  Figure: SHAP beeswarm plot")

    shap_raw_path = REVISION_DIR / "revision_shap_raw_values.csv"
    feat_path = REVISION_DIR / "revision_shap_feature_matrix.csv"

    if not shap_raw_path.exists() or not feat_path.exists():
        print("    SHAP data not found — skipping")
        return

    shap_values = pd.read_csv(shap_raw_path).values
    X_display = pd.read_csv(feat_path)
    feature_names = X_display.columns.tolist()

    # Sort features by mean |SHAP|
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1]

    # Top 15 features
    top_n = min(15, len(feature_names))
    top_idx = order[:top_n]

    fig, ax = plt.subplots(figsize=(8, 6))

    for plot_pos, feat_idx in enumerate(reversed(top_idx)):
        shap_vals = shap_values[:, feat_idx]
        feat_vals = X_display.iloc[:, feat_idx].values

        # Normalise feature values to [0, 1] for colour
        fmin, fmax = np.nanmin(feat_vals), np.nanmax(feat_vals)
        if fmax > fmin:
            colours = (feat_vals - fmin) / (fmax - fmin)
        else:
            colours = np.full_like(feat_vals, 0.5)

        # Add jitter
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(shap_vals))

        sc = ax.scatter(shap_vals, plot_pos + jitter, c=colours,
                       cmap="coolwarm", s=8, alpha=0.6, edgecolors="none",
                       vmin=0, vmax=1)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in reversed(top_idx)])
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.axvline(x=0, color="black", linewidth=0.5, alpha=0.3)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Feature value\n(Low → High)")

    ax.set_title("SHAP Feature Importance (XGBoost)")
    fig.tight_layout()
    save_fig(fig, "revision_shap_beeswarm")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2: SHAP BAR PLOT
# ═══════════════════════════════════════════════════════════════════════════════


def plot_shap_bar():
    """Bar plot of mean |SHAP| per feature."""
    print("\n  Figure: SHAP bar plot")

    shap_path = REVISION_DIR / "revision_shap_importance.csv"
    if not shap_path.exists():
        print("    SHAP importance data not found — skipping")
        return

    shap_df = pd.read_csv(shap_path).sort_values("mean_abs_shap", ascending=True)

    # Top 15
    top = shap_df.tail(15)

    fig, ax = plt.subplots(figsize=(7, 5))
    colours = ["#2196F3" if "prr" in f or "ror" in f or "ebgm" in f or
               "eb05" in f or "ic" in f else "#FF9800"
               for f in top["feature"]]
    ax.barh(range(len(top)), top["mean_abs_shap"], color=colours, alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Mean Absolute SHAP Value per Feature")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="#2196F3", label="DPA features"),
                       Patch(facecolor="#FF9800", label="Non-DPA features")]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.tight_layout()
    save_fig(fig, "revision_shap_bar")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3: PRIOR DISTRIBUTION PLOT
# ═══════════════════════════════════════════════════════════════════════════════


def plot_prior_distribution():
    """Empirical vs fitted mixture prior density."""
    print("\n  Figure: MGPS prior distribution")

    disp_df = pd.read_csv(OUTPUT_DIR / "disproportionality_full.csv")
    n_obs = disp_df["a"].values.astype(float)
    E = disp_df["expected"].values

    # Observed/expected ratios (lambda estimates)
    ratios = n_obs / np.maximum(E, 1e-10)
    log_ratios = np.log10(np.clip(ratios, 1e-3, 1e5))

    # Fitted prior parameters
    priors = {
        "Fitted (DAEN)": (0.2444, 0.52, 0.006, 1.17, 0.41),
        "DuMouchel FAERS": (0.2, 0.25, 0.5, 1.5, 2.0),
        "Vague": (0.5, 1.0, 1.0, 1.0, 1.0),
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    # Empirical distribution
    ax.hist(log_ratios, bins=100, density=True, alpha=0.4,
            color="#9E9E9E", edgecolor="none", label="Empirical")

    # Overlay fitted densities
    x_range = np.linspace(-3, 5, 500)
    colours = {"Fitted (DAEN)": "#2196F3",
               "DuMouchel FAERS": "#FF5722",
               "Vague": "#4CAF50"}

    for name, (al, a1, b1, a2, b2) in priors.items():
        # Gamma mixture density on the log10 scale
        lam = 10 ** x_range
        density = (al * stats.gamma.pdf(lam, a1, scale=1/b1) +
                   (1 - al) * stats.gamma.pdf(lam, a2, scale=1/b2))
        # Jacobian for log10 transform: d(lam)/d(log10_lam) = lam * ln(10)
        density_log10 = density * lam * np.log(10)
        ax.plot(x_range, density_log10, linewidth=2, color=colours[name],
                label=name)

    ax.set_xlabel("log$_{10}$(Observed / Expected)")
    ax.set_ylabel("Density")
    ax.set_title("MGPS Prior Distribution: Empirical vs Fitted Gamma Mixtures")
    ax.legend(frameon=True, fontsize=9)
    ax.set_xlim(-3, 5)

    fig.tight_layout()
    save_fig(fig, "revision_prior_distribution")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 4: CALIBRATION PLOT
# ═══════════════════════════════════════════════════════════════════════════════


def plot_calibration():
    """Calibration curve for XGBoost classifier."""
    print("\n  Figure: Calibration plot")

    cal_path = REVISION_DIR / "revision_calibration_data.csv"
    if not cal_path.exists():
        print("    Calibration data not found — skipping")
        return

    cal_data = pd.read_csv(cal_path)
    y = cal_data["label"].values
    y_prob = cal_data["xgb_probability"].values

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Calibration curve
    try:
        prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=5,
                                                  strategy="uniform")
        ax1.plot(prob_pred, prob_true, "s-", color="#2196F3",
                linewidth=2, markersize=8, label="XGBoost")
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
        ax1.set_xlabel("Mean predicted probability")
        ax1.set_ylabel("Fraction of positives")
        ax1.set_title("Calibration Curve (XGBoost)")
        ax1.legend(loc="lower right")
        ax1.set_xlim(-0.05, 1.05)
        ax1.set_ylim(-0.05, 1.05)
    except Exception:
        ax1.text(0.5, 0.5, "Insufficient data\nfor calibration",
                ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Calibration Curve")

    # Distribution of predicted probabilities
    ax2.hist(y_prob[y == 1], bins=20, alpha=0.7, color="#4CAF50",
            edgecolor="white", label=f"Positive (n={int((y==1).sum())})")
    ax2.hist(y_prob[y == 0], bins=20, alpha=0.7, color="#F44336",
            edgecolor="white", label=f"Negative (n={int((y==0).sum())})")
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Predicted Probabilities")
    ax2.legend()

    fig.tight_layout()
    save_fig(fig, "revision_calibration")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 5: TEMPORAL NETWORK EVOLUTION
# ═══════════════════════════════════════════════════════════════════════════════


def plot_temporal_network():
    """Temporal network evolution panel."""
    print("\n  Figure: Temporal network evolution")

    temp_path = REVISION_DIR / "revision_temporal_network.csv"
    if not temp_path.exists():
        print("    Temporal network data not found — skipping")
        return

    temp_df = pd.read_csv(temp_path)
    if len(temp_df) == 0:
        print("    Empty temporal network data — skipping")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    years = temp_df["cutoff_year"]

    # Panel A: Nodes
    ax = axes[0, 0]
    ax.plot(years, temp_df["n_drug_nodes"], "o-", color="#2196F3",
           label="Drug nodes", linewidth=2)
    ax.plot(years, temp_df["n_reaction_nodes"], "s-", color="#FF5722",
           label="Reaction nodes", linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of nodes")
    ax.set_title("A. Network Growth (Nodes)")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Panel B: Edges (pairs)
    ax = axes[0, 1]
    ax.plot(years, temp_df["n_edges"], "o-", color="#4CAF50", linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of edges (drug–AE pairs)")
    ax.set_title("B. Network Growth (Edges)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Panel C: Communities
    ax = axes[1, 0]
    ax.bar(years, temp_df["n_communities"], color="#9C27B0", alpha=0.7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Louvain communities")
    ax.set_title("C. Community Structure Evolution")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Panel D: Density
    ax = axes[1, 1]
    ax.plot(years, temp_df["density"], "o-", color="#FF9800", linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("Network density")
    ax.set_title("D. Network Density Over Time")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("Temporal Evolution of the Drug–AE Network (Cumulative)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    save_fig(fig, "revision_temporal_network")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 6: BORDER-ZONE CHARACTERISATION
# ═══════════════════════════════════════════════════════════════════════════════


def plot_border_zone():
    """Distribution of border-zone vs consensus signals."""
    print("\n  Figure: Border-zone characterisation")

    disp_df = pd.read_csv(OUTPUT_DIR / "disproportionality_full.csv")

    consensus = disp_df[disp_df["n_methods_signal"] == 4]
    border = disp_df[(disp_df["n_methods_signal"] >= 1) &
                     (disp_df["n_methods_signal"] < 4)]
    non_signal = disp_df[disp_df["n_methods_signal"] == 0]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Report count distribution
    ax = axes[0]
    bins = np.logspace(np.log10(3), np.log10(10000), 50)
    ax.hist(consensus["a"], bins=bins, alpha=0.7, color="#4CAF50",
           label=f"Consensus (n={len(consensus):,})", density=True)
    ax.hist(border["a"], bins=bins, alpha=0.7, color="#FF9800",
           label=f"Border zone (n={len(border):,})", density=True)
    ax.hist(non_signal["a"], bins=bins, alpha=0.5, color="#9E9E9E",
           label=f"Non-signal (n={len(non_signal):,})", density=True)
    ax.set_xscale("log")
    ax.set_xlabel("Report count (log scale)")
    ax.set_ylabel("Density")
    ax.set_title("A. Report Count Distribution")
    ax.legend(fontsize=8)

    # Panel B: EBGM distribution
    ax = axes[1]
    bins_ebgm = np.logspace(np.log10(0.01), np.log10(5000), 50)
    ax.hist(consensus["ebgm"], bins=bins_ebgm, alpha=0.7, color="#4CAF50",
           label="Consensus", density=True)
    ax.hist(border["ebgm"], bins=bins_ebgm, alpha=0.7, color="#FF9800",
           label="Border zone", density=True)
    ax.set_xscale("log")
    ax.set_xlabel("EBGM (log scale)")
    ax.set_ylabel("Density")
    ax.set_title("B. EBGM Distribution")
    ax.legend(fontsize=8)

    # Panel C: Methods agreement stacked bar
    ax = axes[2]
    method_counts = disp_df["n_methods_signal"].value_counts().sort_index()
    colours = ["#9E9E9E", "#FFEB3B", "#FF9800", "#FF5722", "#4CAF50"]
    labels = ["0 methods", "1 method", "2 methods", "3 methods", "4 methods"]
    bars = ax.bar(method_counts.index, method_counts.values,
                  color=[colours[i] for i in method_counts.index],
                  edgecolor="white")
    ax.set_xlabel("Number of DPA methods signalling")
    ax.set_ylabel("Number of drug–AE pairs")
    ax.set_title("C. Signal Agreement Distribution")
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels, rotation=30, ha="right")

    # Add count labels
    for bar, val in zip(bars, method_counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
               f"{val:,}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Three-Tier Signal Classification: Consensus vs Border Zone "
                 "vs Non-Signal", fontsize=12, y=1.02)
    fig.tight_layout()
    save_fig(fig, "revision_border_zone")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 7: UPDATED FIGURE 5 WITH DELONG ANNOTATIONS
# ═══════════════════════════════════════════════════════════════════════════════


def plot_updated_ml_comparison():
    """Updated AUC-ROC bar chart with DeLong significance annotations."""
    print("\n  Figure: Updated ML comparison with DeLong")

    delong_path = REVISION_DIR / "revision_delong_tests.csv"
    eval_path = OUTPUT_DIR / "ml_evaluation_results.csv"

    if not eval_path.exists():
        print("    ML evaluation results not found — skipping")
        return

    eval_df = pd.read_csv(eval_path)

    # Simplify model names for plotting
    plot_data = eval_df.sort_values("auc_roc", ascending=True).tail(10)

    fig, ax = plt.subplots(figsize=(10, 6))

    colours = []
    for model in plot_data["model"]:
        if "single" in model.lower():
            colours.append("#9E9E9E")
        elif "xgboost" in model.lower():
            colours.append("#2196F3")
        else:
            colours.append("#FF9800")

    bars = ax.barh(range(len(plot_data)), plot_data["auc_roc"],
                   color=colours, alpha=0.85, edgecolor="white")

    ax.set_yticks(range(len(plot_data)))
    ax.set_yticklabels(plot_data["model"], fontsize=8)
    ax.set_xlabel("AUC-ROC")
    ax.set_title("Model Comparison: AUC-ROC on Reference Set")
    ax.set_xlim(0.8, 1.02)

    # Add value labels
    for bar, auc in zip(bars, plot_data["auc_roc"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
               f"{auc:.3f}", ha="left", va="center", fontsize=8)

    # Add DeLong annotation if available
    if delong_path.exists():
        delong_df = pd.read_csv(delong_path)
        # Find XGB vs EBGM comparison
        xgb_ebgm = delong_df[delong_df["comparison"].str.contains("EBGM standalone",
                                                                    na=False)]
        if len(xgb_ebgm) > 0:
            p = xgb_ebgm.iloc[0]["p_value"]
            sig = "n.s." if p >= 0.05 else f"p={p:.3f}"
            ax.text(0.98, 0.02, f"DeLong XGB vs EBGM: {sig}",
                   transform=ax.transAxes, fontsize=8, ha="right",
                   style="italic", color="#666666")

    fig.tight_layout()
    save_fig(fig, "revision_ml_comparison")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    print("=" * 70)
    print("  TGA DAEN — Revision Figures")
    print("=" * 70)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_shap_beeswarm()
    plot_shap_bar()
    plot_prior_distribution()
    plot_calibration()
    plot_temporal_network()
    plot_border_zone()
    plot_updated_ml_comparison()

    print(f"\n{'=' * 70}")
    print("  ALL REVISION FIGURES COMPLETE")
    print(f"{'=' * 70}")

    # List output files
    revision_figs = sorted(FIGURE_DIR.glob("revision_*"))
    print(f"\n  Revision figures generated:")
    for p in revision_figs:
        size_kb = p.stat().st_size / 1024
        print(f"    {p.name:50s} {size_kb:>8.1f} KB")


if __name__ == "__main__":
    main()
