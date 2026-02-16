"""
Script 05: Network Analysis — Bipartite Drug–AE Network

Constructs a bipartite drug–adverse event co-reporting network from the
disproportionality results, then:

  1. Computes network-level statistics (density, components, degree distribution)
  2. Computes node-level centrality metrics (degree, betweenness, eigenvector)
  3. Projects onto the drug side to build a drug–drug similarity network
  4. Runs Louvain community detection to identify drug clusters
  5. Generates summary tables and a filtered network visualisation

Outputs (in outputs/):
  tables/network_drug_centrality.csv     Drug centrality rankings
  tables/network_reaction_centrality.csv Reaction centrality rankings
  tables/network_communities.csv         Community assignments
  tables/network_summary.txt             Network statistics report
  figures/degree_distribution.png        Degree distribution plot
  figures/network_top_drugs.png          Filtered network visualisation

Usage:
    python scripts/05_network_analysis.py
"""

import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms import bipartite, community as nx_comm
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "tables"
FIGURE_DIR = PROJECT_DIR / "outputs" / "figures"


# ═══════════════════════════════════════════════════════════════════════════════
#  NETWORK CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════


def build_bipartite_network(disp_df, signal_only=False):
    """
    Build a bipartite graph with drug and reaction nodes.

    Nodes have attribute 'bipartite': 0 = drug, 1 = reaction.
    Edges are weighted by observed count (a) and EBGM score.
    """
    if signal_only:
        df = disp_df[disp_df["n_methods_signal"] == 4].copy()
    else:
        df = disp_df.copy()

    G = nx.Graph()

    # Add drug nodes
    drugs = df["active_ingredient"].unique()
    for d in drugs:
        G.add_node(f"D:{d}", bipartite=0, node_type="drug", label=d)

    # Add reaction nodes
    reactions = df["reaction"].unique()
    for r in reactions:
        G.add_node(f"R:{r}", bipartite=1, node_type="reaction", label=r)

    # Add edges
    for _, row in df.iterrows():
        d_node = f"D:{row['active_ingredient']}"
        r_node = f"R:{row['reaction']}"
        G.add_edge(d_node, r_node,
                   weight=row["a"],
                   ebgm=row.get("ebgm", 1.0),
                   prr=row.get("prr", 1.0))

    return G


def compute_network_stats(G, name="Network"):
    """Compute and return network-level statistics."""
    drug_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 0}
    rxn_nodes = {n for n, d in G.nodes(data=True) if d.get("bipartite") == 1}

    stats = {
        "name": name,
        "n_nodes": G.number_of_nodes(),
        "n_drug_nodes": len(drug_nodes),
        "n_reaction_nodes": len(rxn_nodes),
        "n_edges": G.number_of_edges(),
        "density": nx.density(G),
        "n_components": nx.number_connected_components(G),
    }

    # Degree statistics
    degrees = dict(G.degree())
    drug_degrees = [degrees[n] for n in drug_nodes if n in degrees]
    rxn_degrees = [degrees[n] for n in rxn_nodes if n in degrees]

    stats["mean_drug_degree"] = np.mean(drug_degrees) if drug_degrees else 0
    stats["median_drug_degree"] = np.median(drug_degrees) if drug_degrees else 0
    stats["max_drug_degree"] = max(drug_degrees) if drug_degrees else 0
    stats["mean_rxn_degree"] = np.mean(rxn_degrees) if rxn_degrees else 0
    stats["median_rxn_degree"] = np.median(rxn_degrees) if rxn_degrees else 0
    stats["max_rxn_degree"] = max(rxn_degrees) if rxn_degrees else 0

    # Largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    stats["largest_component_size"] = len(largest_cc)
    stats["largest_component_pct"] = 100 * len(largest_cc) / G.number_of_nodes()

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
#  CENTRALITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════


def compute_centrality(G):
    """
    Compute centrality metrics for all nodes.

    - Degree centrality
    - Betweenness centrality (approximate, k=500 samples)
    - Eigenvector centrality (with fallback)
    """
    print("  Computing degree centrality ...")
    degree_cent = nx.degree_centrality(G)

    print("  Computing betweenness centrality (approximate, k=500) ...")
    k = min(500, G.number_of_nodes())
    betweenness = nx.betweenness_centrality(G, k=k, seed=42, weight="weight")

    print("  Computing eigenvector centrality ...")
    try:
        eigen = nx.eigenvector_centrality(G, max_iter=500, weight="weight")
    except nx.PowerIterationFailedConvergence:
        print("    Eigenvector centrality did not converge; using degree centrality as proxy.")
        eigen = degree_cent

    return degree_cent, betweenness, eigen


def build_centrality_table(G, degree_cent, betweenness, eigen, node_type):
    """Build a ranked centrality DataFrame for a specific node type."""
    bipartite_val = 0 if node_type == "drug" else 1
    nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == bipartite_val]

    rows = []
    for n in nodes:
        label = G.nodes[n].get("label", n)
        rows.append({
            "name": label,
            "degree": G.degree(n),
            "weighted_degree": G.degree(n, weight="weight"),
            "degree_centrality": degree_cent.get(n, 0),
            "betweenness_centrality": betweenness.get(n, 0),
            "eigenvector_centrality": eigen.get(n, 0),
        })

    df = pd.DataFrame(rows).sort_values("degree", ascending=False).reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  DRUG PROJECTION AND COMMUNITY DETECTION
# ═══════════════════════════════════════════════════════════════════════════════


def build_drug_projection(G_signal):
    """
    Project the bipartite signal network onto the drug side.

    Two drugs are connected if they share at least one significant AE.
    Edge weight = number of shared significant AEs.
    """
    drug_nodes = {n for n, d in G_signal.nodes(data=True) if d.get("bipartite") == 0}

    if len(drug_nodes) == 0:
        return nx.Graph()

    proj = bipartite.weighted_projected_graph(G_signal, drug_nodes)
    return proj


def detect_communities(drug_proj, resolution=1.0):
    """
    Run Louvain community detection on the drug projection.

    Returns a dict mapping node → community_id.
    """
    if drug_proj.number_of_nodes() == 0:
        return {}

    communities = nx_comm.louvain_communities(
        drug_proj, weight="weight", resolution=resolution, seed=42
    )

    # Map node → community ID
    node_community = {}
    for i, comm in enumerate(communities):
        for node in comm:
            node_community[node] = i

    return node_community, communities


def summarise_communities(communities, G_signal, drug_proj):
    """
    Create a summary of each community: top drugs, shared AEs, size.
    """
    rows = []
    for i, comm in enumerate(communities):
        # Drug labels
        drug_labels = sorted([G_signal.nodes[n].get("label", n) for n in comm])

        # Find shared AEs for this community's drugs
        ae_counter = Counter()
        for drug_node in comm:
            for neighbor in G_signal.neighbors(drug_node):
                if G_signal.nodes[neighbor].get("bipartite") == 1:
                    ae_label = G_signal.nodes[neighbor].get("label", neighbor)
                    ae_counter[ae_label] += 1

        top_aes = [ae for ae, _ in ae_counter.most_common(5)]

        rows.append({
            "community": i,
            "n_drugs": len(comm),
            "top_drugs": "; ".join(drug_labels[:8]),
            "top_shared_aes": "; ".join(top_aes),
            "internal_edges": drug_proj.subgraph(comm).number_of_edges(),
        })

    return pd.DataFrame(rows).sort_values("n_drugs", ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════


def plot_degree_distribution(G, save_path):
    """Plot degree distributions for drug and reaction nodes."""
    drug_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 0]
    rxn_nodes = [n for n, d in G.nodes(data=True) if d.get("bipartite") == 1]

    drug_deg = [G.degree(n) for n in drug_nodes]
    rxn_deg = [G.degree(n) for n in rxn_nodes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(drug_deg, bins=50, color="#2196F3", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Degree (number of connected AEs)")
    axes[0].set_ylabel("Number of drugs")
    axes[0].set_title("Drug Degree Distribution")
    axes[0].set_yscale("log")

    axes[1].hist(rxn_deg, bins=50, color="#FF5722", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Degree (number of connected drugs)")
    axes[1].set_ylabel("Number of reactions")
    axes[1].set_title("Reaction Degree Distribution")
    axes[1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_top_network(G_signal, drug_centrality_df, node_community, save_path, top_n=30):
    """
    Plot a filtered network of the top N most-connected drugs
    and their significant AEs, coloured by community.
    """
    # Select top drugs by degree
    top_drugs = drug_centrality_df.head(top_n)["name"].tolist()
    top_drug_nodes = [f"D:{d}" for d in top_drugs]

    # Build subgraph with these drugs and their AE neighbours
    sub_nodes = set(top_drug_nodes)
    for dn in top_drug_nodes:
        if dn in G_signal:
            # Only add top AE neighbors (by edge weight)
            neighbors = sorted(
                G_signal[dn].items(), key=lambda x: x[1].get("weight", 0), reverse=True
            )[:5]  # top 5 AEs per drug
            for rn, _ in neighbors:
                sub_nodes.add(rn)

    sub = G_signal.subgraph(sub_nodes).copy()

    if sub.number_of_nodes() == 0:
        return

    # Layout
    pos = nx.spring_layout(sub, k=2.0, iterations=50, seed=42, weight="weight")

    # Node colours and sizes
    node_colors = []
    node_sizes = []
    cmap = plt.cm.Set3
    max_comm = max(node_community.values()) + 1 if node_community else 1

    for n in sub.nodes():
        if sub.nodes[n].get("bipartite") == 0:  # drug
            comm_id = node_community.get(n, 0)
            node_colors.append(cmap(comm_id % 12 / 12))
            node_sizes.append(max(100, sub.degree(n) * 30))
        else:  # reaction
            node_colors.append("#CCCCCC")
            node_sizes.append(max(40, sub.degree(n) * 15))

    # Labels: only for drug nodes
    labels = {}
    for n in sub.nodes():
        if sub.nodes[n].get("bipartite") == 0:
            label = sub.nodes[n].get("label", n)
            labels[n] = label[:20]

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    # Draw edges
    nx.draw_networkx_edges(sub, pos, alpha=0.15, width=0.5, ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(sub, pos, node_color=node_colors, node_size=node_sizes,
                           alpha=0.85, edgecolors="white", linewidths=0.5, ax=ax)

    # Draw labels
    nx.draw_networkx_labels(sub, pos, labels, font_size=6, font_weight="bold", ax=ax)

    ax.set_title(f"Top {top_n} Hub Drugs and Their Significant Adverse Events\n"
                 f"(coloured by Louvain community)", fontsize=13)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()

    print("=" * 70)
    print("  TGA DAEN Network Analysis")
    print("  Bipartite Drug–AE Network · Centrality · Community Detection")
    print("=" * 70)

    # ── Load data ────────────────────────────────────────────────────────
    print("\nLoading disproportionality results ...")
    disp_df = pd.read_csv(OUTPUT_DIR / "disproportionality_full.csv")
    print(f"  Total pairs: {len(disp_df):,}")
    print(f"  Consensus signals (all 4 methods): {(disp_df['n_methods_signal'] == 4).sum():,}")

    # ── Build networks ───────────────────────────────────────────────────
    print("\nBuilding full bipartite network (all pairs ≥ 3 reports) ...")
    G_full = build_bipartite_network(disp_df, signal_only=False)

    print("Building signal bipartite network (consensus signals only) ...")
    G_signal = build_bipartite_network(disp_df, signal_only=True)

    # ── Network statistics ───────────────────────────────────────────────
    stats_full = compute_network_stats(G_full, "Full Network (≥3 reports)")
    stats_signal = compute_network_stats(G_signal, "Signal Network (all 4 methods)")

    print(f"\n{'=' * 70}")
    print("  NETWORK STATISTICS")
    print(f"{'=' * 70}")
    for stats in [stats_full, stats_signal]:
        print(f"\n  {stats['name']}")
        print(f"  {'─' * 50}")
        print(f"  Nodes:              {stats['n_nodes']:>8,}  "
              f"({stats['n_drug_nodes']:,} drugs, {stats['n_reaction_nodes']:,} reactions)")
        print(f"  Edges:              {stats['n_edges']:>8,}")
        print(f"  Density:            {stats['density']:>11.6f}")
        print(f"  Components:         {stats['n_components']:>8,}")
        print(f"  Largest component:  {stats['largest_component_size']:>8,} nodes "
              f"({stats['largest_component_pct']:.1f}%)")
        print(f"  Drug degree — mean: {stats['mean_drug_degree']:.1f}  "
              f"median: {stats['median_drug_degree']:.0f}  max: {stats['max_drug_degree']}")
        print(f"  Rxn degree  — mean: {stats['mean_rxn_degree']:.1f}  "
              f"median: {stats['median_rxn_degree']:.0f}  max: {stats['max_rxn_degree']}")

    # ── Centrality (on full network) ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  CENTRALITY ANALYSIS (full network)")
    print(f"{'=' * 70}")

    degree_cent, betweenness, eigen = compute_centrality(G_full)

    drug_cent = build_centrality_table(G_full, degree_cent, betweenness, eigen, "drug")
    rxn_cent = build_centrality_table(G_full, degree_cent, betweenness, eigen, "reaction")

    print(f"\n  TOP 15 HUB DRUGS (by degree)")
    print(f"  {'─' * 75}")
    print(f"  {'Drug':<35s} {'Deg':>5s} {'WtDeg':>8s} {'Between':>10s} {'Eigen':>8s}")
    print(f"  {'─' * 75}")
    for _, row in drug_cent.head(15).iterrows():
        name = str(row["name"])[:34]
        print(f"  {name:<35s} {row['degree']:>5.0f} {row['weighted_degree']:>8.0f} "
              f"{row['betweenness_centrality']:>10.5f} {row['eigenvector_centrality']:>8.5f}")

    print(f"\n  TOP 15 HUB REACTIONS (by degree)")
    print(f"  {'─' * 75}")
    print(f"  {'Reaction':<35s} {'Deg':>5s} {'WtDeg':>8s} {'Between':>10s} {'Eigen':>8s}")
    print(f"  {'─' * 75}")
    for _, row in rxn_cent.head(15).iterrows():
        name = str(row["name"])[:34]
        print(f"  {name:<35s} {row['degree']:>5.0f} {row['weighted_degree']:>8.0f} "
              f"{row['betweenness_centrality']:>10.5f} {row['eigenvector_centrality']:>8.5f}")

    # ── Drug projection and community detection ──────────────────────────
    print(f"\n{'=' * 70}")
    print("  COMMUNITY DETECTION (Louvain on drug projection of signal network)")
    print(f"{'=' * 70}")

    print("\n  Projecting signal network onto drug side ...")
    drug_proj = build_drug_projection(G_signal)
    print(f"  Drug projection: {drug_proj.number_of_nodes()} nodes, "
          f"{drug_proj.number_of_edges()} edges")

    print("  Running Louvain community detection ...")
    node_community, communities = detect_communities(drug_proj, resolution=1.0)
    print(f"  Communities found: {len(communities)}")

    # Community summary
    comm_summary = summarise_communities(communities, G_signal, drug_proj)

    # Filter to communities with ≥5 drugs for display
    large_comms = comm_summary[comm_summary["n_drugs"] >= 5]
    print(f"  Communities with ≥ 5 drugs: {len(large_comms)}")

    print(f"\n  TOP COMMUNITIES (≥ 5 drugs)")
    print(f"  {'─' * 85}")
    print(f"  {'#':>3s} {'Size':>5s} {'Edges':>6s}  {'Top drugs':<40s} {'Top shared AEs'}")
    print(f"  {'─' * 85}")
    for _, row in large_comms.head(15).iterrows():
        drugs_str = str(row["top_drugs"])[:39]
        aes_str = str(row["top_shared_aes"])[:35]
        print(f"  {row['community']:>3.0f} {row['n_drugs']:>5.0f} {row['internal_edges']:>6.0f}"
              f"  {drugs_str:<40s} {aes_str}")

    # ── Visualisation ────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("Generating visualisations ...")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    deg_path = FIGURE_DIR / "degree_distribution.png"
    plot_degree_distribution(G_full, deg_path)
    print(f"  {deg_path.name}")

    net_path = FIGURE_DIR / "network_top_drugs.png"
    # Use signal centrality for the visualisation
    signal_deg, signal_bet, signal_eig = {}, {}, {}
    for n in G_signal.nodes():
        signal_deg[n] = 0
        signal_bet[n] = 0
        signal_eig[n] = 0
    signal_deg = nx.degree_centrality(G_signal)
    signal_drug_cent = build_centrality_table(
        G_signal, signal_deg, signal_deg, signal_deg, "drug"
    )
    plot_top_network(G_signal, signal_drug_cent, node_community, net_path, top_n=30)
    print(f"  {net_path.name}")

    # ── Save outputs ─────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("Saving results ...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Centrality tables
    drug_cent_path = OUTPUT_DIR / "network_drug_centrality.csv"
    drug_cent.to_csv(drug_cent_path, index=False, float_format="%.6f")

    rxn_cent_path = OUTPUT_DIR / "network_reaction_centrality.csv"
    rxn_cent.to_csv(rxn_cent_path, index=False, float_format="%.6f")

    # Community assignments
    comm_path = OUTPUT_DIR / "network_communities.csv"
    comm_rows = []
    for node, comm_id in node_community.items():
        label = G_signal.nodes[node].get("label", node) if node in G_signal else node
        comm_rows.append({"drug": label, "community": comm_id})
    pd.DataFrame(comm_rows).sort_values(["community", "drug"]).to_csv(comm_path, index=False)

    # Community summary
    comm_sum_path = OUTPUT_DIR / "network_community_summary.csv"
    comm_summary.to_csv(comm_sum_path, index=False)

    # Network statistics report
    report_lines = []
    report_lines.append("TGA DAEN Network Analysis — Summary Report\n")
    for stats in [stats_full, stats_signal]:
        report_lines.append(f"\n{stats['name']}")
        report_lines.append(f"  Nodes: {stats['n_nodes']:,} ({stats['n_drug_nodes']:,} drugs, {stats['n_reaction_nodes']:,} reactions)")
        report_lines.append(f"  Edges: {stats['n_edges']:,}")
        report_lines.append(f"  Density: {stats['density']:.6f}")
        report_lines.append(f"  Connected components: {stats['n_components']:,}")
        report_lines.append(f"  Largest component: {stats['largest_component_size']:,} ({stats['largest_component_pct']:.1f}%)")
        report_lines.append(f"  Drug degree — mean: {stats['mean_drug_degree']:.1f}, median: {stats['median_drug_degree']:.0f}, max: {stats['max_drug_degree']}")
        report_lines.append(f"  Reaction degree — mean: {stats['mean_rxn_degree']:.1f}, median: {stats['median_rxn_degree']:.0f}, max: {stats['max_rxn_degree']}")
    report_lines.append(f"\nCommunities detected: {len(communities)}")
    report_lines.append(f"Communities with ≥5 drugs: {len(large_comms)}")

    stats_path = OUTPUT_DIR / "network_summary.txt"
    with open(stats_path, "w") as f:
        f.write("\n".join(report_lines))

    for path in [drug_cent_path, rxn_cent_path, comm_path, comm_sum_path, stats_path]:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.name:45s}  {size_mb:>6.1f} MB")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  NETWORK ANALYSIS COMPLETE  ({elapsed:.0f}s)")
    print(f"{'=' * 70}")
    print(f"\n  Next step: python scripts/06_visualisation.py\n")


if __name__ == "__main__":
    main()
