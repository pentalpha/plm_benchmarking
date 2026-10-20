import sys
import os
import json

import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

def _format_param_val(val):
    """Formats floats to 4 significant digits / clean decimals, leaving ints and strings intact."""
    if isinstance(val, (int, np.integer)):
        return str(val)
    if isinstance(val, (float, np.floating)):
        if float(val).is_integer():
            return str(int(val))
        return f"{val:.4f}".rstrip("0").rstrip(".")
    return str(val)


def _assign_bounded_tsne_clusters(coords, max_size=9):
    """
    Clusters 2D t-SNE coordinates using Ward hierarchical clustering,
    recursively splitting any cluster with > max_size points, and merging
    adjacent small clusters if their combined size <= max_size.
    Strictly guarantees 1 <= cluster_size <= max_size.
    """
    n_samples = len(coords)
    init_k = max(2, int(np.ceil(n_samples / (max_size * 0.75))))
    labels = AgglomerativeClustering(n_clusters=init_k, linkage="ward").fit_predict(coords)

    # Step 1: Recursively split any cluster larger than max_size
    next_label = int(labels.max()) + 1
    while True:
        unique_labels, counts = np.unique(labels, return_counts=True)
        oversized = unique_labels[counts > max_size]
        if len(oversized) == 0:
            break
        for cid in oversized:
            idx = np.where(labels == cid)[0]
            sub_labels = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(coords[idx])
            labels[idx[sub_labels == 1]] = next_label
            next_label += 1

    # Step 2: Greedily merge very close small clusters if combined size <= max_size
    # Set a distance threshold based on median nearest-cluster distance
    unique_labels = np.unique(labels)
    if len(unique_labels) > 1:
        centroids = np.array([coords[labels == c].mean(axis=0) for c in unique_labels])
        pw_dists = cdist(centroids, centroids)
        np.fill_diagonal(pw_dists, np.inf)
        merge_radius = np.percentile(pw_dists.min(axis=1), 65) * 1.35

        while True:
            unique_labels, counts = np.unique(labels, return_counts=True)
            if len(unique_labels) <= 1:
                break
            centroids = np.array([coords[labels == c].mean(axis=0) for c in unique_labels])
            dists = cdist(centroids, centroids)
            np.fill_diagonal(dists, np.inf)

            best_pair = None
            best_dist = np.inf
            for i in range(len(unique_labels)):
                for j in range(i + 1, len(unique_labels)):
                    if counts[i] + counts[j] <= max_size and dists[i, j] < best_dist:
                        best_dist = dists[i, j]
                        best_pair = (unique_labels[i], unique_labels[j])

            if best_pair is not None and best_dist <= merge_radius:
                c_keep, c_merge = best_pair
                labels[labels == c_merge] = c_keep
            else:
                break

    return labels


def create_tsne_clusters_table(df_evi_tsne, results_dir, max_cluster_size=9):
    """
    1. Clusters t-SNE points into groups of at most `max_cluster_size` rows.
    2. Ranks clusters by Mean Sort Score (Cluster 0 = Best) and adds 'Cluster' column to df_evi_tsne.
    3. Builds a summary table with Mean Sort Score, Rule averages, and per-parameter columns
       (single value if constant across the cluster, or 'param=sort_score;...' if varied).
    4. Saves the table to `tsne_clusters.csv` in `results_dir`.
    """
    df = df_evi_tsne.copy()
    coords = df[["TSNE_1", "TSNE_2"]].values

    # 1. Assign raw clusters with <= max_cluster_size rows
    raw_clusters = _assign_bounded_tsne_clusters(coords, max_size=max_cluster_size)
    df["_raw_cluster"] = raw_clusters

    # 2. Sort cluster IDs by Max Sort Score (descending) so Cluster 0 is highest quality
    cluster_order = (
        df.groupby("_raw_cluster")["Sort Score"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    cluster_id_map = {old_id: new_id for new_id, old_id in enumerate(cluster_order)}
    df["Cluster"] = df["_raw_cluster"].map(cluster_id_map)
    df_evi_tsne["Cluster"] = df["Cluster"]

    # 3. Collect all parameter names in deterministic order
    evi_param_order = [
        "derived_not", "phylo_not", "curated_not",
        "Random False Val", "conditional_not", "curated", "phylo",
        "Random Falses Min Perc"
    ]
    all_param_keys = set()
    parsed_params_list = []
    for _, row in df.iterrows():
        p_dict = json.loads(row["Parameters"])
        parsed_params_list.append(p_dict)
        all_param_keys.update(p_dict.keys())
    df["_parsed_params"] = parsed_params_list

    # Put soft-label parameters first, followed by GBDT hyperparameters alphabetically
    ordered_param_names = [k for k in evi_param_order if k in all_param_keys] + sorted(
        [k for k in all_param_keys if k not in evi_param_order]
    )

    # 4. Build one summary row per cluster
    cluster_rows = []
    rule_cols = [c for c in ["R1_Score", "R2_Score", "R3_Score", "R4_Score", "Rule_Sum"] if c in df.columns]

    for cid in range(len(cluster_order)):
        # Sort lines inside the cluster by Sort Score descending
        c_df = df[df["Cluster"] == cid].sort_values(by="Sort Score", ascending=False)
        scores = c_df["Sort Score"].values

        row_summary = {
            "Cluster": cid,
            "N_Lines": len(c_df),
            "Mean_Sort_Score": round(c_df["Sort Score"].mean(), 2),
            "Max_Sort_Score": round(c_df["Sort Score"].max(), 2),
            "Min_Sort_Score": round(c_df["Sort Score"].min(), 2),
            "Mean_OWA_Score": round(c_df["OWA Score"].mean(), 2),
            "Mean_CWA_Score": round(c_df["CWA Score"].mean(), 2),
        }

        row_summary["Quality Variance"] = row_summary["Max_Sort_Score"] - row_summary["Min_Sort_Score"]

        # Add average for every rule
        for r_col in rule_cols:
            row_summary[f"Avg_{r_col}"] = round(c_df[r_col].mean(), 2)

        # Include Ontology and Evi Sorting using the same single-vs-series convention
        for meta_col in ["Ontology", "Evi Sorting"]:
            if meta_col in c_df.columns:
                m_vals = c_df[meta_col].tolist()
                if len(set(m_vals)) == 1:
                    row_summary[meta_col] = str(m_vals[0])
                else:
                    row_summary[meta_col] = ";".join(
                        f"{m}={s:.1f}" for m, s in zip(m_vals, scores)
                    )

        # Format each parameter column: single value if constant, else "param=sort_score;..."
        for p_name in ordered_param_names:
            p_vals = [d.get(p_name, None) for d in c_df["_parsed_params"]]
            fmt_vals = [_format_param_val(v) for v in p_vals]
            unique_fmt = set(fmt_vals)

            if len(unique_fmt) == 1:
                row_summary[p_name] = fmt_vals[0]
            else:
                row_summary[p_name] = ";".join(
                    f"{fv}={sc:.1f}" for fv, sc in zip(fmt_vals, scores)
                )

        cluster_rows.append(row_summary)

    clusters_df = pd.DataFrame(cluster_rows)
    clusters_df = clusters_df.sort_values(by="Max_Sort_Score", ascending=False).reset_index(drop=True)

    csv_path = os.path.join(results_dir, "tsne_clusters.csv")
    clusters_df.to_csv(csv_path, index=False)
    print(f"\nSaved {len(clusters_df)} t-SNE clusters (max size <= {max_cluster_size}) to: {csv_path}")
    print(clusters_df[["Cluster", "N_Lines", "Mean_Sort_Score", "Mean_OWA_Score", "Avg_Rule_Sum", "Evi Sorting"]].head(10).to_string(index=False))

    return df_evi_tsne, clusters_df

STR_MARKER_MAP = {
    "Classic": "s",       # Square
    "Conditional Negatives": "^",         # 
    "Conditional Negatives + RNS": "v",       # 
    "Soft": "o",       # 
    "Soft + RNS": "p"    # 
}

MARKER_MAP = {
    "MF": "s", #square
    "BP": "^", #triangle-up
    "CC": "o", #circle
}

MARKER_COLOR_MAP = {
    "Classic": "royalblue",
    "Conditional Negatives": "red",
    "Conditional Negatives + RNS": "firebrick",
    "Soft": "palegreen",
    "Soft + RNS": "seagreen"
}

def scatter_owa_cwa_scores(ax, df_best_recalc, base_dir=None):
    y_col = "OWA Score"
    x_col = "CWA Score"

    color_col = "Algorithm"
    marker_col = "Ontology"

    for ont, df in df_best_recalc.groupby(marker_col):
        y = (df[y_col] * 100).values
        x = (df[x_col] * 100).values

        c = [MARKER_COLOR_MAP[ont] for ont in df[color_col].values]

        ax.scatter(
            x, y, c=c, marker=MARKER_MAP[ont],
            s=180, edgecolors="white", linewidths=1, zorder=3, alpha=0.85
        )
    
    legend_elements = []

    # Color patches for Ontologies (no `marker` argument in mpatches.Patch)
    for strategy, color in MARKER_COLOR_MAP.items():
        legend_elements.append(
            mpatches.Patch(facecolor=color, edgecolor="black", linewidth=0, label=strategy)
        )
        
    # Line2D handles for Algorithm marker shapes
    for ont, marker in MARKER_MAP.items():
        legend_elements.append(
            Line2D(
                [0], [0],
                marker=marker,
                color="none",          # hides the line
                label=ont,
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=7,
            )
        )

    ax.set_xlabel("CWA Score (%)", fontweight="bold", fontsize=10)
    ax.set_ylabel("OWA Score (%)", fontweight="bold", fontsize=10)
    ax.set_xlim(42, 67)
    ax.set_ylim(49, 93)
    ax.set_title("B: Optimized OWA and CWA Performance", fontsize=12, fontweight="bold", y=1.02)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

def compute_structural_metrics(val_dict, other_parameters, tie_eps=0.025):
    """
    Computes both graded categorical rules (with fractional + near-tie states)
    and 4 continuous hypothesis scores for a single soft-labeling configuration.
    """
    params = json.loads(other_parameters)
    py_pos = val_dict.get("PY+", np.nan)
    cu_pos = val_dict.get("CU+", np.nan)
    de_neg = val_dict.get("DE-", np.nan)
    py_neg = val_dict.get("PY-", np.nan)
    cu_neg = val_dict.get("CU-", np.nan)
    rd_neg = val_dict.get("RD-", np.nan)
    co_neg = val_dict.get("CO-", np.nan)
    has_rd = not np.isnan(rd_neg)

    # --- R1: Positive Ordering (CU+ < PY+) ---
    gap_r1 = py_pos - cu_pos
    if gap_r1 > tie_eps:
        r1_score = 1.0
    elif abs(gap_r1) <= tie_eps and py_pos > cu_pos:
        r1_score = 0.5
    else:
        r1_score = 0.0

    # --- R2: Taxonomic/Derived Negs before Curated Neg (DE- & PY- < CU-) ---
    gap_de = cu_neg - de_neg
    gap_py = cu_neg - py_neg

    r2_score = 0.0
    r2_max = 2.0
    if gap_de > tie_eps:
        r2_score += 1.0
    elif cu_neg > de_neg and abs(gap_de) <= tie_eps:
        r2_score += 0.5
    
    if gap_py > tie_eps:
        r2_score += 1.0
    elif cu_neg > py_neg and abs(gap_py) <= tie_eps:
        r2_score += 0.5
    
    r2_score = r2_score / r2_max

    # --- R3: Explicit Negs (DE-, PY-, CU-) before Pseudo-Negs (RD-, CO-) ---
    first_pseudo = min(rd_neg, co_neg) if has_rd else co_neg
    max_explicit = max(de_neg, py_neg, cu_neg)
    gap_r3 = first_pseudo - max_explicit  # H1: Tier Buffer Gap
    relevantly_below = [(first_pseudo - x) > tie_eps for x in [de_neg, py_neg, cu_neg]]
    below = [(first_pseudo - x) > 0 for x in [de_neg, py_neg, cu_neg]]

    r3_sum = 0.0
    for is_below, relevantly_below in zip(below, relevantly_below):
        if relevantly_below:
            r3_sum += 1.0
        elif is_below:
            r3_sum += 0.5
    
    r3_score = r3_sum / 3.0

    # --- R4: CO- is the Softest Negative (All other negs < CO-) ---
    other_negs = [de_neg, py_neg, cu_neg, rd_neg] if has_rd else [de_neg, py_neg, cu_neg]
    max_other_neg = max(other_negs)
    gap_r4 = co_neg - max_other_neg
    if gap_r4 > tie_eps:
        r4_score = 1.0
    elif abs(gap_r4) <= tie_eps and co_neg > max_other_neg:
        r4_score = 0.5
    else:
        r4_score = 0.0

    # --- Hard Negative Anchor Ceiling (max(DE-, PY-) <= 0.11 ---
    max_hard_neg = max(de_neg, py_neg)
    if max_hard_neg <= 0.11:
        r5_score = 1.0
    elif max_hard_neg <= 0.11 + tie_eps:
        r5_score = 0.5
    else:
        r5_score = 0.0

    #Moderate RNS Density
    rns_perc = float(params.get("Random Falses Min Perc", 0.0))
    if rns_perc <= 0.28 + tie_eps:
        if rns_perc <= 0.28:
            r6_score = 1.0
        else:
            r6_score = 0.5
    else:
        r6_score = 0.0

    #max_depth >= 7
    max_depth = float(params.get("max_depth", 0.0))
    if max_depth >= 7:
        r7_score = 1.0
    else:
        r7_score = 0.0
    
    # --- R8: Gap between Explicit Negs (DE-, PY-, CU-) and (RD-, CO-) >= 0.12 and <= 0.19
    min_gap = 0.12
    max_gap = 0.19
    gap_r8 = first_pseudo - max_explicit  # H1: Tier Buffer Gap

    if gap_r8 >= min_gap and gap_r8 <= max_gap:
        r8_score = 1.0
    elif gap_r8 >= min_gap - tie_eps and gap_r8 <= max_gap + tie_eps:
        r8_score = 0.5
    else:
        r8_score = 0.0

    
    rule_sum = r1_score + r2_score + r3_score + r4_score + r5_score + r6_score + r7_score

    return {
        "R1_Score": r1_score,
        "R2_Score": r2_score,
        "R3_Score": r3_score,
        "R4_Score": r4_score,
        "R5_Score": r5_score,
        "R6_Score": r6_score,
        "R7_Score": r7_score,
        #"R8_Score": r8_score,
        "Rule_Sum": rule_sum,
    }


def plot_evi_sortings(grouping_call, sorting_col, sorting_results, results_dir):
    print(f"i\t\t{grouping_call}\t{sorting_col}\t\tStd\t\tN Tests")
    for i, row in enumerate(sorting_results):
        print(i, row["Evi Sorting"], f"{row[sorting_col]:.2f}", f"{row['Sort Std']:.2f}", row["N Tests"])
        print("\tBest Config: ", row["Best Evi Vals"])
        print("\tWorst Config:", row["Worst Evi Vals"])

    names = [f"{row['Evi Sorting']} (n={row['N Tests']})" for row in sorting_results]
    is_owa = "OWA" in sorting_col
    values = [row["All OWA"] if is_owa else row["All Sort"] for row in sorting_results]
    means = [row["Mean OWA"] if is_owa else row["Mean Sort"] for row in sorting_results]

    n_rows = len(sorting_results)
    y_positions = np.arange(1, n_rows + 1)

    if grouping_call == "Evi Sorting":
        # TASK 1 FIX: Wider figure, increased right panel ratio [1.9, 1.5], tight wspace
        fig, axes = plt.subplots(
            nrows=1, ncols=2, figsize=(24, max(10, n_rows * 0.36)),
            width_ratios=[1.4, 1.6], sharey=True,
            gridspec_kw={"wspace": 0.05}
        )
        ax = axes[0]
        ax_patterns = axes[1]
    else:
        fig, ax = plt.subplots(figsize=(13, 8))
        ax_patterns = None

    # --- Left Plot: Horizontal Boxplot + Jittered Points ---
    ax.boxplot(
        values,
        positions=y_positions,
        orientation="horizontal",
        tick_labels=names,
        patch_artist=True,
        boxprops=dict(facecolor="#cfe2f3", edgecolor="#2b5c8f", alpha=0.85),
        medianprops=dict(color="#b22222", linewidth=2),
        flierprops=dict(marker="o", markersize=4, alpha=0.5)
    )

    rng = np.random.default_rng(42)
    for y_pos, vals in zip(y_positions, values):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(
            vals, np.full(len(vals), y_pos) + jitter,
            color="#1f4e78", alpha=0.55, s=18, zorder=3
        )

    ax.scatter(
        means, y_positions,
        marker="D", color="#d95f02", s=35, zorder=4, label="Mean Score"
    )

    x_label = "OWA Score (Normalized %)" if is_owa else "Overall Score (Normalized %)"
    ax.set_xlabel(x_label, fontsize=11, fontweight="bold")
    ax.set_ylabel("Evi. Type Sorting", fontsize=11, fontweight="bold")
    ax.set_title(f"{grouping_call} vs. {x_label} (Sorted by {sorting_col})", fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")
    ax.invert_yaxis()

    # --- Right Plot: Graded Pattern Matrix (Fractional + Margin) ---
    if grouping_call == "Evi Sorting" and ax_patterns is not None:
        # Concise, wrapped 2-line column headers so they never overlap
        col_names = [
            "R1: Positives\n(CU+ < PY+)",
            "R2: Hard Negs\n(DE-, PY- < CU-)",
            "R3: Tier Buffer\n(Negs < RD-/CO-)",
            "R4: Softest Neg\n(All < CO-)",
            "R5: Hard Neg Anch\n(DE-, PY- <= 0.11)",
            "R6: RNS Density\n(20-25%)",
            "R7: Max Depth\n(>= 7)",
            #"R8: Tier Buffer Gap\n(0.12 <= Gap <= 0.19)",
            "Rules Met"
        ]

        pattern_scores = np.array([
            [
                row["R1_Score"],
                row["R2_Score"],
                row["R3_Score"],
                row["R4_Score"],
                row["R5_Score"],
                row["R6_Score"],
                row["R7_Score"],
                #row["R8_Score"],
                row["Rule_Sum"] / 7.0
            ]
            for row in sorting_results
        ], dtype=float)

        pattern_texts = [
            [
                f"{row["R1_Score"]:.1f}",
                f"{row["R2_Score"]:.1f}",
                f"{row["R3_Score"]:.1f}",
                f"{row["R4_Score"]:.1f}",
                f"{row["R5_Score"]:.1f}",
                f"{row["R6_Score"]:.1f}",
                f"{row["R7_Score"]:.1f}",
                #f"{row["R8_Score"]:.1f}",
                f"{row["Rule_Sum"]:.1f}"
            ]
            for row in sorting_results
        ]

        # Smooth 4-stop color scale: Soft Red (0.0) -> Amber (0.45) -> Pale Green (0.75) -> Rich Green (1.0)
        cmap = LinearSegmentedColormap.from_list(
            "graded_rules",
            ["#f4cccc", "#ffe599", "#d9ead3", "#b6d7a8"]
        )
        ax_patterns.imshow(
            pattern_scores,
            aspect="auto",
            cmap=cmap,
            vmin=0.0, vmax=1.0,
            extent=[-0.5, len(col_names) - 0.5, n_rows + 0.5, 0.5]
        )

        for row_idx in range(n_rows):
            y_pos = row_idx + 1
            for col_idx in range(len(col_names)):
                score_val = pattern_scores[row_idx, col_idx]
                text_color = "#1c450c" if score_val >= 0.70 else ("#7f6000" if score_val >= 0.35 else "#78281f")
                ax_patterns.text(
                    col_idx, y_pos, pattern_texts[row_idx][col_idx],
                    ha="center", va="center",
                    color=text_color, fontweight="bold", fontsize=8.5
                )

        ax_patterns.set_xticks(np.arange(len(col_names)))
        ax_patterns.set_xticklabels(col_names, fontsize=9.5, fontweight="bold", ha="center")
        ax_patterns.tick_params(axis="x", pad=6)
        ax_patterns.set_xticks(np.arange(-0.5, len(col_names), 1), minor=True)
        ax_patterns.set_yticks(np.arange(0.5, n_rows + 1, 1), minor=True)
        ax_patterns.grid(which="minor", color="white", linestyle="-", linewidth=2)
        ax_patterns.tick_params(which="minor", bottom=False, left=False)
        ax_patterns.set_title("Graded Structural Rules & Best-Config Margins", fontsize=12, fontweight="bold")

    fig_name = f"Evidence {sorting_col} vs {grouping_call} - boxplot.png"
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, fig_name), dpi=300, bbox_inches="tight")
    plt.close(fig)

def get_evi_tsne_coords(
    df_evi,
    perplexity=25,
    random_state=1337
):
    """
    Task 2: Projects the soft-labeling configurations into 2D t-SNE space based on their
    evidence weights, plotting two side-by-side panels:
      - Left:  Colored by model performance (`Sort Score` or `OWA Score`)
      - Right: Colored by total normalized rule fit (`Rule_Fit_Sum` in [0, 4])
    """
    all_names = set()
    for _, row in df_evi.iterrows():
        param_names = json.loads(row["Parameters"])
        all_names.update(param_names.keys())
        
    all_names = list(all_names)
    print("Parameters for tsne: ", all_names)
    params_matrix = []
    for _, row in df_evi.iterrows():
        params_dict = json.loads(row["Parameters"])
        is_mf = 1 if "MF" in row["Ontology"] else -1
        is_bp = 1 if "BP" in row["Ontology"] else -1
        is_cc = 1 if "CC" in row["Ontology"] else -1
        vals = [is_mf, is_bp, is_cc] + [params_dict.get(name, -random_state) for name in all_names if name in params_dict]
        print(vals)
        params_matrix.append(vals)
    params_matrix = np.array(params_matrix)

    X = StandardScaler().fit_transform(params_matrix)
    n_samples = len(df_evi)
    eff_perplexity = min(perplexity, max(5, (n_samples - 1) // 3))

    tsne = TSNE(
        n_components=2,
        perplexity=eff_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=random_state
    )
    coords = tsne.fit_transform(X)
    df_evi["TSNE_1"] = coords[:, 0]
    df_evi["TSNE_2"] = coords[:, 1]
    return df_evi

def plot_tsne_from_df(df, color_cols: list, results_dir, col_x = "TSNE_1", col_y = "TSNE_2", title = "TSNE Plot"):
    fig, axes = plt.subplots(nrows=1, ncols=len(color_cols), figsize=(len(color_cols)*7, 7), sharex=True, sharey=True)

    if len(color_cols) == 1:
        axes = [axes]

    for ax, score_col in zip(axes, color_cols):
        sc_perf = ax.scatter(
            df["TSNE_1"],
            df["TSNE_2"],
            c=df[score_col],
            cmap="gnuplot",
            s=120,
            alpha=0.8,
            edgecolors="black",
            linewidths=0,
            zorder=3
        )
        cbar = fig.colorbar(sc_perf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(score_col, fontsize=10, fontweight="bold", labelpad=12)
        ax.set_xlabel(col_x.replace("_", " "), fontsize=10, fontweight="bold")
        ax.set_ylabel(col_y.replace("_", " "), fontsize=10, fontweight="bold")
        ax.set_title(f"Colored by {score_col.replace('_', ' ').title()}", fontsize=12, fontweight="bold")
        if "Cluster" in df.columns:
            for cid, c_group in df.groupby("Cluster"):
                cx, cy = c_group["TSNE_1"].mean(), c_group["TSNE_2"].mean()
                ax.text(
                    cx, cy, str(cid),
                    fontsize=7.5, fontweight="bold", ha="center", va="center",
                    color="white", bbox=dict(boxstyle="circle,pad=0.15", fc="black", ec="white", lw=0.6, alpha=0.65),
                    zorder=5
                )
    
    

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "params_tsne.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    #Scatter of color_col 1 and 2
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 7), sharex=True, sharey=True)
    sc_perf = ax.scatter(
        df[color_cols[0]],
        df[color_cols[1]],
        c=df["OWA Score"],
        cmap="gnuplot",
        s=120,
        alpha=0.8,
        edgecolors="black",
        linewidths=0,
        zorder=3
    )
    cbar = fig.colorbar(sc_perf, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("OWA Score", fontsize=10, fontweight="bold", labelpad=12)
    ax.set_xlabel(color_cols[0], fontsize=10, fontweight="bold")
    ax.set_ylabel(color_cols[1], fontsize=10, fontweight="bold")
    ax.set_title(f"{color_cols[0].title()} vs {color_cols[1].title()} (Colored by OWA Score)", fontsize=12, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "correlation.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

def analyze_evi_sortings(df, results_dir):
    labels = ["phylo", "curated", "conditional_not", "curated_not", "derived_not", "phylo_not", "Random False Val"]
    labels_short = {
        "phylo": "PY+", "curated": "CU+", "conditional_not": "CO-",
        "curated_not": "CU-", "derived_not": "DE-", "phylo_not": "PY-",
        "Random False Val": "RD-",
    }

    df_valid = df[df["Sort Score"] > 0].copy()
    df_norm_list = []
    for _, df_group in df_valid.groupby(["Ontology"]):
        df_group = df_group.copy()
        for col in ["Sort Score", "OWA Score", "CWA Score"]:
            bottom = df_group[col].min()
            top = df_group[col].max()
            total_range = top - bottom
            if total_range > 0:
                df_group[col] = ((df_group[col] - bottom) / total_range) * 100.0
        df_norm_list.append(df_group)
    df_norm = pd.concat(df_norm_list, ignore_index=True)

    df_evi = []
    for _, row in df_norm.iterrows():
        prev_k = ["Algorithm", "Ontology", "Sort Score", "OWA Score", "CWA Score", "Parameters"]
        new_row = {k: row[k] for k in prev_k}
        evi_vals = []
        val_dict = {}
        has_rand = False

        params_dict = json.loads(row["Parameters"])
        for label in labels:
            if label in params_dict:
                if labels_short[label] == "RD-":
                    has_rand = True
                label_short = labels_short[label]
                val = float(params_dict[label])
                evi_vals.append((label_short, val))
                val_dict[label_short] = val

        if len(evi_vals) > 1 and has_rand:
            evi_vals.sort(key=lambda x: x[1])
            pos_vals = [(k, v) for k, v in evi_vals if "+" in k]
            neg_vals = [(k, v) for k, v in evi_vals if "-" in k]

            new_row["Evi Sorting"] = " < ".join([k for k, v in evi_vals])
            new_row["Evi Sorting Pos"] = " < ".join([k for k, v in pos_vals])
            new_row["Evi Sorting Neg"] = " < ".join([k for k, v in neg_vals])
            new_row["Evi Vals"] = " < ".join([f"{k}={v:.2f}" for k, v in evi_vals])
            for k, v in evi_vals:
                new_row[k] = v

            # Compute graded categorical rules and continuous hypothesis metrics
            new_row.update(compute_structural_metrics(val_dict, new_row["Parameters"]))
            df_evi.append(new_row)

    df_evi = pd.DataFrame(df_evi)
    df_evi_tsne = get_evi_tsne_coords(df_evi)
    df_evi_tsne, clusters_df = create_tsne_clusters_table(
        df_evi_tsne, results_dir=results_dir, max_cluster_size=9
    )
    clusters_df.to_csv(os.path.join(results_dir, "clusters.csv"), index=False)
    plot_tsne_from_df(df_evi_tsne, color_cols=["Sort Score", "Rule_Sum"], results_dir=results_dir)

    for grouping_call in ["Evi Sorting"]:
        for sorting_col in ["Max Sort", "Max OWA", "Median Sort"]:
            sorting_results = []
            print("-------\n", grouping_call, "sorted by", sorting_col)
            for tp, group_df in df_evi.groupby([grouping_call]):
                non_null = group_df[group_df["Sort Score"].notna()].copy()
                # Sort ascending so iloc[0] is WORST and iloc[-1] is BEST
                score_key = "OWA Score" if "OWA" in sorting_col else "Sort Score"
                non_null = non_null.sort_values(by=[score_key], ascending=True)
                sorting = tp[0]

                unique_evi_vals = non_null["Evi Vals"].unique()
                n_unique_evi_vals = len(unique_evi_vals)
                
                
                worst_sort_row = non_null.iloc[0]   # FIXED: iloc[0] is lowest score
                best_sort_row = non_null.iloc[-1]   # FIXED: iloc[-1] is highest score

                row = {
                    "Evi Sorting": sorting,
                    "Median Sort": non_null["Sort Score"].median(),
                    "Mean Sort": non_null["Sort Score"].mean(),
                    "Max Sort": non_null["Sort Score"].max(),
                    "Min Sort": non_null["Sort Score"].min(),
                    "Sort Std": non_null["Sort Score"].std(),
                    "N Tests": n_unique_evi_vals,
                    "All Sort": non_null["Sort Score"].tolist(),
                    "All OWA": non_null["OWA Score"].tolist(),
                    "Mean OWA": non_null["OWA Score"].mean(),
                    "Max OWA": non_null["OWA Score"].max(),
                    "Best Evi Vals": best_sort_row["Evi Vals"],
                    "Worst Evi Vals": worst_sort_row["Evi Vals"],
                    # Use the best-performing config's graded rule strings/margins for display
                    "R1_Score": non_null["R1_Score"].mean(),
                    "R2_Score": non_null["R2_Score"].mean(),
                    "R3_Score": non_null["R3_Score"].mean(),
                    "R4_Score": non_null["R4_Score"].mean(),
                    "R5_Score": non_null["R5_Score"].mean(),
                    "R6_Score": non_null["R6_Score"].mean(),
                    "R7_Score": non_null["R7_Score"].mean(),
                    #"R8_Score": non_null["R8_Score"].mean(),
                    "Rule_Sum": non_null["Rule_Sum"].mean(),
                }
                sorting_results.append(row)

            sorting_results.sort(key=lambda x: x[sorting_col], reverse=True)
            plot_evi_sortings(grouping_call, sorting_col, sorting_results, results_dir)
    
    quit(1)
    #return pd.DataFrame(sorting_results)

def plot_evi_values_scale(ax, df_best_recalc, base_dir, strategy = "Soft + RNS"):
    # 1. Carregamento dos dados
    df_best_str = df_best_recalc[df_best_recalc['Algorithm'] == strategy].copy()
    print(df_best_recalc)
    print(df_best_recalc['Algorithm'].unique())
    print(df_best_str)

    labels = ["phylo",
        "curated",
        "conditional_not",
        "curated_not",
        "derived_not",
        "phylo_not",
        #"Random Falses Min Perc",
        "Random False Val",]
    
    df_lines = []
    for ont in ['MF', 'CC', 'BP']:
        df_ont = df_best_str[df_best_str['Ontology'] == ont]
        df_ont_row = None if len(df_ont) == 0 else df_ont.iloc[0]
        
        params_json = df_ont_row['Parameters']
        params = json.loads(params_json)
        for label in labels:
            if label in params:
                if not ("False" in label or "_not" in label):
                    label2 = label+'_yes'
                else:
                    label2 = label
                if label2 == "Random False Val":
                    label2 = "random_not"
                df_lines.append({"evi_type": label2, "label_value": params[label], 'ontology': ont})
    
    df = pd.DataFrame(df_lines)

    # Formatação limpa dos textos (ex: 'experimental_not' -> 'Experimental\n(Not)')
    def format_evi(name):
        parts = name.split('_')
        if len(parts) == 2:
            return f"{parts[0].capitalize()}\n({parts[1].capitalize()})"
        return name

    # 2. Definições de Estética e Layout
    # Paleta de Cores (Cores quentes para Not, Frias para Yes)
    color_map = {
        'random_not': '#E8F5E9', # green-ish
        'experimental_not': '#9c171e', # Vermelho escuro
        'derived_not': '#db5341',      # Vermelho
        'phylo_not': '#f28b4c',        # Laranja
        'curated_not': '#f9b670',      # Laranja claro
        'conditional_not': '#f7cc57',  # Amarelo
        'curated_yes': '#4eb19e',      # Turquesa
        'phylo_yes': '#4a7eb3',        # Azul
        'experimental_yes': '#287d6f'  # Verde/Teal escuro
    }

    # Criando a figura
    #fig, ax = plt.subplots(figsize=(7, 3.6))

    # Calculando a posição x média (Visão Conceitual)
    avg_x = df.groupby('evi_type')['label_value'].mean().sort_values()

    # Desenhando as barras cinzas horizontais
    for y in [0, 1, 2]:
        ax.hlines(y, xmin=-0.05, xmax=1.05, color='#b0b0b0', linewidth=5, zorder=1)

    # 3. Lógica Anti-Sobreposição dos Labels Inferiores
    label_xs = {}
    # Distanciando itens do lado esquerdo (valores baixos)
    current_x = -0.01
    for evi, x_val in avg_x.items():
        if x_val < 0.5:
            target_x = max(x_val, current_x)
            label_xs[evi] = target_x
            current_x = target_x + 0.12
        else:
            label_xs[evi] = x_val

    # Distanciando itens do lado direito (valores altos) de trás pra frente
    current_x = 1.05
    for evi in reversed(list(avg_x.index)):
        x_val = avg_x[evi]
        if x_val >= 0.5:
            target_x = min(x_val, current_x)
            label_xs[evi] = target_x
            current_x = target_x - 0.1

    # 4. Plotagem dos pontos e linhas de conexão
    y_label = -0.7 # Altura das caixinhas de texto
    Path = mpath.Path

    for evi, _ in avg_x.items(): # Descartamos a variável x_val (média) aqui
        c = color_map[evi]
        target_x = label_xs[evi]
        display_name = format_evi(evi)
        
        # Capturando os valores exatos de peso para cada ontologia (CC em y=0, BP em y=1, MF em y=2)
        x_cc = df[(df['evi_type'] == evi) & (df['ontology'] == 'CC')]['label_value'].values[0]
        x_bp = df[(df['evi_type'] == evi) & (df['ontology'] == 'BP')]['label_value'].values[0]
        x_mf = df[(df['evi_type'] == evi) & (df['ontology'] == 'MF')]['label_value'].values[0]
        
        # Linha conectando os valores exatos (Cria o efeito zig-zag)
        ax.plot([x_cc, x_bp, x_mf], [0, 1, 2], color=c, linewidth=2.5, zorder=2)
        
        # Pontos de dispersão (Círculos) nos locais exatos
        ax.scatter([x_cc, x_bp, x_mf], [0, 1, 2], color=c, s=220, zorder=3, edgecolors='white', linewidth=2)
        
        # Conector Suave (Drop-down) a partir de CC (y=0) para o rótulo
        if abs(target_x - x_cc) < 0.005:
            # Se for praticamente reto
            ax.plot([x_cc, x_cc], [0, y_label + 0.15], color=c, linewidth=2, zorder=2)
        else:
            # Curva de Bézier a partir do ponto CC exato
            pp1 = (x_cc, 0)
            pp2 = (x_cc, -0.3)
            pp3 = (target_x, -0.3)
            pp4 = (target_x, y_label + 0.15)
            
            verts = [pp1, pp2, pp3, pp4]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            patch = mpatches.PathPatch(path, facecolor='none', lw=2, edgecolor=c, zorder=2)
            ax.add_patch(patch)
        
        # Caixas de Texto (Labels)
        ax.text(target_x, y_label, display_name,
                ha='center', va='center',
                fontsize=8, #fontweight='medium',
                color='black',
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=c, lw=1, alpha=1.0),
                zorder=4)

    # 5. Adição dos Textos Estáticos (Títulos e Ontologias)
    ax.text(-0.08, 2, 'MF', fontsize=10, fontweight='bold', ha='right', va='center', color='black')
    ax.text(-0.08, 1, 'BP', fontsize=10, fontweight='bold', ha='right', va='center', color='black')
    ax.text(-0.08, 0, 'CC', fontsize=10, fontweight='bold', ha='right', va='center', color='black')

    #fig.suptitle("Simplified Evidence-Specific Weight Mappings Across Gene Ontologies", 
    #             fontsize=22, fontweight='bold', y=1.02)
    #ax.set_title("(Illustrative Conceptual View with Distinct Color Encoding)", 
    #             fontsize=18, pad=30, color='#333333')

    # 5.5 Adição da Régua Superior (Top Ruler)
    ruler_y = 2.5 # Posição Y da régua (acima da barra MF)
    # Linha principal da régua
    ax.plot([0, 1], [ruler_y, ruler_y], color='#555555', linewidth=2, zorder=1)

    # Valores que queremos delimitar
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]

    for t in ticks:
        # Pequenos traços (ticks) apontando para cima
        ax.plot([t, t], [ruler_y, ruler_y + 0.08], color='#555555', linewidth=2, zorder=1)
        
        # Textos dos valores (0.00, 0.25, etc.)
        ax.text(t, ruler_y + 0.14, f"{t:.2f}", ha='center', va='bottom', 
                fontsize=10, color='#333333', fontweight='bold')
        
        # Opcional (Altamente Recomendado): Linhas tracejadas verticais muito sutis 
        # descendo da régua pelo gráfico para guiar o olhar até os pontos
        ax.plot([t, t], [0, ruler_y], color='#cccccc', linestyle='--', linewidth=1.2, zorder=0, alpha=0.6)

    # 6. Limpeza e Finalização do Eixo
    ax.axis('off') # Remove réguas e espinhas do gráfico padrão
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.9, 2.8)
    ax.set_title("C: Optimized Soft Labeling Scheme",
                 fontsize=12, fontweight='bold', y=1.02)

def get_marker(alg_name: str) -> str:
    """Case-insensitive marker lookup for algorithm families."""
    if alg_name in STR_MARKER_MAP:
        return STR_MARKER_MAP[alg_name]
    lower_map = {k.lower(): v for k, v in STR_MARKER_MAP.items()}
    return lower_map.get(str(alg_name).lower(), "o")

def parse_test_id(val) -> float:
    """Safely parse timestamp/test order, handling None and NaN values."""
    if val is None or pd.isna(val):
        return -1.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return -1.0


def _decollide_y_positions(y_vals, min_gap):
    """Adjust vertical label positions slightly so end-of-line annotations do not overlap."""
    if len(y_vals) <= 1:
        return list(y_vals)
    order = np.argsort(y_vals)
    adjusted = np.array(y_vals, dtype=float)[order]
    for i in range(1, len(adjusted)):
        if adjusted[i] - adjusted[i - 1] < min_gap:
            adjusted[i] = adjusted[i - 1] + min_gap
    # Center the group shift around original mean
    shift = np.mean(adjusted) - np.mean(y_vals)
    adjusted -= shift
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result.tolist()


def prepare_cumulative_df(df_all: pd.DataFrame, y_col: str = "Sort Score") -> pd.DataFrame:
    """Pivots and computes per-ontology and overall cumulative maximum scores."""
    df = df_all.copy()
    df["test_order"] = df["test_order"].apply(parse_test_id)

    print(df)
    print(df["test_order"])

    cum_df = []
    for group_tp, group_df in df.groupby(by=["Algorithm", "Ontology"]):
        overalls = set([0.0])
        timeline_sorted = group_df.sort_values(by="test_order")
        sort_scores = timeline_sorted[y_col].values
        cummax = []
        for v in sort_scores:
            overalls.add(v)
            cummax.append(max(overalls))
        timeline_sorted["cummax_score"] = cummax
        cum_df.append(timeline_sorted)

    cum_df = pd.concat(cum_df)
    mf_df = cum_df[cum_df["Ontology"] == "MF"]
    bp_df = cum_df[cum_df["Ontology"] == "BP"]
    cc_df = cum_df[cum_df["Ontology"] == "CC"]

    return mf_df, bp_df, cc_df

def plot_cumulative_performance_by_ontology(
    axes,
    mf_df: pd.DataFrame, bp_df: pd.DataFrame, cc_df: pd.DataFrame,
    color_dict: dict = None,
    y_col: str = "Sort Score",
):
    """Plots cumulative maximum Sort Score broken down per Ontology (MF, BP, CC) to match best_results3.tsv directly."""
    bg_color = "#FCFBF9"
    ontologies = ["MF", "BP", "CC"]

    dfs = {
        "MF": mf_df,
        "BP": bp_df,
        "CC": cc_df
    }

    '''fig, axes = plt.subplots(1, len(ontologies), figsize=(4 * len(ontologies), 5), sharey=False)
    if len(ontologies) == 1:
        axes = [axes]
    fig.patch.set_facecolor(bg_color)'''

    algorithms = list(mf_df["Algorithm"].unique())
    #if color_dict is None:
    #    cmap = sns.color_palette("deep", len(algorithms))
    #    color_dict = dict(zip(algorithms, cmap))

    max_x = 72
    for ax, ont in zip(axes, ontologies):
        pivot_df = dfs[ont]
        pivot_df = pivot_df[pivot_df["test_order"] < max_x]
        ax.set_facecolor(bg_color)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.7)
        ax.xaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.7)
        ax.set_axisbelow(True)

        col = "cummax_score"
        endpoints = []
        for name, group in pivot_df.groupby("Algorithm", sort=False):
            x = group["test_order"].values
            y = group[col].values * 100.0
            marker = get_marker(name)

            ax.plot(x, y, color=MARKER_COLOR_MAP[name], linewidth=2.2, alpha=0.8, zorder=3)
            ax.plot(x[-1], y[-1], marker=marker, color=MARKER_COLOR_MAP[name], markersize=8, zorder=4)
            endpoints.append((name, x[-1], y[-1]))

        if endpoints:
            y_vals = [ep[2] for ep in endpoints]
            y_span = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 1.0
            adj_y = _decollide_y_positions(y_vals, min_gap=max(0.45, y_span * 0.04))
            for (name, end_x, end_y), label_y in zip(endpoints, adj_y):
                ax.annotate(
                    f"{end_y:.1f}%",
                    (end_x, label_y),
                    xytext=(6, 0),
                    textcoords="offset points",
                    fontsize=8,
                    va="center",
                    color="#333333",
                    path_effects=[pe.withStroke(linewidth=2.5, foreground=bg_color)],
                )

        ax.set_title(f"{ont}", fontsize=10, weight="bold", pad=12)
        if ax is axes[1]:
            ax.set_xlabel("Parameter Configurations Evaluated", fontweight="bold", color="#444444", fontsize=10, labelpad=8)
        if ax is axes[0]:
            ax.set_ylabel(f"Max Overall Score Seen So Far (%)", fontweight="bold", color="#444444", fontsize=10, labelpad=8)
        ax.set_xlim(0, max_x + (max_x * 0.12))
        ax.set_ylim(45, 76.5)

    handles = [
        plt.Line2D([0], [0], color=MARKER_COLOR_MAP[f], marker=get_marker(f), lw=3, markersize=8, label=f)
        for f in algorithms
    ]
    axes[-1].legend(handles=handles, frameon=True, loc="lower right", fontsize=8,
        title="Evidence Representation\nStrategy", title_fontproperties={"weight": "bold"})

    axes[1].text(
        0.5,
        1.15,
        "A: Timeline of Parameter Optimization",
        transform=axes[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

def plot_cumulative_performance(axes, df_all: pd.DataFrame, y_col: str = "Sort Score"):
    """Plots the cumulative maximum overall score across hyperparameter tests to show diminishing returns."""
    '''bg_color = "#FCFBF9"
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    # Clean up axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.7)
    ax.xaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.7)
    ax.set_axisbelow(True)'''

    mf_df, bp_df, cc_df = prepare_cumulative_df(df_all, y_col=y_col)
    algorithms = list(mf_df["Algorithm"].unique())
    cmap = sns.color_palette("deep", len(algorithms))
    color_dict = dict(zip(algorithms, cmap))

    plot_cumulative_performance_by_ontology(
        axes, mf_df, bp_df, cc_df, color_dict=color_dict, y_col=y_col
    )


if __name__ == "__main__":
    if ".tsv" in sys.argv[1]:
        all_results_path = sys.argv[1]
        evi_exp_res_dir = os.path.dirname(all_results_path)
    else:
        evi_exp_res_dir = sys.argv[1]
        all_results_path = os.path.join(evi_exp_res_dir, "all_results_wtimes.tsv")
    df = pd.read_csv(all_results_path, sep="\t")

    analyze_evi_sortings(df, evi_exp_res_dir)

    df_best_recalc = []
    for group_tp, group_df in df.groupby(["Algorithm", "Ontology"]):
        group_df = group_df.sort_values(by="Sort Score", ascending=False)
        top_test = group_df.iloc[[0]]
        top_test = {
            k: top_test[k].item()
            for k in ["Algorithm", "Ontology", "Sort Score", "OWA Score", "CWA Score", "Parameters"]
        }
        df_best_recalc.append(top_test)
        print(group_tp, len(group_df))
    df_best_recalc = pd.DataFrame(df_best_recalc)
    df_best_recalc = df_best_recalc.sort_values(by="Sort Score", ascending=False)
    df_best_recalc.to_csv(os.path.join(evi_exp_res_dir, "best_results3.tsv"), sep="\t", index=False)

    # Also save an overall summary table averaged across ontologies (MF, BP, CC)
    # so the table can be compared 1-to-1 against the overall cumulative performance plot.
    df_overall_summary = (
        df_best_recalc.groupby("Algorithm", as_index=False)[["Sort Score", "OWA Score", "CWA Score"]]
        .mean()
        .sort_values(by="Sort Score", ascending=False)
    )
    df_overall_summary.to_csv(os.path.join(evi_exp_res_dir, "best_results3_overall_mean.tsv"), sep="\t", index=False)
    print("\nBest Sort Score Averaged Across Ontologies (MF, BP, CC):")
    print(df_overall_summary.to_string(index=False))

    bg_color = "#FCFBF9"
    fig = plt.figure(figsize=(13, 9.5), facecolor=bg_color)

    # Outer 2-row grid
    outer_gs = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=[1, 1], hspace=0.32
    )

    # Row 1: 3 equal columns for MF, BP, CC
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer_gs[0], wspace=0.22
    )
    axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(3)]

    # Row 2: 2 columns with 40% / 60% width ratio for B and C
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_gs[1], width_ratios=[4, 6], wspace=0.025
    )
    ax_b = fig.add_subplot(gs_bot[0, 0])
    ax_c = fig.add_subplot(gs_bot[0, 1])

    plot_cumulative_performance(axes_top, df)
    scatter_owa_cwa_scores(ax_b, df_best_recalc, evi_exp_res_dir)
    plot_evi_values_scale(ax_c, df_best_recalc, evi_exp_res_dir)

    fig.savefig(
        os.path.join(evi_exp_res_dir, "best_results.png"),
        dpi=400,
        facecolor=bg_color,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(evi_exp_res_dir, "best_results.svg"),
        dpi=400,
        facecolor=bg_color,
        bbox_inches="tight",
    )
    plt.close(fig)