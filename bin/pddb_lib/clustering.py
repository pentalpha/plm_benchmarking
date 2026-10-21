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