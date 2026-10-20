from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
import numpy as np
import json
import os
import pandas as pd

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