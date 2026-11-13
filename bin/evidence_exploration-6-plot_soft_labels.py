import os
import sys
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import seaborn as sns
from scipy import stats
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    mutual_info_regression,
    f_regression,
    chi2,
)
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import KBinsDiscretizer
from mlxtend.frequent_patterns import apriori, association_rules

# =============================================================================
# CONSTANTS & LABEL MAPPINGS
# =============================================================================
SOFT_LABEL_COLORS = {
    "phylo": "navy",
    "curated": "cornflowerblue",
    "conditional_not": "mediumvioletred",
    "Random False Val": "yellow",
    "curated_not": "red",
    "derived_not": "firebrick",
    "phylo_not": "maroon",
}

NEG_LABELS = [
    "conditional_not",
    "Random False Val",
    "curated_not",
    "derived_not",
    "phylo_not",
]

POS_LABELS = [
    "phylo",
    "curated",
]

REAL_NEG_COLS = ["derived_not", "phylo_not", "curated_not"]
VIRT_NEG_COLS = ["Random False Val", "conditional_not"]

GOOD_LEVELS_PERCENTILE = {
    "VERY_GOOD": 0.9,
    "GOOD": 0.8,
    "MODERATE": 0.7,
}

BAD_LEVELS_PERCENTILE = {
    "MEDIOCRE": 0.7,
    "BAD": 0.3,
    "VERY_BAD": 0.2,
    "HORRIBLE": 0.1,
}

# Core symmetric pairwise comparisons where both True and False (NOT_) polarities
# represent direct ordering reversals (e.g., CU+ < PY+ vs PY+ <= CU+).
CORE_SYMMETRIC_BOOL_COLS = [
    "neg_below_pos",
    "phylo_pos_gt_curated_pos",
    "virtual_neg_higher",
    "curated_neg_lt_phylo_neg",
    "random_false_val_lt_conditional_not",
]

CONSEQUENT_COLORS = {
    "VERY_GOOD": "#1b7837",
    "GOOD": "#4daf4a",
    "NOT_VERY_GOOD": "#762a83",
    "MODERATE": "#99d8c9",
    "MEDIOCRE": "#e68a00",
    "BAD": "#de2d26",
    "VERY_BAD": "#a50f15",
    "HORRIBLE": "#67000d",
}

thin_vertical_rect = [(-1, -5), (1, -5), (1, 5), (-1, 5)]
thin_horizontal_rect = [(-5, -1), (5, -1), (5, 1), (-5, 1)]


def parse_label_name(name):
    """Map raw column names to readable display names."""
    alt_names = {
        "phylo": "Phylogenetic+",
        "curated": "Curated+",
        "conditional_not": "Conditional-",
        "Random False Val": "Random-",
        "curated_not": "Curated-",
        "derived_not": "Derived-",
        "phylo_not": "Phylogenetic-",
        "pos_neg_inversion_count": "Pos/Neg Inversion Count",
        "pos_to_virtual_margin": "Pos-to-Virtual Margin",
        "virtual_to_real_margin": "Virtual-to-Real Margin",
        "neg_virt_real_center_diff": "Neg Virt-to-Real Center Diff",
        "real_neg_mean": "Real Negatives Mean",
        "real_neg_spread": "Real Negatives Spread",
        "neg_below_pos": "All Neg < All Pos",
        "phylo_pos_gt_curated_pos": "CU+ < PY+",
        "virtual_neg_higher": "Real Negs < Virtual Negs",
        "curated_neg_lt_phylo_neg": "CU- < PY-",
        "random_false_val_lt_conditional_not": "RA- < CO-",
        "derived_neg_above_pos": "DE- >= Min(Pos)",
        "derived_neg_is_highest_real_neg": "DE- == Max(Real Negs)",
        "virtual_neg_anchored_lowest": "Min(Virt Neg) == Min(All Neg)",
        "severe_pos_neg_inversion": "Pos/Neg Inversions >= 3",
        "derived_neg_lowest": "DE- == Min(All Negs)",
        "derived_neg_le_012": "DE- <= 0.12",
        "real_negs_compressed_low": "Max(Real Negs) <= 0.22",
        "neg_virt_real_center_diff_large": "Neg Virt-to-Real Center Diff >= 0.05",
        "pos_to_virtual_margin_relevant": "Pos-to-Virtual Margin > 0.09",
        "strict_3tier_hierarchy": "Strict 3-Tier Hierarchy",
        "stricter_3tier_hierarchy": "Stricter 3-Tier Hierarchy",
        "strict_4tier_hierarchy": "Strict 4-Tier Hierarchy",
        "strict_5tier_hierarchy": "Strict 5-Tier Hierarchy",
        "soft_tiers_1": "Soft Tiers 1",
        "IS_MF": "Ontology = MF",
        "IS_CC": "Ontology = CC",
        "IS_BP": "Ontology = BP",
    }
    return alt_names.get(name, name)


def parse_label_name_small(name):
    """Map raw soft label names to compact two-letter codes."""
    alt_names = {
        "phylo": "PY+",
        "curated": "CU+",
        "conditional_not": "CO-",
        "Random False Val": "RA-",
        "curated_not": "CU-",
        "derived_not": "DE-",
        "phylo_not": "PY-",
    }
    return alt_names.get(name, name)


def format_compact_item(item_str):
    """Format an antecedent item into a concise, readable token for plot axes."""
    short_map = {
        "neg_below_pos": "Neg<Pos",
        "NOT_neg_below_pos": "NOT(Neg<Pos)",
        "phylo_pos_gt_curated_pos": "CU+<PY+",
        "NOT_phylo_pos_gt_curated_pos": "NOT(CU+<PY+)",
        "virtual_neg_higher": "RealNeg<VirtNeg",
        "NOT_virtual_neg_higher": "NOT(RealNeg<VirtNeg)",
        "curated_neg_lt_phylo_neg": "CU-<PY-",
        "NOT_curated_neg_lt_phylo_neg": "NOT(CU-<PY-)",
        "random_false_val_lt_conditional_not": "RA-<CO-",
        "NOT_random_false_val_lt_conditional_not": "NOT(RA-<CO-)",
        "derived_neg_above_pos": "DE->=MinPos",
        "derived_neg_is_highest_real_neg": "DE-==MaxRealNeg",
        "virtual_neg_anchored_lowest": "VirtNeg==MinNeg",
        "severe_pos_neg_inversion": "Inversions>=3",
        "derived_neg_lowest": "DE-==MinNeg",
        "derived_neg_le_012": "DE-<=0.12",
        "real_negs_compressed_low": "RealNegs<=0.22",
        "strict_4tier_hierarchy": "Strict4Tier",
        "strict_5tier_hierarchy": "Strict5Tier",
    }
    if item_str in short_map:
        return short_map[item_str]
    return (
        item_str.replace("pos_to_virtual_margin", "PosVirtMargin")
        .replace("virtual_to_real_margin", "VirtRealMargin")
        .replace("neg_virt_real_center_diff", "NegVirtRealDiff")
        .replace("pos_neg_inversion_count", "InversionCnt")
        .replace("real_neg_mean", "RealNegMean")
        .replace("real_neg_spread", "RealNegSpread")
    )


def format_pvalue_stars(p_val):
    """Return compact significance stars for a p-value."""
    if p_val < 1e-4:
        return "****"
    if p_val < 1e-3:
        return "***"
    if p_val < 1e-2:
        return "**"
    if p_val < 0.05:
        return "*"
    return "ns"


# =============================================================================
# 1. DATA LOADING & PREPROCESSING
# =============================================================================
def resolve_paths_and_load_data(arg_path):
    """Load results TSV, filter to MF/CC/BP, and min-max normalize scores per ontology."""
    if ".tsv" in arg_path:
        all_results_path = arg_path
        evi_exp_res_dir = os.path.dirname(all_results_path)
    else:
        evi_exp_res_dir = arg_path
        all_results_path = os.path.join(
            evi_exp_res_dir, "soft_optimization_test", "all_results.tsv"
        )

    df = pd.read_csv(all_results_path, sep="\t")
    df = df[df["Ontology"].isin(["MF", "CC", "BP"])].copy()
    print("Loaded columns:", df.columns.tolist())

    score_cols = ["Sort Score", "OWA Score", "CWA Score"]
    ont_normalized_chunks = []
    for _, ont_rows in df.groupby("Ontology"):
        new_chunk = ont_rows.copy()
        for sc_col in score_cols:
            if sc_col in new_chunk.columns:
                min_val = new_chunk[sc_col].min()
                max_val = new_chunk[sc_col].max()
                denom = (max_val - min_val) if max_val != min_val else 1.0
                new_chunk[f"{sc_col} Norm"] = (
                    (new_chunk[sc_col] - min_val) / denom
                ).fillna(0.0)
        ont_normalized_chunks.append(new_chunk)

    df = pd.concat(ont_normalized_chunks, ignore_index=True)
    return df, evi_exp_res_dir


# =============================================================================
# 2. SCATTER PLOT OF RAW SOFT LABELS VS. SCORE
# =============================================================================
def plot_soft_label_scatter(df, evi_exp_res_dir, score_cols=None):
    """Generate the soft_optimization_test.png scatter plot."""
    if score_cols is None:
        score_cols = ["Sort Score"]

    df_plot = []
    for _, row in df.iterrows():
        for label_name, labelc in SOFT_LABEL_COLORS.items():
            new_row = {
                "legend": parse_label_name(label_name),
                "color": labelc,
                "value": row[label_name],
                "Ontology": row["Ontology"],
                "OWA Score": row["OWA Score"],
                "CWA Score": row["CWA Score"],
                "Sort Score": row["Sort Score"],
            }
            df_plot.append(new_row)
    df_plot = pd.DataFrame(df_plot)

    fig, axes = plt.subplots(1, len(score_cols), figsize=(10, 8))
    axes_list = [axes[i] for i in range(len(score_cols))] if len(score_cols) > 1 else [axes]

    for label, label_df in df_plot.groupby("legend"):
        for ax, score_col in zip(axes_list, score_cols):
            ax.scatter(
                label_df["value"],
                label_df[score_col] * 100.0,
                s=280,
                alpha=0.75,
                c=label_df["color"],
                marker=thin_horizontal_rect,
                label=label,
            )
            ax.set_xlabel("Soft Label Value", fontsize=10, fontweight="bold")
            score_label = score_col if score_col != "Sort Score" else "Overall Score"
            ax.set_ylabel(f"{score_label} (%)", fontsize=10, fontweight="bold")
            ax.set_title(f"Soft Label Value vs {score_col}", fontsize=12, fontweight="bold")
            ax.grid(True, linestyle="--", alpha=0.5, zorder=0)
            ax.set_xlim(0, 1)

    axes_list[-1].legend(loc="lower right", bbox_to_anchor=(0.98, 0.02))
    fig.tight_layout()
    out_path = os.path.join(evi_exp_res_dir, "soft_optimization_test.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Saved scatter figure to: {out_path}]")


# =============================================================================
# 3. FEATURE ENGINEERING (QUALITY TIERS, BOOLEAN PATTERNS, MARGINS & SEQUENCES)
# =============================================================================
def engineer_features(df):
    import json
    """Engineer quality targets, boolean hierarchy patterns, continuous margins, and sequences."""
    target_score_col = "Sort Score Norm" if "Sort Score Norm" in df.columns else "Sort Score"

    for new_col, min_percentile in GOOD_LEVELS_PERCENTILE.items():
        qt_val = df[target_score_col].quantile(min_percentile)
        print(f"Percentile for {new_col} (>= {min_percentile}) is {qt_val:.4f}")
        df[new_col] = df[target_score_col] >= qt_val

    df["NOT_VERY_GOOD"] = df[target_score_col] < df[target_score_col].quantile(0.9)

    for new_col, max_percentile in BAD_LEVELS_PERCENTILE.items():
        qt_val = df[target_score_col].quantile(max_percentile)
        print(f"Percentile for {new_col} (< {max_percentile}) is {qt_val:.4f}")
        df[new_col] = df[target_score_col] < qt_val

    # Core hierarchy comparisons
    df["neg_below_pos"] = df[NEG_LABELS].max(axis=1) < df[POS_LABELS].min(axis=1)
    #df["phylo_pos_gt_curated_pos"] = df["curated"] < df["phylo"]
    df["virtual_neg_higher"] = df[REAL_NEG_COLS].max(axis=1) < df[VIRT_NEG_COLS].min(axis=1)
    #df["curated_neg_lt_phylo_neg"] = df["curated_not"] < df["phylo_not"]
    #df["random_false_val_lt_conditional_not"] =  (df["conditional_not"] - df["Random False Val"]) >= -0.02

    # Label ordering sequences
    sequences_all, sequences_pos, sequences_neg = [], [], []
    for _, row in df.iterrows():
        tuples_all = sorted([(row[lab], lab) for lab in NEG_LABELS + POS_LABELS], key=lambda x: x[0])
        sequences_all.append("<".join(parse_label_name_small(t[1]) for t in tuples_all))

        tuples_pos = sorted([(row[lab], lab) for lab in POS_LABELS], key=lambda x: x[0])
        sequences_pos.append("<".join(parse_label_name_small(t[1]) for t in tuples_pos))

        tuples_neg = sorted([(row[lab], lab) for lab in NEG_LABELS], key=lambda x: x[0])
        sequences_neg.append("<".join(parse_label_name_small(t[1]) for t in tuples_neg))

    df["label_sequence"] = sequences_all
    df["label_sequence_pos"] = sequences_pos
    df["label_sequence_neg"] = sequences_neg

    # Failure-mode and top-tier determinant features
    #df["derived_neg_above_pos"] = df["derived_not"] >= df[POS_LABELS].min(axis=1)
    #df["derived_neg_is_highest_real_neg"] = df["derived_not"] == df[REAL_NEG_COLS].max(axis=1)
    #df["virtual_neg_anchored_lowest"] = df[VIRT_NEG_COLS].min(axis=1) == df[NEG_LABELS].min(axis=1)
    df["pos_neg_inversion_count"] = sum(
        (df[n] >= df[p]).astype(int) for n in NEG_LABELS for p in POS_LABELS
    )
    #df["severe_pos_neg_inversion"] = df["pos_neg_inversion_count"] >= 3

    #df["derived_neg_lowest"] = df["derived_not"] == df[NEG_LABELS].min(axis=1)
    #df["derived_neg_le_012"] = df["derived_not"] <= 0.12
    #df["real_negs_compressed_low"] = df[REAL_NEG_COLS].max(axis=1) <= 0.22
    '''df["strict_3tier_hierarchy"] = (
        df["neg_below_pos"]
        & df["virtual_neg_higher"]
    )
    df["strict_4tier_hierarchy"] = (
        df["strict_3tier_hierarchy"]
        & df["random_false_val_lt_conditional_not"]
    )'''

    # Continuous margins and spreads
    df["pos_to_virtual_margin"] = df[POS_LABELS].min(axis=1) - df[VIRT_NEG_COLS].max(axis=1)
    df["pos_to_virtual_margin_relevant"] = df["pos_to_virtual_margin"] >= 0.126
    #df["stricter_3tier_hierarchy"] = df["pos_to_virtual_margin_relevant"] & df["strict_3tier_hierarchy"]
    df["virtual_to_real_margin"] = df[VIRT_NEG_COLS].min(axis=1) - df[REAL_NEG_COLS].max(axis=1)
    df["real_neg_mean"] = df[REAL_NEG_COLS].mean(axis=1)
    df["virtual_neg_mean"] = df[VIRT_NEG_COLS].mean(axis=1)
    df["pos_far_from_middle"] = df[POS_LABELS].min(axis=1) >= 0.56
    df["neg_virt_real_center_diff"] = df["virtual_neg_mean"] - df["real_neg_mean"]
    #df["neg_virt_real_center_diff_large"] = df["neg_virt_real_center_diff"] >= 0.16
    df["real_neg_spread"] = df[REAL_NEG_COLS].max(axis=1) - df[REAL_NEG_COLS].min(axis=1)
    df["derived_neg_below_virtual_neg_mean"] = df["derived_not"] < df["virtual_neg_mean"]

    '''df["soft_tiers_1"] = (df["pos_to_virtual_margin_relevant"] 
        & df["derived_neg_below_virtual_neg_mean"] & df["random_false_val_lt_conditional_not"])'''

    all_neg_max = df[NEG_LABELS].max(axis=1)
    pos_min = df[POS_LABELS].min(axis=1)
    virt_min = df[VIRT_NEG_COLS].min(axis=1)
    all_neg_max = df[NEG_LABELS].max(axis=1)

    # 1. Separacao natural de polaridade e ancoragem do negativo dominante (DE-)
    df["natural_polarity_split"] = (all_neg_max < 0.50) & (pos_min > 0.50)
    df["derived_neg_below_virtual_negs"] = df["derived_not"] <= virt_min
    df["random_false_val_lt_conditional_not"] = df["Random False Val"] <= df["conditional_not"]

    # Opcao A (30 runs: 28 VERY_GOOD, 2 GOOD, 0 MODERATE, 0 MEDIOCRE -> 93.3% VG recall, 100% >=GOOD)
    df["soft_tiers_strict_chain"] = (
        df["neg_below_pos"]
        & df["derived_neg_below_virtual_negs"]
        & df["random_false_val_lt_conditional_not"]
    )

    # Opcao B (36 runs: 29 VERY_GOOD, 7 GOOD, 0 MODERATE, 0 MEDIOCRE -> 96.7% VG recall, 100% >=GOOD)
    df["soft_tiers_2"] = (
        df["natural_polarity_split"]
        & df["derived_neg_below_virtual_negs"]
        & (df["random_false_val_lt_conditional_not"] | df["virtual_neg_higher"])
    )

    df["real_neg_median"] = df[REAL_NEG_COLS].median(axis=1)
    df["conditional_neg_gt_real_neg_median"] = (
        df["conditional_not"] > df["real_neg_median"]
    )
    df["derived_neg_le_conditional_neg"] = (
        df["derived_not"] <= df["conditional_not"]
    )

    df["conditional_neg_top_anchor"]= (
        df["natural_polarity_split"]
        & df["derived_neg_le_conditional_neg"]
        & df["random_false_val_lt_conditional_not"]
        & df["conditional_neg_gt_real_neg_median"]
    )

    for ont in ["MF", "CC", "BP"]:
        if ont in df["Ontology"].unique():
            df[f"IS_{ont}"] = df["Ontology"] == ont

    n_soft_tier1 = len(df[df["soft_tiers_2"]])
    n_not_soft_tier1 = len(df[~df["soft_tiers_2"]])
    n_very_good = len(df[df["VERY_GOOD"]])
    not_soft_tier_but_good = df[~df["soft_tiers_2"] & df["VERY_GOOD"]]
    in_soft_tier_but_not_good = df[df["soft_tiers_2"] & ~df["VERY_GOOD"]]
    in_soft_tier_but_not_bad = df[df["soft_tiers_2"] & ~df["GOOD"]]
    in_soft_tier_but_mediocre = df[df["soft_tiers_2"] & df["MEDIOCRE"]]
    n_not_soft_tier_but_good = len(not_soft_tier_but_good)
    n_in_soft_tier_but_not_good = len(in_soft_tier_but_not_good)
    n_in_soft_tier_but_not_bad = len(in_soft_tier_but_not_bad)
    n_in_soft_tier_but_mediocre = len(in_soft_tier_but_mediocre)

    print(f"{n_soft_tier1} in soft tier 1 ({n_soft_tier1/len(df):.2%})")
    print(f"{(n_very_good-n_not_soft_tier_but_good)/n_very_good:.2%} of VERY_GOOD are in soft tier 1")
    print(f"{n_in_soft_tier_but_not_good} in soft tier 1 but not VERY_GOOD ({n_in_soft_tier_but_not_good/n_soft_tier1:.2%})")
    print(f"{n_in_soft_tier_but_not_bad} in soft tier 1 but not GOOD ({n_in_soft_tier_but_not_bad/n_soft_tier1:.2%})")
    print(f"{n_in_soft_tier_but_mediocre} in soft tier 1 and MEDIOCRE ({n_in_soft_tier_but_mediocre/n_soft_tier1:.2%})")
    print(f"N={n_not_soft_tier_but_good} examples where not soft_tiers_2 but VERY_GOOD is true ({n_not_soft_tier_but_good/n_very_good:.2%})")
    
    relevant_to_print = [
        "Ontology", target_score_col] + REAL_NEG_COLS + VIRT_NEG_COLS + POS_LABELS + ["pos_to_virtual_margin", "neg_virt_real_center_diff"]
    print("\nNot soft tier 2 but VERY_GOOD:")
    not_soft_tier_but_good.sort_values(by=target_score_col, ascending=False, inplace=True)
    for idx, row in not_soft_tier_but_good.iterrows():
        row_dict = {col: row[col] for col in relevant_to_print}
        print(json.dumps(row_dict, indent=2))
    
    print("In soft tier 2 and MEDIOCRE:")
    in_soft_tier_but_mediocre.sort_values(by=target_score_col, ascending=False, inplace=True)
    for idx, row in in_soft_tier_but_mediocre.iterrows():
        row_dict = {col: row[col] for col in relevant_to_print}
        print(json.dumps(row_dict, indent=2))

    #quit(1)
        
    bool_pattern_cols = [
        c for c in [
            "IS_MF",
            "IS_CC",
            "IS_BP",
            "soft_tiers_strict_chain",
            "soft_tiers_2",
            "conditional_neg_top_anchor",
            "neg_below_pos",
            "natural_polarity_split",
            "pos_to_virtual_margin_relevant",
            "derived_neg_below_virtual_negs",
            "virtual_neg_higher",
            "random_false_val_lt_conditional_not",
            "conditional_neg_gt_real_neg_median",
            "derived_neg_le_conditional_neg",
            #"curated_neg_lt_phylo_neg",
            #"derived_neg_above_pos",
            #"derived_neg_is_highest_real_neg",
            #"virtual_neg_anchored_lowest",
            #"severe_pos_neg_inversion",
            #"derived_neg_lowest",
            #"derived_neg_le_012",
            #"real_negs_compressed_low",
            #"neg_virt_real_center_diff_large",
            #"strict_3tier_hierarchy",
            #"stricter_3tier_hierarchy",
            #"strict_4tier_hierarchy",
            #"soft_tiers_1",
        ]
        if c in df.columns
    ]

    cont_engineered_cols = [
        #"pos_neg_inversion_count",
        "pos_to_virtual_margin",
        "virtual_to_real_margin",
        #"neg_virt_real_center_diff",
        #"real_neg_mean",
        "real_neg_median",
        #"real_neg_spread",
    ]

    for col in bool_pattern_cols:
        df[col] = df[col].astype(bool)
        df[f"NOT_{col}"] = ~df[col]
        print(f"Mean for {col}: {df[col].mean():.4f}")

    for col in cont_engineered_cols:
        print(f"Mean for {col}: {df[col].mean():.4f}")

    quality_cols = list(GOOD_LEVELS_PERCENTILE.keys()) + list(BAD_LEVELS_PERCENTILE.keys())
    return df, quality_cols, bool_pattern_cols, cont_engineered_cols, target_score_col


# =============================================================================
# 4. TRANSACTION MATRIX & REDUNDANCY / IMPLICATION DETECTION
# =============================================================================
def build_apriori_transactions(
    df, bool_pattern_cols, cont_engineered_cols, quality_cols, min_seq_support=0.03
):
    """
    Build boolean transaction dataframe and deduplicate strictly identical columns.
    Only include NOT_ polarities for the 5 core symmetric hierarchy comparisons so
    high-prevalence complements of rare indicators (like NOT_strict_4tier_hierarchy
    or NOT_IS_MF) do not generate trivial noise rules.
    """
    raw_trans = pd.DataFrame(index=df.index)

    # 1. Explicit boolean pattern features (+ NOT_ polarity for symmetric ordering features)
    for col in bool_pattern_cols:
        raw_trans[col] = df[col].astype(bool)
        if col in CORE_SYMMETRIC_BOOL_COLS:
            raw_trans[f"NOT_{col}"] = ~df[col].astype(bool)

    # 2. Discretize continuous engineered features into median High/Low boolean items
    kbd = KBinsDiscretizer(
        n_bins=2, encode="ordinal", strategy="quantile", subsample=None
    )
    binned_cont = kbd.fit_transform(df[cont_engineered_cols])
    for idx, col in enumerate(cont_engineered_cols):
        edges = kbd.bin_edges_[idx]
        if len(edges) >= 3:
            cut_val = float(edges[1])
            high_col = f"{col}_gt_{cut_val:.2f}"
            low_col = f"{col}_le_{cut_val:.2f}"
            raw_trans[high_col] = binned_cont[:, idx] >= 1.0
            raw_trans[low_col] = binned_cont[:, idx] < 1.0

    # 3. Frequent negative sequences
    for seq_val, count in df["label_sequence_neg"].value_counts().items():
        if count / len(df) >= min_seq_support:
            raw_trans[f"seq_neg={seq_val}"] = df["label_sequence_neg"] == seq_val

    # 4. Deduplicate columns that have identical boolean vectors across all rows
    trans_df = pd.DataFrame(index=df.index)
    seen_vectors = {}
    dropped_aliases = []
    for col in raw_trans.columns:
        vec_key = tuple(raw_trans[col].tolist())
        if vec_key in seen_vectors:
            dropped_aliases.append((col, seen_vectors[vec_key]))
        else:
            seen_vectors[vec_key] = col
            trans_df[col] = raw_trans[col]

    if dropped_aliases:
        print("\n[Redundancy Check] Dropped identical alias columns before Apriori:")
        for alias_col, canon_col in dropped_aliases:
            print(f"  - {alias_col} == {canon_col} (100% identical rows)")

    antecedent_items = list(trans_df.columns)
    for q_col in quality_cols:
        trans_df[q_col] = df[q_col].astype(bool)

    return trans_df, antecedent_items


def find_redundant_feature_pairs(trans_df, antecedent_items):
    """
    Detect all pairs of antecedent columns (u, v) where u is only True when v is also True
    (i.e. u => v, meaning count(u & ~v) == 0). Any rule antecedent containing both u and v
    is logically redundant because u & v == u.
    """
    redundant_pairs = set()
    implication_list = []

    mat = trans_df[antecedent_items].to_numpy(dtype=bool)
    col_sums = mat.sum(axis=0)

    n_cols = len(antecedent_items)
    for i in range(n_cols):
        if col_sums[i] == 0:
            continue
        u = antecedent_items[i]
        u_vec = mat[:, i]
        for j in range(n_cols):
            if i == j or col_sums[j] == 0:
                continue
            v = antecedent_items[j]
            v_vec = mat[:, j]
            if not np.any(u_vec & ~v_vec):
                pair = tuple(sorted((u, v)))
                redundant_pairs.add(pair)
                implication_list.append((u, v, int(col_sums[i]), int(col_sums[j])))

    print(
        f"\n[Redundancy Check] Detected {len(implication_list)} directed logical implications "
        f"(u => v, i.e., u is only True when v is True) across {len(redundant_pairs)} feature pairs."
    )
    return redundant_pairs, implication_list


def filter_redundant_rules(rules_df, trans_df, redundant_pairs, cont_engineered_cols):
    """
    Remove all redundant association rules:
      1. Pairwise logical implication: antecedent contains (u, v) where u => v or v => u
         (e.g., neg_below_pos & strict_4tier_hierarchy).
      2. Multi-column subset implication (non-minimal generator): antecedent A has the exact
         same support count as a proper subset B < A (meaning B => A \\ B).
      3. Identical row-mask deduplication: if multiple antecedents select the exact same subset
         of rows across the dataset, keep only the single canonical (shortest, cleanest) antecedent.
      4. Semantic subset & sub-rule confidence dominance: for the same Consequent Q, if an
         antecedent A is a syntactic sub-rule of B OR its active row mask is a strict subset of
         another rule B (len(B) <= len(A)) whose confidence is already >= Confidence(A => Q),
         then A => Q adds no predictive gain over B => Q and is pruned.
      5. Cumulative tier deduplication: if A => Q_broad has the exact same match count as
         A => Q_strict (where Q_strict implies Q_broad, e.g. VERY_GOOD => GOOD => MODERATE),
         prune the redundant broader-tier copy.
    """
    if rules_df.empty:
        return rules_df

    raw_count = len(rules_df)

    # Step 1: Reject rules with pairwise implied columns (u => v) in antecedent
    def has_pairwise_implication(ant_set):
        if len(ant_set) <= 1:
            return False
        for u, v in combinations(sorted(ant_set), 2):
            if (u, v) in redundant_pairs:
                return True
        return False

    filtered = rules_df[~rules_df["antecedents"].apply(has_pairwise_implication)].copy()

    # Precompute exact boolean row mask and support count for every candidate antecedent
    unique_ants = list({r for r in filtered["antecedents"]})
    ant_to_mask_arr = {}
    ant_to_mask_tuple = {}
    ant_to_count = {}
    for ant in unique_ants:
        row_mask = trans_df[list(ant)].all(axis=1).to_numpy(dtype=bool)
        ant_to_mask_arr[ant] = row_mask
        ant_to_mask_tuple[ant] = tuple(row_mask.tolist())
        ant_to_count[ant] = int(row_mask.sum())

    # Step 2: Reject non-minimal generators where a proper subset B < ant has identical support
    def is_minimal_generator(ant):
        k = len(ant)
        if k <= 1:
            return True
        ant_cnt = ant_to_count[ant]
        for r_len in range(1, k):
            for sub in combinations(ant, r_len):
                if int(trans_df[list(sub)].all(axis=1).sum()) == ant_cnt:
                    return False
        return True

    filtered = filtered[filtered["antecedents"].apply(is_minimal_generator)].copy()

    # Step 3: Canonical antecedent selection per unique row-mask vector
    def antecedent_priority_key(ant):
        n_binned = sum(
            any(item.startswith(c + "_gt_") or item.startswith(c + "_le_") for c in cont_engineered_cols)
            for item in ant
        )
        n_neg = sum(item.startswith("NOT_") for item in ant)
        n_seq = sum(item.startswith("seq_neg=") for item in ant)
        return (len(ant), n_binned, n_seq, n_neg, tuple(sorted(ant)))

    mask_to_canonical_ant = {}
    for ant in {r for r in filtered["antecedents"]}:
        m_tup = ant_to_mask_tuple[ant]
        if (
            m_tup not in mask_to_canonical_ant
            or antecedent_priority_key(ant) < antecedent_priority_key(mask_to_canonical_ant[m_tup])
        ):
            mask_to_canonical_ant[m_tup] = ant

    filtered = filtered[
        filtered["antecedents"].apply(lambda a: mask_to_canonical_ant[ant_to_mask_tuple[a]] == a)
    ].copy()

    # Step 4: Prune rules dominated by a syntactic sub-rule OR a broader semantic superset rule
    #         (if B is broader than A with len(B) <= len(A) and Confidence(B => Q) >= Confidence(A => Q))
    kept_indices = []
    for cons_val, grp in filtered.groupby("Consequent"):
        grp_records = [
            (idx, row["antecedents"], float(row["confidence"]), ant_to_mask_arr[row["antecedents"]], ant_to_count[row["antecedents"]])
            for idx, row in grp.iterrows()
        ]
        for idx, ant, conf, mask_a, cnt_a in grp_records:
            k = len(ant)
            dominated = False
            # 4a. Check all syntactic proper subsets in trans_df
            if k > 1:
                for r_len in range(1, k):
                    for sub in combinations(ant, r_len):
                        sub_mask = trans_df[list(sub)].all(axis=1)
                        sub_n = int(sub_mask.sum())
                        sub_conf = (
                            float((sub_mask & trans_df[cons_val]).sum()) / sub_n
                            if sub_n > 0
                            else 0.0
                        )
                        if sub_conf >= conf - 1e-9:
                            dominated = True
                            break
                    if dominated:
                        break
            # 4b. Check semantic row-subset dominance against broader rules in the same consequent
            if not dominated:
                for other_idx, other_ant, other_conf, mask_b, cnt_b in grp_records:
                    if other_idx == idx:
                        continue
                    if cnt_b > cnt_a and len(other_ant) <= k and other_conf >= conf - 1e-9:
                        # Check if mask_a is only True when mask_b is True (mask_a subset of mask_b)
                        if not np.any(mask_a & ~mask_b):
                            dominated = True
                            break
            if not dominated:
                kept_indices.append(idx)

    filtered = filtered.loc[kept_indices].copy()

    # Step 5: Prune cumulative consequent duplicates where all matching rows already fall
    #         in a stricter nested quality tier (e.g. VERY_GOOD => GOOD => MODERATE)
    strict_tier_parents = {
        "GOOD": ["VERY_GOOD"],
        "MODERATE": ["VERY_GOOD", "GOOD"],
        "VERY_BAD": ["HORRIBLE"],
        "BAD": ["HORRIBLE", "VERY_BAD"],
        "MEDIOCRE": ["HORRIBLE", "VERY_BAD", "BAD"],
        "NOT_VERY_GOOD": ["HORRIBLE", "VERY_BAD", "BAD", "MEDIOCRE"]
    }
    ant_tier_support = {
        (row["antecedents"], row["Consequent"]): round(float(row["support"]), 6)
        for _, row in filtered.iterrows()
    }
    final_indices = []
    for idx, row in filtered.iterrows():
        ant = row["antecedents"]
        cons = row["Consequent"]
        supp = round(float(row["support"]), 6)
        stricter_tiers = strict_tier_parents.get(cons, [])
        if any(ant_tier_support.get((ant, st), -1.0) == supp for st in stricter_tiers):
            continue
        final_indices.append(idx)

    pruned_df = filtered.loc[final_indices].copy()
    print(
        f"[Redundancy Check] Pruned {raw_count - len(pruned_df)} redundant rules "
        f"({raw_count} raw -> {len(pruned_df)} non-redundant rules)."
    )
    return pruned_df


# =============================================================================
# 5. APRIORI ASSOCIATION RULE MINING
# =============================================================================
def mine_association_rules(
    trans_df,
    antecedent_items,
    quality_cols,
    cont_engineered_cols,
    evi_exp_res_dir,
    min_support=0.03,
    min_confidence=0.35,
    min_lift=1.25,
    max_antecedent_len=3,
):
    """Mine non-redundant association rules predicting quality tiers via mlxtend."""
    redundant_pairs, _ = find_redundant_feature_pairs(trans_df, antecedent_items)
    antecedent_set = set(antecedent_items)

    freq_itemsets = apriori(
        trans_df,
        min_support=min_support,
        use_colnames=True,
        max_len=max_antecedent_len + 1,
    )
    raw_rules = association_rules(
        freq_itemsets,
        metric="confidence",
        min_threshold=min_confidence,
    )

    def is_valid_target_rule(row):
        ants = row["antecedents"]
        cons = row["consequents"]
        if len(cons) != 1 or len(ants) > max_antecedent_len:
            return False
        cons_item = next(iter(cons))
        if cons_item not in quality_cols or not ants.issubset(antecedent_set):
            return False
        return row["lift"] >= min_lift

    rules_df = raw_rules[raw_rules.apply(is_valid_target_rule, axis=1)].copy()
    if rules_df.empty:
        print("No association rules met the support/confidence/lift thresholds.")
        return rules_df

    rules_df["Consequent"] = rules_df["consequents"].apply(lambda s: next(iter(s)))
    rules_df = filter_redundant_rules(
        rules_df, trans_df, redundant_pairs, cont_engineered_cols
    )

    if not rules_df.empty:
        n_total = len(trans_df)
        rules_df["Antecedent"] = rules_df["antecedents"].apply(
            lambda s: " & ".join(sorted(s))
        )
        rules_df["Antecedent_Len"] = rules_df["antecedents"].apply(len)
        rules_df["Count"] = (rules_df["support"] * n_total).round().astype(int)
        rules_df["Antecedent_Count"] = (
            (rules_df["antecedent support"] * n_total).round().astype(int)
        )
        rules_df["Support"] = rules_df["support"].round(4)
        rules_df["Antecedent_Support"] = rules_df["antecedent support"].round(4)
        rules_df["Consequent_Support"] = rules_df["consequent support"].round(4)
        rules_df["Confidence"] = rules_df["confidence"].round(4)
        rules_df["Lift"] = rules_df["lift"].round(4)
        rules_df["Conviction"] = rules_df["conviction"].round(4)

        fisher_p_vals = []
        for _, r in rules_df.iterrows():
            a = int(r["Count"])
            ant_n = int(r["Antecedent_Count"])
            cons_n = int(round(r["consequent support"] * n_total))
            b = max(0, ant_n - a)
            c = max(0, cons_n - a)
            d = max(0, n_total - a - b - c)
            _, p_val = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
            fisher_p_vals.append(p_val)
        rules_df["Fisher_p_value"] = fisher_p_vals

        export_cols = [
            "Antecedent",
            "Consequent",
            "Antecedent_Len",
            "Count",
            "Antecedent_Count",
            "Support",
            "Antecedent_Support",
            "Consequent_Support",
            "Confidence",
            "Lift",
            "Conviction",
            "Fisher_p_value",
        ]
        rules_df = rules_df.sort_values(
            by=["Consequent", "Lift", "Confidence", "Support"],
            ascending=[True, False, False, False],
        )
        rules_out_path = os.path.join(evi_exp_res_dir, "soft_label_association_rules.tsv")
        rules_df[export_cols].to_csv(rules_out_path, sep="\t", index=False)

        print("\n" + "=" * 80)
        print("NON-REDUNDANT APRIORI ASSOCIATION RULES (mlxtend)")
        print("=" * 80)
        for q_col in quality_cols:
            sub_rules = rules_df[rules_df["Consequent"] == q_col].head(8)
            if not sub_rules.empty:
                print(f"\n--- Top Non-Redundant Rules for Consequent: {q_col} ---")
                cols_show = [
                    "Antecedent",
                    "Count",
                    "Antecedent_Count",
                    "Support",
                    "Confidence",
                    "Lift",
                    "Fisher_p_value",
                ]
                print(sub_rules[cols_show].to_string(index=False))
        print(f"\n[Saved non-redundant association rules table to: {rules_out_path}]")

    return rules_df


# =============================================================================
# 6. STATISTICAL HYPOTHESIS TESTS ON BOOLEAN PATTERNS
# =============================================================================
def analyze_boolean_patterns(df, bool_pattern_cols, target_score_col, evi_exp_res_dir):
    """Run Mann-Whitney U, Point-Biserial, Mutual Info, and Chi2 tests on boolean features."""
    print("\n" + "=" * 80)
    print("MANN-WHITNEY U, POINT-BISERIAL & CHI2 TESTS FOR BOOLEAN PATTERN FEATURES")
    print("=" * 80)
    pattern_stats_rows = []
    for subset_name, sub_df in [
        ("All_Runs", df),
        #("Within_neg_below_pos", df[df["neg_below_pos"]]),
    ]:
        X_bool = sub_df[bool_pattern_cols].astype(int)
        y_sub = sub_df[target_score_col]
        mi_bool = mutual_info_regression(
            X_bool, y_sub, discrete_features=True, random_state=42
        )
        chi2_good, _ = chi2(X_bool, sub_df["GOOD"].astype(int))
        chi2_bad, _ = chi2(X_bool, sub_df["BAD"].astype(int))

        for idx, pat in enumerate(bool_pattern_cols):
            if subset_name == "Within_neg_below_pos" and pat == "neg_below_pos":
                continue
            mask = sub_df[pat].astype(bool)
            grp_true = sub_df.loc[mask, target_score_col]
            grp_false = sub_df.loc[~mask, target_score_col]
            if len(grp_true) == 0 or len(grp_false) == 0:
                continue
            u_stat, p_val = stats.mannwhitneyu(grp_true, grp_false, alternative="two-sided")
            pb_r, _ = stats.pointbiserialr(mask.astype(int), y_sub)
            r_rb = (2.0 * u_stat) / (len(grp_true) * len(grp_false)) - 1.0
            pattern_stats_rows.append({
                "Subset": subset_name,
                "Pattern": pat,
                "N_True": len(grp_true),
                "N_False": len(grp_false),
                "Mean_Score_True": round(grp_true.mean(), 4),
                "Mean_Score_False": round(grp_false.mean(), 4),
                "Mean_Diff": round(grp_true.mean() - grp_false.mean(), 4),
                "Median_Score_True": round(grp_true.median(), 4),
                "Median_Score_False": round(grp_false.median(), 4),
                "Point_Biserial_r": round(pb_r, 4),
                "Rank_Biserial_Effect": round(r_rb, 4),
                "Mutual_Info": round(mi_bool[idx], 4),
                "Chi2_GOOD": round(float(chi2_good[idx]), 3) if np.isfinite(chi2_good[idx]) else 0.0,
                "Chi2_BAD": round(float(chi2_bad[idx]), 3) if np.isfinite(chi2_bad[idx]) else 0.0,
                "MannWhitney_p_value": p_val,
            })

    pattern_stats_df = pd.DataFrame(pattern_stats_rows)
    pattern_stats_path = os.path.join(evi_exp_res_dir, "soft_label_pattern_stats.tsv")
    pattern_stats_df.to_csv(pattern_stats_path, sep="\t", index=False)
    print(pattern_stats_df.to_string(index=False))
    print(f"\n[Saved pattern statistical tests to: {pattern_stats_path}]")
    return pattern_stats_df


# =============================================================================
# 7. CONTINUOUS FEATURE IMPORTANCE & CORRELATION ANALYSIS
# =============================================================================
def analyze_continuous_features(df, cont_engineered_cols, target_score_col, evi_exp_res_dir):
    """Evaluate continuous soft labels and engineered margins via Spearman, RF, and Permutation Importance."""
    print("\n" + "=" * 80)
    print("CONTINUOUS FEATURE IMPORTANCE & CORRELATION (RAW LABELS + ENGINEERED MARGINS)")
    print("=" * 80)
    float_cols = list(SOFT_LABEL_COLORS.keys()) + cont_engineered_cols
    X_float = df[float_cols]
    y_score = df[target_score_col]

    mi_scores = mutual_info_regression(X_float, y_score, random_state=42)
    f_vals, _ = f_regression(X_float, y_score)
    rf_reg = RandomForestRegressor(n_estimators=500, random_state=42)
    rf_reg.fit(X_float, y_score)
    rf_importances = rf_reg.feature_importances_
    perm_imp = permutation_importance(
        rf_reg, X_float, y_score, n_repeats=20, random_state=42
    ).importances_mean

    #sub_nbp = df[df["neg_below_pos"]]
    #rf_nbp = RandomForestRegressor(n_estimators=500, random_state=42)
    #rf_nbp.fit(sub_nbp[float_cols], sub_nbp[target_score_col])
    #rf_nbp_importances = rf_nbp.feature_importances_

    float_rows = []
    for i, col in enumerate(float_cols):
        sp_r, sp_p = stats.spearmanr(df[col], y_score)
        pe_r, _ = stats.pearsonr(df[col], y_score)
        #if sub_nbp[col].nunique() > 1:
        #    sp_nbp_r, sp_nbp_p = stats.spearmanr(sub_nbp[col], sub_nbp[target_score_col])
        #else:
        #    sp_nbp_r, sp_nbp_p = 0.0, 1.0
        float_rows.append({
            "Soft_Label": col,
            "Display_Name": parse_label_name(col),
            "Feature_Group": "Engineered_Continuous" if col in cont_engineered_cols else "Raw_Soft_Label",
            "Spearman_r_All": round(float(sp_r), 4),
            "Spearman_p_All": float(sp_p),
            "Pearson_r_All": round(float(pe_r), 4),
            "F_Stat": round(float(f_vals[i]), 2),
            #"Spearman_r_NegBelowPos": round(float(sp_nbp_r), 4),
            #"Spearman_p_NegBelowPos": float(sp_nbp_p),
            "Mutual_Info": round(float(mi_scores[i]), 4),
            "RF_Importance_All": round(float(rf_importances[i]), 4),
            "Permutation_Imp_All": round(float(perm_imp[i]), 4),
            #"RF_Importance_NegBelowPos": round(float(rf_nbp_importances[i]), 4),
        })

    float_stats_df = pd.DataFrame(float_rows).sort_values(
        by="RF_Importance_All", ascending=False
    )
    float_stats_path = os.path.join(evi_exp_res_dir, "soft_label_float_correlations.tsv")
    float_stats_df.to_csv(float_stats_path, sep="\t", index=False)
    print(float_stats_df.to_string(index=False))
    print(f"\n[Saved continuous feature correlations & importances to: {float_stats_path}]")
    return float_stats_df


# =============================================================================
# 8. DECISION TREE RULE EXTRACTION & SEQUENCE SUMMARIES
# =============================================================================
def extract_decision_tree_rules(df, bool_pattern_cols, cont_engineered_cols, target_score_col):
    """Fit shallow Decision Trees to extract interpretable threshold rules."""
    float_cols = list(SOFT_LABEL_COLORS.keys()) + cont_engineered_cols
    X_float = df[float_cols]
    y_score = df[target_score_col]

    print("\n" + "=" * 80)
    print("DECISION TREE EXTRACTED RULES (Predicting Score from Continuous Features)")
    print("=" * 80)
    dt_reg = DecisionTreeRegressor(max_depth=3, min_samples_leaf=9, random_state=42)
    dt_reg.fit(X_float, y_score)
    print(f"Decision Tree R^2 on {target_score_col}: {dt_reg.score(X_float, y_score):.4f}")
    print(export_text(dt_reg, feature_names=float_cols))

    all_model_features = bool_pattern_cols + cont_engineered_cols
    X_all_features = df[all_model_features].astype(float)

    print("=" * 80)
    print("DECISION TREE EXTRACTED RULES (Predicting GOOD Performance from All Features)")
    print("=" * 80)
    dt_clf_good = DecisionTreeClassifier(max_depth=3, min_samples_leaf=6, random_state=42)
    dt_clf_good.fit(X_all_features, df["GOOD"])
    print(
        f"Decision Tree Accuracy on GOOD (>=80th pct): "
        f"{dt_clf_good.score(X_all_features, df['GOOD']):.4f}"
    )
    print(export_text(dt_clf_good, feature_names=all_model_features))

    print("=" * 80)
    print("DECISION TREE EXTRACTED RULES (Predicting BAD Performance from All Features)")
    print("=" * 80)
    dt_clf_bad = DecisionTreeClassifier(max_depth=3, min_samples_leaf=6, random_state=42)
    dt_clf_bad.fit(X_all_features, df["BAD"])
    print(
        f"Decision Tree Accuracy on BAD (<30th pct): "
        f"{dt_clf_bad.score(X_all_features, df['BAD']):.4f}"
    )
    print(export_text(dt_clf_bad, feature_names=all_model_features))


def summarize_label_sequences(df, evi_exp_res_dir):
    """Aggregate and export full and negative soft-label sequence rankings."""
    print("\n" + "=" * 80)
    print("TOP 10 FULL LABEL SEQUENCES BY MEAN SORT SCORE")
    print("=" * 80)
    seq_summary = (
        df.groupby("label_sequence")["Sort Score"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )
    seq_summary_path = os.path.join(evi_exp_res_dir, "soft_label_sequence_summary.tsv")
    seq_summary.to_csv(seq_summary_path, sep="\t", index=False)
    print(seq_summary.head(10).to_string(index=False))

    print("\n--- Negative Label Sequence Summary (Top 10 by Mean Sort Score) ---")
    seq_neg_summary = (
        df.groupby("label_sequence_neg")["Sort Score"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
        .sort_values(by="mean", ascending=False)
    )
    print(seq_neg_summary.head(10).to_string(index=False))
    return seq_summary, seq_neg_summary


# =============================================================================
# 9. ENHANCED MULTI-PANEL RELATIONSHIPS VISUALIZATION
# =============================================================================
def select_diverse_top_rules(rules_df, target_tiers, max_rules=13, per_tier_quota=5):
    """
    Select top non-redundant rules across multiple consequent tiers with balanced coverage,
    ensuring each Antecedent appears at most once (assigned to its highest-Lift tier).
    """
    if rules_df.empty:
        return pd.DataFrame()

    sub = rules_df[rules_df["Consequent"].isin(target_tiers)].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values(
        by=["Lift", "Confidence", "Support", "Fisher_p_value"],
        ascending=[False, False, False, True],
    ).drop_duplicates(subset=["Antecedent"], keep="first")

    selected_chunks = []
    used_idx = set()
    for tier in target_tiers:
        t_rows = sub[sub["Consequent"] == tier].head(per_tier_quota)
        selected_chunks.append(t_rows)
        used_idx.update(t_rows.index.tolist())

    selected = pd.concat(selected_chunks, axis=0) if selected_chunks else pd.DataFrame()
    if len(selected) < max_rules:
        rem = sub.loc[~sub.index.isin(used_idx)].head(max_rules - len(selected))
        selected = pd.concat([selected, rem], axis=0)

    selected = selected.head(max_rules).sort_values(
        by=["Lift", "Confidence", "Support"], ascending=[False, False, False]
    )
    return selected.reset_index(drop=True)


def parse_item_polarity_and_base(item_str, cont_engineered_cols):
    """
    Parse an antecedent item into (base_feature_name, polarity) where:
      +1 = True / High (> median) / Positive sequence match
      -1 = False (NOT_) / Low (<= median)
    """
    if item_str.startswith("NOT_"):
        return item_str[4:], -1
    for c in cont_engineered_cols:
        if item_str.startswith(c + "_gt_"):
            return f"{c} (High vs Low)", +1
        if item_str.startswith(c + "_le_"):
            return f"{c} (High vs Low)", -1
    return item_str, +1


def plot_relationships(
    df,
    pattern_stats_df,
    float_stats_df,
    rules_df,
    bool_pattern_cols,
    cont_engineered_cols,
    target_score_col,
    evi_exp_res_dir,
):
    """
    Create an informative, collision-free 4-panel visualization (soft_optimization_relationships.png):
      - Panel A (Top-Left): Boolean Pattern Features vs. Normalized Score (sorted by effect size + stats)
      - Panel B (Top-Right): Continuous Soft Labels & Engineered Margins Impact (All vs Within neg_below_pos)
      - Panel C (Bottom-Left): UpSet-Style Association Rule x Feature Membership Blueprint (up to 26 rules)
      - Panel D (Bottom-Right, row-aligned with Panel C): Apriori Lift, Confidence, Match Count & Fisher p-value
    """
    fig = plt.figure(figsize=(23, 19.5))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.58],
        width_ratios=[1.18, 1.0],
        hspace=0.31,
        wspace=0.28,
    )
    ax_pat = fig.add_subplot(gs[0, 0])
    ax_flt = fig.add_subplot(gs[0, 1])
    ax_mat = fig.add_subplot(gs[1, 0])
    ax_bar = fig.add_subplot(gs[1, 1])

    # -------------------------------------------------------------------------
    # PANEL A: Boolean Pattern Features vs. Normalized Score (Sorted by Mean_Diff)
    # -------------------------------------------------------------------------
    all_runs_stats = (
        pattern_stats_df[pattern_stats_df["Subset"] == "All_Runs"]
        .set_index("Pattern")
    )
    plot_bool_cols = [c for c in bool_pattern_cols if not c.startswith("IS_")]
    plot_bool_cols = sorted(
        plot_bool_cols,
        key=lambda c: all_runs_stats.loc[c, "Mean_Diff"] if c in all_runs_stats.index else 0.0,
        reverse=True,
    )

    pat_long_df = df.melt(
        id_vars=[target_score_col],
        value_vars=plot_bool_cols,
        var_name="Pattern",
        value_name="State_Bool",
    )
    pat_long_df["State"] = np.where(pat_long_df["State_Bool"], "True", "False")
    pat_long_df["Score (%)"] = pat_long_df[target_score_col] * 100.0
    pat_long_df["Pattern_Label"] = pat_long_df["Pattern"].apply(
        lambda c: f"{parse_label_name(c)}\n({c})" if parse_label_name(c) != c else c
    )
    ordered_labels = [
        f"{parse_label_name(c)}\n({c})" if parse_label_name(c) != c else c
        for c in plot_bool_cols
    ]

    sns.boxplot(
        data=pat_long_df,
        x="Score (%)",
        y="Pattern_Label",
        order=ordered_labels,
        hue="State",
        hue_order=["True", "False"],
        palette={"True": "#2e8b57", "False": "#cd5c5c"},
        width=0.68,
        fliersize=3,
        ax=ax_pat,
    )
    ax_pat.set_xlim(-3, 136)
    ax_pat.axvline(50.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)

    for y_idx, pat_col in enumerate(plot_bool_cols):
        if pat_col in all_runs_stats.index:
            s_row = all_runs_stats.loc[pat_col]
            diff_pct = s_row["Mean_Diff"] * 100.0
            n_true = int(s_row["N_True"])
            stars = format_pvalue_stars(s_row["MannWhitney_p_value"])
            sign_str = "+" if diff_pct >= 0 else ""
            txt_color = "#1b7837" if diff_pct >= 0 else "#a50f15"
            ax_pat.text(
                102.5,
                y_idx,
                f"Δ={sign_str}{diff_pct:.1f}% ({stars}, n={n_true})",
                va="center",
                ha="left",
                fontsize=8.5,
                fontweight="bold",
                color=txt_color,
            )

    ax_pat.set_title(
        "A. Boolean Pattern Features vs. Normalized Score (Sorted by Effect Size Δ)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_pat.set_xlabel("Normalized Overall Score (%)", fontsize=10, fontweight="bold")
    ax_pat.set_ylabel("")
    ax_pat.tick_params(axis="y", labelsize=8.5)
    ax_pat.legend(title="Feature State", loc="lower left", framealpha=0.9, fontsize=8.5)
    ax_pat.grid(True, linestyle="--", alpha=0.4)

    # -------------------------------------------------------------------------
    # PANEL B: Continuous Soft Labels & Engineered Margins Impact
    # -------------------------------------------------------------------------
    float_plot_df = float_stats_df.sort_values(by="RF_Importance_All", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(float_plot_df))
    bar_h = 0.26

    ax_flt.barh(
        y_pos - bar_h,
        float_plot_df["RF_Importance_All"],
        height=bar_h,
        label="RF Importance (All Runs)",
        color="#2b5c8f",
        alpha=0.9,
    )
    ax_flt.barh(
        y_pos,
        float_plot_df["Spearman_r_All"].abs(),
        height=bar_h,
        label="|Spearman r| (All Runs)",
        color="#e67e22",
        alpha=0.9,
    )
    '''ax_flt.barh(
        y_pos + bar_h,
        float_plot_df["Spearman_r_NegBelowPos"].abs(),
        height=bar_h,
        label="|Spearman r| (Within neg_below_pos)",
        color="#7d5fff",
        alpha=0.85,
    )'''

    for i, r_row in float_plot_df.iterrows():
        r_all = r_row["Spearman_r_All"]
        #r_nbp = r_row["Spearman_r_NegBelowPos"]
        ax_flt.text(
            abs(r_all) + 0.012,
            i,
            f"{r_all:+.2f}",
            va="center",
            ha="left",
            fontsize=8,
            color="#b35900",
            fontweight="bold",
        )
        '''if abs(r_nbp) > 0.001:
            ax_flt.text(
                abs(r_nbp) + 0.012,
                i + bar_h,
                f"{r_nbp:+.2f}",
                va="center",
                ha="left",
                fontsize=8,
                color="#4b30b3",
                fontweight="bold",
            )'''

    ax_flt.set_yticks(y_pos)
    ax_flt.set_yticklabels(float_plot_df["Display_Name"], fontsize=9.5, fontweight="bold")
    ax_flt.set_xlim(0, max(0.92, float_plot_df["Spearman_r_All"].abs().max() + 0.14))
    ax_flt.set_xlabel("Importance / |Spearman Correlation| (Signed r Annotated)", fontsize=10, fontweight="bold")
    ax_flt.set_title(
        "B. Continuous Soft Labels & Engineered Margins Impact",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax_flt.legend(loc="lower right", framealpha=0.9, fontsize=8.5)
    ax_flt.grid(True, linestyle="--", alpha=0.4)

    # -------------------------------------------------------------------------
    # PANELS C & D: Row-Aligned Rule Blueprint Matrix (Left) + Lift/Confidence Bars (Right)
    # -------------------------------------------------------------------------
    pos_tiers = ["VERY_GOOD", "GOOD", "MODERATE"]
    neg_tiers = ["HORRIBLE", "VERY_BAD", "BAD", "MEDIOCRE", "NOT_VERY_GOOD"]

    top_good = select_diverse_top_rules(rules_df, pos_tiers, max_rules=13, per_tier_quota=5)
    top_bad = select_diverse_top_rules(rules_df, neg_tiers, max_rules=13, per_tier_quota=4)

    combined_rules = pd.concat([top_good, top_bad], ignore_index=True)
    if not combined_rules.empty:
        combined_rules["Rule_ID"] = [f"R{i + 1}" for i in range(len(combined_rules))]
        n_rules = len(combined_rules)

        feature_order = []
        for _, r in combined_rules.iterrows():
            for item in sorted(r["antecedents"]):
                base_feat, _ = parse_item_polarity_and_base(item, cont_engineered_cols)
                if base_feat not in feature_order:
                    feature_order.append(base_feat)

        mat_grid = np.zeros((n_rules, len(feature_order)), dtype=int)
        for row_idx, (_, r) in enumerate(combined_rules.iterrows()):
            for item in r["antecedents"]:
                base_feat, pol = parse_item_polarity_and_base(item, cont_engineered_cols)
                col_idx = feature_order.index(base_feat)
                mat_grid[row_idx, col_idx] = pol

        # --- PANEL C: Rule x Feature Polarity Matrix ---
        cmap = ListedColormap(["#e57373", "#f4f6f8", "#66bb6a"])
        ax_mat.imshow(mat_grid, aspect="auto", cmap=cmap, vmin=-1, vmax=1)

        rule_y_labels = [
            f"[{r['Rule_ID']}] "
            + " & ".join(format_compact_item(x) for x in sorted(r["antecedents"]))
            for _, r in combined_rules.iterrows()
        ]
        ax_mat.set_yticks(np.arange(n_rules))
        ax_mat.set_yticklabels(rule_y_labels, fontsize=8.5, fontweight="bold")

        col_labels = [
            parse_label_name(f.replace(" (High vs Low)", "")) + (" (High/Low)" if "(High vs Low)" in f else "")
            for f in feature_order
        ]
        ax_mat.set_xticks(np.arange(len(feature_order)))
        ax_mat.set_xticklabels(col_labels, rotation=38, ha="right", fontsize=8.5, fontweight="bold")

        ax_mat.set_xticks(np.arange(-0.5, len(feature_order), 1), minor=True)
        ax_mat.set_yticks(np.arange(-0.5, n_rules, 1), minor=True)
        ax_mat.grid(which="minor", color="white", linestyle="-", linewidth=1.6)
        ax_mat.tick_params(which="minor", bottom=False, left=False)

        for r_i in range(n_rules):
            for c_i in range(len(feature_order)):
                val = mat_grid[r_i, c_i]
                if val == 1:
                    ax_mat.text(c_i, r_i, "+", ha="center", va="center", color="#0d3b14", fontsize=11, fontweight="bold")
                elif val == -1:
                    ax_mat.text(c_i, r_i, "−", ha="center", va="center", color="#5c0000", fontsize=11, fontweight="bold")

        if len(top_good) > 0 and len(top_bad) > 0:
            ax_mat.axhline(len(top_good) - 0.5, color="#111111", linestyle="-", linewidth=2.4)

        legend_elements = [
            Patch(facecolor="#66bb6a", edgecolor="gray", label="True / High (> Median) [+]"),
            Patch(facecolor="#e57373", edgecolor="gray", label="False (NOT_) / Low (<= Median) [−]"),
            Patch(facecolor="#f4f6f8", edgecolor="gray", label="Not in Rule"),
        ]
        ax_mat.legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.012),
            ncol=3,
            fontsize=8.5,
            framealpha=0.95,
        )
        ax_mat.set_title(
            f"C. Non-Redundant Rule Antecedent Blueprint ({len(top_good)} Positive & {len(top_bad)} Failure-Mode Rules)",
            fontsize=12,
            fontweight="bold",
            pad=36,
        )

        # --- PANEL D: Row-Aligned Horizontal Bar Chart of Lift, Confidence & Support ---
        y_idx = np.arange(n_rules)
        bar_colors = [CONSEQUENT_COLORS.get(c, "slategray") for c in combined_rules["Consequent"]]
        ax_bar.barh(
            y_idx,
            combined_rules["Lift"],
            color=bar_colors,
            alpha=0.9,
            height=0.72,
            edgecolor="black",
            linewidth=0.5,
        )
        ax_bar.axvline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax_bar.set_ylim(n_rules - 0.5, -0.5)

        bar_y_labels = [
            f"[{r['Rule_ID']}] => {r['Consequent']}"
            for _, r in combined_rules.iterrows()
        ]
        ax_bar.set_yticks(y_idx)
        ax_bar.set_yticklabels(bar_y_labels, fontsize=8.5, fontweight="bold")

        max_lift = float(combined_rules["Lift"].max())
        ax_bar.set_xlim(0, max_lift * 1.68)

        for i, r in combined_rules.iterrows():
            conf_pct = int(round(r["Confidence"] * 100))
            supp_pct = r["Support"] * 100.0
            cnt = int(r["Count"])
            ant_cnt = int(r["Antecedent_Count"])
            stars = format_pvalue_stars(r["Fisher_p_value"])
            ax_bar.text(
                r["Lift"] + max_lift * 0.015,
                i,
                f"Lift={r['Lift']:.1f}x | Conf={conf_pct}% ({cnt}/{ant_cnt}, Supp={supp_pct:.1f}%) {stars}",
                va="center",
                ha="left",
                fontsize=8.2,
                fontweight="bold",
                color="#1f1f1f",
            )

        if len(top_good) > 0 and len(top_bad) > 0:
            ax_bar.axhline(len(top_good) - 0.5, color="#111111", linestyle="-", linewidth=2.4)

        all_tiers_order = pos_tiers + neg_tiers
        tier_handles = [
            Patch(facecolor=CONSEQUENT_COLORS[t], edgecolor="black", label=t)
            for t in all_tiers_order
            if t in combined_rules["Consequent"].values
        ]
        if tier_handles:
            ax_bar.legend(
                handles=tier_handles,
                title="Consequent Quality Tier",
                loc="lower right",
                fontsize=8.5,
                framealpha=0.95,
            )

        ax_bar.set_xlabel(
            "Apriori Lift (Annotated with Confidence, Match Count, Support & Fisher p-value)",
            fontsize=10,
            fontweight="bold",
        )
        ax_bar.set_title(
            f"D. Rule Lift, Confidence & Statistical Significance (Aligned with R1–R{n_rules})",
            fontsize=12,
            fontweight="bold",
            pad=36,
        )
        ax_bar.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle(
        "Soft Label Hierarchy, Margin Importance & Non-Redundant Association Rule Analysis",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.16, right=0.98, top=0.94, bottom=0.08)
    rel_fig_path = os.path.join(evi_exp_res_dir, "soft_optimization_relationships.png")
    fig.savefig(rel_fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Saved enhanced relationship summary figure to: {rel_fig_path}]")


# =============================================================================
# 10. MAIN ORCHESTRATOR
# =============================================================================
def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python evidence_exploration-6-plot_soft_labels.py <path_to_results_dir_or_tsv>")

    df, evi_exp_res_dir = resolve_paths_and_load_data(sys.argv[1])
    plot_soft_label_scatter(df, evi_exp_res_dir)

    df, quality_cols, bool_pattern_cols, cont_engineered_cols, target_score_col = engineer_features(df)
    trans_df, antecedent_items = build_apriori_transactions(
        df, bool_pattern_cols, cont_engineered_cols, quality_cols
    )
    df.to_csv(os.path.join(evi_exp_res_dir, "df_engineered.csv"), index=False)

    rules_df = mine_association_rules(
        trans_df, antecedent_items, quality_cols, cont_engineered_cols, evi_exp_res_dir
    )
    pattern_stats_df = analyze_boolean_patterns(
        df, bool_pattern_cols, target_score_col, evi_exp_res_dir
    )
    float_stats_df = analyze_continuous_features(
        df, cont_engineered_cols, target_score_col, evi_exp_res_dir
    )
    extract_decision_tree_rules(
        df, bool_pattern_cols, cont_engineered_cols, target_score_col
    )
    summarize_label_sequences(df, evi_exp_res_dir)
    plot_relationships(
        df,
        pattern_stats_df,
        float_stats_df,
        rules_df,
        bool_pattern_cols,
        cont_engineered_cols,
        target_score_col,
        evi_exp_res_dir,
    )


if __name__ == "__main__":
    main()