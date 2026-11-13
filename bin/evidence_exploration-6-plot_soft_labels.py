import os
import sys
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False

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
ONTOLOGIES = ["MF", "BP", "CC"]
ONTOLOGY_TITLES = {
    "MF": "Molecular Function (MF)",
    "BP": "Biological Process (BP)",
    "CC": "Cellular Component (CC)",
}

# Simplified, boxplot-aligned quality thresholds on Normalized Sort Score [0, 1]:
# - ELITE (>= 0.90): Top 10% of the score span (tight peak cluster in boxplots)
# - GOOD  (>= 0.85): Top 15% of the score span (above the upper quartile of False boxplots)
# - POOR  (<  0.50): Bottom half of the score span (polarity-inversion failure regime)
QUALITY_THRESHOLDS = {
    "ELITE": (">=", 0.90),
    "GOOD": (">=", 0.85),
    "POOR": ("<", 0.50),
}

# Nuclear boolean rules (assigned letters A-H in descending order of overall effect size)
NUCLEAR_RULE_LETTERS = {
    "neg_below_pos": "A",
    "natural_polarity_split": "B",
    "pos_to_virtual_margin_relevant": "C",
    "virtual_neg_higher": "D",
    "derived_neg_below_virtual_negs": "E",
    "derived_neg_le_conditional_neg": "F",
    "conditional_neg_gt_real_neg_median": "G",
    "random_false_val_lt_conditional_not": "H",
}

LETTER_TO_NUCLEAR_COL = {v: k for k, v in NUCLEAR_RULE_LETTERS.items()}

# Logical expressions for the 3 key composite rules (discovered via rule mining on A-H)
COMPOSITE_RULE_EXPRESSIONS = {
    "soft_tiers_strict_chain": "A ∧ E ∧ H",
    "soft_tiers_2": "B ∧ E ∧ (D ∨ H)",
    "conditional_neg_top_anchor": "B ∧ F ∧ G ∧ H",
    "strict_4tier": "A ∧ D ∧ H"
}

# Shades for Panel A (|Spearman r| across MF, BP, CC)
ONTOLOGY_SPEARMAN_COLORS = {
    "MF": "#a04000",  # Dark rust orange
    "BP": "#e67e22",  # Medium amber orange
    "CC": "#f5b041",  # Light golden orange
}

thin_horizontal_rect = [(-5, -1), (5, -1), (5, 1), (-5, 1)]


def parse_label_name(name):
    """Map raw column names to readable display names."""
    alt_names = {
        # Raw soft labels
        "phylo": "Phylogenetic+",
        "curated": "Curated+",
        "conditional_not": "Conditional-",
        "Random False Val": "Random-",
        "curated_not": "Curated-",
        "derived_not": "Derived-",
        "phylo_not": "Phylogenetic-",
        # Continuous engineered features
        "pos_neg_inversion_count": "Pos/Neg Inversion Count",
        "pos_to_virtual_margin": "Pos-to-Virtual Margin",
        "virtual_to_real_margin": "Virtual-to-Real Margin",
        "neg_virt_real_center_diff": "Neg Virt-to-Real Center Diff",
        "real_neg_mean": "Real Negatives Mean",
        "real_neg_median": "Real Negatives Median",
        "virtual_neg_mean": "Virtual Negatives Mean",
        "real_neg_spread": "Real Negatives Spread",
        # Boolean pattern features
        "soft_tiers_strict_chain": "Soft Tiers: Strict Chain",
        "soft_tiers_2": "Soft Tiers: Relaxed 3-Tier",
        "conditional_neg_top_anchor": "Conditional- Top Neg Anchor",
        "strict_4tier": "Strict 4-Tier Hierarchy",
        "neg_below_pos": "All Neg. < All Pos.",
        "natural_polarity_split": "All Neg. < 0.50 < All Pos.",
        "pos_to_virtual_margin_relevant": "Margin Between Virt. and Pos. >= 0.13",
        "pos_far_from_middle": "Min(Pos.) >= 0.56",
        "derived_neg_below_virtual_negs": "Derived Neg. <= Min(Virtual Neg.)",
        "derived_neg_below_virtual_neg_mean": "Derived Neg. < Mean(Virtual Neg.)",
        "derived_neg_le_conditional_neg": "Derived Neg. <= Conditional Neg.",
        "conditional_neg_gt_real_neg_median": "Conditional Neg. > Median(Real Neg.)",
        "virtual_neg_higher": "Real Neg. < Virtual Neg.",
        "random_false_val_lt_conditional_not": "Random False Val. <= Conditional Neg.",
        "phylo_pos_gt_curated_pos": "Curated Pos. < Phylogenetic Pos.",
        "curated_neg_lt_phylo_neg": "Curated Neg. < Phylogenetic Neg.",
        # Ontology indicators
        "IS_MF": "Ontology = MF",
        "IS_CC": "Ontology = CC",
        "IS_BP": "Ontology = BP",
    }
    return alt_names.get(name, name)


def format_rule_axis_label(col_name):
    """
    Format boolean rule names for Panel B without raw variable names:
      - Nuclear rules: '[A] All Neg. < All Pos.'
      - Composite rules: 'Soft Tiers: Strict Chain\n(A ∧ E ∧ H)'
    """
    base_name = parse_label_name(col_name)
    if col_name in NUCLEAR_RULE_LETTERS:
        letter = NUCLEAR_RULE_LETTERS[col_name]
        return f"[{letter}] {base_name}"
    if col_name in COMPOSITE_RULE_EXPRESSIONS:
        expr = COMPOSITE_RULE_EXPRESSIONS[col_name]
        return f"{base_name}\n({expr})"
    return base_name


def adjust_pvalues_bh(p_vals):
    """Apply Benjamini-Hochberg FDR multiple-testing correction to an array of p-values."""
    p_arr = np.asarray(p_vals, dtype=float)
    if hasattr(stats, "false_discovery_control"):
        return stats.false_discovery_control(p_arr, method="bh")
    n = len(p_arr)
    order = np.argsort(p_arr)
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1)
    adj = p_arr * n / ranks
    adj_sorted = np.minimum.accumulate(adj[order][::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(adj_sorted, 0.0, 1.0)
    return out


def format_pvalue_rounded(p_val, prefix="p"):
    """Return a compact, rounded numeric p-value string."""
    if pd.isna(p_val):
        return f"{prefix}=NA"
    if p_val < 1e-3:
        return f"{prefix}={p_val:.1e}"
    if p_val < 0.01:
        return f"{prefix}={p_val:.3f}"
    return f"{prefix}={p_val:.2f}"


def assign_exclusive_quality_tier(row):
    """Assign each run to its mutually exclusive simplified quality tier."""
    if row["ELITE"]:
        return "1. ELITE (Norm >= 0.90)"
    if row["GOOD"]:
        return "2. GOOD (Norm 0.85-0.90)"
    if row["POOR"]:
        return "4. POOR (Norm < 0.50)"
    return "3. MODERATE (Norm 0.50-0.85)"


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
    df = df[df["Ontology"].isin(ONTOLOGIES)].copy()
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
            df_plot.append({
                "legend": parse_label_name(label_name),
                "color": labelc,
                "value": row[label_name],
                "Ontology": row["Ontology"],
                "OWA Score": row["OWA Score"],
                "CWA Score": row["CWA Score"],
                "Sort Score": row["Sort Score"],
            })
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
# 3. FEATURE ENGINEERING (SIMPLIFIED QUALITY TIERS, NUCLEAR & COMPOSITE RULES)
# =============================================================================
def engineer_features(df):
    """Engineer simplified boxplot-aligned quality tiers, nuclear rules [A]-[H], and margins."""
    target_score_col = "Sort Score Norm" if "Sort Score Norm" in df.columns else "Sort Score"

    print("\n--- Simplified Boxplot-Aligned Quality Tiers (on Normalized Score [0, 1]) ---")
    for tier_name, (op, thr) in QUALITY_THRESHOLDS.items():
        if op == ">=":
            df[tier_name] = df[target_score_col] >= thr
        else:
            df[tier_name] = df[target_score_col] < thr
        cnt_all = int(df[tier_name].sum())
        cnt_by_ont = {
            ont: int(df.loc[df["Ontology"] == ont, tier_name].sum())
            for ont in ONTOLOGIES
        }
        print(
            f"  {tier_name:6s} ({op} {thr:.2f}): Total={cnt_all:3d}/300 ({cnt_all/len(df):.1%}) "
            f"| MF={cnt_by_ont['MF']:2d}, BP={cnt_by_ont['BP']:2d}, CC={cnt_by_ont['CC']:2d}"
        )

    all_neg_max = df[NEG_LABELS].max(axis=1)
    pos_min = df[POS_LABELS].min(axis=1)
    virt_min = df[VIRT_NEG_COLS].min(axis=1)
    virt_max = df[VIRT_NEG_COLS].max(axis=1)
    real_min = df[REAL_NEG_COLS].min(axis=1)
    real_max = df[REAL_NEG_COLS].max(axis=1)

    # Continuous margins and summary statistics
    df["pos_neg_inversion_count"] = sum(
        (df[n] >= df[p]).astype(int) for n in NEG_LABELS for p in POS_LABELS
    )
    df["pos_to_virtual_margin"] = pos_min - virt_max
    df["virtual_to_real_margin"] = virt_min - real_max
    df["real_neg_mean"] = df[REAL_NEG_COLS].mean(axis=1)
    df["real_neg_median"] = df[REAL_NEG_COLS].median(axis=1)
    df["virtual_neg_mean"] = df[VIRT_NEG_COLS].mean(axis=1)
    df["neg_virt_real_center_diff"] = df["virtual_neg_mean"] - df["real_neg_mean"]
    df["real_neg_spread"] = real_max - real_min

    # Nuclear boolean hierarchy rules [A]-[H]
    df["neg_below_pos"] = all_neg_max < pos_min                                      # [A]
    df["natural_polarity_split"] = (all_neg_max < 0.50) & (pos_min > 0.50)           # [B]
    df["pos_to_virtual_margin_relevant"] = df["pos_to_virtual_margin"] >= 0.126      # [C]
    df["virtual_neg_higher"] = real_max < virt_min                                   # [D]
    df["derived_neg_below_virtual_negs"] = df["derived_not"] <= virt_min             # [E]
    df["derived_neg_le_conditional_neg"] = df["derived_not"] <= df["conditional_not"]  # [F]
    df["conditional_neg_gt_real_neg_median"] = df["conditional_not"] > df["real_neg_median"]  # [G]
    df["random_false_val_lt_conditional_not"] = df["Random False Val"] <= df["conditional_not"]  # [H]

    # Composite soft-tier rules (evaluated in Panel B and Table S2, but EXCLUDED from rule mining inputs)
    # 1. Soft Tiers: Strict Chain (A ∧ E ∧ H)
    df["soft_tiers_strict_chain"] = (
        df["neg_below_pos"]
        & df["derived_neg_below_virtual_negs"]
        & df["random_false_val_lt_conditional_not"]
    )
    # 2. Soft Tiers: Relaxed 3-Tier (B ∧ E ∧ (D ∨ H))
    df["soft_tiers_2"] = (
        df["natural_polarity_split"]
        & df["derived_neg_below_virtual_negs"]
        & (df["virtual_neg_higher"] | df["random_false_val_lt_conditional_not"])
    )
    # 3. Conditional- Top Neg Anchor (B ∧ F ∧ G ∧ H)
    df["conditional_neg_top_anchor"] = (
        df["natural_polarity_split"]
        & df["derived_neg_le_conditional_neg"]
        & df["conditional_neg_gt_real_neg_median"]
        & df["random_false_val_lt_conditional_not"]
    )
    df["strict_4tier"] = (
        df["neg_below_pos"]
        & df["virtual_neg_higher"]
        & df["random_false_val_lt_conditional_not"]
    )

    nuclear_cols = list(NUCLEAR_RULE_LETTERS.keys())
    composite_cols = list(COMPOSITE_RULE_EXPRESSIONS.keys())
    bool_pattern_cols = composite_cols + nuclear_cols

    cont_engineered_cols = [
        "pos_to_virtual_margin",
        "virtual_to_real_margin",
        "real_neg_median",
    ]

    for col in bool_pattern_cols:
        df[col] = df[col].astype(bool)

    df["Ontology_Rank"] = (
        df.groupby("Ontology")["Sort Score"]
        .rank(ascending=False, method="min")
        .astype(int)
    )
    df["Quality_Tier"] = df.apply(assign_exclusive_quality_tier, axis=1)

    return df, nuclear_cols, bool_pattern_cols, cont_engineered_cols, target_score_col


# =============================================================================
# 4. ALGORITHMIC COMPOSITE RULE MINING FROM NUCLEAR RULES [A]-[H] ONLY
# =============================================================================
def mine_composite_rules_from_nuclear(df, evi_exp_res_dir, min_support=0.07, min_confidence=0.74):
    """
    Mine composite boolean rules predicting simplified quality tiers (ELITE, GOOD)
    strictly from the 8 nuclear rules [A]-[H] and the elementary virtual-ordering
    disjunction (D ∨ H), WITHOUT including pre-defined composite rules as input features.

    Uses mlxtend.frequent_patterns (apriori + association_rules) when available,
    followed by minimal-generator and strict-confidence-gain pruning so only
    non-redundant composite rules remain.
    """
    print("\n" + "=" * 80)
    print("SCIENTIFIC RULE MINING FROM NUCLEAR RULES [A]-[H] (EXCLUDING COMPOSITE FEATURES)")
    print("=" * 80)

    # Build atomic rule basis A..H + elementary virtual-tier disjunction (D ∨ H)
    atom_df = pd.DataFrame(index=df.index)
    for col, letter in NUCLEAR_RULE_LETTERS.items():
        atom_df[letter] = df[col].astype(bool)
    atom_df["(D ∨ H)"] = atom_df["D"] | atom_df["H"]
    cand_atoms = ["A", "B", "C", "D", "E", "F", "G", "H", "(D ∨ H)"]

    known_composite_tags = {
        "A ∧ E ∧ H": "Soft Tiers: Strict Chain [Target #1]",
        "B ∧ E ∧ (D ∨ H)": "Soft Tiers: Relaxed 3-Tier [Target #2]",
        "B ∧ F ∧ G ∧ H": "Conditional- Top Neg Anchor [Target #3]",
    }

    subsets = [("All_Runs", df.index)] + [
        (ont, df[df["Ontology"] == ont].index) for ont in ONTOLOGIES
    ]
    target_tiers = ["ELITE", "GOOD"]

    all_mined_rows = []
    for subset_name, sub_idx in subsets:
        sub_atoms = atom_df.loc[sub_idx]
        sub_df = df.loc[sub_idx]
        n_tot = len(sub_df)

        for t_col in target_tiers:
            t_vec = sub_df[t_col].to_numpy(dtype=bool)
            n_target = int(t_vec.sum())
            if n_target == 0:
                continue
            base_rate = n_target / n_tot

            # Enumerate candidate itemsets via mlxtend.apriori (or exact enumeration fallback)
            trans = sub_atoms[cand_atoms].copy()
            trans[t_col] = t_vec
            if HAS_MLXTEND:
                freq = apriori(trans, min_support=min_support, use_colnames=True, max_len=5)
                raw_ar = association_rules(freq, metric="confidence", min_threshold=min_confidence)
                raw_ar = raw_ar[
                    raw_ar["consequents"].apply(lambda s: s == frozenset([t_col]))
                ]
                candidate_sets = [
                    tuple(sorted(ants, key=lambda x: cand_atoms.index(x)))
                    for ants in raw_ar["antecedents"]
                    if len(ants) <= 4 and not ("(D ∨ H)" in ants and ("D" in ants or "H" in ants))
                ]
            else:
                candidate_sets = [
                    comb
                    for k in range(1, 5)
                    for comb in combinations(cand_atoms, k)
                    if not ("(D ∨ H)" in comb and ("D" in comb or "H" in comb))
                ]

            cand_records = []
            for comb in candidate_sets:
                mask = sub_atoms[list(comb)].all(axis=1).to_numpy(dtype=bool)
                n_ant = int(mask.sum())
                if n_ant == 0:
                    continue
                tp = int((mask & t_vec).sum())
                if tp / n_tot < min_support:
                    continue
                conf = tp / n_ant
                if conf < min_confidence:
                    continue

                # 1. Minimal generator check: reject if any proper subset has the exact same support
                is_minimal = True
                for sub_k in range(1, len(comb)):
                    for sub_c in combinations(comb, sub_k):
                        if int(sub_atoms[list(sub_c)].all(axis=1).sum()) == n_ant:
                            is_minimal = False
                            break
                    if not is_minimal:
                        break
                if not is_minimal:
                    continue

                rec = tp / n_target
                lift = conf / base_rate
                f1 = 2.0 * conf * rec / (conf + rec) if (conf + rec) > 0 else 0.0
                b = n_ant - tp
                c = n_target - tp
                d = n_tot - tp - b - c
                _, p_val = stats.fisher_exact([[tp, b], [c, d]], alternative="greater")

                expr_str = " ∧ ".join(comb)
                cand_records.append({
                    "Subset": subset_name,
                    "Quality_Target": t_col,
                    "Mined_Formula": expr_str,
                    "Rule_Identity": known_composite_tags.get(expr_str, "Additional Mined Rule"),
                    "Atoms": comb,
                    "Mask": tuple(mask.tolist()),
                    "Rule_Length": len(comb),
                    "N_Match": tp,
                    "N_Rule_True": n_ant,
                    "Support": round(tp / n_tot, 4),
                    "Confidence": round(conf, 4),
                    "Recall": round(rec, 4),
                    "Lift": round(lift, 4),
                    "F1_Score": round(f1, 4),
                    "Mean_Norm_Score": round(float(sub_df.loc[mask, "Sort Score Norm"].mean()), 4),
                    "Min_Norm_Score": round(float(sub_df.loc[mask, "Sort Score Norm"].min()), 4),
                    "Fisher_p_value": float(p_val),
                })

            if not cand_records:
                continue

            c_df = pd.DataFrame(cand_records)
            # 2. Canonical mask deduplication: if two formulas select identical rows, prefer
            #    shorter length, threshold-free polarity rules (A/B over C), and simpler clauses
            def priority_key(row):
                atoms = row["Atoms"]
                has_c = 1 if "C" in atoms else 0
                has_disj = 1 if "(D ∨ H)" in atoms else 0
                return (row["Rule_Length"], has_c, has_disj, atoms)

            c_df["_prio"] = c_df.apply(priority_key, axis=1)
            c_df = (
                c_df.sort_values("_prio")
                .drop_duplicates(subset=["Mask"], keep="first")
                .drop(columns=["_prio"])
            )

            # 3. Sub-rule confidence dominance: keep a composite rule only if it strictly
            #    improves Confidence over all of its proper sub-rules
            kept_idx = []
            for idx, r in c_df.iterrows():
                comb = r["Atoms"]
                conf = r["Confidence"]
                dominated = False
                for sub_k in range(1, len(comb)):
                    for sub_c in combinations(comb, sub_k):
                        sub_m = sub_atoms[list(sub_c)].all(axis=1).to_numpy(dtype=bool)
                        sub_n = int(sub_m.sum())
                        if sub_n > 0:
                            sub_conf = int((sub_m & t_vec).sum()) / sub_n
                            if sub_conf >= conf - 1e-9:
                                dominated = True
                                break
                    if dominated:
                        break
                if not dominated:
                    kept_idx.append(idx)

            c_df = c_df.loc[kept_idx].copy()
            if not c_df.empty:
                c_df["Fisher_p_adj_BH"] = adjust_pvalues_bh(c_df["Fisher_p_value"].to_numpy())
                c_df = c_df.sort_values(
                    by=["Confidence", "F1_Score", "Lift", "Mean_Norm_Score"],
                    ascending=[False, False, False, False],
                ).drop(columns=["Atoms", "Mask"])
                all_mined_rows.append(c_df)

    mined_rules_df = pd.concat(all_mined_rows, ignore_index=True) if all_mined_rows else pd.DataFrame()
    mined_out_path = os.path.join(evi_exp_res_dir, "soft_label_association_rules.tsv")
    mined_rules_df.to_csv(mined_out_path, sep="\t", index=False)

    for (sub_name, t_col), grp in mined_rules_df.groupby(["Subset", "Quality_Target"], sort=False):
        print(f"\n--- Mined Non-Redundant Rules | Subset: {sub_name} | Target: {t_col} ---")
        show_cols = [
            "Mined_Formula",
            "Rule_Identity",
            "N_Match",
            "N_Rule_True",
            "Confidence",
            "Recall",
            "Lift",
            "F1_Score",
            "Mean_Norm_Score",
            "Fisher_p_adj_BH",
        ]
        print(grp[show_cols].head(10).to_string(index=False))

    print(f"\n[Saved algorithmically mined composite rules to: {mined_out_path}]")
    return mined_rules_df


# =============================================================================
# 5. STATISTICAL HYPOTHESIS TESTS ON BOOLEAN PATTERNS (ALL & PER ONTOLOGY)
# =============================================================================
def analyze_boolean_patterns(df, bool_pattern_cols, target_score_col, evi_exp_res_dir):
    """Run Mann-Whitney U, Point-Biserial, Rank-Biserial, and BH-FDR correction."""
    print("\n" + "=" * 80)
    print("MANN-WHITNEY U & RANK-BISERIAL TESTS FOR BOOLEAN PATTERN FEATURES")
    print("=" * 80)

    subsets = [("All_Runs", df)] + [
        (ont, df[df["Ontology"] == ont]) for ont in ONTOLOGIES
    ]

    pattern_stats_rows = []
    for subset_name, sub_df in subsets:
        score_col = target_score_col if subset_name == "All_Runs" else "Sort Score"
        y_sub = sub_df[score_col]

        tot_elite = int(sub_df["ELITE"].sum())
        subset_rows = []
        for pat in bool_pattern_cols:
            mask = sub_df[pat].astype(bool)
            grp_true = sub_df.loc[mask, score_col]
            grp_false = sub_df.loc[~mask, score_col]
            if len(grp_true) == 0 or len(grp_false) == 0:
                continue
            u_stat, p_val = stats.mannwhitneyu(grp_true, grp_false, alternative="two-sided")
            pb_r, _ = stats.pointbiserialr(mask.astype(int), y_sub)
            r_rb = (2.0 * u_stat) / (len(grp_true) * len(grp_false)) - 1.0

            elite_in_true = int(sub_df.loc[mask, "ELITE"].sum())
            good_in_true = int(sub_df.loc[mask, "GOOD"].sum())
            poor_in_true = int(sub_df.loc[mask, "POOR"].sum())
            elite_recall = elite_in_true / max(tot_elite, 1)
            good_prec = good_in_true / max(len(grp_true), 1)
            rule_cat = "Composite Rule" if pat in COMPOSITE_RULE_EXPRESSIONS else "Nuclear Rule"

            subset_rows.append({
                "Subset": subset_name,
                "Rule_Label": format_rule_axis_label(pat).replace("\n", " "),
                "Rule_Category": rule_cat,
                "Mean_Diff": round(grp_true.mean() - grp_false.mean(), 4),
                "MannWhitney_p_adj_BH": np.nan,
                "Rank_Biserial_Effect": round(r_rb, 4),
                "Mean_Score_True": round(grp_true.mean(), 4),
                "Mean_Score_False": round(grp_false.mean(), 4),
                "Median_Score_True": round(grp_true.median(), 4),
                "Median_Score_False": round(grp_false.median(), 4),
                "N_True": len(grp_true),
                "N_False": len(grp_false),
                "ELITE_in_True": elite_in_true,
                "ELITE_Recall": round(elite_recall, 4),
                "GOOD_Precision": round(good_prec, 4),
                "POOR_in_True": poor_in_true,
                "Point_Biserial_r": round(pb_r, 4),
                "MannWhitney_p_value": p_val,
                "Pattern": pat,
            })

        if subset_rows:
            adj_p = adjust_pvalues_bh([r["MannWhitney_p_value"] for r in subset_rows])
            for r, p_adj in zip(subset_rows, adj_p):
                r["MannWhitney_p_adj_BH"] = float(p_adj)
            pattern_stats_rows.extend(subset_rows)

    pattern_stats_df = pd.DataFrame(pattern_stats_rows)
    pattern_stats_path = os.path.join(evi_exp_res_dir, "soft_label_pattern_stats.tsv")
    pattern_stats_df.to_csv(pattern_stats_path, sep="\t", index=False)
    print(pattern_stats_df.to_string(index=False))
    print(f"\n[Saved pattern statistical tests to: {pattern_stats_path}]")
    return pattern_stats_df


# =============================================================================
# 6. CONTINUOUS FEATURE CORRELATION ANALYSIS (SPEARMAN & PEARSON + BH-FDR)
# =============================================================================
def analyze_continuous_features(df, cont_engineered_cols, target_score_col, evi_exp_res_dir):
    """Evaluate continuous soft labels and engineered margins via Spearman & Pearson correlations."""
    print("\n" + "=" * 80)
    print("CONTINUOUS FEATURE CORRELATIONS (ALL RUNS & PER ONTOLOGY — NO RF IMPORTANCE)")
    print("=" * 80)
    float_cols = list(SOFT_LABEL_COLORS.keys()) + cont_engineered_cols

    subsets = [("All_Runs", df)] + [
        (ont, df[df["Ontology"] == ont]) for ont in ONTOLOGIES
    ]

    float_rows = []
    for subset_name, sub_df in subsets:
        score_col = target_score_col if subset_name == "All_Runs" else "Sort Score"
        y_score = sub_df[score_col]

        subset_rows = []
        for col in float_cols:
            sp_r, sp_p = stats.spearmanr(sub_df[col], y_score)
            pe_r, pe_p = stats.pearsonr(sub_df[col], y_score)
            subset_rows.append({
                "Subset": subset_name,
                "Display_Name": parse_label_name(col),
                "Feature_Group": "Engineered Margin" if col in cont_engineered_cols else "Raw Soft Label",
                "Spearman_r": round(float(sp_r), 4),
                "Spearman_p_adj_BH": np.nan,
                "Pearson_r": round(float(pe_r), 4),
                "Pearson_p_adj_BH": np.nan,
                "Spearman_p": float(sp_p),
                "Pearson_p": float(pe_p),
                "Soft_Label": col,
            })

        sp_adj = adjust_pvalues_bh([r["Spearman_p"] for r in subset_rows])
        pe_adj = adjust_pvalues_bh([r["Pearson_p"] for r in subset_rows])
        for r, s_a, p_a in zip(subset_rows, sp_adj, pe_adj):
            r["Spearman_p_adj_BH"] = float(s_a)
            r["Pearson_p_adj_BH"] = float(p_a)
        float_rows.extend(subset_rows)

    float_stats_df = pd.DataFrame(float_rows)
    float_stats_path = os.path.join(evi_exp_res_dir, "soft_label_float_correlations.tsv")
    float_stats_df.to_csv(float_stats_path, sep="\t", index=False)
    print(float_stats_df.to_string(index=False))
    print(f"\n[Saved continuous feature correlations to: {float_stats_path}]")
    return float_stats_df


# =============================================================================
# 7. PUBLICATION-READY MULTI-SHEET EXCEL EXPORT (PANDAS + OPENPYXL)
# =============================================================================
def apply_sheet_styling(ws, col_formats, freeze_cell="C2"):
    """Apply publication-ready formatting, explicit number formats, and auto-fitted widths."""
    ws.views.sheetView[0].showGridLines = True
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.freeze_panes = freeze_cell

    max_col = ws.max_column
    max_row = ws.max_row

    hdr_fill = PatternFill(start_color="1B365D", end_color="1B365D", fill_type="solid")
    zebra_fill = PatternFill(start_color="F7F9FC", end_color="F7F9FC", fill_type="solid")
    yes_fill = PatternFill(start_color="D1E7DD", end_color="D1E7DD", fill_type="solid")
    no_fill = PatternFill(start_color="F8D7DA", end_color="F8D7DA", fill_type="solid")
    vg_fill = PatternFill(start_color="D1E7DD", end_color="D1E7DD", fill_type="solid")
    good_fill = PatternFill(start_color="E2F0D9", end_color="E2F0D9", fill_type="solid")
    bad_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")

    hdr_font = Font(name="Calibri", size=10.5, bold=True, color="FFFFFF")
    data_font = Font(name="Calibri", size=10, bold=False, color="111111")
    bold_data_font = Font(name="Calibri", size=10, bold=True, color="111111")
    yes_font = Font(name="Calibri", size=10, bold=True, color="0F5132")
    no_font = Font(name="Calibri", size=10, bold=False, color="842029")

    thin_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="thin", color="D9D9D9"),
        bottom=Side(style="thin", color="D9D9D9"),
    )
    group_top_border = Border(
        left=Side(style="thin", color="D9D9D9"),
        right=Side(style="thin", color="D9D9D9"),
        top=Side(style="medium", color="1B365D"),
        bottom=Side(style="thin", color="D9D9D9"),
    )

    # Header row at Row 1
    ws.row_dimensions[1].height = 28
    for col_idx in range(1, max_col + 1):
        c = ws.cell(row=1, column=col_idx)
        c.fill = hdr_fill
        c.font = hdr_font
        c.border = thin_border
        col_name = str(c.value or "")
        _, align_h = col_formats.get(col_name, ("@", "left"))
        c.alignment = Alignment(
            horizontal=align_h, vertical="center", wrap_text=True
        )

    # Data rows from Row 2 onward
    prev_group_val = None
    for row_idx in range(2, max_row + 1):
        ws.row_dimensions[row_idx].height = 20
        first_col_val = str(ws.cell(row=row_idx, column=1).value or "")
        is_new_group = (
            prev_group_val is not None and first_col_val != prev_group_val
        )
        prev_group_val = first_col_val

        for col_idx in range(1, max_col + 1):
            c = ws.cell(row=row_idx, column=col_idx)
            col_name = str(ws.cell(row=1, column=col_idx).value or "")
            num_fmt, align_h = col_formats.get(col_name, ("@", "left"))

            c.border = group_top_border if is_new_group else thin_border
            c.font = bold_data_font if col_idx <= 2 else data_font
            c.alignment = Alignment(horizontal=align_h, vertical="center")
            if num_fmt != "@":
                c.number_format = num_fmt

            if row_idx % 2 == 0:
                c.fill = zebra_fill

            val_str = str(c.value or "").strip()
            if val_str == "YES":
                c.fill = yes_fill
                c.font = yes_font
            elif val_str == "NO":
                c.fill = no_fill
                c.font = no_font
            elif val_str.startswith("1. ELITE") or "[Target #" in val_str:
                c.fill = vg_fill
                c.font = yes_font
            elif val_str.startswith("2. GOOD"):
                c.fill = good_fill
            elif val_str.startswith("4. POOR"):
                c.fill = bad_fill

    # Auto-fit column widths
    for col_idx in range(1, max_col + 1):
        col_letter = get_column_letter(col_idx)
        hdr_val = str(ws.cell(row=1, column=col_idx).value or "")
        max_len = len(hdr_val)
        for row_idx in range(2, max_row + 1):
            val = ws.cell(row=row_idx, column=col_idx).value
            if val is None:
                continue
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            max_len = max(max_len, len(val_str))
        ws.column_dimensions[col_letter].width = min(max(max_len + 4, 12), 40)


def export_publication_excel(df, float_stats_df, pattern_stats_df, mined_rules_df, out_excel_path):
    """
    Build a publication-ready multi-sheet Excel workbook using pandas.ExcelWriter + openpyxl:
      - Sheet 1 ('1_Continuous_Features'): Continuous soft labels & margins sorted by |Spearman r|
      - Sheet 2 ('2_Boolean_Rules'): Composite & nuclear boolean rules + mined rule discovery metrics
      - Sheet 3 ('3_All_300_Tests'): All 300 benchmark runs sorted by Ontology & Overall Sort Score
      - Sheet 4 ('4_Mined_Composite_Rules'): Complete algorithmic rule-mining table from [A]-[H]
    """
    subset_order = {"All_Runs": 0, "MF": 1, "BP": 2, "CC": 3}

    # -------------------------------------------------------------------------
    # SHEET 1 DATAFRAME: Continuous Features Analysis (Correlations & p-values only)
    # -------------------------------------------------------------------------
    s1 = float_stats_df.copy()
    s1["_sub_ord"] = s1["Subset"].map(subset_order).fillna(9)
    s1["_abs_r"] = s1["Spearman_r"].abs()
    s1 = s1.sort_values(
        by=["_sub_ord", "_abs_r"],
        ascending=[True, False],
    ).drop(columns=["_sub_ord", "_abs_r"])

    s1_rename = {
        "Subset": "Ontology Subset",
        "Display_Name": "Feature Name",
        "Feature_Group": "Feature Category",
        "Spearman_r": "Spearman r",
        "Spearman_p_adj_BH": "Spearman p (BH-FDR)",
        "Pearson_r": "Pearson r",
        "Pearson_p_adj_BH": "Pearson p (BH-FDR)",
        "Spearman_p": "Spearman p (Raw)",
        "Pearson_p": "Pearson p (Raw)",
        "Soft_Label": "Variable Code",
    }
    s1 = s1[list(s1_rename.keys())].rename(columns=s1_rename)

    # -------------------------------------------------------------------------
    # SHEET 2 DATAFRAME: Boolean Pattern Rules Analysis
    # -------------------------------------------------------------------------
    s2 = pattern_stats_df.copy()
    s2["_sub_ord"] = s2["Subset"].map(subset_order).fillna(9)
    s2["_cat_ord"] = np.where(s2["Rule_Category"] == "Composite Rule", 0, 1)
    s2 = s2.sort_values(
        by=["_sub_ord", "_cat_ord", "Mean_Diff"],
        ascending=[True, True, False],
    ).drop(columns=["_sub_ord", "_cat_ord"])

    s2_rename = {
        "Subset": "Ontology Subset",
        "Rule_Label": "Boolean Rule & Formula",
        "Rule_Category": "Rule Category",
        "Mean_Diff": "Mean Score Diff (Δ)",
        "MannWhitney_p_adj_BH": "Mann-Whitney p (BH-FDR)",
        "Rank_Biserial_Effect": "Rank-Biserial Effect",
        "Mean_Score_True": "Mean Score (True)",
        "Mean_Score_False": "Mean Score (False)",
        "Median_Score_True": "Median Score (True)",
        "Median_Score_False": "Median Score (False)",
        "N_True": "N (True)",
        "N_False": "N (False)",
        "ELITE_in_True": "ELITE Count (True)",
        "ELITE_Recall": "ELITE Recall",
        "GOOD_Precision": "GOOD Precision (True)",
        "POOR_in_True": "POOR Count (True)",
        "Point_Biserial_r": "Point-Biserial r",
        "MannWhitney_p_value": "Mann-Whitney p (Raw)",
        "Pattern": "Variable Code",
    }
    s2 = s2[list(s2_rename.keys())].rename(columns=s2_rename)

    # -------------------------------------------------------------------------
    # SHEET 3 DATAFRAME: All 300 Benchmark Runs Detailed
    # -------------------------------------------------------------------------
    s3 = df.copy()
    ont_order = {"MF": 0, "BP": 1, "CC": 2}
    s3["_ont_ord"] = s3["Ontology"].map(ont_order).fillna(9)
    s3 = s3.sort_values(
        by=["_ont_ord", "Sort Score"], ascending=[True, False]
    ).drop(columns=["_ont_ord"])

    bool_cols_to_yes_no = [
        "soft_tiers_strict_chain",
        "soft_tiers_2",
        "conditional_neg_top_anchor",
        "strict_4tier",
        "neg_below_pos",
        "natural_polarity_split",
        "pos_to_virtual_margin_relevant",
        "virtual_neg_higher",
        "derived_neg_below_virtual_negs",
        "derived_neg_le_conditional_neg",
        "conditional_neg_gt_real_neg_median",
        "random_false_val_lt_conditional_not",
    ]
    for b_col in bool_cols_to_yes_no:
        if b_col in s3.columns:
            s3[b_col] = np.where(s3[b_col].astype(bool), "YES", "NO")

    s3_rename = {
        "Ontology": "Ontology",
        "Ontology_Rank": "Ontology Rank",
        "Quality_Tier": "Quality Tier",
        "Sort Score": "Overall Sort Score",
        "Sort Score Norm": "Normalized Sort Score",
        "OWA Score": "OWA Score",
        "CWA Score": "CWA Score",
        "OWA Fmax (Inverse-Weighted)": "OWA Fmax (Inv-Weighted)",
        "CAFA Weighted Fmax": "CAFA Weighted Fmax",
        "soft_tiers_strict_chain": "Strict Chain (A ∧ E ∧ H)",
        "soft_tiers_2": "Relaxed 3-Tier (B ∧ E ∧ (D ∨ H))",
        "conditional_neg_top_anchor": "Cond- Top Anchor (B ∧ F ∧ G ∧ H)",
        "strict_4tier": "Strict 4-Tier Hierarchy",
        "phylo": "Phylogenetic+ (PY+)",
        "curated": "Curated+ (CU+)",
        "conditional_not": "Conditional- (CO-)",
        "Random False Val": "Random- (RA-)",
        "curated_not": "Curated- (CU-)",
        "phylo_not": "Phylogenetic- (PY-)",
        "derived_not": "Derived- (DE-)",
        "pos_to_virtual_margin": "Pos-to-Virtual Margin",
        "virtual_to_real_margin": "Virtual-to-Real Margin",
        "real_neg_median": "Real Negatives Median",
        "real_neg_mean": "Real Negatives Mean",
        "virtual_neg_mean": "Virtual Negatives Mean",
        "neg_virt_real_center_diff": "Neg Virt-to-Real Center Diff",
        "real_neg_spread": "Real Negatives Spread",
        "pos_neg_inversion_count": "Pos/Neg Inversion Count",
        "neg_below_pos": "[A] All Neg. < All Pos.",
        "natural_polarity_split": "[B] All Neg. < 0.50 < All Pos.",
        "pos_to_virtual_margin_relevant": "[C] Margin Virt-Pos >= 0.13",
        "virtual_neg_higher": "[D] Real Neg. < Virtual Neg.",
        "derived_neg_below_virtual_negs": "[E] Derived Neg. <= Min(Virt)",
        "derived_neg_le_conditional_neg": "[F] Derived Neg. <= Cond. Neg.",
        "conditional_neg_gt_real_neg_median": "[G] Cond. Neg. > Median(Real)",
        "random_false_val_lt_conditional_not": "[H] Random- <= Cond. Neg.",
        "OWA Weighted MCC": "OWA Weighted MCC",
        "OWA Weighted AUPRC": "OWA Weighted AUPRC",
        "CAFA Weighted MCC": "CAFA Weighted MCC",
        "CAFA AUPRC": "CAFA AUPRC",
    }
    s3_cols = [c for c in s3_rename.keys() if c in s3.columns]
    s3 = s3[s3_cols].rename(columns=s3_rename)

    # -------------------------------------------------------------------------
    # SHEET 4 DATAFRAME: Algorithmically Mined Composite Rules from [A]-[H]
    # -------------------------------------------------------------------------
    s4 = mined_rules_df.copy()
    s4_rename = {
        "Subset": "Ontology Subset",
        "Quality_Target": "Quality Tier Target",
        "Mined_Formula": "Mined Logical Formula",
        "Rule_Identity": "Rule Classification",
        "Confidence": "Precision / Confidence",
        "Recall": "Target Recall",
        "F1_Score": "F1 Score",
        "Lift": "Apriori Lift",
        "Fisher_p_adj_BH": "Fisher p (BH-FDR)",
        "N_Match": "Target Matches",
        "N_Rule_True": "Total Rule Matches",
        "Support": "Support",
        "Mean_Norm_Score": "Mean Norm Score",
        "Min_Norm_Score": "Min Norm Score",
        "Rule_Length": "Clause Count",
        "Fisher_p_value": "Fisher p (Raw)",
    }
    s4 = s4[list(s4_rename.keys())].rename(columns=s4_rename)

    with pd.ExcelWriter(out_excel_path, engine="openpyxl") as writer:
        s1.to_excel(writer, sheet_name="1_Continuous_Features", index=False)
        s2.to_excel(writer, sheet_name="2_Boolean_Rules", index=False)
        s3.to_excel(writer, sheet_name="3_All_300_Tests", index=False)
        s4.to_excel(writer, sheet_name="4_Mined_Composite_Rules", index=False)

        wb = writer.book
        s1_formats = {
            "Ontology Subset": ("@", "left"),
            "Feature Name": ("@", "left"),
            "Feature Category": ("@", "left"),
            "Spearman r": ("+0.0000;-0.0000;0.0000", "right"),
            "Spearman p (BH-FDR)": ("0.00E+00", "right"),
            "Pearson r": ("+0.0000;-0.0000;0.0000", "right"),
            "Pearson p (BH-FDR)": ("0.00E+00", "right"),
            "Spearman p (Raw)": ("0.00E+00", "right"),
            "Pearson p (Raw)": ("0.00E+00", "right"),
            "Variable Code": ("@", "left"),
        }
        apply_sheet_styling(wb["1_Continuous_Features"], s1_formats, freeze_cell="C2")

        s2_formats = {
            "Ontology Subset": ("@", "left"),
            "Boolean Rule & Formula": ("@", "left"),
            "Rule Category": ("@", "left"),
            "Mean Score Diff (Δ)": ("+0.0000;-0.0000;0.0000", "right"),
            "Mann-Whitney p (BH-FDR)": ("0.00E+00", "right"),
            "Rank-Biserial Effect": ("+0.0000;-0.0000;0.0000", "right"),
            "Mean Score (True)": ("0.0000", "right"),
            "Mean Score (False)": ("0.0000", "right"),
            "Median Score (True)": ("0.0000", "right"),
            "Median Score (False)": ("0.0000", "right"),
            "N (True)": ("#,##0", "right"),
            "N (False)": ("#,##0", "right"),
            "ELITE Count (True)": ("#,##0", "right"),
            "ELITE Recall": ("0.0%", "right"),
            "GOOD Precision (True)": ("0.0%", "right"),
            "POOR Count (True)": ("#,##0", "right"),
            "Point-Biserial r": ("+0.0000;-0.0000;0.0000", "right"),
            "Mann-Whitney p (Raw)": ("0.00E+00", "right"),
            "Variable Code": ("@", "left"),
        }
        apply_sheet_styling(wb["2_Boolean_Rules"], s2_formats, freeze_cell="C2")

        s3_formats = {
            "Ontology": ("@", "left"),
            "Ontology Rank": ("#,##0", "right"),
            "Quality Tier": ("@", "left"),
            "Overall Sort Score": ("0.0000", "right"),
            "Normalized Sort Score": ("0.0000", "right"),
            "OWA Score": ("0.0000", "right"),
            "CWA Score": ("0.0000", "right"),
            "OWA Fmax (Inv-Weighted)": ("0.0000", "right"),
            "CAFA Weighted Fmax": ("0.0000", "right"),
            "Strict Chain (A ∧ E ∧ H)": ("@", "center"),
            "Relaxed 3-Tier (B ∧ E ∧ (D ∨ H))": ("@", "center"),
            "Cond- Top Anchor (B ∧ F ∧ G ∧ H)": ("@", "center"),
            "Phylogenetic+ (PY+)": ("0.00", "right"),
            "Curated+ (CU+)": ("0.00", "right"),
            "Conditional- (CO-)": ("0.00", "right"),
            "Random- (RA-)": ("0.00", "right"),
            "Curated- (CU-)": ("0.00", "right"),
            "Phylogenetic- (PY-)": ("0.00", "right"),
            "Derived- (DE-)": ("0.00", "right"),
            "Pos-to-Virtual Margin": ("+0.00;-0.00;0.00", "right"),
            "Virtual-to-Real Margin": ("+0.00;-0.00;0.00", "right"),
            "Real Negatives Median": ("0.0000", "right"),
            "Real Negatives Mean": ("0.0000", "right"),
            "Virtual Negatives Mean": ("0.0000", "right"),
            "Neg Virt-to-Real Center Diff": ("+0.0000;-0.0000;0.0000", "right"),
            "Real Negatives Spread": ("0.00", "right"),
            "Pos/Neg Inversion Count": ("#,##0", "right"),
            "[A] All Neg. < All Pos.": ("@", "center"),
            "[B] All Neg. < 0.50 < All Pos.": ("@", "center"),
            "[C] Margin Virt-Pos >= 0.13": ("@", "center"),
            "[D] Real Neg. < Virtual Neg.": ("@", "center"),
            "[E] Derived Neg. <= Min(Virt)": ("@", "center"),
            "[F] Derived Neg. <= Cond. Neg.": ("@", "center"),
            "[G] Cond. Neg. > Median(Real)": ("@", "center"),
            "[H] Random- <= Cond. Neg.": ("@", "center"),
            "OWA Weighted MCC": ("0.0000", "right"),
            "OWA Weighted AUPRC": ("0.0000", "right"),
            "CAFA Weighted MCC": ("0.0000", "right"),
            "CAFA AUPRC": ("0.0000", "right"),
        }
        apply_sheet_styling(wb["3_All_300_Tests"], s3_formats, freeze_cell="E2")

        s4_formats = {
            "Ontology Subset": ("@", "left"),
            "Quality Tier Target": ("@", "left"),
            "Mined Logical Formula": ("@", "left"),
            "Rule Classification": ("@", "left"),
            "Precision / Confidence": ("0.0%", "right"),
            "Target Recall": ("0.0%", "right"),
            "F1 Score": ("0.0000", "right"),
            "Apriori Lift": ("0.00", "right"),
            "Fisher p (BH-FDR)": ("0.00E+00", "right"),
            "Target Matches": ("#,##0", "right"),
            "Total Rule Matches": ("#,##0", "right"),
            "Support": ("0.0%", "right"),
            "Mean Norm Score": ("0.0000", "right"),
            "Min Norm Score": ("0.0000", "right"),
            "Clause Count": ("#,##0", "right"),
            "Fisher p (Raw)": ("0.00E+00", "right"),
        }
        apply_sheet_styling(wb["4_Mined_Composite_Rules"], s4_formats, freeze_cell="D2")

    print(f"[Saved publication-ready multi-sheet Excel workbook to: {out_excel_path}]")


# =============================================================================
# 8. COMPACT TWO-ROW MULTI-ONTOLOGY RELATIONSHIPS VISUALIZATION (PANELS A & B)
# =============================================================================
def plot_relationships(
    df,
    float_stats_df,
    bool_pattern_cols,
    evi_exp_res_dir,
):
    """
    Create a compact, high-readability 2-row figure (soft_optimization_relationships.png):
      - Top Row (Panel A): Continuous Soft Labels & Engineered Margins Impact across Ontologies
        (|Spearman r| for MF, BP, and CC with signed r and BH-FDR adjusted p-values — no RF bars).
      - Bottom Row (Panel B): 3 seamless side-by-side boxplot panels (MF, BP, CC) over unnormalized
        Overall Score (%), showing readable rule names (nuclear letters [A]-[H] and composite boolean
        expressions) only on the leftmost panel, with no separating vertical spines.
    """
    fig = plt.figure(figsize=(12.0, 12.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.25)
    gs_bot = gs[1].subgridspec(1, 3, wspace=0.04)

    ax_flt = fig.add_subplot(gs[0])
    ax_a_list = [fig.add_subplot(gs_bot[0, i]) for i in range(3)]

    # -------------------------------------------------------------------------
    # PANEL A (TOP ROW): Continuous Soft Labels & Engineered Margins (|Spearman r| Only)
    # -------------------------------------------------------------------------
    all_runs_flt = float_stats_df[float_stats_df["Subset"] == "All_Runs"].copy()
    all_runs_flt["_abs_r"] = all_runs_flt["Spearman_r"].abs()
    all_float_order = (
        all_runs_flt.sort_values(by="_abs_r", ascending=True)["Soft_Label"].tolist()
    )

    y_pos = np.arange(len(all_float_order))
    bar_h = 0.25
    offsets = {
        "MF": -1.0 * bar_h,
        "BP": 0.0,
        "CC": +1.0 * bar_h,
    }

    flt_lookup = float_stats_df.set_index(["Subset", "Soft_Label"])

    for ont in ONTOLOGIES:
        sp_abs_vals = [
            abs(flt_lookup.loc[(ont, col), "Spearman_r"]) for col in all_float_order
        ]
        ax_flt.barh(
            y_pos + offsets[ont],
            sp_abs_vals,
            height=bar_h,
            label=f"|Spearman r| ({ont})",
            color=ONTOLOGY_SPEARMAN_COLORS[ont],
            alpha=0.92,
            edgecolor="white",
            linewidth=0.5,
        )

        for i, col in enumerate(all_float_order):
            r_val = flt_lookup.loc[(ont, col), "Spearman_r"]
            p_adj = flt_lookup.loc[(ont, col), "Spearman_p_adj_BH"]
            p_str = format_pvalue_rounded(p_adj, prefix="p")
            ax_flt.text(
                abs(r_val) + 0.008,
                i + offsets[ont],
                f"{ont}: r={r_val:+.2f} ({p_str})",
                va="center",
                ha="left",
                fontsize=9.2,
                color="#7a2e00" if ont == "MF" else ("#9c4c00" if ont == "BP" else "#7d5a00"),
                fontweight="bold",
            )

    for idx in range(len(all_float_order) - 1):
        ax_flt.axhline(idx + 0.5, color="#e0e0e0", linestyle="-", linewidth=0.8)

    y_display_labels = [parse_label_name(col) for col in all_float_order]
    ax_flt.set_yticks(y_pos)
    ax_flt.set_yticklabels(y_display_labels, fontsize=11.0, fontweight="bold")
    ax_flt.tick_params(axis="x", labelsize=10.5)
    max_sp = float_stats_df["Spearman_r"].abs().max()
    ax_flt.set_xlim(0, max(0.98, max_sp + 0.22))
    ax_flt.set_xlabel(
        "|Spearman Rank Correlation| (Annotated with Signed r & BH-Adjusted p-value)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax_flt.set_title(
        "A. Continuous Soft Labels & Engineered Margins Impact Across Ontologies (MF, BP, CC)",
        fontsize=13.2,
        fontweight="bold",
        pad=10,
    )
    ax_flt.legend(
        title="Ontology",
        loc="lower right",
        framealpha=0.95,
        fontsize=10.0,
        title_fontsize=10.5,
    )
    ax_flt.grid(True, axis="x", linestyle="--", alpha=0.4)

    # -------------------------------------------------------------------------
    # PANEL B (BOTTOM ROW): 3 Seamless Subplots (MF, BP, CC) vs. Raw Overall Score (%)
    # -------------------------------------------------------------------------
    composite_cols = [
        "soft_tiers_strict_chain",
        "soft_tiers_2",
        "conditional_neg_top_anchor",
        "strict_4tier"
    ]
    nuclear_cols = [
        "neg_below_pos",
        "natural_polarity_split",
        "pos_to_virtual_margin_relevant",
        "virtual_neg_higher",
        "derived_neg_below_virtual_negs",
        "derived_neg_le_conditional_neg",
        "conditional_neg_gt_real_neg_median",
        "random_false_val_lt_conditional_not",
    ]
    plot_bool_cols = [c for c in composite_cols + nuclear_cols if c in bool_pattern_cols]
    ordered_labels = [format_rule_axis_label(c) for c in plot_bool_cols]

    pat_long_df = df.melt(
        id_vars=["Ontology", "Sort Score"],
        value_vars=plot_bool_cols,
        var_name="Pattern",
        value_name="State_Bool",
    )
    pat_long_df["State"] = np.where(pat_long_df["State_Bool"], "True", "False")
    pat_long_df["Overall Score (%)"] = pat_long_df["Sort Score"] * 100.0
    pat_long_df["Pattern_Label"] = pat_long_df["Pattern"].apply(format_rule_axis_label)

    for idx_ont, (ont, ax_a) in enumerate(zip(ONTOLOGIES, ax_a_list)):
        ont_sub = pat_long_df[pat_long_df["Ontology"] == ont]

        sns.boxplot(
            data=ont_sub,
            x="Overall Score (%)",
            y="Pattern_Label",
            order=ordered_labels,
            hue="State",
            hue_order=["True", "False"],
            palette={"True": "#2e8b57", "False": "#cd5c5c"},
            width=0.72,
            fliersize=2.8,
            linewidth=1.0,
            ax=ax_a,
        )

        # Separate composite rules from nuclear rules [A]-[H] with a subtle dashed line
        if len(composite_cols) > 0:
            ax_a.axhline(
                len(composite_cols) - 0.5,
                color="#666666",
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
            )

        # Light horizontal row separators
        for r_i in range(len(plot_bool_cols) - 1):
            if r_i != len(composite_cols) - 1:
                ax_a.axhline(r_i + 0.5, color="#e6e6e6", linestyle="-", linewidth=0.7)

        # Set ontology-specific X limits with clean padding and keep X ticks
        ont_scores = df.loc[df["Ontology"] == ont, "Sort Score"] * 100.0
        s_min, s_max = float(ont_scores.min()), float(ont_scores.max())
        pad = (s_max - s_min) * 0.05
        ax_a.set_xlim(s_min - pad, s_max + pad)

        ax_a.set_title(ONTOLOGY_TITLES[ont], fontsize=11.8, fontweight="bold", pad=7)
        ax_a.tick_params(axis="x", labelsize=10.5)
        ax_a.grid(True, axis="x", linestyle="--", alpha=0.45)

        # Show rule names only on the first subplot; remove inner vertical spines
        if idx_ont == 0:
            ax_a.set_ylabel("")
            ax_a.tick_params(axis="y", labelsize=10.5)
            for lbl in ax_a.get_yticklabels():
                lbl.set_fontweight("bold")
            ax_a.spines["right"].set_visible(False)
            ax_a.legend(
                title="Rule State",
                loc="lower left",
                framealpha=0.95,
                fontsize=9.6,
                title_fontsize=10.2,
            )
        else:
            ax_a.set_ylabel("")
            ax_a.set_yticklabels([])
            ax_a.tick_params(axis="y", left=False)
            ax_a.spines["left"].set_visible(False)
            if idx_ont < len(ONTOLOGIES) - 1:
                ax_a.spines["right"].set_visible(False)
            if ax_a.get_legend() is not None:
                ax_a.get_legend().remove()

        # Center the shared X-axis label on the middle ontology subplot
        if idx_ont == 1:
            ax_a.set_xlabel("Overall Score (%)", fontsize=11.5, fontweight="bold", labelpad=7)
        else:
            ax_a.set_xlabel("")

    # Add unified Panel B header above the 3 ontology subplots
    ax_a_list[1].text(
        0.5,
        1.095,
        "B. Boolean Pattern Rules vs. Overall Score (%) by Ontology",
        transform=ax_a_list[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=13.2,
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.25, right=0.98, top=0.955, bottom=0.045)
    rel_fig_path = os.path.join(evi_exp_res_dir, "soft_optimization_relationships.png")
    fig.savefig(rel_fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Saved relationship summary figure to: {rel_fig_path}]")


# =============================================================================
# 9. MAIN ORCHESTRATOR
# =============================================================================
def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python evidence_exploration-6-plot_soft_labels.py <path_to_results_dir_or_tsv>")

    df, evi_exp_res_dir = resolve_paths_and_load_data(sys.argv[1])
    plot_soft_label_scatter(df, evi_exp_res_dir)

    df, nuclear_cols, bool_pattern_cols, cont_engineered_cols, target_score_col = engineer_features(df)
    df.to_csv(os.path.join(evi_exp_res_dir, "df_engineered.csv"), index=False)

    mined_rules_df = mine_composite_rules_from_nuclear(df, evi_exp_res_dir)
    pattern_stats_df = analyze_boolean_patterns(
        df, bool_pattern_cols, target_score_col, evi_exp_res_dir
    )
    float_stats_df = analyze_continuous_features(
        df, cont_engineered_cols, target_score_col, evi_exp_res_dir
    )
    excel_out_path = os.path.join(evi_exp_res_dir, "soft_label_supplementary_tables.xlsx")
    export_publication_excel(
        df, float_stats_df, pattern_stats_df, mined_rules_df, excel_out_path
    )

    plot_relationships(
        df,
        float_stats_df,
        bool_pattern_cols,
        evi_exp_res_dir,
    )


if __name__ == "__main__":
    main()