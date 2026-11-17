import os
import sys
import json
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import to_rgba
import seaborn as sns
from scipy import stats
import openpyxl

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    HAS_MLXTEND = True
except ImportError:
    HAS_MLXTEND = False

# =============================================================================
# CONSTANTS & LABEL MAPPINGS
# =============================================================================
SOFT_LABEL_COLORS = {
    "phylo": "#250C6F",             # Deep Ultramarine Navy
    "curated": "#2D5CCB",           # Bright Royal Blue
    "conditional_not": "#6D863C",   # Olive Green
    "Random False Val": "#FFF200",  # Luminous Golden Yellow
    "curated_not": "#ff5d00",       # Bright Tangerine-Orange
    "derived_not": "#d90429",       # Pure Crimson Red
    "phylo_not": "#4D1400",         # Deep Espresso / Umber
}

# Explicit intuitive legend order requested for Panel A: py-, de-, cu-, rd-, co-, cu+, py+
LEGEND_LABEL_ORDER = [
    "phylo_not",
    "derived_not",
    "curated_not",
    "Random False Val",
    "conditional_not",
    "curated",
    "phylo",
]

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

QUALITY_THRESHOLDS = {
    "ELITE": (">=", 0.90),
    "GOOD": (">=", 0.85),
    "POOR": ("<", 0.50),
}

# Nuclear rules ordered strictly by ascending BH-adjusted p-value (A = lowest p-value, H = largest p-value):
#   A: neg_below_pos (p = 3.9e-32)
#   B: natural_polarity_split (p = 1.3e-30)
#   C: pos_to_virtual_margin_relevant (p = 2.0e-30)
#   D: derived_neg_le_conditional_neg (p = 1.2e-15)
#   E: derived_neg_below_virtual_negs (p = 1.4e-15)
#   F: virtual_neg_higher (p = 8.5e-11)
#   G: conditional_neg_gt_real_neg_median (p = 5.6e-08)
#   H: random_false_val_lt_conditional_not (p = 8.5e-04)
NUCLEAR_RULE_LETTERS = {
    "neg_below_pos": "A",
    "natural_polarity_split": "B",
    "pos_to_virtual_margin_relevant": "C",
    "derived_neg_le_conditional_neg": "D",
    "derived_neg_below_virtual_negs": "E",
    "virtual_neg_higher": "F",
    "conditional_neg_gt_real_neg_median": "G",
    "random_false_val_lt_conditional_not": "H",
}

LETTER_TO_NUCLEAR_COL = {v: k for k, v in NUCLEAR_RULE_LETTERS.items()}

# Apriori / composite rules named R1-R4 (ordered by ascending BH-adjusted p-value, matching A-H)
COMPOSITE_RULE_NAMES = {
    "conditional_neg_top_anchor": "R1",  # p = 6.8e-23 (lowest p-value)
    "soft_tiers_2": "R2",                # p = 6.8e-20
    "soft_tiers_strict_chain": "R3",     # p = 6.8e-19
    "strict_4tier": "R4",                # p = 7.7e-14
}

COMPOSITE_RULE_EXPRESSIONS = {
    "conditional_neg_top_anchor": "B ∧ D ∧ G ∧ H",
    "soft_tiers_2": "B ∧ E ∧ (F ∨ H)",
    "soft_tiers_strict_chain": "A ∧ E ∧ H",
    "strict_4tier": "A ∧ F ∧ H",
}

# Colors for Panel C (MF, BP, CC bars) and Panel D (Blue for True, Orange for False)
# selected directly from SOFT_LABEL_COLORS in Panels A & B
ONTOLOGY_SPEARMAN_COLORS = {
    "MF": SOFT_LABEL_COLORS["phylo"],        # "#250C6F" Deep Ultramarine Navy
    "BP": SOFT_LABEL_COLORS["curated"],      # "#2D5CCB" Bright Royal Blue
    "CC": SOFT_LABEL_COLORS["curated_not"],  # "#ff5d00" Bright Tangerine-Orange
}

BOXPLOT_RULE_PALETTE = {
    "True": SOFT_LABEL_COLORS["curated"],       # "#2D5CCB" Bright Royal Blue
    "False": SOFT_LABEL_COLORS["curated_not"],  # "#ff5d00" Bright Tangerine-Orange
}


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
        "real_neg_median": "Real Negatives Median",
        "virtual_neg_mean": "Virtual Negatives Mean",
        "real_neg_spread": "Real Negatives Spread",
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
        "IS_MF": "Ontology = MF",
        "IS_CC": "Ontology = CC",
        "IS_BP": "Ontology = BP",
    }
    return alt_names.get(name, name)


def format_rule_axis_label(col_name):
    """Format boolean rule names for Panel D with [A]-[H] for atomic rules and [R1]-[R4] for apriori rules."""
    base_name = parse_label_name(col_name)
    if col_name in NUCLEAR_RULE_LETTERS:
        letter = NUCLEAR_RULE_LETTERS[col_name]
        return f"[{letter}] {base_name}"
    if col_name in COMPOSITE_RULE_EXPRESSIONS:
        r_tag = COMPOSITE_RULE_NAMES.get(col_name, "")
        expr = COMPOSITE_RULE_EXPRESSIONS[col_name]
        prefix = f"[{r_tag}] " if r_tag else ""
        return f"{prefix}{base_name}\n({expr})"
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


def format_pvalue_number(p_val):
    """Return a compact, rounded numeric p-value string (without prefix)."""
    if pd.isna(p_val):
        return "NA"
    if p_val < 1e-3:
        return f"{p_val:.1e}"
    if p_val < 0.01:
        return f"{p_val:.3f}"
    return f"{p_val:.2f}"


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
# 2. PANEL A & PANEL B HELPER FUNCTIONS
# =============================================================================
def build_long_scatter_df(df, palette=None):
    """Flatten the 7 soft-label columns into a long DataFrame of 2,100 points."""
    if palette is None:
        palette = SOFT_LABEL_COLORS
    rows = []
    for run_idx, row in df.iterrows():
        for label_name, labelc in palette.items():
            rows.append({
                "run_idx": run_idx,
                "raw_label": label_name,
                "legend": parse_label_name(label_name),
                "color": labelc,
                "value": float(row[label_name]),
                "Ontology": row["Ontology"],
                "OWA Score": float(row["OWA Score"]),
                "CWA Score": float(row["CWA Score"]),
                "Sort Score": float(row["Sort Score"]),
                "Sort Score Norm": float(row.get("Sort Score Norm", row["Sort Score"])),
            })
    return pd.DataFrame(rows)


def draw_panel_a_scatter(
    ax,
    df,
    fill_alpha=0.70,
    bg_color="#dcdcdc",
    seed=42,
    title="A. Classification Performance with Random Soft Label Values",
):
    """
    Render Panel A: Classification Performance with Random Soft Label Values.
      - Legend ordered intuitively: py-, de-, cu-, rd-, co-, cu+, py+
      - Elite (90%-100%) vertical text box shifted further left for clean separation from ticks
    """
    df_plot = build_long_scatter_df(df, palette=SOFT_LABEL_COLORS)
    df_shuffled = df_plot.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    sizes = [300] * len(df_shuffled)
    face_rgba = [to_rgba(c, alpha=fill_alpha) for c in df_shuffled["color"]]

    ax.set_facecolor(bg_color)
    ax.scatter(
        df_shuffled["value"],
        df_shuffled["Sort Score Norm"] * 100.0,
        s=sizes,
        facecolors=face_rgba,
        edgecolors="none",
        linewidths=0,
        marker="o",
        zorder=3,
    )

    # Show only the Elite range (90% to 100%) and vertical label on the left axis
    n_elite = int((df["Sort Score Norm"] >= 0.90).sum())
    elite_color = "#1f4e1b"
    ax.axhline(
        90.0,
        color=elite_color,
        linestyle="--",
        linewidth=1.8,
        alpha=0.90,
        zorder=4,
    )
    ax.axhspan(90.0, 100.0, color="#6D863C", alpha=0.14, zorder=1)

    # Vertical bracket and "Elite ({n} tests)" label placed with extra separation to the left of ticks
    trans_y = ax.get_yaxis_transform()
    ax.plot(
        [-0.082, -0.082],
        [90.0, 100.0],
        transform=trans_y,
        color=elite_color,
        linewidth=2.8,
        solid_capstyle="butt",
        clip_on=False,
        zorder=6,
    )
    ax.plot(
        [-0.082, -0.064],
        [90.0, 90.0],
        transform=trans_y,
        color=elite_color,
        linewidth=2.2,
        clip_on=False,
        zorder=6,
    )
    ax.plot(
        [-0.082, -0.064],
        [100.0, 100.0],
        transform=trans_y,
        color=elite_color,
        linewidth=2.2,
        clip_on=False,
        zorder=6,
    )
    ax.text(
        -0.122,
        95.0,
        f"Elite ({n_elite} tests)",
        transform=trans_y,
        rotation=90,
        ha="center",
        va="center",
        fontsize=9.8,
        fontweight="bold",
        color=elite_color,
        bbox=dict(
            boxstyle="round,pad=0.22",
            fc="#eef4e8",
            ec=elite_color,
            lw=1.1,
            alpha=0.96,
        ),
        clip_on=False,
        zorder=7,
    )

    ax.set_xlabel("Soft Label Value", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Normalized Overall Score (%)", fontsize=10.5, fontweight="bold", labelpad=20)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=10)
    ax.set_yticks([0, 20, 40, 60, 80, 90, 100])
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="both", labelsize=9.8)
    ax.grid(True, linestyle="--", alpha=0.35, color="#888888", zorder=0)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-2.5, 103.0)

    # Order legend explicitly: py-, de-, cu-, rd-, co-, cu+, py+
    legend_handles = [
        plt.scatter(
            [],
            [],
            s=160,
            facecolors=to_rgba(SOFT_LABEL_COLORS[col_key], alpha=0.80),
            edgecolors="none",
            linewidths=0,
            marker="o",
            label=parse_label_name(col_key),
        )
        for col_key in LEGEND_LABEL_ORDER
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        framealpha=0.92,
        facecolor="white",
        fontsize=9.2,
    )


def plot_soft_label_scatter(df, evi_exp_res_dir):
    """Standalone Panel A export (`soft_optimization_test.png`)."""
    fig, ax = plt.subplots(figsize=(8, 12), facecolor="white")
    draw_panel_a_scatter(
        ax,
        df,
        title="Classification Performance with Random Soft Label Values",
    )
    fig.tight_layout()
    out_path = os.path.join(evi_exp_res_dir, "soft_optimization_test.png")
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved scatter figure to: {out_path}]")
    return out_path


def draw_panel_b_evi_values_scale(ax, df):
    """
    Render Panel B: Soft Labeling Scheme With Best Performance (extra compact height)
    Adapted from `plot_evi_values_scale` in `evidence_exploration-4-plot.py`,
    using the exact `SOFT_LABEL_COLORS` from Panel A.
    """
    labels = [
        "phylo",
        "curated",
        "conditional_not",
        "curated_not",
        "derived_not",
        "phylo_not",
        "Random False Val",
    ]

    df_lines = []
    for ont in ["MF", "CC", "BP"]:
        df_ont = df[df["Ontology"] == ont].sort_values(by="Sort Score", ascending=False)
        if len(df_ont) == 0:
            continue
        top_row = df_ont.iloc[0]

        params = {}
        if "Parameters" in top_row and pd.notna(top_row["Parameters"]):
            try:
                params = json.loads(top_row["Parameters"])
            except Exception:
                params = {}

        for label in labels:
            val = params.get(label, top_row.get(label, np.nan))
            if pd.isna(val):
                continue
            if not ("False" in label or "_not" in label):
                label2 = label + "_yes"
            else:
                label2 = label
            if label2 == "Random False Val":
                label2 = "random_not"
            df_lines.append({"evi_type": label2, "label_value": float(val), "ontology": ont})

    df_scale = pd.DataFrame(df_lines)

    def format_evi(name):
        parts = name.split("_")
        if len(parts) == 2:
            return f"{parts[0].capitalize()}\n({parts[1].capitalize()})"
        return name

    color_map = {
        "phylo_not": SOFT_LABEL_COLORS["phylo_not"],
        "derived_not": SOFT_LABEL_COLORS["derived_not"],
        "curated_not": SOFT_LABEL_COLORS["curated_not"],
        "random_not": SOFT_LABEL_COLORS["Random False Val"],
        "conditional_not": SOFT_LABEL_COLORS["conditional_not"],
        "curated_yes": SOFT_LABEL_COLORS["curated"],
        "phylo_yes": SOFT_LABEL_COLORS["phylo"],
    }

    avg_x = df_scale.groupby("evi_type")["label_value"].mean().sort_values()

    # Compact horizontal gray ontology bars
    for y in [0, 1, 2]:
        ax.hlines(y, xmin=-0.05, xmax=1.05, color="#b0b0b0", linewidth=4.2, zorder=1)

    # Anti-overlap logic for bottom text boxes
    label_xs = {}
    current_x = -0.02
    for evi, x_val in avg_x.items():
        if x_val < 0.5:
            target_x = max(x_val, current_x)
            label_xs[evi] = target_x
            current_x = target_x + 0.125
        else:
            label_xs[evi] = x_val

    current_x = 1.03
    for evi in reversed(list(avg_x.index)):
        x_val = avg_x[evi]
        if x_val >= 0.5:
            target_x = min(x_val, current_x)
            label_xs[evi] = target_x
            current_x = target_x - 0.125

    y_label = -0.70
    Path = mpath.Path

    for evi, _ in avg_x.items():
        c = color_map[evi]
        edge_c = "#d4a800" if evi == "random_not" else c
        target_x = label_xs[evi]
        display_name = format_evi(evi)

        x_cc = df_scale[(df_scale["evi_type"] == evi) & (df_scale["ontology"] == "CC")]["label_value"].values[0]
        x_bp = df_scale[(df_scale["evi_type"] == evi) & (df_scale["ontology"] == "BP")]["label_value"].values[0]
        x_mf = df_scale[(df_scale["evi_type"] == evi) & (df_scale["ontology"] == "MF")]["label_value"].values[0]

        ax.plot([x_cc, x_bp, x_mf], [0, 1, 2], color=edge_c, linewidth=2.2, zorder=2)
        ax.scatter(
            [x_cc, x_bp, x_mf],
            [0, 1, 2],
            color=c,
            s=175,
            zorder=3,
            edgecolors="white" if evi != "random_not" else "#b58e00",
            linewidth=1.6,
        )

        if abs(target_x - x_cc) < 0.005:
            ax.plot([x_cc, x_cc], [0, y_label + 0.18], color=edge_c, linewidth=1.7, zorder=2)
        else:
            pp1 = (x_cc, 0)
            pp2 = (x_cc, -0.28)
            pp3 = (target_x, -0.28)
            pp4 = (target_x, y_label + 0.18)
            verts = [pp1, pp2, pp3, pp4]
            codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
            path = Path(verts, codes)
            patch = mpatches.PathPatch(path, facecolor="none", lw=1.7, edgecolor=edge_c, zorder=2)
            ax.add_patch(patch)

        ax.text(
            target_x,
            y_label,
            display_name,
            ha="center",
            va="center",
            fontsize=7.9,
            color="black",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=edge_c, lw=1.1, alpha=1.0),
            zorder=4,
        )

    ax.text(-0.08, 2, "MF", fontsize=9.8, fontweight="bold", ha="right", va="center", color="black")
    ax.text(-0.08, 1, "BP", fontsize=9.8, fontweight="bold", ha="right", va="center", color="black")
    ax.text(-0.08, 0, "CC", fontsize=9.8, fontweight="bold", ha="right", va="center", color="black")

    ruler_y = 2.40
    ax.plot([0, 1], [ruler_y, ruler_y], color="#555555", linewidth=1.7, zorder=1)
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    for t in ticks:
        ax.plot([t, t], [ruler_y, ruler_y + 0.08], color="#555555", linewidth=1.7, zorder=1)
        ax.text(
            t,
            ruler_y + 0.12,
            f"{t:.2f}",
            ha="center",
            va="bottom",
            fontsize=9.3,
            color="#333333",
            fontweight="bold",
        )
        ax.plot([t, t], [0, ruler_y], color="#cccccc", linestyle="--", linewidth=1.0, zorder=0, alpha=0.6)

    ax.axis("off")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.95, 2.72)
    ax.set_title(
        "B. Soft Labeling Scheme With Best Performance",
        fontsize=12.5,
        fontweight="bold",
        y=1.00,
    )


# =============================================================================
# 3. FEATURE ENGINEERING (SIMPLIFIED QUALITY TIERS, NUCLEAR & COMPOSITE RULES)
# =============================================================================
def engineer_features(df):
    """Engineer simplified boxplot-aligned quality tiers, nuclear rules [A]-[H], and margins."""
    target_score_col = "Sort Score Norm" if "Sort Score Norm" in df.columns else "Sort Score"

    for tier_name, (op, thr) in QUALITY_THRESHOLDS.items():
        if op == ">=":
            df[tier_name] = df[target_score_col] >= thr
        else:
            df[tier_name] = df[target_score_col] < thr

    all_neg_max = df[NEG_LABELS].max(axis=1)
    pos_min = df[POS_LABELS].min(axis=1)
    virt_min = df[VIRT_NEG_COLS].min(axis=1)
    virt_max = df[VIRT_NEG_COLS].max(axis=1)
    real_min = df[REAL_NEG_COLS].min(axis=1)
    real_max = df[REAL_NEG_COLS].max(axis=1)

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

    # Nuclear rules [A]-[H] ordered by ascending BH-adjusted p-value (A = lowest p-value)
    df["neg_below_pos"] = all_neg_max < pos_min                                      # [A]
    df["natural_polarity_split"] = (all_neg_max < 0.50) & (pos_min > 0.50)           # [B]
    df["pos_to_virtual_margin_relevant"] = df["pos_to_virtual_margin"] >= 0.126      # [C]
    df["derived_neg_le_conditional_neg"] = df["derived_not"] <= df["conditional_not"]  # [D]
    df["derived_neg_below_virtual_negs"] = df["derived_not"] <= virt_min             # [E]
    df["virtual_neg_higher"] = real_max < virt_min                                   # [F]
    df["conditional_neg_gt_real_neg_median"] = df["conditional_not"] > df["real_neg_median"]  # [G]
    df["random_false_val_lt_conditional_not"] = df["Random False Val"] <= df["conditional_not"]  # [H]

    # Composite / apriori rules [R1]-[R4]
    df["conditional_neg_top_anchor"] = (
        df["natural_polarity_split"]
        & df["derived_neg_le_conditional_neg"]
        & df["conditional_neg_gt_real_neg_median"]
        & df["random_false_val_lt_conditional_not"]
    )  # [R1] (B ∧ D ∧ G ∧ H)
    df["soft_tiers_2"] = (
        df["natural_polarity_split"]
        & df["derived_neg_below_virtual_negs"]
        & (df["virtual_neg_higher"] | df["random_false_val_lt_conditional_not"])
    )  # [R2] (B ∧ E ∧ (F ∨ H))
    df["soft_tiers_strict_chain"] = (
        df["neg_below_pos"]
        & df["derived_neg_below_virtual_negs"]
        & df["random_false_val_lt_conditional_not"]
    )  # [R3] (A ∧ E ∧ H)
    df["strict_4tier"] = (
        df["neg_below_pos"]
        & df["virtual_neg_higher"]
        & df["random_false_val_lt_conditional_not"]
    )  # [R4] (A ∧ F ∧ H)

    nuclear_cols = list(NUCLEAR_RULE_LETTERS.keys())
    composite_cols = list(COMPOSITE_RULE_EXPRESSIONS.keys())
    bool_pattern_cols = nuclear_cols + composite_cols

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
    """Mine composite boolean rules strictly from nuclear rules [A]-[H] and (F ∨ H)."""
    atom_df = pd.DataFrame(index=df.index)
    for col, letter in NUCLEAR_RULE_LETTERS.items():
        atom_df[letter] = df[col].astype(bool)
    atom_df["(F ∨ H)"] = atom_df["F"] | atom_df["H"]
    cand_atoms = ["A", "B", "C", "D", "E", "F", "G", "H", "(F ∨ H)"]

    known_composite_tags = {
        "B ∧ D ∧ G ∧ H": "[R1] Conditional- Top Neg Anchor",
        "B ∧ E ∧ (F ∨ H)": "[R2] Soft Tiers: Relaxed 3-Tier",
        "A ∧ E ∧ H": "[R3] Soft Tiers: Strict Chain",
        "A ∧ F ∧ H": "[R4] Strict 4-Tier Hierarchy",
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
                    if len(ants) <= 4 and not ("(F ∨ H)" in ants and ("F" in ants or "H" in ants))
                ]
            else:
                candidate_sets = [
                    comb
                    for k in range(1, 5)
                    for comb in combinations(cand_atoms, k)
                    if not ("(F ∨ H)" in comb and ("F" in comb or "H" in comb))
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

            def priority_key(row):
                atoms = row["Atoms"]
                has_c = 1 if "C" in atoms else 0
                has_disj = 1 if "(F ∨ H)" in atoms else 0
                return (row["Rule_Length"], has_c, has_disj, atoms)

            c_df["_prio"] = c_df.apply(priority_key, axis=1)
            c_df = (
                c_df.sort_values("_prio")
                .drop_duplicates(subset=["Mask"], keep="first")
                .drop(columns=["_prio"])
            )

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
    return mined_rules_df


# =============================================================================
# 5. STATISTICAL HYPOTHESIS TESTS ON BOOLEAN PATTERNS (ALL & PER ONTOLOGY)
# =============================================================================
def analyze_boolean_patterns(df, bool_pattern_cols, target_score_col, evi_exp_res_dir):
    """Run Mann-Whitney U, Point-Biserial, Rank-Biserial, and BH-FDR correction."""
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
    return pattern_stats_df


# =============================================================================
# 6. CONTINUOUS FEATURE CORRELATION ANALYSIS (SPEARMAN & PEARSON + BH-FDR)
# =============================================================================
def analyze_continuous_features(df, cont_engineered_cols, target_score_col, evi_exp_res_dir):
    """Evaluate continuous soft labels and engineered margins via Spearman & Pearson correlations."""
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
    return float_stats_df


# =============================================================================
# 7. PUBLICATION-READY MULTI-SHEET EXCEL EXPORT
# =============================================================================
def export_publication_excel(df, float_stats_df, pattern_stats_df, mined_rules_df, out_excel_path):
    """Build the multi-sheet Excel workbook."""
    with pd.ExcelWriter(out_excel_path, engine="openpyxl") as writer:
        float_stats_df.to_excel(writer, sheet_name="1_Continuous_Features", index=False)
        pattern_stats_df.to_excel(writer, sheet_name="2_Boolean_Rules", index=False)
        df.to_excel(writer, sheet_name="3_All_300_Tests", index=False)
        mined_rules_df.to_excel(writer, sheet_name="4_Mined_Composite_Rules", index=False)


# =============================================================================
# 8. UNIFIED 4-PANEL PUBLICATION FIGURE (PANELS A, B, C, D + FLOW ARROWS)
# =============================================================================
def plot_relationships(
    df,
    float_stats_df,
    pattern_stats_df,
    bool_pattern_cols,
    evi_exp_res_dir,
):
    """
    Create the unified 4-panel figure (`soft_optimization_relationships.png`):
      - Panel A (top-left): Classification Performance with Random Soft Label Values,
        with legend ordered [py-, de-, cu-, rd-, co-, cu+, py+] and vertical 'Elite ({n} tests)'
        separated cleanly on the left axis.
      - Panel B (extra compact bottom-left): Soft Labeling Scheme With Best Performance.
      - Panel C (top-right): Diverging horizontal bars centered at 0.0 (negative correlations
        extending left, positive correlations extending right), annotated with 'R=X, P-value=Y'.
      - Panel D (bottom-right): Blue ('True') and Orange ('False') boxplots with thicker lines
        and circles across MF, BP, CC, with atomic rules [H]..[A] and apriori rules [R4]..[R1]
        sorted strictly by BH-adjusted p-value descending (largest p-value at the top).
    """
    fig = plt.figure(figsize=(24.0, 15.2), facecolor="white")

    # Preserve user's parallel wspace=0.48
    outer_gs = gridspec.GridSpec(
        1, 2, figure=fig, width_ratios=[0.92, 1.52], wspace=0.48
    )

    # Left column: A even taller on top, B extra compact on bottom
    gs_left = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[0], height_ratios=[2.50, 0.52], hspace=0.22
    )
    ax_a = fig.add_subplot(gs_left[0, 0])
    ax_b = fig.add_subplot(gs_left[1, 0])

    # Right column: C on top, D on bottom (preserve user's hspace=0.32)
    gs_right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs[1], height_ratios=[0.95, 1.25], hspace=0.32
    )
    ax_c = fig.add_subplot(gs_right[0, 0])
    gs_d = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_right[1, 0], wspace=0.04
    )
    ax_d_list = [fig.add_subplot(gs_d[0, i]) for i in range(3)]

    # -------------------------------------------------------------------------
    # PANEL A: Classification Performance with Random Soft Label Values
    # -------------------------------------------------------------------------
    draw_panel_a_scatter(
        ax_a,
        df,
        title="A. Classification Performance with Random Soft Label Values",
    )

    # -------------------------------------------------------------------------
    # PANEL B: Soft Labeling Scheme With Best Performance (Extra Compact)
    # -------------------------------------------------------------------------
    draw_panel_b_evi_values_scale(ax_b, df)

    # -------------------------------------------------------------------------
    # PANEL C: Diverging Continuous Soft Labels & Margins Impact (Left=Neg, Right=Pos)
    # -------------------------------------------------------------------------
    all_runs_flt = float_stats_df[float_stats_df["Subset"] == "All_Runs"].copy()
    all_runs_flt["_abs_r"] = all_runs_flt["Spearman_r"].abs()
    all_float_order = (
        all_runs_flt.sort_values(by="_abs_r", ascending=True)["Soft_Label"].tolist()
    )

    y_pos = np.arange(len(all_float_order))
    bar_h = 0.25
    offsets = {"MF": -1.0 * bar_h, "BP": 0.0, "CC": +1.0 * bar_h}
    flt_lookup = float_stats_df.set_index(["Subset", "Soft_Label"])

    text_colors_c = {
        "MF": "#250C6F",
        "BP": "#1f4599",
        "CC": "#c44500",
    }

    # Central zero vertical axis separating negative (left) from positive (right) correlations
    ax_c.axvline(0.0, color="#222222", linestyle="-", linewidth=1.3, zorder=3)

    for ont in ONTOLOGIES:
        sp_signed_vals = [flt_lookup.loc[(ont, col), "Spearman_r"] for col in all_float_order]
        ax_c.barh(
            y_pos + offsets[ont],
            sp_signed_vals,
            height=bar_h,
            label=f"Spearman R ({ont})",
            color=ONTOLOGY_SPEARMAN_COLORS[ont],
            alpha=0.90,
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        for i, col in enumerate(all_float_order):
            r_val = flt_lookup.loc[(ont, col), "Spearman_r"]
            p_adj = flt_lookup.loc[(ont, col), "Spearman_p_adj_BH"]
            p_num = format_pvalue_number(p_adj)
            if r_val >= 0:
                x_txt = r_val + 0.015
                ha_align = "left"
            else:
                x_txt = r_val - 0.015
                ha_align = "right"
            ax_c.text(
                x_txt,
                i + offsets[ont],
                f"R={r_val:+.2f}, P-value={p_num}",
                va="center",
                ha=ha_align,
                fontsize=8.3,
                color=text_colors_c[ont],
                fontweight="bold",
            )

    for idx in range(len(all_float_order) - 1):
        ax_c.axhline(idx + 0.5, color="#e0e0e0", linestyle="-", linewidth=0.8)

    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels([parse_label_name(col) for col in all_float_order], fontsize=10.5, fontweight="bold")
    ax_c.tick_params(axis="x", labelsize=10.0)
    ax_c.set_xlim(-1.26, 1.12)
    ax_c.set_xlabel(
        "Spearman Rank Correlation R (Negative Left, Positive Right; BH-Adjusted P-value)",
        fontsize=10.8,
        fontweight="bold",
    )
    ax_c.set_title(
        "C. Correlation Between Soft Label Characteristics and Classification Performance",
        fontsize=12.5,
        fontweight="bold",
        pad=10,
    )
    ax_c.legend(title="Ontology", loc="lower right", framealpha=0.95, fontsize=9.5, title_fontsize=10.0)
    ax_c.grid(True, axis="x", linestyle="--", alpha=0.4)

    # -------------------------------------------------------------------------
    # PANEL D: Patterns Associated With 'Elite' Performance (>= 90%)
    #          - Blue for True (#2D5CCB), Orange for False (#ff5d00)
    #          - Thicker boxplot lines and outlier circles
    #          - Atomic rules [H]..[A] and Apriori rules [R4]..[R1] sorted by p-value descending
    # -------------------------------------------------------------------------
    pat_p_lookup = pattern_stats_df.set_index(["Subset", "Pattern"])

    nuclear_cols_raw = [c for c in NUCLEAR_RULE_LETTERS.keys() if c in bool_pattern_cols]
    composite_cols_raw = [
        c
        for c in [
            "conditional_neg_top_anchor",
            "soft_tiers_2",
            "soft_tiers_strict_chain",
            "strict_4tier",
        ]
        if c in bool_pattern_cols
    ]

    # Sort both groups by overall BH-adjusted p-value descending (larger p-value at the top)
    nuclear_cols = sorted(
        nuclear_cols_raw,
        key=lambda c: float(pat_p_lookup.loc[("All_Runs", c), "MannWhitney_p_adj_BH"]),
        reverse=True,
    )
    composite_cols = sorted(
        composite_cols_raw,
        key=lambda c: float(pat_p_lookup.loc[("All_Runs", c), "MannWhitney_p_adj_BH"]),
        reverse=True,
    )

    plot_bool_cols = nuclear_cols + composite_cols
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

    for idx_ont, (ont, ax_d) in enumerate(zip(ONTOLOGIES, ax_d_list)):
        ont_sub = pat_long_df[pat_long_df["Ontology"] == ont]
        sns.boxplot(
            data=ont_sub,
            x="Overall Score (%)",
            y="Pattern_Label",
            order=ordered_labels,
            hue="State",
            hue_order=["True", "False"],
            palette=BOXPLOT_RULE_PALETTE,
            width=0.72,
            fliersize=4.5,
            linewidth=1.55,
            ax=ax_d,
            boxprops=dict(alpha=0.88, linewidth=1.55),
            whiskerprops=dict(linewidth=1.55),
            capprops=dict(linewidth=1.55),
            medianprops=dict(linewidth=1.8, color="#1a1a1a"),
            flierprops=dict(
                marker="o",
                markersize=4.5,
                markeredgewidth=1.25,
                markeredgecolor="#333333",
                markerfacecolor="none",
                alpha=0.85,
            ),
        )

        # Dashed divider separating atomic rules (top) from composite rules (bottom)
        if len(nuclear_cols) > 0:
            ax_d.axhline(
                len(nuclear_cols) - 0.5,
                color="#555555",
                linestyle="--",
                linewidth=1.4,
                alpha=0.80,
            )

        for r_i in range(len(plot_bool_cols) - 1):
            if r_i != len(nuclear_cols) - 1:
                ax_d.axhline(r_i + 0.5, color="#e6e6e6", linestyle="-", linewidth=0.7)

        ont_scores = df.loc[df["Ontology"] == ont, "Sort Score"] * 100.0
        s_min, s_max = float(ont_scores.min()), float(ont_scores.max())
        pad = (s_max - s_min) * 0.05
        ax_d.set_xlim(s_min - pad, s_max + pad)
        ax_d.set_title(ONTOLOGY_TITLES[ont], fontsize=11.0, fontweight="bold", pad=6)
        ax_d.tick_params(axis="x", labelsize=10.0)
        ax_d.grid(True, axis="x", linestyle="--", alpha=0.45)

        if idx_ont == 0:
            ax_d.set_ylabel("")
            ax_d.tick_params(axis="y", labelsize=9.6)
            for lbl in ax_d.get_yticklabels():
                lbl.set_fontweight("bold")
            ax_d.spines["right"].set_visible(False)
            ax_d.legend(
                title="Rule State",
                loc="lower left",
                framealpha=0.95,
                fontsize=9.0,
                title_fontsize=9.5,
            )
        else:
            ax_d.set_ylabel("")
            ax_d.set_yticklabels([])
            ax_d.tick_params(axis="y", left=False)
            ax_d.spines["left"].set_visible(False)
            if idx_ont < len(ONTOLOGIES) - 1:
                ax_d.spines["right"].set_visible(False)
            if ax_d.get_legend() is not None:
                ax_d.get_legend().remove()

        if idx_ont == 1:
            ax_d.set_xlabel("Overall Score (%)", fontsize=10.8, fontweight="bold", labelpad=6)
        else:
            ax_d.set_xlabel("")

        # On the rightmost subplot (CC), display only the overall BH-adjusted p-value ("P-Value=X")
        if idx_ont == len(ONTOLOGIES) - 1:
            for r_i, pat_col in enumerate(plot_bool_cols):
                p_all = pat_p_lookup.loc[("All_Runs", pat_col), "MannWhitney_p_adj_BH"]
                p_num = format_pvalue_number(p_all)
                ax_d.text(
                    1.025,
                    r_i,
                    f"P-Value={p_num}",
                    transform=ax_d.get_yaxis_transform(),
                    ha="left",
                    va="center",
                    fontsize=9.2,
                    fontweight="bold",
                    color="#1f4599" if p_all < 0.001 else "#333333",
                    clip_on=False,
                )

    ax_d_list[1].text(
        0.5,
        1.09,
        "D. Patterns Associated With 'Elite' Performance (>= 90%)",
        transform=ax_d_list[1].transAxes,
        ha="center",
        va="bottom",
        fontsize=12.5,
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.062, right=0.925, top=0.945, bottom=0.05)
    fig.canvas.draw()

    # -------------------------------------------------------------------------
    # FLOW ARROWS CONNECTING A -> B, A -> C, and C -> D
    # -------------------------------------------------------------------------
    pos_a = ax_a.get_position()
    pos_b = ax_b.get_position()
    pos_c = ax_c.get_position()
    pos_d_mid = ax_d_list[1].get_position()

    arrow_style = "Simple,tail_width=3.2,head_width=12,head_length=11"
    arrow_color = "#2c3e50"

    # 1. Arrow from bottom of A to top of B
    x_ab = pos_a.x0 + 0.50 * pos_a.width
    y_a_bot = pos_a.y0 - 0.036
    y_b_top = pos_b.y1 + 0.022
    arr_ab = FancyArrowPatch(
        (x_ab, y_a_bot),
        (x_ab, y_b_top),
        transform=fig.transFigure,
        arrowstyle=arrow_style,
        color=arrow_color,
        alpha=0.88,
        zorder=20,
    )
    fig.add_artist(arr_ab)

    # 2. Arrow from right of A to left of C
    x_a_right = pos_a.x1 + 0.012
    x_c_left = pos_c.x0 - 0.095
    y_ac = pos_c.y0 + 0.50 * pos_c.height
    arr_ac = FancyArrowPatch(
        (x_a_right, y_ac),
        (x_c_left, y_ac),
        transform=fig.transFigure,
        arrowstyle=arrow_style,
        color=arrow_color,
        alpha=0.88,
        zorder=20,
    )
    fig.add_artist(arr_ac)

    # 3. Arrow from bottom of C to top of D
    x_cd = pos_d_mid.x0 + 0.50 * pos_d_mid.width
    y_c_bot = pos_c.y0 - 0.036
    y_d_top = pos_d_mid.y1 + 0.052
    arr_cd = FancyArrowPatch(
        (x_cd, y_c_bot),
        (x_cd, y_d_top),
        transform=fig.transFigure,
        arrowstyle=arrow_style,
        color=arrow_color,
        alpha=0.88,
        zorder=20,
    )
    fig.add_artist(arr_cd)

    rel_fig_path = os.path.join(evi_exp_res_dir, "soft_optimization_relationships.png")
    fig.savefig(rel_fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Saved unified 4-panel figure to: {rel_fig_path}]")


# =============================================================================
# 9. MAIN ORCHESTRATOR
# =============================================================================
def main():
    if len(sys.argv) < 2:
        raise ValueError("Usage: python evidence_exploration-6-plot_soft_labels.py <path_to_results_dir_or_tsv>")

    df, evi_exp_res_dir = resolve_paths_and_load_data(sys.argv[1])
    df, nuclear_cols, bool_pattern_cols, cont_engineered_cols, target_score_col = engineer_features(df)

    plot_soft_label_scatter(df, evi_exp_res_dir)
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
        pattern_stats_df,
        bool_pattern_cols,
        evi_exp_res_dir,
    )


if __name__ == "__main__":
    main()