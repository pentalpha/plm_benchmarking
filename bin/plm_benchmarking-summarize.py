import json
from glob import glob
import sys
import os
import matplotlib.gridspec as gridspec
import seaborn as sns
import math
import pandas as pd
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties
import numpy as np

from pddb_lib.sample_metaparameters import gdbt_params_list

#When two param_comb_id of very close values in Sort Score are present, choose the one with less forest complexity

free_serif_fontprop = FontProperties(family="FreeSerif", weight="light")

metrics = [
    ("Overall Score", ["Sort Score"]),
    ("OWA Score", ["OWA Fmax (Inverse-Weighted)", "OWA Weighted MCC", "OWA Weighted AUPRC"]),
    ("CWA Score", ["CAFA Weighted Fmax", "CAFA Weighted MCC", "CAFA AUPRC"]),
    ("Fmax", ["OWA Fmax (Inverse-Weighted)", "CAFA Weighted Fmax"]),
    ("MCC", ["OWA Weighted MCC", "CAFA Weighted MCC"]),
    ("AUPRC", ["OWA Weighted AUPRC", "CAFA AUPRC"])
]

model_simple_names = {
    "ElnaggarLab/ankh-base": "ankh_base",
    "ElnaggarLab/ankh-large": "ankh_large",
    "ElnaggarLab/ankh2-ext2": "ankh2_large",
    "ElnaggarLab/ankh3-large": "ankh3_large",
    "Profluent-Bio/E1-150m": "e1_150",
    "Profluent-Bio/E1-300m": "e1_300",
    "Profluent-Bio/E1-600m": "e1_600",
    "facebook/esm2_t30_150M_UR50D": "esm2_150",
    "facebook/esm2_t33_650M_UR50D": "esm2_650",
    "facebook/esm2_t36_3B_UR50D": "esm2_3b",
    "biohub/ESMC-300M-hf": "esmc_300",
    "biohub/ESMC-600M-hf": "esmc_600",
    "flair-bio/amplify-350m": "amplify_350",
    "biohub/ESMC-300M-hf": "esmc_300",
    "biohub/ESMC-600M-hf": "esmc_600",
    "flair-bio/amplify-120m": "amplify_120",
    "flair-bio/amplify-350m": "amplify_350",
}

model_emb_w = {
    "ElnaggarLab/ankh-base": 768,
    "ElnaggarLab/ankh-large": 1536,
    "ElnaggarLab/ankh2-ext2": 1536,
    "ElnaggarLab/ankh3-large": 1536,
    "Profluent-Bio/E1-150m": 768,
    "Profluent-Bio/E1-300m": 1024,
    "Profluent-Bio/E1-600m": 1280,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "facebook/esm2_t36_3B_UR50D": None,
    "biohub/ESMC-300M-hf": 960,
    "biohub/ESMC-600M-hf": 1152,
    "flair-bio/amplify-120m": 640,
    "flair-bio/amplify-350m": 960,
    "biohub/ESMC-300M-hf": 960,
    "biohub/ESMC-600M-hf": 1152,
}

model_original_names = {v: k for k, v in model_simple_names.items()}


# Define specific marker shapes for each family
MARKER_MAP = {
    "ankh": "o",       # Circle
    "e1": "s",         # Square
    "esm2": "^",       # Triangle Up
    "esmc": "D",       # Diamond
    "amplify": "v",    # Triangle Down
    "unknown": "x"     # Cross
}


def pyboost_complexity_idx(row):
    # Parameters that increase complexity (Structural Capacity)
    max_depth = row["param_max_depth"]
    ntrees = row["param_ntrees"]
    max_bin = row["param_max_bin"]
    
    # Parameters that decrease complexity (Growth Constraints)
    min_data_in_leaf = row["param_min_data_in_leaf"]
    min_gain_to_split = row["param_min_gain_to_split"]
    
    # 1. Calculate the unconstrained potential of the forest
    # 2^max_depth represents the maximum possible leaves per tree
    tree_capacity = 2 ** max_depth
    forest_capacity = ntrees * tree_capacity
    resolution = max_bin 
    
    numerator = forest_capacity * resolution
    
    # 2. Calculate the penalizing constraints
    # (1 + min_gain_to_split) prevents division by zero
    denominator = min_data_in_leaf * (1 + min_gain_to_split)
    
    # 3. Calculate Index
    # Use log10 to compress the scale into a highly readable float
    complexity_ratio = numerator / denominator
    
    return math.log10(complexity_ratio)

def model_family(model_name: str):
    m_low = model_name.lower()
    if "ankh3" in m_low:
        return "ankh3"
    elif "ankh2" in m_low:
        return "ankh2"
    elif "ankh" in m_low:
        return "ankh"
    elif "e1" in m_low:
        return "e1"
    elif "esm2" in m_low:
        return "esm2"
    elif "esmc" in m_low:
        return "esmc"
    elif "amplify" in m_low:
        return "amplify"
    else:
        print(f"Unknown model family: {model_name}")
        return "unknown"

def pretty_names(model_name):
    m = model_name.replace('_', ' ').title()
    if "nlu" in m.lower():
        m = m.replace(" Nlu", "")
    m = m.replace("Esm", "ESM").replace('SMc', "SMC")
    return m

def get_convex_pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    df_sorted = df.sort_values(by=[x_col, y_col], ascending=[True, False]).reset_index(drop=True)
    
    pareto = []
    max_y = -float('inf')
    for _, row in df_sorted.iterrows():
        if row[y_col] > max_y:
            pareto.append(row)
            max_y = row[y_col]
            
    def cross_product(o, a, b):
        return (a[x_col] - o[x_col]) * (b[y_col] - a[y_col]) - (a[y_col] - o[y_col]) * (b[x_col] - a[x_col])

    hull = []
    for p in pareto:
        while len(hull) >= 2 and cross_product(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(p)
        
    return pd.DataFrame(hull)

def add_gradient_fill(ax, x, y, color, drop_height, max_alpha, zorder=1, n_steps=100):
    """Simulates a gradient fade that strictly follows the line's topography."""
    x_vals = np.array(x)
    y_vals = np.array(y)
    
    # Linearly decrease alpha from max_alpha to 0
    alphas = np.linspace(max_alpha, 0.0, n_steps)
    
    for i in range(n_steps - 1):
        y_top = y_vals - (i / n_steps) * drop_height
        y_bottom = y_vals - ((i + 1) / n_steps) * drop_height
        
        ax.fill_between(x_vals, y_bottom, y_top, 
                        color=color, alpha=alphas[i], 
                        zorder=zorder, linewidth=0, edgecolor='none')

def pareto_frontier_plot(df_all: pd.DataFrame, output_path: str,
                        x_col: str = "size_millions", 
                        y_col: str = "Sort Score",
                        color_col: str = "model_family"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    df_all[y_col] = df_all[y_col] * 100

    ontology_to_coords = {
        "MF": (0,0),
        "CC": (0,1),
        "BP": (1,0),
        "DEEPLOC": (1,1)
    }
    
    frontier_line_color = "#e74c3c"
    frontier_bg_color = "#ea9999"
    bg_color = "#FCFBF9"
    fig.patch.set_facecolor(bg_color)

    global_top_y = df_all[y_col].max() + 1.5
    global_bottom_y = df_all[y_col].min() - 1.5
    global_y_range = global_top_y - global_bottom_y

    for ont, ax_coords in ontology_to_coords.items():
        df = df_all[df_all["ontology"] == ont]
        ax = axes[ax_coords[0], ax_coords[1]]
        ax.set_facecolor(bg_color)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linestyle='-', which='major', 
            color='lightgrey', alpha=0.7)
        ax.set_axisbelow(True)

        y_min = df[y_col].min() - (df[y_col].max() - df[y_col].min()) * 0.1
        
        # Set the drop height to exactly 50% of the Y-axis range so it fades out completely mid-air
        y_range = df[y_col].max() - df[y_col].min()
        gradient_drop = y_range * 0.50

        # --- 1. Compute and Plot the Global Frontier ---
        frontier_df = get_convex_pareto_frontier(df, x_col, y_col)
        
        ax.plot(frontier_df[x_col], frontier_df[y_col]-0.0005, 
                color=frontier_line_color, linestyle="--", linewidth=5,
                 label="Efficiency Frontier", zorder=2)
        
        # EXCLUSIVELY use the gradient fill. The static flat-bottom fill_between has been deleted.
        add_gradient_fill(ax, frontier_df[x_col], frontier_df[y_col]-0.0005, 
                        color=frontier_bg_color, drop_height=gradient_drop, max_alpha=0.45, zorder=1)

        # --- 2. Plot all models individually by family ---
        groups = df.groupby(color_col)
        
        for name, group in groups:
            marker = MARKER_MAP.get(name.lower(), "o")
            group = group.sort_values(by=x_col)
            
            line, = ax.plot(group[x_col], group[y_col], label=name.upper(), 
                            marker=marker, markersize=9, linewidth=2, zorder=3)
            
            # Add the parallel gradient for the individual lines, making it slightly shorter so it doesn't clutter
            add_gradient_fill(ax, group[x_col], group[y_col], 
                            color=line.get_color(), drop_height=gradient_drop * 0.6, max_alpha=0.15, zorder=1)
            
            for i, row in group.iterrows():
                pos_dot = (row[x_col], row[y_col])
                
                is_frontier = (row['model_name'] in frontier_df['model_name'].values)
                y_offset = 12 if is_frontier else 8
                
                ax.annotate(row["pretty_name"], pos_dot, 
                            xytext=(0, y_offset), textcoords="offset points", 
                            fontsize=9, color="#333333",
                            ha='center', va='bottom', zorder=4,
                            path_effects=[pe.withStroke(linewidth=3, foreground=bg_color)])
        if ont == "DEEPLOC":
            ax.legend(frameon=False, loc="lower right")
        ax.set_title(ont, pad=15, fontsize=14)
        
        ax.set_ylim(bottom=global_bottom_y, top=global_top_y)
    # Sup-labels with explicit position coordinates to push them outward
    fig.supylabel("Overall Score (%)", fontweight='bold', color='#444444', x=0.015)
    fig.supxlabel("Model Size (millions of parameters)", fontweight='bold', color='#444444', y=0.02)
    
    # Adjust layout padding so the outward-pushed labels don't get cropped
    fig.tight_layout(rect=[0.02, 0.02, 1, 1])
    fig.savefig(output_path, dpi=400, facecolor=bg_color)

def pareto_frontier_oneplot(df: pd.DataFrame,
                            ax: plt.Axes,
                            x_col: str = "size_millions", 
                            y_col: str = "Sort Score",
                            color_col: str = "model_family",
                            global_top_y: float = 90.0,
                            global_bottom_y: float = 60.0,
                            bg_color: str = "#FCFBF9"):
    """Plots the Pareto frontier and scatter points for model performances."""
    frontier_line_color = "#e74c3c"
    frontier_bg_color = "#ea9999"

    ax.set_facecolor(bg_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.7)
    ax.set_axisbelow(True)

    y_range = df[y_col].max() - df[y_col].min()
    gradient_drop = y_range * 0.50

    # --- 1. Compute and Plot the Global Frontier ---
    frontier_df = get_convex_pareto_frontier(df, x_col, y_col)
    
    ax.plot(frontier_df[x_col], frontier_df[y_col] - 0.0005, 
            color=frontier_line_color, linestyle="--", linewidth=5,
            label="Efficiency Frontier", zorder=2)
    
    add_gradient_fill(ax, frontier_df[x_col], frontier_df[y_col] - 0.0005, 
                      color=frontier_bg_color, drop_height=gradient_drop, max_alpha=0.45, zorder=1)

    # --- 2. Plot all models individually by family ---
    groups = df.groupby(color_col)
    
    for name, group in groups:
        marker = MARKER_MAP.get(name.lower(), "o")
        group = group.sort_values(by=x_col)
        
        line, = ax.plot(group[x_col], group[y_col], label=name.upper(), 
                        marker=marker, markersize=9, linewidth=2, zorder=3)
        
        add_gradient_fill(ax, group[x_col], group[y_col], 
                          color=line.get_color(), drop_height=gradient_drop * 0.6, max_alpha=0.15, zorder=1)
        
        for _, row in group.iterrows():
            pos_dot = (row[x_col], row[y_col])
            is_frontier = (row['model_name'] in frontier_df['model_name'].values)
            y_offset = 13 if is_frontier else 8
            
            ax.annotate(row["pretty_name"], pos_dot, 
                        xytext=(0, y_offset), textcoords="offset points", 
                        fontsize=9 if is_frontier else 8,
                        color="#333333",
                        ha='center', va='bottom', zorder=5 if is_frontier else 4,
                        fontweight="bold" if is_frontier else "normal",
                        alpha=0.8 if is_frontier else 0.95,
                        path_effects=[pe.withStroke(linewidth=2, foreground=bg_color)])
            
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(bottom=global_bottom_y, top=global_top_y)
    
    ax.set_ylabel(f"Overall Score (%)", fontweight='bold', color='#444444')
    ax.set_xlabel("Model Size (millions of parameters)", fontweight='bold', color='#444444')
    
    # Use ScalarFormatter on a log scale to safely show 500, 1000 instead of 10^x
    #ax.set_xscale("log")
    formatter = mtick.ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(formatter)
    
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=7)
    ax.xaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.7)
    # ignore minors:
    ax.xaxis.set_minor_locator(mtick.MultipleLocator(3000))
    # X major is every 500:
    ax.xaxis.set_major_locator(mtick.MultipleLocator(400))


def plot_benchmark_heatmap(df_all: pd.DataFrame, ax: plt.Axes, y_col: str = "Sort Score"):
    """Generates a stylized heatmap matching the aesthetics of standard PLM benchmark papers."""
    # 1. Create a pivot table calculating the mean score for each model and ontology
    pivot_df = df_all.pivot_table(index="pretty_name", columns="ontology", values=y_col, aggfunc="mean")
    
    # Extract model metadata (size and embed width)
    meta_df = df_all[['pretty_name', 'embed_width', 'size_millions']].drop_duplicates().set_index('pretty_name')
    
    # Merge metadata into pivot table and rename for display
    pivot_df = pivot_df.join(meta_df)
    pivot_df = pivot_df.rename(columns={
        'embed_width': 'Embedding\nWidth',
        'size_millions': 'Million\nParameters'
    })
    
    # 2. Enforce the exact column order requested
    cols = ["Embedding\nWidth", "Million\nParameters", "MF", "CC", "BP", "DEEPLOC"]
    existing_cols = [c for c in cols if c in pivot_df.columns]
    pivot_df = pivot_df[existing_cols]
    #pivot_df.insert(2, " ", np.nan)
    
    # 3. Sort models by their average performance on the metrics (excluding meta columns)
    metric_cols = [c for c in ["MF", "CC", "BP", "DEEPLOC"] if c in pivot_df.columns]
    pivot_df["Average"] = pivot_df[metric_cols].mean(axis=1)
    pivot_df = pivot_df.sort_values(by="Average", ascending=False)
    pivot_df = pivot_df.drop(columns=["Average"]) 
    
    # --- NEW: Inject empty rows to leave space for future models ---
    #future_rows = pd.DataFrame(np.nan, index=["Future Model A", "Future Model B"], columns=pivot_df.columns)
    #pivot_df = pd.concat([pivot_df, future_rows])
    
    # 4. Normalize data per-column (0 to 1) so each column has its own color scale
    # NaNs will be ignored during min/max calculation and remain NaNs
    normalized_df = (pivot_df - pivot_df.min()) / (pivot_df.max() - pivot_df.min())

    # For ["Embedding\nWidth", "Million\nParameters"], less is better. So we should invert these values
    #normalized_df[["Embedding\nWidth", "Million\nParameters"]] = 1 - normalized_df[["Embedding\nWidth", "Million\nParameters"]]
    
    # Build a custom annotation array handling the empty future rows gracefully
    annot_array = []
    for _, row in pivot_df.iterrows():
        row_annots = []
        for col in pivot_df.columns:
            val = row[col]
            if pd.isna(val):
                row_annots.append("") # Leave cell blank if NaN
            elif col in ["Embedding\nWidth", "Million\nParameters"]:
                row_annots.append(f"{val:.0f}") # No decimals for size/width
            else:
                row_annots.append(f"{val:.1f}") # 1 decimal for scores
        annot_array.append(row_annots)
    
    # Create masks to isolate the two sections
    mask_dimensions = np.zeros_like(normalized_df, dtype=bool)
    mask_dimensions[:, 2:] = True  # Hide the score columns

    mask_scores = np.zeros_like(normalized_df, dtype=bool)
    mask_scores[:, :2] = True      # Hide the dimension columns
    
    # Plot Dimensions with a neutral/different colormap (e.g., 'bone_r', 'Blues', 'Greys')
    sns.heatmap(normalized_df, 
                annot=annot_array, fmt="", cmap="flare",          
                linewidths=3, linecolor='white', cbar=False, 
                mask=mask_dimensions, ax=ax)

    # Plot Scores with Viridis
    sns.heatmap(normalized_df, 
                annot=annot_array, fmt="", cmap="viridis",          
                linewidths=3, linecolor='white', cbar=False, 
                mask=mask_scores, ax=ax)
    
    # 6. Format the axes
    ax.set_ylabel("") 
    ax.set_xlabel("")
    
    #ax.xaxis.tick_top()
    ax.tick_params(axis='x', rotation=0, labelsize=9.5, labelcolor='black', length=0)
    
    # --- NEW: Reduced font size to 9.5 and increased padding to 12 to prevent overlap ---
    ax.tick_params(axis='y', rotation=0, labelsize=9.5, labelcolor='black', length=0, pad=12)

    # 7. Add grouping brackets and labels below the x-axis using TextPath
    def add_stretched_brace(ax, x_center, x_width, y_pos, text_label, y_text_offset):
        # 1. Create the base TextPath for the brace using your preferred font
        tp = TextPath((0, 0), "}", size=40, prop=free_serif_fontprop)
        bbox = tp.get_extents()
        
        # 2. Extract original dimensions of the glyph
        orig_width = bbox.width   # Will map to the vertical thickness
        orig_height = bbox.height # Will map to the horizontal span
        
        # Calculate the exact center of the original glyph
        cx = bbox.x0 + orig_width / 2.0
        cy = bbox.y0 + orig_height / 2.0
        
        # 3. Calculate scale factors
        # Lock the vertical thickness of the bracket to a constant axes fraction (e.g., 2.5% of figure height)
        thickness = 0.019 
        
        # sy applies to the original height (which rotates to become the horizontal width across columns)
        sy = x_width / orig_height
        
        # sx applies to the original width (which rotates to become the elegant vertical thickness)
        sx = thickness / orig_width
        
        # 4. Build the transform: center -> scale -> rotate (point up) -> translate -> axes coordinates
        transform = (Affine2D()
                     .translate(-cx, -cy)
                     .scale(sx=sx, sy=sy)
                     .rotate_deg(-90)
                     .translate(x_center, y_pos)
                     + ax.get_xaxis_transform())
                     
        # 5. Create and add the path patch
        # clip_on=False ensures it renders outside the bounds of the main graph
        patch = PathPatch(tp, transform=transform, facecolor="#111111", edgecolor="none", clip_on=False)
        ax.add_patch(patch)
        
        # 6. Add the label below the brace
        ax.text(x_center, y_pos + y_text_offset, text_label, ha='center', va='top', 
                fontsize=11, fontfamily="FreeSerif", color="#111111",
                transform=ax.get_xaxis_transform())

    # --- 1st Bracket: Model Dimensions ---
    # Center is at x=1.0. We set the width to 1.9 to leave a tiny gap at the edges of the 2 columns.
    add_stretched_brace(ax, x_center=1.0, x_width=1.7, y_pos=-0.08, 
                        text_label="Model Dimensions", y_text_offset=-0.018)

    # --- 2nd Bracket: Task-Specific Scores ---
    # Center is at x=4.0. We set the width to 3.9 so it perfectly blankets the 4 columns.
    add_stretched_brace(ax, x_center=4.0, x_width=3.7, y_pos=-0.08, 
                        text_label="Task-specific Scores", y_text_offset=-0.018)


def pareto_heatmap_combined_plot(df_all: pd.DataFrame, output_path: str,
                                 x_col: str = "size_millions", 
                                 y_col: str = "Sort Score",
                                 color_col: str = "model_family"):
    """Main wrapper function to generate a 1x2 figure with scatter and heatmap."""
    bg_color = "#FCFBF9"
    
    # 1. Setup Figure and GridSpec
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor(bg_color)
    
    # --- NEW: Increased wspace from 0.15 to 0.28 to push the heatmap further right ---
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.3, 1], wspace=0.28)
    ax_scatter = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])

    # 2. Aggregate data elegantly using pandas groupby instead of a manual loop
    df_agg = df_all.groupby(["model_name", "pretty_name", color_col], as_index=False)[[x_col, y_col]].mean()

    global_top_y = df_all[y_col].max()
    global_bottom_y = df_all[y_col].min()

    # 3. Populate axes via subfunctions
    pareto_frontier_oneplot(df_agg, ax_scatter, x_col=x_col, y_col=y_col, color_col=color_col, 
                            global_top_y=global_top_y, global_bottom_y=global_bottom_y, 
                            bg_color=bg_color)
    
    plot_benchmark_heatmap(df_all, ax_heat, y_col=y_col)

    ax_scatter.set_title("Efficiency Frontier of PLMs", fontsize=14, weight='bold')
    ax_heat.set_title("Heatmap of Task-Specific Scores", fontsize=14, weight='bold')

    # 4. Finalize and save
    fig.tight_layout()
    fig.savefig(output_path, dpi=400, facecolor=bg_color, bbox_inches='tight')
    fig.savefig(output_path.replace('.png', '.svg'), dpi=400, facecolor=bg_color, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    benchmarking_dir = "outputs/remote_results/plm_benchmarking/"#sys.argv[1]

    model_sizes = json.load(open("input_data/model_sizes.json"))
    for prefix in ["nlu", "s2s"]:
        name2 = "ElnaggarLab/ankh3-large"+'_'+prefix
        model_original_names[f"ankh3_large_{prefix}"] = name2
        model_emb_w[name2] = 1536
        model_sizes[name2] = model_sizes["ElnaggarLab/ankh3-large"]
    
    results_files = glob(os.path.join(benchmarking_dir, "*", "results_eval.json"))
    results_files += glob(os.path.join(benchmarking_dir, "*", "model_*", "results_eval.json"))

    print("Results files:", len(results_files))
    
    lines = []
    for results_file in results_files:
        row = json.load(open(results_file))
        model_dir = os.path.basename(os.path.dirname(results_file))
        if "ankh3" in results_file and not "nlu" in results_file:
            #Ignore ankh3-large when NLU token is not used
            continue
        if "model_" in model_dir:
            param_comb_id = model_dir.split("_")[-1]
            model_dir = os.path.basename(os.path.dirname(os.path.dirname(results_file)))
        else:
            param_comb_id = "default"
        
        row["model_name"] = model_dir
        original_name = model_original_names[model_dir]
        row["original_name"] = original_name
        row["embed_width"] = model_emb_w[original_name]
        row["large_emb"] = model_emb_w[original_name] > 950
        row["size_millions"] = model_sizes[original_name]
        row["model_family"] = model_family(original_name)
        row["pretty_name"] = pretty_names(model_dir)
        row["param_comb_id"] = param_comb_id

        if "parameters" in row:
            for key, v in json.loads(row["parameters"]).items():
                if key in gdbt_params_list:
                    row[f'param_{key}'] = v

            row["forest_complexity_index"] = pyboost_complexity_idx(row)
        else:
            row["forest_complexity_index"] = float('nan')

        ont_rows = []
        for ont in ['DEEPLOC', "MF", "CC", "BP"]:
            ont_row = {k.replace(f'{ont} - ', ''): v for k, v in row.items() if (not ' - ' in k) or (f'{ont} - ' in k)}
            if "Sort Score" in ont_row:
                ont_row['ontology'] = ont
                ont_rows.append(ont_row)
            else:
                if "parameters" in row:
                    if len(row["parameters"]) > 10:
                        ont_row["Sort Score"] = -99999
                        ont_row["ontology"] = ont
                        ont_rows.append(ont_row)
        lines += ont_rows

    
    df = pd.DataFrame(lines)
    df["Sort_Score_Binned"] = ((df["Sort Score"] * 100)*10).round() / 10  # Groups by 0.05 increments
    df = df.sort_values(
        by=["Sort_Score_Binned", "forest_complexity_index"], 
        ascending=[False, True]
    )
    df.to_csv(f"{benchmarking_dir}/results_all.tsv", sep="\t", index=False)
    
    #Get best Sort Score by model_name
    df_best = []
    no_oom_df = df[df['Sort Score'] >= 0]
    
    for model_name, group_df in no_oom_df.groupby(["model_name", "ontology"]):
        #already sorted, get first value
        best_row = group_df.iloc[0]
        best_row = {k: v for k, v in best_row.items()}
        best_row["N_Tests"] = len(group_df)
        df_best.append(best_row)

    df_best = pd.DataFrame(df_best)
    df_best = df_best.sort_values(by="Sort Score", ascending=False)
    df_best.to_csv(f"{benchmarking_dir}/results_best_overall.tsv", sep="\t", index=False)
    pareto_frontier_plot(df_best, f"{benchmarking_dir}/pareto_frontier_overall.png", y_col="Sort Score")
    pareto_heatmap_combined_plot(df_best, f"{benchmarking_dir}/pareto_frontier_overall_combined.png", y_col="Sort Score")
    

    cols_simple = ["original_name", "size_millions", "embed_width", "model_family", "ontology", "Sort Score", 
        "N_Tests", "param_comb_id", "forest_complexity_index", "parameters"]
    cols_simple2 = ["original_name", "size_millions", "embed_width", "model_family", "ontology", "Sort Score", 
        "param_comb_id", "forest_complexity_index", "parameters"]

    df_best[cols_simple].to_csv(f"{benchmarking_dir}/results_best_simple.tsv", sep="\t", index=False)
    df[cols_simple2].to_csv(f"{benchmarking_dir}/results_all_simple.tsv", sep="\t", index=False)

    #Find out at which metrics large_emb=True beats large_emb=False more or less
    

    m_comp_results = []

    for m_name, m_cols in metrics:
        print(m_name, m_cols)
        large_emb_true_df = df_best[df_best["size_millions"] > 1100]
        large_emb_false_df = df_best[df_best["size_millions"] < 1100]

        l_true_scores = large_emb_true_df[m_cols].mean(axis=1)
        #print(l_true_scores)
        l_true_mean = l_true_scores.mean()
        #print(l_true_mean)
        l_false_scores = large_emb_false_df[m_cols].mean(axis=1)
        #print(l_false_scores)
        l_false_mean = l_false_scores.mean()
        #print(l_false_mean)
        
        diff = l_true_mean - l_false_mean
        print(diff)
        
        print("\n\n")

        m_comp_results.append(
            {
                "metric": m_name,
                "diff": diff
            }
        )

    m_comp_results = pd.DataFrame(m_comp_results)
    m_comp_results.to_csv(f"{benchmarking_dir}/metric_comparison.tsv", sep="\t", index=False)
        
        
        
        
        
        
        