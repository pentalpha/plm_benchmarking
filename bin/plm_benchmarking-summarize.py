import json
from glob import glob
import sys
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

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
    "flair-bio/amplify-350m": "amplify_350"
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

def model_family(model_name: str):
    if "ankh" in model_name:
        return "ankh"
    elif "e1" in model_name.lower():
        return "e1"
    elif "esm2" in model_name:
        return "esm2"
    elif "esmc" in model_name:
        return "esmc"
    elif "amplify" in model_name:
        return "amplify"
    else:
        print(f"Unknown model family: {model_name}")
        return "unknown"

def pretty_names(model_name):
    m = model_name.replace('_', ' ').title()
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

def pareto_frontier_plot(df: pd.DataFrame, output_path: str,
                        x_col: str = "size_millions", 
                        y_col: str = "Overall Score",
                        color_col: str = "model_family"):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    frontier_line_color = "#e74c3c"
    frontier_bg_color = "#ea9999"
    bg_color = "#FCFBF9"
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.7)
    ax.set_axisbelow(True)

    y_min = df[y_col].min() - (df[y_col].max() - df[y_col].min()) * 0.1
    
    # Set the drop height to exactly 50% of the Y-axis range so it fades out completely mid-air
    y_range = df[y_col].max() - df[y_col].min()
    gradient_drop = y_range * 0.50

    # --- 1. Compute and Plot the Global Frontier ---
    frontier_df = get_convex_pareto_frontier(df, x_col, y_col)
    
    ax.plot(frontier_df[x_col], frontier_df[y_col]-0.0005, 
            color=frontier_line_color, linestyle="--", linewidth=5, label="Efficiency Frontier", zorder=2)
    
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
        
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlabel("Model Size (millions of parameters)", labelpad=10, fontweight='bold', color='#444444')
    ax.set_ylabel("Overall Score (%)", labelpad=10, fontweight='bold', color='#444444')
    ax.set_title("Efficiency Frontier of Protein Language Models ", pad=15, fontsize=14)
    ax.set_ylim(bottom=y_min)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=400, facecolor=bg_color)

if __name__ == "__main__":
    benchmarking_dir = sys.argv[1]

    model_sizes = json.load(open("input_data/model_sizes.json"))
    results_files = glob(os.path.join(benchmarking_dir, "*", "all_results.tsv"))
    
    lines = []
    for results_file in results_files:
        df = pd.read_csv(results_file, sep="\t")
        dirname = os.path.basename(os.path.dirname(results_file))
        df["model_name"] = dirname
        original_name = model_original_names[dirname]
        df["original_name"] = original_name
        df["size_millions"] = model_sizes[original_name]
        df["model_family"] = model_family(original_name)
        df["pretty_name"] = pretty_names(dirname)
        lines.append(df)
    
    df = pd.concat(lines)
    df["Overall Score"] = df[["BP - Sort Score", "MF - Sort Score",
        "CC - Sort Score", "DEEPLOC - Sort Score"]].mean(axis=1)*100
    
    df = df.sort_values(by="Overall Score", ascending=False)
    df.to_csv(f"{benchmarking_dir}/results.tsv", sep="\t", index=False)
    pareto_frontier_plot(df, f"{benchmarking_dir}/pareto_frontier.png")
        
        
        