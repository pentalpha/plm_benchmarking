import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SOFT_LABEL_COLORS = {
    "phylo": "navy",
    "curated": "cornflowerblue",
    "conditional_not": "mediumvioletred",
    "Random False Val": "yellow",
    "curated_not": "red",
    "derived_not": "firebrick",
    "phylo_not": "maroon",
}

def parse_label_name(name):
    alt_names = {
        "phylo": "Phylogenetic+",
        "curated": "Curated+",
        "conditional_not": "Conditional-",
        "Random False Val": "Random-",
        "curated_not": "Curated-",
        "derived_not": "Derived-",
        "phylo_not": "Phylogenetic-",
    }
    return alt_names.get(name, name)

if __name__ == "__main__":
    if ".tsv" in sys.argv[1]:
        all_results_path = sys.argv[1]
        evi_exp_res_dir = os.path.dirname(all_results_path)
    else:
        evi_exp_res_dir = sys.argv[1]
        all_results_path = os.path.join(evi_exp_res_dir, "soft_optimization_test", "all_results.tsv")
    df = pd.read_csv(all_results_path, sep="\t")
    df = df[df['Ontology'].isin(['MF', 'CC', 'BP'])]
    print(df.columns)

    df_plot = []
    for _, row in df.iterrows():
        for label_name, labelc in SOFT_LABEL_COLORS.items():
            new_row = {'legend': parse_label_name(label_name), "color": labelc,
                "value": row[label_name],
                "Ontology": row['Ontology'], "OWA Score": row['OWA Score'], "CWA Score": row['CWA Score'],
                "Sort Score": row['Sort Score']}
            df_plot.append(new_row)
    df_plot = pd.DataFrame(df_plot)
    #score_cols = ['OWA Score', 'CWA Score', 'Sort Score']
    score_cols = ['Sort Score']

    fig, axes = plt.subplots(1, len(score_cols), figsize=(10, 8))
    axes_list = [axes[i] for i in range(len(score_cols))] if len(score_cols) > 1 else [axes]

    for label, label_df in df_plot.groupby('legend'):
        
        for ax, score_col in zip(axes_list, score_cols):
            ax.scatter(label_df['value'], label_df[score_col] * 100, s=280, alpha=0.75,
                c=label_df['color'], marker='o', label=label)

            ax.set_xlabel("Soft Label Value", fontsize=10, fontweight="bold")
            score_label = score_col if score_col != "Sort Score" else "Overall Score"
            ax.set_ylabel(f"{score_label} (%)", fontsize=10, fontweight="bold")

            ax.set_title(f"Soft Label Value vs {score_col}", fontsize=12, fontweight="bold")

            ax.grid(True, linestyle="--", alpha=0.5, zorder=0)

            ax.set_xlim(0, 1)
    
    axes_list[-1].legend(loc="lower right", bbox_to_anchor=(0.98, 0.02))
    fig.tight_layout()
    fig.savefig(os.path.join(evi_exp_res_dir, "soft_optimization_test.png"))
    plt.close()
    
    
        


        
    