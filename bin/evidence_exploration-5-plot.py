import sys
import os
import json
from collections import Counter

import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib import colormaps as cm
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns

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
    "Conditional Negatives": "lightpink",
    "Conditional Negatives + RNS": "red",
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
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

    ax.set_xlabel("CWA Score (%)", fontweight="bold", fontsize=10)
    ax.set_ylabel("OWA Score (%)", fontweight="bold", fontsize=10)
    ax.set_xlim(42, 67)
    ax.set_ylim(49, 93)
    ax.set_title("B: Optimized OWA and CWA Performance", fontsize=12, fontweight="bold", y=1.02)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)

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
    
    alg_names = [x for x in df_evi["Algorithm"].unique()]
    all_names = list(all_names)
    params_matrix = []
    for _, row in df_evi.iterrows():
        params_dict = json.loads(row["Parameters"])
        alg_cols = [int(alg_name in row["Algorithm"]) for alg_name in alg_names]
        is_mf = 1 if "MF" in row["Ontology"] else 0
        is_bp = 1 if "BP" in row["Ontology"] else 0
        is_cc = 1 if "CC" in row["Ontology"] else 0
        vals = [params_dict.get(name, -1) for name in all_names]
        params_matrix.append(vals)
    print([len(r) for r in params_matrix])
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

def plot_scatter(
    ax,
    df,
    x_col,
    y_col,
    c_col,
    x_label,
    y_label,
    title,
    marker_col="Ontology",
    marker_map=MARKER_MAP,
    perc_x = None,
    perc_y = None
):
    """
    Plots OWA vs CWA metric scatter points encoded by Strategy (color),
    Ontology (marker shape), and Top-Performing status (black outline + halo).
    """
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    ax.grid(True, linestyle="--", color="#D5D5D5", alpha=0.65, zorder=0)
    ax.set_axisbelow(True)

    multi = 1 if 'TSNE' in x_col else 100.0

    # 1. Non-top configurations (background layer)
    df_bg = df[~df["is_top_performance"]]
    for marker_val, group_df in df_bg.groupby(marker_col, sort=False):
        if group_df.empty:
            continue
        ax.scatter(
            group_df[x_col] * multi,
            group_df[y_col] * multi,
            c=group_df[c_col],
            s=55,
            alpha=0.68,
            edgecolors="#333333",
            linewidths=0.3,
            zorder=3,
            marker=marker_map[marker_val],
        )

    '''# 2. Top-performing configurations (foreground layer with white halo + black outline)
    df_top = df[df["is_top_performance"]]
    for marker_val, group_df in df_top.groupby(marker_col, sort=False):
        if group_df.empty:
            continue
        x_vals = group_df[x_col] * multi
        y_vals = group_df[y_col] * multi
        marker = marker_map[marker_val]

        ax.scatter(
            x_vals,
            y_vals,
            facecolors="none",
            edgecolors="white",
            s=195,
            linewidths=2.6,
            alpha=0.95,
            zorder=5,
            marker=marker,
        )
        ax.scatter(
            x_vals,
            y_vals,
            c=group_df[c_col],
            s=145,
            alpha=1.0,
            edgecolors="black",
            linewidths=1.8,
            zorder=6,
            marker=marker,
        )'''
    
    if perc_x is not None:
        percx_line_y_start = df[y_col].min()*100
        percx_line_y_end = df[y_col].max()*100
        percy_line_x_start = df[x_col].min()*100
        percy_line_x_end = df[x_col].max()*100
        
        ax.plot([perc_x, perc_x], [percx_line_y_start, percx_line_y_end], c="#333333", ls='--', lw=1)
        ax.plot([percy_line_x_start, percy_line_x_end], [perc_y, perc_y], c="#333333", ls='--', lw=1)

    ax.set_xlabel(x_label, fontsize=10, fontweight="bold", color="#333333", labelpad=5)
    ax.set_ylabel(y_label, fontsize=10, fontweight="bold", color="#333333", labelpad=5)
    ax.set_title(title, fontsize=12, fontweight="bold", color="#222222", pad=8)
    ax.tick_params(axis="both", which="major", labelsize=8, colors="#333333")
    ax.margins(0.06)


def plot_metrics_from_df(df, results_dir, percentiles=None):
    plot_defs = [
        [
            {
                "x": "OWA Weighted MCC",
                "xlabel": "OWA MCC (%)",
                "y": "CAFA Weighted MCC",
                "ylabel": "CWA MCC (%)",
                "c": "AlgColors",
                "title": "A: Matthews Correlation Coefficient (MCC)",
            },
            {
                "x": "OWA Weighted AUPRC",
                "xlabel": "OWA AUPRC (%)",
                "y": "CAFA AUPRC",
                "ylabel": "CWA AUPRC (%)",
                "c": "AlgColors",
                "title": "Individual Metrics:\n\nB: Area Under Precision-Recall Curve (AUPRC)",
            },
            {
                "x": "OWA Fmax (Inverse-Weighted)",
                "xlabel": "OWA Fmax (%)",
                "y": "CAFA Weighted Fmax",
                "ylabel": "CWA Fmax (%)",
                "c": "AlgColors",
                "title": "C: Maximum F-Measure (Fmax)",
            },
        ],
        [
            
            {
                "x": "OWA Score",
                "xlabel": "Avg. OWA (%)",
                "y": "CWA Score",
                "ylabel": "Avg. CWA (%)",
                "c": "AlgColors",
                "title": "D: Molecular Function (MF)",
                "single_ont": "MF"
            },
            {
                "x": "OWA Score",
                "xlabel": "Avg. OWA (%)",
                "y": "CWA Score",
                "ylabel": "Avg. CWA (%)",
                "c": "AlgColors",
                "title": "Overall CWA vs OWA Trade-off:\n\nE: Biological Process (BP)",
                "single_ont": "BP"
            },
            {
                "x": "OWA Score",
                "xlabel": "Avg. OWA (%)",
                "y": "CWA Score",
                "ylabel": "Avg. CWA (%)",
                "c": "AlgColors",
                "title": "F: Cellular Component (CC)",
                "single_ont": "CC"
            },
        ],
    ]

    df = df.copy()
    df["AlgColors"] = [MARKER_COLOR_MAP[n] for n in df["Algorithm"].tolist()]

    bg_color = "#FCFBF9"
    fig = plt.figure(figsize=(16, 10), facecolor=bg_color)

    # Outer GridSpec with explicit margins:
    # - top=0.935 brings the legend right up under suptitle
    # - left=0.07, right=0.98 eliminates side margins
    # - hspace=0.08 keeps the legend close to the top row of plots
    outer_gs = gridspec.GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[0.15, 1.0],
        hspace=0.10,
        top=0.935,
        bottom=0.05,
        left=0.07,
        right=0.98,
    )

    # --- Compact 3-Row Legend Panel (Row 0) ---
    ax_leg = fig.add_subplot(outer_gs[0])
    ax_leg.axis("off")

    # Row 1: Strategy Colors
    color_handles = [
        Line2D([0], [0], color="none", label="Strategy (Color):"),
    ] + [
        mpatches.Patch(
            facecolor=color,
            edgecolor="#555555",
            linewidth=0.8,
            label=strategy,
        )
        for strategy, color in MARKER_COLOR_MAP.items()
    ]

    # Row 2: Ontology Shapes
    ONT_LABELS = {
        "MF": "MF (Molecular Function)",
        "BP": "BP (Biological Process)",
        "CC": "CC (Cellular Component)",
    }
    shape_handles = (
        [
    #    Line2D([0], [0], color="none", label="Ontology (Shape):"),
    #] + [
        
        Line2D(
            [0], [0],
            marker=marker,
            color="none",
            label=ONT_LABELS.get(ont, ont),
            markerfacecolor="#777777",
            markeredgecolor="#333333",
            markeredgewidth=0.8,
            markersize=8,
        )
        for ont, marker in MARKER_MAP.items()
        ]
        +
        [
            #Explain percentile line: c="#333333", ls='--'
            Line2D(
                [0], [0],
                color="#333333",
                ls='--',
                lw=1,
                label="Percentile 95"
            )
        ]
    )

    '''
    not_used anymore
    # Row 3: Selection / Top-Performing Outline
    sel_handles = [
        Line2D([0], [0], color="none", label="Selection (Outline):"),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            label="Evaluated Configuration",
            markerfacecolor="#999999",
            markeredgecolor="#333333",
            markeredgewidth=0.4,
            markersize=7.5,
            alpha=0.7,
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            label="Top-Performing Choice (Black Outline)",
            markerfacecolor="#999999",
            markeredgecolor="black",
            markeredgewidth=1.8,
            markersize=10.5,
        ),
    ]'''

    legend_rows = [
        (color_handles, "upper center", (0.5, 1.00)),
        (shape_handles, "center",       (0.5, 0.50)),
        #(sel_handles,   "lower center", (0.5, 0.00)),
    ]

    for handles, loc, bbox in legend_rows:
        leg = ax_leg.legend(
            handles=handles,
            loc=loc,
            bbox_to_anchor=bbox,
            ncol=len(handles),
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor="#DDDDDD",
            fontsize=10,
            borderpad=0.2,
            handlelength=1.3,
            handletextpad=0.45,
            columnspacing=1.1,
        )
        leg.get_texts()[0].set_fontweight("bold")
        ax_leg.add_artist(leg)

    # --- 2x2 Scatter Grid (Row 1) with independent hspace=0.32 between plot rows ---
    n_plot_rows = len(plot_defs)
    n_plot_cols = max(len(r) for r in plot_defs)
    gs_plots = gridspec.GridSpecFromSubplotSpec(
        n_plot_rows,
        n_plot_cols,
        subplot_spec=outer_gs[1],
        hspace=0.39,
        wspace=0.20,
    )

    for row_i, plots_row in enumerate(plot_defs):
        for col_i, plot_dict in enumerate(plots_row):
            ax = fig.add_subplot(gs_plots[row_i, col_i])
            if "cmap" in plot_dict:
                cmap_name = plot_dict["cmap"]
                values_for_scale = df[plot_dict["c"]]
                cmap = cm[cmap_name]
                color_matrix = cmap(values_for_scale / values_for_scale.max())
                color_strs = [
                    f"#{int(r[0]*255):02x}{int(r[1]*255):02x}{int(r[2]*255):02x}"
                    for r in color_matrix
                ]
                c_col = f"CmapValues_{row_i}_{col_i}"
                df[c_col] = color_strs
            else:
                c_col = plot_dict["c"]
            
            ont = plot_dict.get("single_ont", None)
            if ont:
                df_fil = df[df["Ontology"] == ont]
                percs = percentiles[ont]
                perc_x = percs[plot_dict["x"]]["value"]*100
                perc_y = percs[plot_dict["y"]]["value"]*100
            else:
                df_fil = df
                perc_x = None
                perc_y = None

            plot_scatter(
                ax,
                df_fil,
                x_col=plot_dict["x"],
                y_col=plot_dict["y"],
                c_col=c_col,
                x_label=plot_dict["xlabel"],
                y_label=plot_dict["ylabel"],
                title=plot_dict["title"],
                perc_x=perc_x,
                perc_y=perc_y,
            )

    fig.suptitle(
        "Quality Metrics of Evidence Representation Strategies",
        fontsize=14,
        fontweight="bold",
        color="#111111",
        y=0.968,
    )
    fig.savefig(
        os.path.join(results_dir, "evi_rep_quality_small.png"),
        dpi=160,
        facecolor=bg_color,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(results_dir, "evi_rep_quality.png"),
        dpi=400,
        facecolor=bg_color,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(results_dir, "evi_rep_quality.svg"),
        dpi=400,
        facecolor=bg_color,
        bbox_inches="tight",
    )
    plt.close(fig)

def add_evi_sorting_col(df):
    labels = ["phylo", "curated", "conditional_not", "curated_not", "derived_not", "phylo_not", "Random False Val"]
    labels_short = {
        "phylo": "PY+", "curated": "CU+", "conditional_not": "CO-",
        "curated_not": "CU-", "derived_not": "DE-", "phylo_not": "PY-",
        "Random False Val": "RD-",
    }
    df_evi = []
    for _, row in df.iterrows():
        new_row = {k: row[k] for k in row.keys()}
        evi_vals = []
        val_dict = {}
        has_rand = False

        params_dict = json.loads(row["Parameters"])
        for label in labels:
            if label in params_dict:
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
            df_evi.append(new_row)

    df2 = pd.DataFrame(df_evi)
    return df2

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
    i = 1
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

        ax.set_title(f"A{i}: {ont}", fontsize=10, weight="bold", pad=12)
        if ax is axes[1]:
            ax.set_xlabel("Parameter Configurations Evaluated", fontweight="bold", color="#444444", fontsize=10, labelpad=8)
        if ax is axes[0]:
            ax.set_ylabel(f"Max Overall Score Seen So Far (%)", fontweight="bold", color="#444444", fontsize=10, labelpad=8)
        ax.set_xlim(0, max_x + (max_x * 0.12))
        ax.set_ylim(45, 76.5)
        i += 1

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

def norm_scores(df, score_cols = ["Sort Score", "OWA Score", "CWA Score"]):
    """
    Norm each score column independently to a [0, 100] scale based on its own min/max
    """
    df_norm = df.copy()
    min_values = {}
    max_values = {}
    for col in score_cols:
        col_min = df_norm[col].min()
        col_max = df_norm[col].max()
        col_range = col_max - col_min
        min_values[col] = col_min
        max_values[col] = col_max
        df_norm[col] = 100.0 if col_range == 0 else ((df_norm[col] - col_min) / col_range) * 100.0

    return df_norm, min_values, max_values

def find_top_percentile(df, score_cols = ["Sort Score", "OWA Score", "CWA Score"], percentile=0.95):
    """
    Find the percentile-th configuration for each ontology
    """
    percentiles = {"MF": {}, "BP": {}, "CC": {}}
    ont_dfs = []
    for ont in percentiles.keys():
        ont_df = df[df["Ontology"] == ont].copy()
        for col in score_cols:
            value = ont_df[col].quantile(percentile)
            maximum = ont_df[col].max()
            percentiles[ont][col] = {"value": value, "max": maximum}
        
        for col in score_cols:
            ont_df[f"in_{col}_top"] = ont_df[col] >= percentiles[ont][col]["value"]
        ont_dfs.append(ont_df)
    
    with_percs_df = pd.concat(ont_dfs)

    best_performance_df = []
    for ont in percentiles.keys():
        new_row = {"Ontology": ont}
        ont_df = with_percs_df[with_percs_df["Ontology"] == ont]
        for col in score_cols:
            in_top_perc_df = ont_df[ont_df[f"in_{col}_top"]]
            algorithms_list = in_top_perc_df["Algorithm"].to_list()
            total = len(algorithms_list)
            freqs = Counter(algorithms_list)
            freqs = [(n, v) for n, v in freqs.items()]
            freqs.sort(key=lambda x: x[1], reverse=True)
            mode, mode_freq = freqs[0]
            new_row[f"{col}_mode"] = mode
            new_row[f"{col}_mode_freq"] = mode_freq
            new_row[f"{col}_mode_perc"] = round(mode_freq / total * 100)
            if len(freqs) > 1:
                second, second_freq = freqs[1]
                new_row[f"{col}_second"] = second
                new_row[f"{col}_second_freq"] = second_freq
                new_row[f"{col}_second_perc"] = round(second_freq / total * 100)
            else:
                new_row[f"{col}_second"] = None
                new_row[f"{col}_second_freq"] = 0
            new_row[f"{col}_perc95_value"] = percentiles[ont][col]["value"]
            new_row[f"{col}_max_value"] = percentiles[ont][col]["max"]
        best_performance_df.append(new_row)
    best_performance_df = pd.DataFrame(best_performance_df)
    
    return with_percs_df, best_performance_df, percentiles

def annotate_tsne_clusters(
    ax,
    df,
    x_col="TSNE_1",
    y_col="TSNE_2",
    cluster_col="Algorithm",
):
    """
    Calcula a posição de cada cluster de estratégia no espaço t-SNE e adiciona
    rótulos com bordas coloridas (seguindo MARKER_COLOR_MAP) sem cobrir os pontos.
    """
    # Abre um pequeno espaço extra no topo e na base para acomodar as caixas de texto
    y_min_ax, y_max_ax = ax.get_ylim()
    y_span = y_max_ax - y_min_ax
    ax.set_ylim(y_min_ax - 0.08 * y_span, y_max_ax + 0.10 * y_span)

    # Nomes levemente compactados para caberem sem invadir o cluster vizinho
    DISPLAY_NAMES = {
        "Classic": "Classic",
        "Conditional Negatives": "Cond. Negatives",
        "Conditional Negatives + RNS": "Cond. Neg. + RNS",
        "Soft": "Soft",
        "Soft + RNS": "Soft + RNS",
    }

    for alg_name, group_df in df.groupby(cluster_col, sort=False):

        #outline spots with algorithm color
        alg_color = MARKER_COLOR_MAP.get(alg_name, "#444444")

        if group_df.empty:
            continue

        x_vals = group_df[x_col].values
        y_vals = group_df[y_col].values

        cx = np.median(x_vals)
        cy = np.median(y_vals)
        y_top = np.max(y_vals)
        y_bot = np.min(y_vals)

        label_text = DISPLAY_NAMES.get(alg_name, alg_name)

        # Se o cluster estiver na faixa inferior (y < -15), coloca o rótulo embaixo;
        # caso contrário, coloca logo acima do topo do cluster.
        if cy < -15 or label_text == "Cond. Negatives":
            label_y = y_bot - 6.5
            va = "top"
            anchor_y = y_bot
        else:
            label_y = y_top + 4.5
            va = "bottom"
            anchor_y = y_top

        

        # Linha conectora sutil entre a borda do cluster e a etiqueta
        ax.plot(
            [cx, cx],
            [anchor_y, label_y],
            color="#666666",
            linestyle=":",
            linewidth=1.0,
            zorder=7,
        )

        # Etiqueta com borda na cor da estratégia (mesma paleta dos gráficos A1-A3)
        ax.text(
            cx,
            label_y,
            label_text,
            fontsize=7,
            fontweight="bold",
            color="#222222",
            ha="center",
            va=va,
            zorder=8,
            bbox=dict(
                boxstyle="round,pad=0.1",
                facecolor="#FFFFFF",
                edgecolor=alg_color,
                linewidth=0.5,
                alpha=0.6,
            ),
        )

        '''#Plotar circulo com bordas semi-transparentes e sem preenchimento, 
        #com o ponto central do cluster e raio, centralizado no centro do cluster
        ratius = min(max((max(x_vals) - cx), (cx - min(x_vals))), max((max(y_vals) - cy), (cy - min(y_vals)))) * 0.8
        circle = mpatches.Circle(
            (
                np.median(x_vals),
                np.median(y_vals)
            ),
            radius=ratius,
            edgecolor=edge_color,
            facecolor="none",
            alpha=0.6,
            zorder=8,
        )
        ax.add_patch(circle)'''

def plot_scatter_tsne(
    ax,
    fig,
    df,
    x_col,
    y_col,
    c_col,
    x_label,
    y_label,
    title,
    marker_col="Ontology",
    marker_map=MARKER_MAP,
    cmap="viridis",
    vmin=0.0,
    vmax=100.0,
):
    """
    Plots scatter points encoded by Strategy or Score (color),
    Ontology (marker shape), and Top-Performing status (black outline + halo).
    """
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_linewidth(1.1)

    ax.grid(True, linestyle="--", color="#D5D5D5", alpha=0.65, zorder=0)
    ax.set_axisbelow(True)

    multi = 1.0 if "TSNE" in x_col else 100.0
    is_numeric_c = pd.api.types.is_numeric_dtype(df[c_col])
    cmap_kwargs = {"cmap": cmap, "vmin": 0.0, "vmax": 100.0} if is_numeric_c else {}
    # 1. Non-top configurations (background layer)
    #df_bg = df[~df["is_top_performance"]]
    df_sorted = df.sort_values(by=c_col, ascending=True)
    for alg_name in df_sorted["Algorithm"].unique():
        df_alg = df_sorted[df_sorted["Algorithm"] == alg_name]
        
        alg_color = MARKER_COLOR_MAP.get(alg_name, "#444444")
        alg_marker = STR_MARKER_MAP.get(alg_name)
        
        '''for marker_val, group_df in df_alg.groupby(marker_col, sort=False):
            if group_df.empty:
                continue'''
        ax.scatter(
            df_alg[x_col] * multi,
            df_alg[y_col] * multi,
            c=df_alg[c_col],
            s=75,
            alpha=0.6,
            edgecolors=alg_color,
            linewidths=1.6,
            zorder=3,
            marker=alg_marker,
            **cmap_kwargs,
        )

    '''# 2. Top-performing configurations (foreground layer with white halo + black outline)
    df_top = df[df["is_top_performance"]]
    for marker_val, group_df in df_top.groupby(marker_col, sort=False):
        if group_df.empty:
            continue
        x_vals = group_df[x_col] * multi
        y_vals = group_df[y_col] * multi
        marker = marker_map[marker_val]

        ax.scatter(
            x_vals,
            y_vals,
            facecolors="none",
            edgecolors="white",
            s=195,
            linewidths=2.6,
            alpha=0.95,
            zorder=5,
            marker=marker,
        )
        ax.scatter(
            x_vals,
            y_vals,
            c=group_df[c_col],
            s=145,
            alpha=1.0,
            edgecolors="black",
            linewidths=1.8,
            zorder=6,
            marker=marker,
            **cmap_kwargs,
        )'''

    sm = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=0.0, vmax=100.0),
        cmap=cm[cmap],
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal")

    # Map normalized [0, 100] tick positions to the true global score range (%)
    tick_locs = np.linspace(0.0, 100.0, 5)
    tick_vals = (vmin + (tick_locs / 100.0) * (vmax - vmin)) * 100.0
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([f"{v:.1f}%" for v in tick_vals])

    if c_col == "Sort Score":
        cbar_label = "Overall Score (%)"
    else:
        cbar_label = f"{c_col} (%)"

    cbar.set_label(
        cbar_label,
        fontsize=10,
        fontweight="bold",
        color="#333333",
        labelpad=6,
    )
    cbar.ax.tick_params(labelsize=8, colors="#333333")
    cbar.outline.set_edgecolor("#444444")
    cbar.outline.set_linewidth(1.0)

    ax.set_xlabel(x_label, fontsize=10, fontweight="bold", color="#333333", labelpad=5)
    ax.set_ylabel(y_label, fontsize=10, fontweight="bold", color="#333333", labelpad=5)
    ax.set_title(title, fontsize=11, fontweight="bold", color="#222222", pad=8)
    ax.tick_params(axis="both", which="major", labelsize=8, colors="#333333")
    ax.margins(0.06)

    # Adiciona os nomes das estratégias em cada cluster t-SNE
    if "TSNE" in x_col:
        annotate_tsne_clusters(ax, df, x_col=x_col, y_col=y_col)


def plot_optimization_fig(df, df_best_recalc, evi_exp_res_dir):
    bg_color = "#FCFBF9"
    cmap_name = "Greys"
    fig = plt.figure(figsize=(16, 13.0), facecolor=bg_color)

    gs_main = gridspec.GridSpec(
        2, 1,
        figure=fig,
        height_ratios=[0.8, 1.0],
        hspace=0.32,
    )

    # --- Row 1: 3 equal columns for MF, BP, CC ---
    gs_top = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_main[0], wspace=0.22
    )
    axes_top = [fig.add_subplot(gs_top[0, i]) for i in range(3)]
    plot_cumulative_performance(axes_top, df)

    # --- Row 2: 3 equal columns for t-SNE landscapes (B, C, D) ---
    gs_bot = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs_main[1], wspace=0.20
    )
    ax_b = fig.add_subplot(gs_bot[0, 0])
    ax_c = fig.add_subplot(gs_bot[0, 1])
    ax_d = fig.add_subplot(gs_bot[0, 2])

    norm_df, min_scores, max_scores = norm_scores(df)

    plot_scatter_tsne(
        ax_b,
        fig,
        norm_df,
        x_col="TSNE_1",
        y_col="TSNE_2",
        c_col="CWA Score",
        x_label="TSNE Dimension 1",
        y_label="TSNE Dimension 2",
        title="B: Avg. CWA of\nCandidate Parameter Configurations",
        cmap=cmap_name,
        vmin=min_scores["CWA Score"],
        vmax=max_scores["CWA Score"],
    )
    plot_scatter_tsne(
        ax_c,
        fig,
        norm_df,
        x_col="TSNE_1",
        y_col="TSNE_2",
        c_col="OWA Score",
        x_label="TSNE Dimension 1",
        y_label="TSNE Dimension 2",
        title="B2: Avg. OWA of\nCandidate Parameter Configurations",
        cmap=cmap_name,
        vmin=min_scores["OWA Score"],
        vmax=max_scores["OWA Score"],
    )
    plot_scatter_tsne(
        ax_d,
        fig,
        norm_df,
        x_col="TSNE_1",
        y_col="TSNE_2",
        c_col="Sort Score",
        x_label="TSNE Dimension 1",
        y_label="TSNE Dimension 2",
        title="B3: Overall Score of\nCandidate Parameter Configurations",
        cmap=cmap_name,
        vmin=min_scores["Sort Score"],
        vmax=max_scores["Sort Score"],
    )

    ONT_LABELS = {
        "MF": "Molecular Function",
        "BP": "Biological Process",
        "CC": "Cellular Component",
    }
    '''legend_handles = (
        [Line2D([0], [0], color="none", label="Ontology (Shape):")]
        + [
            Line2D(
                [0], [0],
                marker=marker,
                color="none",
                label=ONT_LABELS.get(ont, ont),
                markerfacecolor="#777777",
                markeredgecolor="#333333",
                markeredgewidth=0.8,
                markersize=8,
            )
            for ont, marker in MARKER_MAP.items()
        ]
        + [
            Line2D([0], [0], color="none", label="Selection (Outline):"),
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                label="Evaluated Configuration",
                markerfacecolor="#999999",
                markeredgecolor="#333333",
                markeredgewidth=0.4,
                markersize=7.5,
                alpha=0.7,
            ),
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                label="Top-Performing Choice (Black Outline)",  # single line
                markerfacecolor="#999999",
                markeredgecolor="black",
                markeredgewidth=1.8,
                markersize=10.5,
            ),
        ]
    )

    leg = fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.085),  # increase/decrease slightly to tune gap under colorbars
        ncol=len(legend_handles),
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#DDDDDD",
        fontsize=8,
        borderpad=0.35,
        handlelength=1.2,
        handletextpad=0.4,
        columnspacing=1.2,
    )
    leg.get_texts()[0].set_fontweight("bold")
    leg.get_texts()[4].set_fontweight("bold")'''

    fig.savefig(
        os.path.join(evi_exp_res_dir, "optimization_fig_small.png"),
        dpi=160,
        facecolor=bg_color,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(evi_exp_res_dir, "optimization_fig.png"),
        dpi=400,
        facecolor=bg_color,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(evi_exp_res_dir, "optimization_fig.svg"),
        dpi=400,
        facecolor=bg_color,
        bbox_inches="tight",
    )
    plt.close(fig)

if __name__ == "__main__":
    if ".tsv" in sys.argv[1]:
        all_results_path = sys.argv[1]
        evi_exp_res_dir = os.path.dirname(all_results_path)
    else:
        evi_exp_res_dir = sys.argv[1]
        all_results_path = os.path.join(evi_exp_res_dir, "all_results_wtimes.tsv")
    df = pd.read_csv(all_results_path, sep="\t")
    df = df[df["test_order"] < 72]
    print(df.columns)

    df = get_evi_tsne_coords(df)
    print(df.columns)

    df_best_recalc = []
    df_alls = []
    for group_tp, group_df in df.groupby(["Algorithm", "Ontology"]):
        group_df = group_df.sort_values(by="Sort Score", ascending=False)
        group_df['is_top_performance'] = [x == 0 for x in range(len(group_df))]
        df_alls.append(group_df)
        top_test = group_df.iloc[[0]]
        top_test = {
            k: top_test[k].item()
            for k in ["Algorithm", "Ontology", "Sort Score", "OWA Score", "CWA Score", "Parameters"]
        }
        df_best_recalc.append(top_test)
        print(group_tp, len(group_df))
    df_best_recalc = pd.DataFrame(df_best_recalc)
    df_best_recalc = df_best_recalc.sort_values(by="Sort Score", ascending=False)
    df_best_recalc.to_csv(os.path.join(evi_exp_res_dir, "evi_rep-top_performance.tsv"), sep="\t", index=False)
    df = pd.concat(df_alls)
    df.to_csv(os.path.join(evi_exp_res_dir, "evi_rep-all_tests.tsv"), sep="\t", index=False)

    df_with_percs, df_best_performance, percentiles = find_top_percentile(df, percentile=0.9)
    df_with_percs.to_csv(os.path.join(evi_exp_res_dir, "evi_rep-with_percs.tsv"), sep="\t", index=False)
    df_best_performance.to_csv(os.path.join(evi_exp_res_dir, "evi_rep-best_performance.tsv"), sep="\t", index=False)
    with open(os.path.join(evi_exp_res_dir, "evi_rep-percentiles.json"), "w") as f:
        json.dump(percentiles, f, indent=4)
    for _, row in df_best_performance.iterrows():
        print({k: v for k, v in row.items()})
    df_best_performance_manuscript = []
    for _, row in df_best_performance.iterrows():
        
        for criteria in ["Sort Score", "OWA Score", "CWA Score"]:
            if criteria == "Sort Score":
                col_final = "Overall Score"
            else:
                col_final = criteria
            new_row = {"Task": row["Ontology"], "Criteria": col_final}
            new_row["Mode"] = f"{row[criteria + '_mode']} ({row[criteria + '_mode_perc']}%)"
            new_row["2nd Most Freq."] = f"{row[criteria + '_second']} ({row[criteria + '_second_perc']}%)"
            min_perc = round(row[f'{criteria}_perc95_value']*100, 2)
            max_perc = round(row[f'{criteria}_max_value']*100, 2)
            new_row[f"Range of Values in Perc95"] = f"{min_perc}% - {max_perc}%"
            df_best_performance_manuscript.append(new_row)
    
    df_best_performance_manuscript = pd.DataFrame(df_best_performance_manuscript)
    df_best_performance_manuscript.to_csv(os.path.join(evi_exp_res_dir, "evi_rep-best_performance_manuscript.tsv"), sep="\t", index=False)

    plot_metrics_from_df(df, evi_exp_res_dir, percentiles=percentiles)

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

    plot_optimization_fig(df, df_best_recalc, evi_exp_res_dir)

    