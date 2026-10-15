import sys
import os
from glob import glob
import json

import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from pddb_lib.parsing import smart_str_parsing
from pddb_lib.sample_metaparameters import GENE_NAMES

# Define specific marker shapes for each family
'''

MARKER_COLOR_MAP = {
    "MF": "#FFFF66", #yellow
    "BP": "#0000FF", #blue
    "CC": "#00FF00", #green
}'''

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
    

def find_datetime(row, param_datetimes):
    alg_name = row["Algorithm"]
    params_dict_str1 = row["Parameters"]

    if alg_name in param_datetimes and params_dict_str1 in param_datetimes[alg_name]:
        return param_datetimes[alg_name][params_dict_str1]
    else:
        print(f"Could not find datetime for {alg_name} with parameters {params_dict_str1}")
        return None

if __name__ == "__main__":
    evi_exp_res_dir = sys.argv[1]
    all_results_path = os.path.join(evi_exp_res_dir, "all_results.tsv")

    df = pd.read_csv(all_results_path, sep="\t")

    param_datetimes = {}
    for alg_dir in glob(f"{evi_exp_res_dir}/*_test"):
        models_trained = []
        alg_name1 = os.path.basename(alg_dir).replace('_test', '')
        genenames = GENE_NAMES[alg_name1.replace('_rns', '+rns')]
        alg_name = alg_name1.replace('_', ' ').title().replace('Rns', '+ RNS')
        #results_tsv = f"{alg_dir}/all_results.tsv"
        pred_parquets = glob(f"{alg_dir}/parameters=*/y_pred.parquet")

        if not alg_name in param_datetimes:
            param_datetimes[alg_name] = {}

        print(f"Loading {len(pred_parquets)} prediction files of {alg_name} from {alg_dir}")

        for pred_path in tqdm(pred_parquets):
            comb = os.path.dirname(pred_path).split("parameters=")[-1]
            param_comb = tuple([smart_str_parsing(x) for x in comb.split("_")])
            params_dict = dict(zip(genenames, param_comb))
            params_str = json.dumps(params_dict, ensure_ascii=False)
            param_datetimes[alg_name][params_str] = os.path.getmtime(pred_path)
    
    param_orders = {}
    for alg_name, datetimes in param_datetimes.items():
        sorted_keys = sorted(datetimes.keys(), key=lambda x: datetimes[x])
        key_to_order = {k: i for i, k in enumerate(sorted_keys)}
        param_orders[alg_name] = key_to_order
    
    df['datetime'] = df.apply(find_datetime, axis=1, param_datetimes=param_datetimes)
    df['test_order'] = df.apply(find_datetime, axis=1, param_datetimes=param_orders)
    df = df.sort_values(by=['Algorithm', 'Ontology', 'test_order'])

    df["Sort Score"] = (df["OWA Score"] + df["CWA Score"]) / 2

    '''for _, row in df.iterrows():
        print(row['Algorithm'], row['Ontology'], row['test_order'])'''

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