import os
import random
import json

import numpy as np

from pddb_lib.gene_ontology import EVIDENCE_REP_STRATEGIES

good_combinations = {}

hyperparameter_space = {
    # Estrutura da Árvore
    "max_depth": [3, 4, 6, 8, 10, 11, 12],
    "min_data_in_leaf": [1, 2, 3, 5, 10, 20, 35, 50, 100],
    "min_gain_to_split": [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0],
    "max_bin": [64, 128, 192, 255],
    # Regularização e Otimização
    "lr": [
        0.1,
        0.08,
        0.05,
        0.03,
        0.01,
        0.0075,
        0.005,
    ],
    "lambda_l2": [0.1, 1, 5, 10, 50, 100],
    "use_hess": [True, False],
    "gd_steps": [1, 2, 3],
    # Amostragem (Controle de Overfitting)
    "colsample": [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # Controle de Treinamento
    "ntrees": [3000, 4500, 6000, 8000, 10000, 12500, 15000, 17500, 20000, 22500, 25000],
    "es": [100, 200, 300],
}

hyperparameter_space2 = {
    # Estrutura da Árvore
    "max_depth": [3, 4, 6, 8],
    "min_data_in_leaf": [1, 2, 3, 5, 10, 20, 35, 50, 100],
    "min_gain_to_split": [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0],
    "max_bin": [64, 128, 192, 255],
    # Regularização e Otimização
    "lr": [
        0.1,
        0.08,
        0.05,
        0.03,
        0.01,
        0.0075,
        0.005,
    ],
    "lambda_l2": [0.1, 1, 5, 10, 50, 100],
    "use_hess": [True, False],
    "gd_steps": [1, 2],
    # Amostragem (Controle de Overfitting)
    "colsample": [0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    # Controle de Treinamento
    "ntrees": [3000, 4500, 6000, 8000, 10000],
    "es": [100, 200, 300],
}

hyperparameter_space3 = {
    # Estrutura da Árvore
    "max_depth": [5,6,7,8],
    "min_data_in_leaf": [1, 3, 5, 10, 20, 50],
    "min_gain_to_split": [0.1, 0.5, 1.0, 2.0],
    "max_bin": [128, 255],
    # Regularização e Otimização
    "lr": [
        0.03,
        0.01,
        0.005,
    ],
    "lambda_l2": [1, 50, 100],
    #"use_hess": [True],
    "gd_steps": [1, 2],
    # Amostragem (Controle de Overfitting)
    "colsample": [0.05, 0.3, 0.4],
    "subsample": [0.6, 0.75, 0.9,],
    # Controle de Treinamento
    "ntrees": [5000, 6000, 12000, 22500],
    "es": [150, 200],
    "phylo": {"min": 0.9, "max": 0.99},
    "curated": {"min": 0.8, "max": 0.99},
    "conditional_not": {"min": 0.3, "max": 0.4},
    "curated_not": {"min": 0.01, "max": 0.35},
    "derived_not": {"min": 0.03, "max": 0.12},
    "phylo_not": {"min": 0.01, "max": 0.2},
    "Random Falses Min Perc": {"min": 0.1, "max": 0.35},
    "Random False Val": {"min": 0.23, "max": 0.25},
}

gdbt_params_list = [
    "max_depth",
    "min_data_in_leaf",
    "min_gain_to_split",
    "max_bin",
    "lr",
    "lambda_l2",
    "gd_steps",
    "colsample",
    "subsample",
    "ntrees",
    "es"
]


GENE_NAMES = {
    "soft+rns": gdbt_params_list + [
        "phylo",
        "curated",
        "conditional_not",
        "curated_not",
        "derived_not",
        "phylo_not",
        "Random Falses Min Perc",
        "Random False Val",
    ],
    "soft": gdbt_params_list + [
        "phylo",
        "curated",
        "conditional_not",
        "curated_not",
        "derived_not",
        "phylo_not",
    ],
    "conditional_negatives+rns": gdbt_params_list + ["Random Falses Min Perc"],
    "conditional_negatives": gdbt_params_list,
    "classic": gdbt_params_list,
}


def generate_for_genelist(n_combinations: int, genenames: list, try_more=True):
    if try_more:
        random.seed(42)
        np.random.seed(42)

    options = []
    for genename in genenames:
        gene_vals_raw = hyperparameter_space3[genename]
        if type(gene_vals_raw) == dict:
            min_val = gene_vals_raw["min"]
            max_val = gene_vals_raw["max"]
            #Sample 100 options in range:
            range_options = [round(float(val), 2) for val in list(np.linspace(min_val, max_val, num=100))]
            range_options = [min_val] + range_options + [max_val]
            options.append(range_options)
        elif type(gene_vals_raw) == list:
            options.append(gene_vals_raw)
    
    choices_by_gene = []
    for options_list in options:
        choices = [x for x in np.random.choice(options_list, size=n_combinations, replace=True)]
        choices_by_gene.append(choices)
    
    param_combs = []
    for comb_index in range(n_combinations):
        comb = [choices_by_gene[gene_index][comb_index] 
            for gene_index in range(len(choices_by_gene))]
        param_combs.append(comb)
    
    param_combs_valid = set()
    for comb in param_combs:
        tp = tuple(comb)
        param_combs_valid.add(tp)
    
    print(
        f"| INFO | Number of fuzzy value combinations created: {len(param_combs_valid)} out of {n_combinations}"
    )
    while len(param_combs_valid) < n_combinations and try_more:
        next_batch = n_combinations - len(param_combs_valid)
        print(f"| INFO | Generating {next_batch} additional combinations...")
        new_combs = generate_for_genelist(n_combinations, genenames, try_more=False)
        actually_new = new_combs - param_combs_valid
        if len(actually_new) > next_batch:
            actually_new = set(random.sample(list(actually_new), next_batch))
        param_combs_valid.update(actually_new)
        print(
            f"| INFO | Total combinations: {len(param_combs_valid)} out of {n_combinations}"
        )
    return param_combs_valid

'''def generate_fuzzy_metaparameters(n_combinations: int, try_more=True):
    # [phylo_true_val, cond_false_val, derivate_false_val, add_random_falses, random_falses_min_perc, random_false_val]
    param_combs = []
    """
    param_combs.append(np.random.uniform(0.75, 1.0, n_combinations))
    param_combs.append(np.random.uniform(0.0, 0.45, n_combinations))
    param_combs.append(np.random.uniform(0.0, 0.4, n_combinations))
    param_combs.append(np.random.choice([True, False], n_combinations))
    param_combs.append(np.random.uniform(0.1, 0.35, n_combinations))
    param_combs.append(np.random.uniform(0.0, 0.45, n_combinations))
    """
    param_combs.append(np.random.uniform(0.88, 1.0, n_combinations))
    param_combs.append(np.random.uniform(0.21, 0.44, n_combinations))
    param_combs.append(np.random.uniform(0.06, 0.33, n_combinations))
    param_combs.append(np.random.choice([True, True], n_combinations))
    param_combs.append(np.random.uniform(0.11, 0.42, n_combinations))
    param_combs.append(np.random.uniform(0.06, 0.43, n_combinations))

    param_combs_valid = set()
    for i in range(n_combinations):
        phylo_true_val = param_combs[0][i]
        cond_false_val = param_combs[1][i]
        derivate_false_val = param_combs[2][i]
        add_random_falses = param_combs[3][i]
        random_falses_min_perc = param_combs[4][i]
        random_false_val = param_combs[5][i]
        comb = (
            round(phylo_true_val, 2),
            round(cond_false_val, 2),
            round(derivate_false_val, 2),
            add_random_falses,
            round(random_falses_min_perc, 2),
            round(random_false_val, 2),
        )
        if derivate_false_val <= cond_false_val:
            param_combs_valid.add(comb)
    print(
        f"| INFO | Number of fuzzy value combinations created: {len(param_combs_valid)} out of {n_combinations}"
    )
    while len(param_combs_valid) < n_combinations and try_more:
        next_batch = n_combinations - len(param_combs_valid)
        print(f"| INFO | Generating {next_batch} additional combinations...")
        new_combs = generate_fuzzy_metaparameters(n_combinations, try_more=False)
        actually_new = new_combs - param_combs_valid
        if len(actually_new) > next_batch:
            actually_new = set(random.sample(list(actually_new), next_batch))
        param_combs_valid.update(actually_new)
        print(
            f"| INFO | Total combinations: {len(param_combs_valid)} out of {n_combinations}"
        )
    return param_combs_valid'''


def save_stats_json_safe(statistics_path: str, stats, ont: str):
    if os.path.exists(statistics_path):
        old_backup_path = statistics_path.replace(".json", ".old.json")
        if os.path.exists(old_backup_path):
            os.remove(old_backup_path)
        os.rename(statistics_path, old_backup_path)
        json.dump(stats, open(statistics_path, "w"), indent=4)

    else:
        json.dump(stats, open(statistics_path, "w"), indent=4)

    owa_metrics = [
        "OWA Weighted Fmax (micro)",
        "OWA Weighted MCC (micro)",
        "OWA Weighted AUPRC",
    ]
    cwa_metrics = ["CAFA Weighted Fmax", "CAFA AUPRC"]

    new_lines = []
    for result in stats:
        norm_stats = result["stats"]
        new_line = {"Config Name": None, "Ontology": ont}
        for i, gene_name in enumerate(gene_names):
            new_line[gene_name] = result["comb"][i]
        owa_sum = 0.0
        for owa_met in owa_metrics:
            new_line[owa_met] = norm_stats[owa_met]
            owa_sum += norm_stats[owa_met]
        cwa_sum = 0.0
        for cwa_met in cwa_metrics:
            new_line[cwa_met] = norm_stats[cwa_met]
            cwa_sum += norm_stats[cwa_met]

        new_line["OWA Sort Score"] = owa_sum / len(owa_metrics)
        new_line["CWA Sort Score"] = cwa_sum / len(cwa_metrics)
        new_line["Sort Score"] = norm_stats["Sort Score"]
        new_lines.append(new_line)

    best_owa = sorted(new_lines, key=lambda x: x["OWA Sort Score"], reverse=True)[0]
    best_cwa = sorted(new_lines, key=lambda x: x["CWA Sort Score"], reverse=True)[0]
    best_sort = sorted(new_lines, key=lambda x: x["Sort Score"], reverse=True)[0]

    best_owa["Config Name"] = "Best OWA"
    best_cwa["Config Name"] = "Best CWA"
    best_sort["Config Name"] = "Best Sort"

    df = pl.DataFrame([best_owa, best_cwa, best_sort])
    df.write_csv(statistics_path.replace(".json", ".tsv"), separator="\t")


def update_y_data_with_new_values(
    y: np.ndarray,
    phylo_true_val: float,
    cond_false_val: float,
    derivate_false_val: float,
):
    derivated_false_orig_val = 0.025
    conditional_false_orig_val = 0.15
    phylogenetic_positive_orig_val = 0.9

    y = np.where(y == derivated_false_orig_val, derivate_false_val, y)
    y = np.where(y == conditional_false_orig_val, cond_false_val, y)
    y = np.where(y == phylogenetic_positive_orig_val, phylo_true_val, y)
    return y

def update_y_data_with_new_values2(
    y: np.ndarray,
    params_dict,
    random_val_original_value = None
):
    #get evi type original values
    evi_types = ["phylo",
        "curated",
        "conditional_not",
        "curated_not",
        "derived_not",
        "phylo_not"]
    
    for evi_type in evi_types:
        original_val = EVIDENCE_REP_STRATEGIES['soft'][evi_type]
        new_val = params_dict[evi_type]
        y = np.where(y == original_val, new_val, y)
    
    # handle random falses
    if random_val_original_value is not None:
        y = np.where(y == random_val_original_value, params_dict['Random False Val'], y)
    
    return y


def make_n_combinations(param_options_dict, n):

    param_names = list(param_options_dict.keys())
    combs = [{} for _ in range(n)]
    for i, param in enumerate(param_names):
        options = param_options_dict[param]
        # check if the list contains lists
        if any(isinstance(o, list) for o in options):
            # list of lists like [[1, 2], [2, 3]]: choose one list randomly
            random_n_vals = random.choices(options, k=n)
        else:
            # list of numbers or booleans
            random_n_vals = random.choices(options, k=n)
        for j in range(n):
            combs[j][param] = random_n_vals[j]
    return combs

def make_default_combinations():
    MAX_COMBINATIONS = 1200
    COMBINATIONS_DIR = "outputs/metaparameters"
    combinations_for_testing = make_n_combinations(hyperparameter_space, MAX_COMBINATIONS)
    combinations_for_testing = list(good_combinations.values()) + combinations_for_testing
    predef_params_loaded = json.load(open("input_data/predef_params.json"))["6GB"]
    combinations_for_testing = predef_params_loaded + combinations_for_testing
    os.makedirs(COMBINATIONS_DIR, exist_ok=True)
    for i, comb in enumerate(combinations_for_testing):
        with open(os.path.join(COMBINATIONS_DIR, f"comb_{i}.json"), "w") as f:
            json.dump(comb, f, indent=4, ensure_ascii=False)
