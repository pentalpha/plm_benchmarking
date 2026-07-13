import os
import random
import json

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
