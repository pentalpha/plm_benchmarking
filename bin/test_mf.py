import polars as pl
import numpy as np

from sklearn.model_selection import train_test_split
import numpy as np
import obonet

from fuzzy_ml import apply_conditional_zeros
from training import eval_param_comb

hyperparameter_space = {
    # Estrutura da Árvore
    "max_depth": [4, 6, 8, 10, 11, 12],
    "min_data_in_leaf": [1, 2, 3, 5, 10, 20, 35, 50, 100],
    "min_gain_to_split": [0.0, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0],
    "max_bin": [64, 128, 192, 256, 384, 512],
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
    "ntrees": [5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 30000],
    "es": [100, 200, 300, 400],
}
"""============================================================
Best performing parameter combination:
============================================================
INTERPRO+TAXID
Fmax: 0.8674
Parameters: {'max_depth': 11, 'min_data_in_leaf': 5, 'min_gain_to_split': 2.0, 
'max_bin': 256, 'lr': 0.01, 'lambda_l2': 1, 'use_hess': True, 'gd_steps': 2, 
'colsample': 0.4, 'subsample': 0.9, 'ntrees': 20000, 'es': 300}
============================================================"""


def show_y_density(y):
    # Show density of cells with 1.0, 0.0 and NaN
    n_cells = y.shape[0] * y.shape[1]
    n_1 = np.sum(y == 1.0)
    n_0 = np.sum(y == 0.0)
    n_nan = np.sum(np.isnan(y))
    print("Density of 1.0:", n_1, "/", n_cells, "=", n_1 / n_cells)
    print("Density of 0.0:", n_0, "/", n_cells, "=", n_0 / n_cells)
    print("Density of NaN:", n_nan, "/", n_cells, "=", n_nan / n_cells)


def make_n_combinations(param_options_dict, n):
    import random

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


if __name__ == "__main__":
    url_ou_caminho_obo = "input_data/go-basic.obo"
    go_graph = obonet.read_obo(url_ou_caminho_obo)

    df = pl.read_parquet("input_data/xy_mf.parquet")
    print(df)
    print(df.columns)
    # Find number of labels in the first element of targets:
    n_labels = len(df["targets"].to_list()[0])
    print("Number of labels:", n_labels)
    # Load label names:
    sorted_goids_path = "input_data/sorted_goids.tsv"
    goids_all = []
    for line in open(sorted_goids_path):
        first_col = line.strip().split("\t")[0]
        if first_col.startswith("GO:"):
            goids_all.append(first_col)

    model_labels = goids_all[:n_labels]
    print(f"first {n_labels} labels: ", model_labels[:n_labels])
    print(f"last {n_labels} labels: ", model_labels[-n_labels:])

    y = np.array(df["targets"].to_list())
    # Replace NaN with 0 in targets
    # df = df.with_columns(pl.col("targets").arr.eval(pl.element().fill_nan(0.0)))
    # print(df)
    # print(df.columns)

    X = np.array(df["emb"].to_list())
    y = np.array(df["targets"].to_list())

    print("Original y density:")
    show_y_density(y)

    print("Conditional y density:")
    y_condicional = apply_conditional_zeros(y, model_labels, go_graph)
    show_y_density(y_condicional)

    print(X.shape)
    print(y.shape)
    print(y_condicional.shape)

    train_x, test_x, train_y, test_y = train_test_split(
        X, y_condicional, test_size=0.2, random_state=42
    )

    combinations_for_testing = make_n_combinations(hyperparameter_space, 32)

    tests = []
    for combination in combinations_for_testing:
        print(f"Trying {combination}...")
        fmax_val, success = eval_param_comb(
            combination, train_x, test_x, train_y, test_y
        )
        if success:
            print(f"Fmax: {fmax_val}")
            tests.append((fmax_val, success, combination))
        else:
            print(f"Failed to train with parameters {combination}")

    # Sort tests by Fmax in descending order
    tests.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "=" * 60)
    print("Best performing parameter combination:")
    print("=" * 60)
    print(f"Fmax: {tests[0][0]:.4f}")
    print(f"Parameters: {tests[0][2]}")
    print("=" * 60)
