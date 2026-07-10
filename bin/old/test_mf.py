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
============================================================"""


def show_y_density(y):
    # Show density of cells with 1.0, 0.0 and NaN
    n_cells = y.shape[0] * y.shape[1]
    # Find possible cells values in matrix
    unique_values = np.unique(y)

    print("Unique values in y:", unique_values)
    for val in unique_values:
        n_val = np.sum(y == val)
        print("Density of", val, ":", n_val, "/", n_cells, "=", n_val / n_cells)


if __name__ == "__main__":
    url_ou_caminho_obo = "input_data/go-basic.obo"
    go_graph = obonet.read_obo(url_ou_caminho_obo)

    # df = pl.read_parquet("input_data/xy_mf.ankh.parquet")
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

    print("Original y density:")
    show_y_density(y)

    df_nan_as_zero = df.with_columns(
        pl.col("targets").arr.eval(pl.element().fill_nan(0.0))
    )
    X_nan_as_zero = np.array(df_nan_as_zero["emb"].to_list())
    y_nan_as_zero = np.array(df_nan_as_zero["targets"].to_list())

    # transform x >= 0.9 into 1.0
    y_nan_as_zero = np.where(y_nan_as_zero >= 0.9, 1.0, y_nan_as_zero)

    print("Nan as zero density:")
    show_y_density(y_nan_as_zero)
    train_x, test_x, train_y, test_y = train_test_split(
        X_nan_as_zero, y_nan_as_zero, test_size=0.2, random_state=42
    )
    fmax_val_orig_f, fmax_val_orig_strict, success_orig = eval_param_comb(
        best_ankh_base, train_x, test_x, train_y, test_y, mask_nan=False
    )

    print("Fmax for original y:", fmax_val_orig_f)
    print("Fmax for original y (strict):", fmax_val_orig_strict)

    """print("Conditional y density:")
    X = np.array(df["emb"].to_list())
    y = np.array(df["targets"].to_list())
    y = np.where(y >= 0.9, 1.0, y)
    y_condicional = apply_conditional_zeros(y, model_labels, go_graph)
    show_y_density(y_condicional)

    print(X.shape)
    print(y.shape)
    print(y_condicional.shape)

    train_x, test_x, train_y, test_y = train_test_split(
        X, y_condicional, test_size=0.2, random_state=42
    )
    fmax_val_cond_f, fmax_val_cond_strict, success_cond = eval_param_comb(
        best_ankh_base, train_x, test_x, train_y, test_y
    )

    print("Fmax for conditional y:", fmax_val_cond_f)
    print("Fmax for conditional y (strict):", fmax_val_cond_strict)
    print("Difference:", fmax_val_cond_f - fmax_val_orig_f)
    print("Strict Difference: ", fmax_val_cond_strict - fmax_val_orig_strict)"""

    print("Conditional y + fuzzy:")
    X = np.array(df["emb"].to_list())
    y = np.array(df["targets"].to_list())
    y_condicional = apply_conditional_zeros(
        y, model_labels, go_graph, custom_inferred_zero=0.15
    )
    show_y_density(y_condicional)

    print(X.shape)
    print(y.shape)
    print(y_condicional.shape)

    train_x, test_x, train_y, test_y = train_test_split(
        X, y_condicional, test_size=0.2, random_state=42
    )
    fmax_val_fuzzy_f, fmax_val_fuzzy_strict, success_fuzzy = eval_param_comb(
        best_ankh_base, train_x, test_x, train_y, test_y
    )

    print("Fmax for conditional + fuzzy:", fmax_val_fuzzy_f)
    print("Fmax for conditional + fuzzy (strict):", fmax_val_fuzzy_strict)
    print("Difference:", fmax_val_fuzzy_f - fmax_val_orig_f)
    print("Strict Difference: ", fmax_val_fuzzy_strict - fmax_val_orig_strict)

    quit(0)

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
