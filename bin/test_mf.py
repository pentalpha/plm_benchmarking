import polars as pl
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from sklearn.model_selection import train_test_split
import numpy as np
import networkx as nx
import obonet

from py_boost import GradientBoosting, SketchBoost

# strategies to deal with multiple outputs
from py_boost.multioutput.sketching import *
from py_boost.multioutput.target_splitter import *

hyperparameter_space = {
    # Estrutura da Árvore
    "max_depth": [4, 6, 8, 10, 11, 12],
    "min_data_in_leaf": [1, 2, 3, 5, 10, 20, 35, 50],
    "min_gain_to_split": [0.0, 0.1, 0.5, 1.0, 2.0],
    "max_bin": [64, 128, 192, 256],
    # Regularização e Otimização
    "lr": [
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
    "ntrees": [5000, 7500, 10000, 12500, 15000, 17500, 20000],
    "es": [100, 200, 300, 400],
}
# Ultima melhor opção:
# {'max_depth': 10,
# 'min_data_in_leaf': 1,
# 'min_gain_to_split': 0.5,
# 'max_bin': 128,
# 'lr': 0.01,
# 'lambda_l2': 5,
# 'use_hess': False,
# 'gd_steps': 2,
# 'colsample': 0.2,
# 'subsample': 0.8,
# 'ntrees': 17500,
# 'es': 300}

from copy import copy

import cupy as cp
from py_boost import Callback
from py_boost.gpu.losses import BCELoss, BCEMetric


class BCEWithNaNLoss(BCELoss):

    def base_score(self, y_true):
        # Replace .mean with nanmean function to calc base score
        means = cp.nanmean(y_true, axis=0)
        means = cp.where(cp.isnan(means), 0, means)
        means = cp.clip(means, self.clip_value, 1 - self.clip_value)

        return cp.log(means / (1 - means))

    def get_grad_hess(self, y_true, y_pred):
        # first, get nan mask for y_true
        mask = cp.isnan(y_true)
        # then, compute loss with any values at nan places just to prevent the exception
        grad, hess = super().get_grad_hess(cp.where(mask, 0, y_true), y_pred)
        # invert mask
        mask = (~mask).astype(cp.float32)
        # multiply grad and hess on inverted mask
        # now grad and hess eq. 0 on NaN points
        # that actually means that prediction on that place should not be updated
        grad = grad * mask
        hess = hess * mask

        return grad, hess


class BCEwithNaNMetric(BCEMetric):

    def __call__(self, y_true, y_pred, sample_weight=None):
        mask = ~cp.isnan(y_true)

        err = super().error(cp.where(mask, y_true, 0), y_pred)
        err = err * mask

        if sample_weight is not None:
            err = err * sample_weight
            mask = mask * sample_weight

        return float(err.sum() / mask.sum())


class WarmStart(Callback):

    def __init__(self, model):
        model.to_cpu()
        self.model = copy(model)
        self.model.postprocess_fn = lambda x: x

    def before_train(self, build_info):
        build_info["model"].base_score = cp.asarray(self.model.base_score)

        train = build_info["data"]["train"]
        train["ensemble"] = cp.asarray(self.model.predict(train["features_cpu"]))

        valid = build_info["data"]["valid"]
        valid["ensemble"] = [
            cp.asarray(self.model.predict(x)) for x in valid["features_cpu"]
        ]

        self.model.to_cpu()

        return

    def after_train(self, build_info):
        build_info["model"].models = self.model.models + build_info["model"].models
        # update the actual iteration
        build_info["num_iter"] = build_info["num_iter"] + len(self.model.models)
        # update the actual best round
        early_stop = build_info["model"].callbacks.callbacks[-1]
        early_stop.best_round = early_stop.best_round + len(self.model.models)

        # not to store old trees multiple times
        self.model = None

        return


def apply_conditional_zeros(
    Y: np.ndarray, terms: list, go_graph: nx.MultiDiGraph
) -> np.ndarray:
    """
    Aplica a estratégia de zeros condicionais na matriz de alvos (targets).

    Args:
        Y: np.ndarray de formato (n_samples, n_terms) contendo 1.0, 0.0 ou np.nan.
        terms: lista de strings com os termos GO, na mesma ordem das colunas de Y.
        go_graph: MultiDiGraph do networkx gerado pela leitura do go-basic.obo (ex: obonet.read_obo).

    Returns:
        np.ndarray: Nova matriz Y atualizada com as fronteiras de 0.0 condicionais.
    """
    print("Applying conditional zeros")
    print(Y.shape, len(terms))

    # 1. Mapeamento de Termo -> Índice para busca rápida (O(1))
    term_to_idx = {term: idx for idx, term in enumerate(terms)}
    n_terms = len(terms)

    # 2. Construção da Matriz de Adjacência de Pais (A)
    # A[i, j] == True significa que o termo j é pai direto do termo i
    A = np.zeros((n_terms, n_terms), dtype=bool)
    print("Building adjacency matrix")
    for i, child_term in enumerate(terms):
        if child_term in go_graph:
            # Em obonet, as arestas direcionadas vão do filho para o pai (child -> parent).
            # Portanto, os 'successors' de um termo são os seus pais imediatos.
            for parent_term in go_graph.successors(child_term):
                # Só nos importamos com pais que fazem parte da nossa sub-lista de targets
                if parent_term in term_to_idx:
                    j = term_to_idx[parent_term]
                    A[i, j] = True

    # 3. Identificação de Pais Verdadeiros
    # Substituímos NaNs temporariamente por 0.0 apenas para a multiplicação matricial não propagar NaNs
    Y_temp = np.nan_to_num(Y, nan=0.0)

    # Criamos uma máscara booleana onde APENAS os 1.0 (True) importam
    Y_ones = (Y_temp == 1.0).astype(float)

    # Multiplicação de Matrizes: (n_samples, n_terms) @ (n_terms, n_terms).T
    # O resultado em (s, i) é a QUANTIDADE de pais diretos com valor 1.0 que o termo i possui para a amostra s.
    parent_true_counts = Y_ones @ A.T

    # 4. Aplicação da Regra Condicional
    Y_updated = Y.copy()

    # A máscara localiza exatamente as posições que obedecem à regra:
    # -> A posição atual é NaN
    # -> A contagem de pais verdadeiros (1.0) é maior que zero
    mask_to_zero = np.isnan(Y_updated) & (parent_true_counts > 0)

    # Aplicamos a fronteira de falsos
    Y_updated[mask_to_zero] = 0.0

    return Y_updated


def fmax(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n_thresholds = 32
    thresholds = np.linspace(0.01, 0.99, n_thresholds)

    fmax_per_threshold = []
    for t in thresholds:
        # Create a boolean array of predictions for this threshold
        pred_bool = y_pred > t

        # Calculate global True Positives, False Positives, and False Negatives
        # Evaluated purely on predicting the 1.0 class correctly
        tp = (pred_bool & (y_true == 1)).sum()
        fp = (pred_bool & (y_true == 0)).sum()
        fn = (~pred_bool & (y_true == 1)).sum()

        # Calculate Precision and Recall with safe division to avoid ZeroDivisionError
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Calculate F1 for this threshold
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        fmax_per_threshold.append(f1)

    # Return the maximum F1 score found across all tested thresholds
    return max(fmax_per_threshold)


def show_y_density(y):
    # Show density of cells with 1.0, 0.0 and NaN
    n_cells = y.shape[0] * y.shape[1]
    n_1 = np.sum(y == 1.0)
    n_0 = np.sum(y == 0.0)
    n_nan = np.sum(np.isnan(y))
    print("Density of 1.0:", n_1, "/", n_cells, "=", n_1 / n_cells)
    print("Density of 0.0:", n_0, "/", n_cells, "=", n_0 / n_cells)
    print("Density of NaN:", n_nan, "/", n_cells, "=", n_nan / n_cells)


def eval_param_comb(params_dict, train_x, test_x, train_y, test_y):
    try:
        model = GradientBoosting(
            # "bce",
            BCEWithNaNLoss(),
            BCEwithNaNMetric(),
            ntrees=params_dict["ntrees"],
            lr=params_dict["lr"],
            min_gain_to_split=params_dict["min_gain_to_split"],
            es=params_dict["es"],
            lambda_l2=params_dict["lambda_l2"],
            gd_steps=params_dict["gd_steps"],
            subsample=params_dict["subsample"],
            colsample=params_dict["colsample"],
            min_data_in_leaf=params_dict["min_data_in_leaf"],
            use_hess=params_dict["use_hess"],
            max_bin=params_dict["max_bin"],
            max_depth=params_dict["max_depth"],
        )

        model.fit(train_x, train_y, eval_sets=[{"X": test_x, "y": test_y}])

        y_pred = model.predict(test_x)

        fmax_val = fmax(test_y, y_pred)

        return fmax_val, True
    except Exception as e:
        print(f"Error in param combination {params_dict}: {e}")
        return -1, False


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


url_ou_caminho_obo = "data/go-basic.obo"
go_graph = obonet.read_obo(url_ou_caminho_obo)

df = pl.read_parquet("data/xy_mf.parquet")
print(df)
print(df.columns)
# Find number of labels in the first element of targets:
n_labels = len(df["targets"].to_list()[0])
print("Number of labels:", n_labels)
# Load label names:
sorted_goids_path = "data/sorted_goids.tsv"
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

combinations_for_testing = make_n_combinations(hyperparameter_space, 20)

tests = []
for combination in combinations_for_testing:
    print(f"Trying {combination}...")
    fmax_val, success = eval_param_comb(combination, train_x, test_x, train_y, test_y)
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

"""print("Initializing model")
model = GradientBoosting(
    # "bce",
    BCEWithNaNLoss(),
    BCEwithNaNMetric(),
    ntrees=20000,
    lr=0.03,
    # min_gain_to_split=0,
    verbose=100,
    es=200,
    lambda_l2=10,
    gd_steps=1,
    subsample=0.8,
    colsample=0.8,
    min_data_in_leaf=10,
    use_hess=True,
    max_bin=256,
    max_depth=6,
    debug=True,
)

print("Starting training")
model.fit(
    train_x,
    train_y,
    eval_sets=[
        {"X": test_x, "y": test_y},
    ],
)

print("Predicting probabilities")
y_pred = model.predict(test_x)

print(test_y[0])
print(y_pred[0])

print("Calculating Fmax...")
fmax_val = fmax(test_y, y_pred)

print(f"Fmax: {fmax_val:.4f}")"""
