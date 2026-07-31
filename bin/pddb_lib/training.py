from copy import copy

from py_boost import Callback
from py_boost.gpu.losses import BCELoss, BCEMetric
from py_boost import GradientBoosting
import numpy as np
import cupy as cp
import networkx as nx
import obonet

from pddb_lib.custom_statistics import fmax, fmax_dual, macro_fmax_dual

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

def show_y_density(y):
    # Show density of cells with 1.0, 0.0 and NaN
    n_cells = y.shape[0] * y.shape[1]
    # Find possible cells values in matrix
    unique_values = np.unique(y)

    print("Unique values in y:", unique_values)
    totals = 0
    for val in unique_values:
        if np.isnan(val):
            n_val = n_cells - totals
            print("Density of", val, ":", n_val, "/", n_cells, "=", n_val / n_cells)
        else:
            # Make mask where 1.0 -> has x = val and 0.0 -> x != val
            mask = np.where(y == val, 1.0, 0.0)
            n_val = np.sum(mask)
            totals += n_val
            print("Density of", val, ":", n_val, "/", n_cells, "=", n_val / n_cells)

def add_random_false_values(train_y, target_min_zeros=0.12, zero_val=0.0):
    # x != NaN and x < 0.5 = negative evi = zero
    train_y = np.ascontiguousarray(train_y).copy()

    zeros_mask = (~np.isnan(train_y)) & (train_y < 0.5)

    n_cells_total = train_y.size
    n_zeros_min = int(n_cells_total * target_min_zeros)
    n_zeros_current = np.sum(zeros_mask)

    if n_zeros_current < n_zeros_min:
        n_zeros_to_add = n_zeros_min - n_zeros_current

        # Encontra as coordenadas (flattened) de todos os NaNs disponíveis
        nan_indices = np.where(np.isnan(train_y).flatten())[0]

        # Garante que não vamos tentar amostrar mais NaNs do que o disponível
        n_zeros_to_add = min(n_zeros_to_add, len(nan_indices))

        if n_zeros_to_add > 0:
            # 2. Seleção exata e aleatória sem repetição (garante o número preciso)
            chosen_nan_indices = np.random.choice(
                nan_indices, size=n_zeros_to_add, replace=False
            )

            # Desachata os índices para o formato da matriz original e substitui
            train_y.ravel()[chosen_nan_indices] = zero_val
        return train_y, True
    else:
        # No need to add more falses
        return train_y, False

def find_conditional_zeros(
    true_labels: set, real_negatives: set, children_lists: dict[str, set]
) -> set:
    new_negatives = set()
    parents = [x for x in true_labels if x in children_lists]
    for l in parents:
        conditional_zeros = children_lists[l] - true_labels - real_negatives
        new_negatives.update(conditional_zeros)

    return new_negatives


def find_conditional_zeros_inverse(
    real_negatives: set, true_labels: set, parent_lists: dict[str, set]
) -> set:
    new_conditional_negatives = set()
    children = [x for x in true_labels if x in parent_lists]
    for l in children:
        conditional_zeros = parent_lists[l] - true_labels - real_negatives
        new_conditional_negatives.update(conditional_zeros)

    return new_conditional_negatives


def apply_conditional_zeros(
    Y: np.ndarray, terms: list, go_graph: nx.MultiDiGraph, custom_inferred_zero=None
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
    if custom_inferred_zero is None:
        Y_updated[mask_to_zero] = 0.0
    else:
        Y_updated[mask_to_zero] = custom_inferred_zero

    return Y_updated

def eval_param_comb(params_dict, train_x, test_x, train_y, test_y, mask_nan=True):
    try:
        if mask_nan:
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
        else:
            model = GradientBoosting(
                "bce",
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

        # fmax_fuzzy, fmax_cafa = fmax_dual(test_y, y_pred)
        fmax_fuzzy, _, fmax_fuzzy_all, _ = macro_fmax_dual(test_y, y_pred)
        print(f"All FMAX values:")
        print(f"Fuzzy: {fmax_fuzzy_all}")

        del model
        import gc

        gc.collect()

        return fmax_fuzzy, fmax_fuzzy_all, True
    except Exception as e:
        print(f"Error in param combination {params_dict}: {e}")
        return -1, -1, False


def train_param_comb(params_dict, train_x, test_x, train_y, test_y):
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

        y_pred_train = model.predict(train_x)
        y_pred_test = model.predict(test_x)

        del model
        import gc

        gc.collect()

        return y_pred_test, y_pred_train, True
    except Exception as e:
        print(f"Error in param combination {params_dict}: {e}")
        return None, None, False

def train_and_pred(train_x, train_y, test_x, test_y, params_dict):
    print("Solving numpy arrays...")
    train_x = np.ascontiguousarray(train_x).copy()
    test_x = np.ascontiguousarray(test_x).copy()
    train_y = np.ascontiguousarray(train_y).copy()
    test_y = np.ascontiguousarray(test_y).copy()

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
        use_hess=True,
        max_bin=params_dict["max_bin"],
        max_depth=params_dict["max_depth"],
    )

    print(params_dict)
    print(train_x.shape, test_x.shape, train_y.shape, test_y.shape)
    print("train_x: ", train_x)
    print("test_x: ", test_x)
    print("train_y: ", train_y)
    print("test_y: ", test_y)

    model.fit(train_x, train_y, eval_sets=[{"X": test_x, "y": test_y}])

    y_pred_test = model.predict(test_x)

    del model
    import gc

    gc.collect()

    return y_pred_test

def train_and_pred_failsafe(train_x, train_y, test_x, test_y, params_dict):
    try:
        return train_and_pred(train_x, train_y, test_x, test_y, params_dict)
    except Exception as e:
        print(f"Error with param combination {params_dict}: {e}")
        return None
