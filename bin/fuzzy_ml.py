from copy import copy
import networkx as nx
import cupy as cp
import numpy as np
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
