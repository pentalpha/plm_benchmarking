import numpy as np
import networkx as nx

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

def calc_y_density(y):
    # Calculate density of cells with 1.0, 0.0 and NaN
    n_cells = y.shape[0] * y.shape[1]
    # Find possible cells values in matrix
    unique_values = np.unique(y)

    print("Unique values in y:", unique_values)
    totals = 0
    n_by_val = {}
    for val in unique_values:
        if np.isnan(val):
            n_val = n_cells - totals
            #perc = n_val / n_cells * 100
            print("Density of", val, ":", n_val, "/", n_cells, "=", n_val / n_cells)
        else:
            # Make mask where 1.0 -> has x = val and 0.0 -> x != val
            mask = np.where(y == val, 1.0, 0.0)
            n_val = np.sum(mask)
            totals += n_val
            #perc = n_val / n_cells * 100
            print("Density of", val, ":", n_val, "/", n_cells, "=", n_val / n_cells)
        n_by_val[val] = n_val
    n_by_val["Total with Evidence"] = totals
    return n_by_val

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