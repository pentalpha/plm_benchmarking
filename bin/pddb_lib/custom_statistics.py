
import warnings
from typing import Tuple, List

from sklearn.metrics import average_precision_score
from tqdm import tqdm
import numpy as np

def get_ia_vector(term_list: List[str], ia_weights) -> np.ndarray:
    """
    Helper to create an aligned numpy array of IA weights corresponding
    to the columns of the matrices.
    """
    return np.array([ia_weights.get(term) for term in term_list])

def create_split_mask(
    split_size: float, n_elements: int, random_state: int
) -> Tuple[list, list]:
    """
    Creates list of indexes to split a dataset into train and test.

    Args:
        split_size (float): proportion of the dataset to be used as test set.
        n_elements (int): total number of elements in the dataset.
    """
    np.random.seed(random_state)

    idx = np.random.choice(n_elements, size=int(n_elements * split_size), replace=False)
    # convert idx to python list
    test_idx = idx.tolist()
    train_idx = list(set(range(n_elements)) - set(test_idx))
    return train_idx, test_idx

def apply_split_mask(X: np.ndarray, train_idx: list, test_idx: list):
    """
    Applies split mask to X.

    Args:
        X (np.ndarray): features matrix.
        train_idx (list): list of indexes for training set.
        test_idx (list): list of indexes for test set.
    """
    return X[train_idx], X[test_idx]

def ia_adapted_metric(metric_func, x, y, w_vec):
    raw_res = metric_func(x, y, average=None)
    semantic_res = np.average(raw_res, weights=w_vec)
    return semantic_res


def fmax_dual(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Calcula o F-max usando duas estratégias simultâneas:
    1. Fuzzy/Binarizado: Valores >= 0.5 são positivos. NaNs são ignorados.
    2. Strict/Crisp: Apenas 1.0 é positivo e 0.0 é negativo. NaNs e valores fuzzy ignorados.

    Retorna: (fmax_fuzzy, fmax_strict)
    """
    n_thresholds = 32
    thresholds = np.linspace(0.01, 0.99, n_thresholds)

    # 1. Máscara: True onde NÃO é NaN (Ignora NaNs de vez)
    mask = ~np.isnan(y_true)

    # 2. Matriz Binarizada: Tudo >= 0.5 vira 1, o resto vira 0.
    # (A máscara garantirá que os NaNs, que virariam 0 aqui, não sejam contados)
    y_true_bin = (y_true >= 0.5).astype(int)

    f1_fuzzy_list = []
    f1_strict_list = []

    for t in thresholds:
        pred_bool = y_pred > t

        # =======================================================
        # ESTRATÉGIA 2: F-MAX FUZZY (Binarizado)
        # Usa y_true_bin. 0.9 vira acerto positivo, 0.15 vira acerto negativo.
        # =======================================================
        tp_fuzzy = (pred_bool & (y_true_bin == 1) & mask).sum()
        fp_fuzzy = (pred_bool & (y_true_bin == 0) & mask).sum()
        fn_fuzzy = (~pred_bool & (y_true_bin == 1) & mask).sum()

        prec_fuzzy = (
            tp_fuzzy / (tp_fuzzy + fp_fuzzy) if (tp_fuzzy + fp_fuzzy) > 0 else 0.0
        )
        rec_fuzzy = (
            tp_fuzzy / (tp_fuzzy + fn_fuzzy) if (tp_fuzzy + fn_fuzzy) > 0 else 0.0
        )
        f1_fuzzy = (
            2 * (prec_fuzzy * rec_fuzzy) / (prec_fuzzy + rec_fuzzy)
            if (prec_fuzzy + rec_fuzzy) > 0
            else 0.0
        )
        f1_fuzzy_list.append(f1_fuzzy)

        # =======================================================
        # AVALIAÇÃO ESTILO CAFA: F-MAX STRICT (Crisp)
        # Usa a matriz original y_true. Só conta se for EXATAMENTE 1.0 ou 0.0.
        # =======================================================
        tp_strict = (pred_bool & (y_true == 1.0) & mask).sum()
        fp_strict = (pred_bool & (y_true < 0.5) & mask).sum()
        fn_strict = (~pred_bool & (y_true == 1.0) & mask).sum()

        prec_strict = (
            tp_strict / (tp_strict + fp_strict) if (tp_strict + fp_strict) > 0 else 0.0
        )
        rec_strict = (
            tp_strict / (tp_strict + fn_strict) if (tp_strict + fn_strict) > 0 else 0.0
        )
        f1_strict = (
            2 * (prec_strict * rec_strict) / (prec_strict + rec_strict)
            if (prec_strict + rec_strict) > 0
            else 0.0
        )
        f1_strict_list.append(f1_strict)

    # Retorna o F-max de ambas as avaliações
    return max(f1_fuzzy_list), max(f1_strict_list)

def macro_fmax0(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, list]:
    n_thresholds = 320
    thresholds = np.linspace(0.01, 0.99, n_thresholds)

    # Ignora NaNs
    mask = ~np.isnan(y_true)
    y_true_bin = (y_true >= 0.5).astype(int)

    n_cols = y_true.shape[1]
    best_f1_per_col = np.zeros(n_cols)

    # Avalia cada coluna (GO term) isoladamente
    for col in range(n_cols):
        y_t_col = y_true_bin[:, col]
        y_p_col = y_pred[:, col]
        m_col = mask[:, col]

        fmax_col = 0.0
        for t in thresholds:
            p_bool = y_p_col > t

            tp = (p_bool & (y_t_col == 1) & m_col).sum()
            fp = (p_bool & (y_t_col == 0) & m_col).sum()
            fn = (~p_bool & (y_t_col == 1) & m_col).sum()

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > fmax_col:
                fmax_col = f1

        best_f1_per_col[col] = fmax_col

    # A média do Fmax das 42 colunas
    return best_f1_per_col.mean(), list(best_f1_per_col)

def macro_fmax(y_true, y_pred, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 100)

    # Adicionamos uma dimensão extra a y_pred e thresholds para broadcasting
    # y_pred: (n_samples, n_classes) -> (1, n_samples, n_classes)
    # thresholds: (100,) -> (100, 1, 1)
    t_3d = thresholds.reshape(-1, 1, 1)

    # Cria matriz booleana para todas as predições de uma vez
    # shape: (100_thresholds, n_samples, n_classes)
    preds_bool = y_pred[np.newaxis, :, :] > t_3d
    y_true_3d = y_true[np.newaxis, :, :]

    # Cálculos simultâneos de TP, FP e FN para todas as amostras
    tp = (preds_bool & (y_true_3d == 1)).sum(axis=1)  # shape: (100, n_classes)
    fp = (preds_bool & (y_true_3d == 0)).sum(axis=1)
    fn = (~preds_bool & (y_true_3d == 1)).sum(axis=1)

    # Precision e Recall
    precision = np.divide(
        tp, tp + fp, out=np.zeros_like(tp, dtype=float), where=(tp + fp) > 0
    )
    recall = np.divide(
        tp, tp + fn, out=np.zeros_like(tp, dtype=float), where=(tp + fn) > 0
    )

    # F1 Score
    denom = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denom,
        out=np.zeros_like(precision, dtype=float),
        where=denom > 0,
    )

    # Max F1 por coluna
    best_f1_per_col = np.array([np.max(f1[:, j]) for j in range(f1.shape[1])])
    fmax_per_col = np.max(f1, axis=0)
    return np.mean(fmax_per_col), best_f1_per_col

def macro_fmax_dual(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, list, list]:
    fmax_fuzzy_mean, fmax_fuzzy_all = macro_fmax(y_true, y_pred)
    y_true_cafa = y_true >= 1.0
    fmax_cafa_mean, fmax_cafa_all = macro_fmax(y_true_cafa, y_pred)

    return fmax_fuzzy_mean, fmax_cafa_mean, fmax_fuzzy_all, fmax_cafa_all

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

def nan_macro_average_precision(
    y_true: np.ndarray, y_score: np.ndarray, weights=None
) -> float:
    """
    Calculates the macro-averaged Average Precision score for multi-label
    data, safely ignoring NaN values on a per-class basis.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples, n_classes)
        Ground truth labels. Missing labels should be represented by np.nan.
    y_score : np.ndarray of shape (n_samples, n_classes)
        Predicted probabilities or target scores.

    Returns
    -------
    float
        The macro-averaged average precision score across all valid classes.
        Returns np.nan if no valid classes are found.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.shape != y_score.shape:
        raise ValueError("y_true and y_score must have the exact same shape.")

    ap_scores = []

    # Iterate through each class (column)
    weights_sum = 0
    for class_idx in range(y_true.shape[1]):
        y_true_class = y_true[:, class_idx]
        y_score_class = y_score[:, class_idx]

        # Mask out NaN values
        valid_mask = ~np.isnan(y_true_class)
        y_true_valid = y_true_class[valid_mask]
        y_score_valid = y_score_class[valid_mask]

        # A valid AP score requires at least two distinct classes (0 and 1)
        if len(np.unique(y_true_valid)) > 1:
            ap = average_precision_score(y_true_valid, y_score_valid)
            if weights is not None:
                ap_w = ap * weights[class_idx]
                weights_sum += weights[class_idx]
            else:
                ap_w = ap
                weights_sum += 1
            ap_scores.append(ap_w)

    # Handle the edgecase where masking left no valid classes to score
    if not ap_scores:
        warnings.warn("No valid classes with both positive and negative samples found.")
        return np.nan

    mean_ap_score = np.sum(ap_scores) / weights_sum
    return float(mean_ap_score)

def faster_fmax_weighted(
    pred_scores, truth_set, weights, n_ths=120, additional_result="threshold"
):
    """
    Calculates the weighted Fmax score for multi-label data, in the most similar way to the
    CAFA methodology. GO IDs are weighted according to Inf. Acc.

    Parameters
    ----------
    pred_scores : np.ndarray of shape (n_samples, n_classes)
        Predicted probabilities or target scores.
    truth_set : np.ndarray of shape (n_samples, n_classes)
        Ground truth labels. Missing labels should be represented by np.nan.
    weights : np.ndarray of shape (n_classes,)
        Weights for each class.
    n_ths : int
        Number of thresholds to test.
    additional_result : str
        Additional result to return. Can be "threshold", "full", or "none".

    Returns
    -------
    Tuple[float, dict]
    """
    thresholds = np.linspace(0, 1, n_ths)

    # Filtro CAFA padrão
    has_positives = np.any(truth_set, axis=1)
    pred_scores = pred_scores[has_positives]
    truth_set = truth_set[has_positives]

    n_samples, n_labels = pred_scores.shape

    # 1. Alinhamento dos Pesos: Shape (n_labels,)
    w = np.array(weights).flatten()

    # 2. Denominador do Recall (Soma dos pesos das labels reais por proteína)
    # Shape: (n_samples,)
    truth_sum_weighted = np.sum(truth_set * w, axis=1)

    # 3. Predições Binárias: (n_thresholds, n_samples, n_labels)
    # Usamos broadcasting: (120, 1, 1) vs (1, samples, labels)
    pred_bin = pred_scores[None, :, :] > thresholds[:, None, None]

    # 4. Numerador (TP Ponderado): (n_thresholds, n_samples)
    # Multiplicamos pred_bin pela linha da verdade e pelos pesos
    # O truth_set * w faz o broadcast de w(labels) para cada sample
    tp_weighted = np.sum(pred_bin * (truth_set * w)[None, :, :], axis=2)

    # 5. Denominador da Precisão (Soma dos pesos das labels preditas)
    # Multiplicamos pred_bin pelo vetor de pesos
    # Shape: (n_thresholds, n_samples)
    pred_sum_weighted = np.sum(pred_bin * w[None, None, :], axis=2)

    # 6. Cálculo de Precision e Recall por amostra
    with np.errstate(divide="ignore", invalid="ignore"):
        # Ambos tp_weighted e pred_sum_weighted são (120, n_samples)
        prec = np.where(pred_sum_weighted > 0, tp_weighted / pred_sum_weighted, 0.0)

        # tp_weighted é (120, n_samples), truth_sum_weighted é (n_samples,)
        # O broadcast ocorre automaticamente no último eixo
        rec = np.where(
            truth_sum_weighted[None, :] > 0,
            tp_weighted / truth_sum_weighted[None, :],
            0.0,
        )

    # print("prec", prec, prec.shape)
    # print("rec", rec, rec.shape)

    # 7. Médias e F-max
    prec_mean = np.mean(prec, axis=1)
    rec_mean = np.mean(rec, axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(
            prec_mean + rec_mean > 0,
            (2 * prec_mean * rec_mean) / (prec_mean + rec_mean),
            0.0,
        )

    best_idx = np.argmax(f)
    f1 = float(f[best_idx])
    max_th = thresholds[best_idx]

    """At moderate FMAX values (0.55 to 0.75) this may get a high recall and low precision threshold
    we must find thresholds which are close to this value but with high precision
    this just represents a preference for almost-equal thresholds, but with slightly higher precision
    tolerance = +-0.02 to f1
    """
    tolerance = 0.005
    for idx, local_f1 in enumerate(f):
        if local_f1 >= f1 - tolerance and local_f1 <= f1 + tolerance:
            if prec_mean[idx] > prec_mean[best_idx]:
                best_idx = idx
                f1 = local_f1
                max_th = thresholds[idx]

    # Get vector with separate fmax by col:
    bool_matrix = pred_scores > max_th
    truth_bool = truth_set == 1.0  # Assumes truth_set contains 1.0 for positives

    # Calculate TP, FP, FN for all columns simultaneously
    tps = np.sum(bool_matrix & truth_bool, axis=0)
    fps = np.sum(bool_matrix & ~truth_bool, axis=0)
    fns = np.sum(~bool_matrix & truth_bool, axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        col_prec = np.where(tps + fps > 0, tps / (tps + fps), 0.0)
        col_rec = np.where(tps + fns > 0, tps / (tps + fns), 0.0)
        fmax_by_col = np.where(
            col_prec + col_rec > 0, (2 * col_prec * col_rec) / (col_prec + col_rec), 0.0
        )

    if additional_result == "threshold":
        return f1, max_th
    elif additional_result == "full":
        return f1, {
            "threshold": max_th,
            "precision": prec_mean[best_idx],
            "recall": rec_mean[best_idx],
            "fmax_by_col": fmax_by_col.tolist(),  # Added to the full return
        }
    else:
        return f1, max_th

def faster_fmax_weighted_nan(
    pred_scores: np.ndarray,
    truth_set: np.ndarray,
    weights: np.ndarray,
    n_ths: int = 120,
    additional_result: str = "threshold",
    average: str = "macro",
):
    """
    Calculates the weighted Fmax score for multi-label data, correctly handling masked NaNs.
    GO IDs are weighted according to Information Accretion (IA).

    If truth_set[i, j] == np.nan, predictions on this label are ignored and do not
    contribute to False Positives or False Negatives.

    Parameters
    ----------
    pred_scores : np.ndarray of shape (n_samples, n_classes)
        Predicted probabilities or target scores.
    truth_set : np.ndarray of shape (n_samples, n_classes)
        Ground truth labels. Missing labels should be represented by np.nan.
    weights : np.ndarray of shape (n_classes,)
        Weights for each class (IA).
    n_ths : int
        Number of thresholds to test.
    average : str
        Average to use. Can be "macro" or "micro".
    additional_result : str
        Additional result to return. Can be "threshold", "full", or "none".
    """
    thresholds = np.linspace(0, 1, n_ths)

    # 1. Cria a máscara para identificar o que é anotação válida (0.0 ou 1.0)
    valid_mask = ~np.isnan(truth_set)

    # Cria uma cópia segura onde NaNs são 0.0 para não quebrar a matemática de matrizes
    safe_truth = np.where(valid_mask, truth_set, 0.0)

    # 2. Filtro CAFA Padrão: removemos amostras que não possuem NENHUM positivo anotado
    has_positives = np.any(safe_truth > 0, axis=1)

    pred_scores = pred_scores[has_positives]
    safe_truth = safe_truth[has_positives]
    valid_mask = valid_mask[has_positives]

    n_samples, n_labels = pred_scores.shape

    # Alinhamento dos Pesos: Shape (n_labels,)
    w = np.array(weights).flatten()

    # 3. Denominador do Recall (Soma dos pesos das labels reais VÁLIDAS por proteína)
    # Shape: (n_samples,)
    truth_sum_weighted = np.sum(safe_truth * w, axis=1)

    # 4. Predições Binárias: (n_thresholds, n_samples, n_labels)
    # Usamos broadcasting: (120, 1, 1) vs (1, samples, labels)
    pred_bin = pred_scores[None, :, :] > thresholds[:, None, None]

    # 5. Numerador (TP Ponderado): (n_thresholds, n_samples)
    # Multiplicamos pred_bin pela verdade e pelos pesos
    tp_weighted = np.sum(pred_bin * (safe_truth * w)[None, :, :], axis=2)

    # 6. Denominador da Precisão (Soma dos pesos das labels preditas VÁLIDAS)
    # IMPORTANTE: Só contamos como "predição feita" (FP ou TP) se a label NÃO for NaN!
    # Ou seja, o pred_bin só passa se a valid_mask for True.
    valid_pred_bin = pred_bin & valid_mask[None, :, :]
    pred_sum_weighted = np.sum(valid_pred_bin * w[None, None, :], axis=2)

    if average == "macro":
        # 7. Cálculo de Precision e Recall por amostra
        with np.errstate(divide="ignore", invalid="ignore"):
            prec = np.where(pred_sum_weighted > 0, tp_weighted / pred_sum_weighted, 0.0)
            rec = np.where(
                truth_sum_weighted[None, :] > 0,
                tp_weighted / truth_sum_weighted[None, :],
                0.0,
            )

        # 8. Médias e F-max
        prec_mean = np.mean(prec, axis=1)
        rec_mean = np.mean(rec, axis=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            f = np.where(
                prec_mean + rec_mean > 0,
                (2 * prec_mean * rec_mean) / (prec_mean + rec_mean),
                0.0,
            )

        best_idx = np.argmax(f)
        f1 = float(f[best_idx])
        max_th = thresholds[best_idx]

        # Tolerância para buscar Precision ligeiramente melhor
        tolerance = 0.005
        for idx, local_f1 in enumerate(f):
            if local_f1 >= f1 - tolerance and local_f1 <= f1 + tolerance:
                if prec_mean[idx] > prec_mean[best_idx]:
                    best_idx = idx
                    f1 = local_f1
                    max_th = thresholds[idx]
    elif average == "micro":
        # ---------------------------------------------------------
        # NOVA ETAPA 7: AGREGAÇÃO GLOBAL (MICRO-AVERAGE)
        # Em vez de calcular prec/rec por amostra, somamos tudo globalmente por threshold
        # ---------------------------------------------------------

        # Soma de TPs para toda a matriz em cada limiar (shape: n_thresholds)
        global_tp = np.sum(tp_weighted, axis=1)

        # Soma de predições válidas para toda a matriz (Denominador da Precisão)
        global_pred_sum = np.sum(pred_sum_weighted, axis=1)

        # Soma de verdadeiros positivos reais (Denominador do Recall) - Constante para todos os limiares
        global_truth_sum = np.sum(truth_sum_weighted)

        with np.errstate(divide="ignore", invalid="ignore"):
            # Arrays de shape (n_thresholds,)
            micro_prec = np.where(global_pred_sum > 0, global_tp / global_pred_sum, 0.0)
            micro_rec = np.where(
                global_truth_sum > 0, global_tp / global_truth_sum, 0.0
            )

            # F-score para cada limiar
            micro_f = np.where(
                micro_prec + micro_rec > 0,
                (2 * micro_prec * micro_rec) / (micro_prec + micro_rec),
                0.0,
            )

        # 8. Seleção do F-max Global
        best_idx = np.argmax(micro_f)
        f1 = float(micro_f[best_idx])
        max_th = thresholds[best_idx]

    else:
        raise ValueError("average must be 'macro' or 'micro'")

    # Retornos condicionais
    if additional_result == "threshold":
        return f1, max_th

    elif additional_result == "full":
        # Extrai as estatísticas individuais por coluna usando o melhor threshold
        bool_matrix = pred_scores > max_th
        truth_bool = safe_truth > 0.5  # Assumes positive is larger than 0.5

        # TP: previsto True, é True, e é Válido
        tps = np.sum(bool_matrix & truth_bool, axis=0)
        # FP: previsto True, é False, e é Válido (ignora prever True num NaN)
        fps = np.sum(bool_matrix & ~truth_bool & valid_mask, axis=0)
        # FN: previsto False, é True, e é Válido
        fns = np.sum(~bool_matrix & truth_bool, axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            col_prec = np.where(tps + fps > 0, tps / (tps + fps), 0.0)
            col_rec = np.where(tps + fns > 0, tps / (tps + fns), 0.0)
            fmax_by_col = np.where(
                col_prec + col_rec > 0,
                (2 * col_prec * col_rec) / (col_prec + col_rec),
                0.0,
            )

        return f1, {
            "threshold": max_th,
            # "precision": prec_mean[best_idx],
            # "recall": rec_mean[best_idx],
            "fmax_by_col": fmax_by_col.tolist(),
        }
    else:
        return f1, max_th

def mcc_bycol_weighted_masked(
    pred_scores: np.ndarray,
    truth_set: np.ndarray,
    weights: np.ndarray,
    n_ths: int = 120,
):
    """
    Calcula o MCC por coluna, com NaN masking e weights.
    MCC = (TP*TN - FP*FN) / sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))

    A função:
    - Ignora NaNs completamente na contagem de TPs, FPs, TNs e FNs.
    - Calcula o melhor MCC para cada termo GO (coluna) variando os limiares.
    - Faz a média macro (simples) e a média macro ponderada pelo Information Accretion (IA).
    """
    thresholds = np.linspace(0, 1, n_ths)

    # 1. Máscara de validade (ignora NaNs da contagem)
    valid_mask = ~np.isnan(truth_set)

    # 2. Verdade Binarizada e Segura
    # (Garante que só valores não NaN e >= 0.5 serão True)
    truth_bool = np.where(valid_mask, truth_set, 0.0) >= 0.5

    # 3. Predições em múltiplos limiares usando Broadcasting
    # pred_scores vira (1, samples, labels)
    # thresholds vira (n_ths, 1, 1)
    # pred_bin final: (n_ths, n_samples, n_labels)
    pred_bin = pred_scores[None, :, :] > thresholds[:, None, None]

    # 4. Cálculo Vetorizado Simultâneo (Matrizes: n_ths x n_labels)
    # A soma ocorre ao longo do eixo das amostras (axis=1)

    # TP: Predito True, Real True
    tp = np.sum(pred_bin & truth_bool[None, :, :], axis=1)

    # FP: Predito True, Real False, e É Válido (Não é NaN)
    fp = np.sum(pred_bin & ~truth_bool[None, :, :] & valid_mask[None, :, :], axis=1)

    # FN: Predito False, Real True
    fn = np.sum(~pred_bin & truth_bool[None, :, :], axis=1)

    # TN: Predito False, Real False, e É Válido (Não é NaN)
    tn = np.sum(~pred_bin & ~truth_bool[None, :, :] & valid_mask[None, :, :], axis=1)

    # 5. Casting para Float64 (Crucial para evitar Integer Overflow no Denominador)
    tp = tp.astype(np.float64)
    fp = fp.astype(np.float64)
    fn = fn.astype(np.float64)
    tn = tn.astype(np.float64)

    # 6. Cálculo do MCC
    num = (tp * tn) - (fp * fn)
    denom_sq = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    denom = np.sqrt(denom_sq)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Se o denominador for 0, o MCC é definido matematicamente como 0.0
        mcc_matrix = np.where(denom > 0, num / denom, 0.0)

    # 7. Extração do Max MCC por Coluna (Termo GO)
    # axis=0 esmaga a dimensão dos thresholds guardando apenas o melhor
    max_mcc_per_col = np.max(mcc_matrix, axis=0)

    # 8. Agregações Finais
    macro_mcc = float(np.mean(max_mcc_per_col))

    w = np.array(weights).flatten()
    if np.sum(w) > 0:
        weighted_macro_mcc = float(np.average(max_mcc_per_col, weights=w))
    else:
        weighted_macro_mcc = 0.0

    return {
        "macro_mcc": macro_mcc,
        "weighted_macro_mcc": weighted_macro_mcc,
        "mcc_by_col": max_mcc_per_col.tolist(),
    }

def calculate_weighted_micro_fmax(y_true: np.ndarray, y_pred: np.ndarray, n_thresholds: int = 100):
    """
    Builds a (n_thresholds, 4, n_cols) confusion matrix and extracts:
    1. Standard Micro F-max
    2. Inverse-Weighted Micro F-max (Weights applied to counts before summing)
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    n_cols = y_true.shape[1]
    
    # Create mask to ignore NaNs
    mask = ~np.isnan(y_true)
    
    # Binarize targets (Strict CAFA style)
    y_true_bin = (y_true == 1.0) 

    # Initialize 3D Matrix: shape (n_thresholds, 4, n_cols)
    # Index map: 0 = TP, 1 = TN, 2 = FP, 3 = FN
    conf_matrix = np.zeros((n_thresholds, 4, n_cols))

    for i, t in enumerate(thresholds):
        pred_bool = y_pred > t
        
        # Calculate boolean overlaps, sum across rows (axis=0) to get counts per column
        conf_matrix[i, 0, :] = (pred_bool & y_true_bin & mask).sum(axis=0)       # TP
        conf_matrix[i, 1, :] = (~pred_bool & ~y_true_bin & mask).sum(axis=0)     # TN
        conf_matrix[i, 2, :] = (pred_bool & ~y_true_bin & mask).sum(axis=0)      # FP
        conf_matrix[i, 3, :] = (~pred_bool & y_true_bin & mask).sum(axis=0)      # FN

    # ==========================================
    # 1. Standard MICRO F-max (Unweighted)
    # ==========================================
    # Sum raw counts across all columns (axis=1)
    tp_micro = conf_matrix[:, 0, :].sum(axis=1)
    fp_micro = conf_matrix[:, 2, :].sum(axis=1)
    fn_micro = conf_matrix[:, 3, :].sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        prec_micro = np.where(tp_micro + fp_micro > 0, tp_micro / (tp_micro + fp_micro), 0.0)
        rec_micro = np.where(tp_micro + fn_micro > 0, tp_micro / (tp_micro + fn_micro), 0.0)
        f1_micro = np.where(prec_micro + rec_micro > 0, 2 * prec_micro * rec_micro / (prec_micro + rec_micro), 0.0)
    
    micro_fmax = f1_micro.max()

    # ==========================================
    # 2. INVERSE-WEIGHTED MICRO F-max
    # ==========================================
    # Calculate how many valid (non-NaN) items exist per column
    valid_counts = mask.sum(axis=0)
    
    # Calculate inverse weights: 1 / Count
    # We use np.where to safely handle columns that are 100% NaN (prevent 1/0)
    inv_weights = np.where(valid_counts > 0, 1.0 / valid_counts, 0.0)
    
    # Normalize weights so they sum to 1.0
    if inv_weights.sum() > 0:
        inv_weights /= inv_weights.sum()

    # Apply the weights directly to the counts in the confusion matrix
    # Broadcasting: (n_thresholds, 4, n_cols) * (n_cols,) -> Multiplies along the last axis
    weighted_conf_matrix = conf_matrix * inv_weights

    # Now sum the WEIGHTED counts across columns (axis=1)
    tp_w_micro = weighted_conf_matrix[:, 0, :].sum(axis=1)
    fp_w_micro = weighted_conf_matrix[:, 2, :].sum(axis=1)
    fn_w_micro = weighted_conf_matrix[:, 3, :].sum(axis=1)

    with np.errstate(divide='ignore', invalid='ignore'):
        prec_w_micro = np.where(tp_w_micro + fp_w_micro > 0, tp_w_micro / (tp_w_micro + fp_w_micro), 0.0)
        rec_w_micro = np.where(tp_w_micro + fn_w_micro > 0, tp_w_micro / (tp_w_micro + fn_w_micro), 0.0)
        f1_w_micro = np.where(prec_w_micro + rec_w_micro > 0, 2 * prec_w_micro * rec_w_micro / (prec_w_micro + rec_w_micro), 0.0)

    inverse_weighted_micro_fmax = f1_w_micro.max()

    return micro_fmax, inverse_weighted_micro_fmax




def run_statistics(y_pred, y_eval_cafa, y_eval_owa, weights, bottom_gos_perc=0.2):
    n_gos = y_eval_cafa.shape[1]
    n_bottom_gos = round(n_gos * bottom_gos_perc)
    # find n_bottom_gos columns in y_true with smallest sums
    # bottom_go_indices = np.argsort(y_eval_cafa.sum(axis=0))[:n_bottom_gos]
    # bottom_go_indices = bottom_go_indices.tolist()

    #fmax_mean_cafa, fmax_all_cafa = macro_fmax(y_eval_cafa, y_pred)
    #bottom_fmaxes = sorted(list(fmax_all_cafa))[:n_bottom_gos]
    #fmax_bottom20percent_cafa = np.mean(bottom_fmaxes)

    """fmax_mean_conditional, fmax_all_conditional = macro_fmax(y_eval_owa, y_pred)
    bottom_fmaxes = sorted(list(fmax_all_conditional))[:n_bottom_gos]
    fmax_bottom20percent_conditional = np.mean(bottom_fmaxes)"""

    # Find macro AUPRC with scikit-learn
    auprc_score_cafa = average_precision_score(y_eval_cafa, y_pred, average="macro")
    auprc_score_owa = nan_macro_average_precision(y_eval_owa, y_pred, weights=weights)
    """
    print(f"Fmax mean (CAFA): {fmax_mean_cafa}")
    print(f"Fmax bottom 20 percent (CAFA): {fmax_bottom20percent_cafa}")
    print(f"Fmax mean (Conditional): {fmax_mean_conditional}")
    print(f"Fmax bottom 20 percent (Conditional): {fmax_bottom20percent_conditional}")
    print(f"AUPRC (Macro) (CAFA): {auprc_score_cafa}")   
    print(f"AUPRC (Macro) (Conditional): {auprc_score_conditional}")"""

    cafa_fmax, other_metrics = faster_fmax_weighted(
        y_pred, y_eval_cafa, weights, additional_result="full"
    )
    #fmax_list = other_metrics["fmax_by_col"]
    #fmax_list_bottom_20 = sorted(fmax_list)[:n_bottom_gos]
    #fmax_bottom20percent_cafa = np.mean(fmax_list_bottom_20)

    #cafa_fmax_owa_macro, other_metrics_owa_macro = faster_fmax_weighted_nan(
    #    y_pred, y_eval_owa, weights, additional_result="full"
    #)
    #fmax_list2 = other_metrics_owa_macro["fmax_by_col"]
    #fmax_list2_bottom_20 = sorted(fmax_list2)[:n_bottom_gos]
    #fmax_bottom20percent_owa = np.mean(fmax_list2_bottom_20)

    #cafa_fmax_owa_micro, other_metrics_owa_micro = faster_fmax_weighted_nan(
    #    y_pred, y_eval_owa, weights, additional_result="full", average="micro"
    #)

    mcc_dict = mcc_bycol_weighted_masked(y_pred, y_eval_owa, weights)
    mcc_cafa_dict = mcc_bycol_weighted_masked(y_pred, y_eval_cafa, weights)
    #macro_mcc = mcc_dict["macro_mcc"]
    weighted_macro_mcc = mcc_dict["weighted_macro_mcc"]
    #cafa_macro_mcc = mcc_cafa_dict["macro_mcc"]
    cafa_weighted_macro_mcc = mcc_cafa_dict["weighted_macro_mcc"]

    fmax_owa_micro, inverse_weighted_micro_fmax = calculate_weighted_micro_fmax(y_eval_owa, y_pred)

    y_stats = {
        #"OWA Weighted Fmax": cafa_fmax_owa_macro,
        #"OWA Weighted Fmax (micro)": cafa_fmax_owa_micro,
        #"OWA Fmax (micro)": fmax_owa_micro,
        "OWA Fmax (Inverse-Weighted)": inverse_weighted_micro_fmax,
        #"OWA Weighted Fmax (lowest 20%)": fmax_bottom20percent_owa,
        #"OWA MCC (Macro)": macro_mcc,
        "OWA Weighted MCC": weighted_macro_mcc,
        "OWA Weighted AUPRC": auprc_score_owa,
        
        "CAFA Weighted Fmax": cafa_fmax,
        #"CAFA Weighted Fmax (lowest 20%)": fmax_bottom20percent_cafa,
        #"CAFA MCC (Macro)": cafa_macro_mcc,
        "CAFA Weighted MCC": cafa_weighted_macro_mcc,
        "CAFA AUPRC": auprc_score_cafa,
    }

    return y_stats

'''metric_weights_for_sorting = {
    "OWA Inverse-Weighted Fmax": 1,
    "OWA Weighted MCC": 1,
    "OWA Weighted AUPRC": 1,
    "CAFA Weighted Fmax": 1,
    "CAFA AUPRC": 1,
}'''

metric_weights_for_sorting = {
    "OWA Fmax (Inverse-Weighted)": 1,
    "OWA Weighted MCC": 1,
    "OWA Weighted AUPRC": 1,
    "CAFA Weighted Fmax": 1,
    "CAFA Weighted MCC": 1,
    "CAFA AUPRC": 1,
}

def get_sorting_score(results: dict) -> float:
    total_score = 0
    for metric_name, metric_value in results.items():
        if metric_name not in metric_weights_for_sorting:
            continue
        total_score += metric_value * metric_weights_for_sorting[metric_name]
    total_score = total_score / sum(metric_weights_for_sorting.values())
    return total_score