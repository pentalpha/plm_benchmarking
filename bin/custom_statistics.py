import numpy as np
import warnings
from sklearn.metrics import average_precision_score
from typing import Tuple


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


def macro_fmax(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, list]:
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


def nan_macro_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
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
            ap_scores.append(ap)

    # Handle the edgecase where masking left no valid classes to score
    if not ap_scores:
        warnings.warn("No valid classes with both positive and negative samples found.")
        return np.nan

    return float(np.mean(ap_scores))
