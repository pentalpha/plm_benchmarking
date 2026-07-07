import numpy as np


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
