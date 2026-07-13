import sys
import json
import os
from glob import glob
import numpy as np
from polars.catalog.unity import models
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata

from training import train_param_comb
from custom_statistics import (
    macro_fmax,
    nan_macro_average_precision,
    faster_fmax_weighted,
    faster_fmax_weighted_nan,
    get_ia_vector,
    faster_fmax_weighted_nan,
    calc_normalized_y_pred,
    nan_macro_average_precision,
    mcc_bycol_weighted_masked,
)

go_ia_path = "input_data/go_ia.tsv"
go_ia_dict = {
    l.strip().split("\t")[0]: float(l.strip().split("\t")[1]) for l in open(go_ia_path)
}


def train_logistic_stacker(y_train, p_train_1, p_train_2, p_test_1, p_test_2):
    """
    Treina um meta-modelo (Regressão Logística) para cada termo GO.
    Usa apenas os dados de treino para aprender os pesos ideais e aplica no teste.
    """
    print("Treinando Meta-Modelo (Logistic Regression Stacker)...")
    n_samples, n_classes = p_test_1.shape
    p_stacker_test = np.zeros_like(p_test_1)

    for i in range(n_classes):
        target_col = y_train[:, i]

        # Evita erro no Scikit-Learn se a coluna tiver apenas uma classe (ex: tudo 0)
        if np.sum(target_col) == 0 or np.sum(target_col) == len(target_col):
            # Fallback de segurança: faz a média simples se não puder treinar
            p_stacker_test[:, i] = (p_test_1[:, i] + p_test_2[:, i]) / 2.0
            continue

        # Features (X): Predições dos modelos base
        X_train_meta = np.column_stack((p_train_1[:, i], p_train_2[:, i]))
        X_test_meta = np.column_stack((p_test_1[:, i], p_test_2[:, i]))

        # Treino estrito no X_train_meta
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train_meta, target_col)

        # Inferência estrita no X_test_meta
        p_stacker_test[:, i] = lr.predict_proba(X_test_meta)[:, 1]

    return p_stacker_test


def rank_average_stacker(p_test_1, p_test_2):
    """
    Combina dois modelos pela média de seus ranks, ignorando a escala de probabilidade bruta.
    Excelente quando modelos têm calibrações muito diferentes (Classic vs Masked).
    """
    print("Aplicando Meta-Modelo (Rank Averaging)...")
    p_stacker_test = np.zeros_like(p_test_1)
    n_samples, n_classes = p_test_1.shape

    for i in range(n_classes):
        # Transforma probabilidades em posições ordenadas (0 a 1)
        r1 = rankdata(p_test_1[:, i]) / n_samples
        r2 = rankdata(p_test_2[:, i]) / n_samples

        # Média simples dos ranks
        p_stacker_test[:, i] = (r1 + r2) / 2.0

    return p_stacker_test


def get_optimal_blend_weights(y_true_original, p_train_1, p_train_2):
    """
    Otimiza pesos garantindo limites de segurança e avaliando o erro
    APENAS nas anotações reais (ignorando NaNs).
    """
    print("Otimizando pesos do Blend com limites de segurança (Fair Evaluation)...")
    n_classes = y_true_original.shape[1]
    alphas = np.zeros(n_classes)

    # LIMITES MENOS EXTREMOS: Busca pesos do Classic apenas entre 25% e 75%
    weight_grid = np.linspace(0.35, 0.65, 21)

    for i in range(n_classes):
        target_col = y_true_original[:, i]

        # Cria uma máscara para focar apenas nos dados que NÃO são NaN
        # Isso impede que o modelo Fuzzy seja punido por prever > 0 em áreas NaN
        valid_mask = ~np.isnan(target_col)

        # Se não houver positivos confirmados, deixa 50/50
        if np.sum(target_col[valid_mask]) == 0 or np.sum(valid_mask) == 0:
            alphas[i] = 0.5
            continue

        best_alpha = 0.5
        best_error = float("inf")

        # Filtra os dados usando a máscara
        target_valid = target_col[valid_mask]
        p1_valid = p_train_1[valid_mask, i]  # Predições Classic
        p2_valid = p_train_2[valid_mask, i]  # Predições Fuzzy/More Negatives

        for alpha in weight_grid:
            p_blend = alpha * p1_valid + (1 - alpha) * p2_valid

            # Erro Quadrático APENAS nas anotações confirmadas
            error = np.mean((target_valid - p_blend) ** 2)

            if error < best_error:
                best_error = error
                best_alpha = alpha

        alphas[i] = best_alpha

    return alphas


def train_and_save_preds(train_x, test_x, y_name, y_data, more_negatives):
    train_y = y_data["train"]
    test_y = y_data["test"]

    if more_negatives:
        np.random.seed(42)  # Mantém benchmarks reprodutíveis
        target_min_zeros = 0.12
        # x != NaN and x < 0.5 = negative evi = zero
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
                train_y.ravel()[chosen_nan_indices] = 0.0

    preds_result_basename = preds_dir + "/" + y_name
    y_pred_path = preds_result_basename + ".npy"
    y_pred_train_path = preds_result_basename + ".train.npy"

    loaded = False

    if os.path.exists(y_pred_path) and os.path.exists(y_pred_train_path):
        try:
            y_pred = np.load(y_pred_path)
            y_pred_train = np.load(y_pred_train_path)
            loaded = True
            success = True
        except Exception as e:
            print(e)
            loaded = False

    if not loaded:
        y_pred, y_pred_train, success = train_param_comb(
            param_comb, train_x, test_x, train_y, test_y
        )
        if success:
            np.save(y_pred_path, y_pred)
            np.save(y_pred_train_path, y_pred_train)

    return y_pred, y_pred_train, success


def run_statistics(y_pred, y_eval_cafa, y_eval_owa, weights):
    n_gos = y_eval_cafa.shape[1]
    n_bottom_gos = round(n_gos * 0.2)
    # find n_bottom_gos columns in y_true with smallest sums
    # bottom_go_indices = np.argsort(y_eval_cafa.sum(axis=0))[:n_bottom_gos]
    # bottom_go_indices = bottom_go_indices.tolist()

    fmax_mean_cafa, fmax_all_cafa = macro_fmax(y_eval_cafa, y_pred)
    bottom_fmaxes = sorted(list(fmax_all_cafa))[:n_bottom_gos]
    fmax_bottom20percent_cafa = np.mean(bottom_fmaxes)

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
    fmax_list = other_metrics["fmax_by_col"]
    fmax_list_bottom_20 = sorted(fmax_list)[:n_bottom_gos]
    fmax_bottom20percent_cafa = np.mean(fmax_list_bottom_20)

    cafa_fmax_owa_macro, other_metrics_owa_macro = faster_fmax_weighted_nan(
        y_pred, y_eval_owa, weights, additional_result="full"
    )
    fmax_list2 = other_metrics_owa_macro["fmax_by_col"]
    fmax_list2_bottom_20 = sorted(fmax_list2)[:n_bottom_gos]
    fmax_bottom20percent_owa = np.mean(fmax_list2_bottom_20)

    cafa_fmax_owa_micro, other_metrics_owa_micro = faster_fmax_weighted_nan(
        y_pred, y_eval_owa, weights, additional_result="full", average="micro"
    )

    mcc_dict = mcc_bycol_weighted_masked(y_pred, y_eval_owa, weights)
    macro_mcc = mcc_dict["macro_mcc"]
    weighted_macro_mcc = mcc_dict["weighted_macro_mcc"]

    y_stats = {
        "OWA Weighted Fmax": cafa_fmax_owa_macro,
        "OWA Weighted Fmax (micro)": cafa_fmax_owa_micro,
        "OWA Weighted MCC": macro_mcc,
        "OWA Weighted MCC (micro)": weighted_macro_mcc,
        "OWA Weighted AUPRC": auprc_score_owa,
        "OWA Weighted Fmax (lowest 20%)": fmax_bottom20percent_owa,
        "CAFA Weighted Fmax": cafa_fmax,
        "CAFA Weighted Fmax (lowest 20%)": fmax_bottom20percent_cafa,
        "CAFA Fmax Macro": fmax_mean_cafa,
        "CAFA AUPRC": auprc_score_cafa,
    }

    return y_stats


if __name__ == "__main__":
    param_comb_path = sys.argv[1]
    processed_inputs_dir = sys.argv[2]
    statistics_path = sys.argv[3]

    test_dir = os.path.dirname(statistics_path)
    preds_dir = os.path.join(test_dir, "predictions")
    os.makedirs(preds_dir, exist_ok=True)

    targets_tests_path = os.path.dirname(
        os.path.dirname(os.path.dirname(processed_inputs_dir))
    )
    targets_name = os.path.basename(targets_tests_path)
    targets_parquet = f"outputs/{targets_name}.parquet"
    label_names_path = targets_parquet + ".targets.txt"
    labels = [l.strip() for l in open(label_names_path)]

    param_comb = json.load(open(param_comb_path))
    train_x = np.load(os.path.join(processed_inputs_dir, "train_x.npy"))
    test_x = np.load(os.path.join(processed_inputs_dir, "test_x.npy"))

    y_np_files = glob(f"{processed_inputs_dir}/*_y_*.npy")
    y_by_name = {}
    for y_file in y_np_files:
        y_name = os.path.basename(y_file).split("_y_")[1].split(".npy")[0]
        is_train = os.path.basename(y_file).startswith("train")
        np_file = np.load(y_file)
        if not y_name in y_by_name.keys():
            y_by_name[y_name] = {"train": None, "test": None}
        if is_train:
            y_by_name[y_name]["train"] = np_file
        else:
            y_by_name[y_name]["test"] = np_file

    for y_name, y_data in y_by_name.items():
        train_y = y_data["train"]
        test_y = y_data["test"]
        # Este é um dataset binário? (apenas 1.0 e 0.0)
        unique_values = np.unique(train_y)
        print("Unique values in train_y:", unique_values)
        is_classic = len(unique_values) == 2
        print("Is classic:", is_classic)
        y_data["is_classic"] = is_classic

    y_eval_cafa = y_by_name["classic"]["test"]
    y_eval_owa = y_by_name["open_world_assumption"]["test"]
    y_by_name = {k: v for k, v in y_by_name.items() if k != "open_world_assumption"}

    assert all([go in go_ia_dict for go in labels])
    weights = get_ia_vector(labels, go_ia_dict)

    models_trained = []

    targets_progress_bar = tqdm(
        y_by_name.items(), total=len(y_by_name.keys()), desc="Training models"
    )

    more_negatives_by_name = {}
    for y_name, y_data in targets_progress_bar:
        is_classic = y_data["is_classic"]
        print("Training model: ", y_name)
        y_pred, y_pred_train, success = train_and_save_preds(
            train_x, test_x, y_name, y_data, False
        )
        if success:
            stats = run_statistics(y_pred, y_eval_cafa, y_eval_owa, weights)

            models_trained.append(
                {
                    "is_classic": is_classic,
                    "y_name": y_name,
                    "y_pred": y_pred,
                    "y_pred_train": y_pred_train,
                    "success": success,
                    "stats": stats,
                }
            )

            if not is_classic:
                more_negatives_by_name[y_name + "-more_negatives"] = y_data

        targets_progress_bar.update(1)

    bar2 = tqdm(
        more_negatives_by_name.items(),
        total=len(more_negatives_by_name.keys()),
        desc="Training models with more negatives",
    )

    for y_name, y_data in bar2:
        print("Training model with more negatives: ", y_name)
        y_pred_more_negatives, y_pred_more_negatives_train, success_neg = (
            train_and_save_preds(train_x, test_x, y_name, y_data, True)
        )
        if success_neg:
            stats = run_statistics(
                y_pred_more_negatives, y_eval_cafa, y_eval_owa, weights
            )

            models_trained.append(
                {
                    "is_classic": False,
                    "y_name": y_name,
                    "y_pred": y_pred_more_negatives,
                    "y_pred_train": y_pred_more_negatives_train,
                    "success": success_neg,
                    "stats": stats,
                }
            )

    if len(models_trained) > 0:

        classic_model = [m for m in models_trained if m["is_classic"]][0]

        blend_models = []

        for m in models_trained:
            if m["is_classic"]:
                continue
            other_y_pred = m["y_pred"]
            blend_name = "classic+" + m["y_name"]
            print("Trying blend", blend_name)

            preds_result_basename = preds_dir + "/" + blend_name
            y_pred_path = preds_result_basename + ".npy"

            composite_preds = None
            if os.path.exists(y_pred_path):
                try:
                    composite_preds = np.load(y_pred_path)
                    print("Loaded composite predictions from", y_pred_path)
                except Exception as e:
                    print(e)

            if composite_preds is None:
                print("Composite predictions not found, computing...")
                composite_preds = classic_model["y_pred"] * 0.5 + other_y_pred * 0.5
                np.save(y_pred_path, composite_preds)

            stats = run_statistics(composite_preds, y_eval_cafa, y_eval_owa, weights)

            blend_models.append(
                {
                    "is_classic": False,
                    "y_name": blend_name,
                    "y_pred": composite_preds,
                    "y_pred_train": None,
                    "success": True,
                    "stats": stats,
                }
            )

        all_models = models_trained + blend_models

        final_stats = {"Model Results": [], "Parameters": param_comb}

        for m in all_models:
            final_stats["Model Results"].append(
                {"name": m["y_name"], "stats": m["stats"]}
            )

        final_stats["Model Results"].sort(
            key=lambda x: (
                x["stats"]["CAFA Fmax Macro"] + x["stats"]["OWA Weighted Fmax"]
            )
        )

        final_stats["success"] = True

        json.dump(final_stats, open(statistics_path, "w"), indent=4)
    else:
        print("No models trained.")
        final_stats = {"Model Results": [], "Parameters": param_comb}
        final_stats["success"] = False
        json.dump(final_stats, open(statistics_path, "w"), indent=4)
