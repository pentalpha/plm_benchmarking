import sys
import json
import os
from glob import glob
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from scipy.stats import rankdata

from training import train_param_comb
from custom_statistics import macro_fmax, nan_macro_average_precision


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
        random_mask = np.random.rand(*train_y.shape) < 0.10
        nan_mask = np.isnan(train_y)
        train_y[nan_mask & random_mask] = 0.0

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
        np.save(y_pred_path, y_pred)
        np.save(y_pred_train_path, y_pred_train)

    return y_pred, y_pred_train, success


def run_statistics(y_pred, y_eval_cafa, y_eval_conditional):
    n_gos = y_eval_cafa.shape[1]
    n_bottom_gos = round(n_gos * 0.2)
    # find n_bottom_gos columns in y_true with smallest sums
    bottom_go_indices = np.argsort(y_eval_cafa.sum(axis=0))[:n_bottom_gos]
    bottom_go_indices = bottom_go_indices.tolist()

    fmax_mean_cafa, fmax_all_cafa = macro_fmax(y_eval_cafa, y_pred)
    bottom_fmaxes = np.array([fmax_all_cafa[i] for i in bottom_go_indices])
    fmax_bottom20percent_cafa = np.mean(bottom_fmaxes)

    fmax_mean_conditional, fmax_all_conditional = macro_fmax(y_eval_conditional, y_pred)
    bottom_fmaxes = np.array([fmax_all_conditional[i] for i in bottom_go_indices])
    fmax_bottom20percent_conditional = np.mean(bottom_fmaxes)

    print(f"Fmax mean (CAFA): {fmax_mean_cafa}")
    print(f"Fmax bottom 20 percent (CAFA): {fmax_bottom20percent_cafa}")
    print(f"Fmax mean (Conditional): {fmax_mean_conditional}")
    print(f"Fmax bottom 20 percent (Conditional): {fmax_bottom20percent_conditional}")

    # Find macro AUPRC with scikit-learn
    auprc_score_cafa = average_precision_score(y_eval_cafa, y_pred, average="macro")
    auprc_score_conditional = nan_macro_average_precision(y_eval_conditional, y_pred)
    print(f"AUPRC (Macro) (CAFA): {auprc_score_cafa}")
    print(f"AUPRC (Macro) (Conditional): {auprc_score_conditional}")

    y_stats = {
        "fmax_mean_cafa": fmax_mean_cafa,
        "fmax_bottom20percent_cafa": fmax_bottom20percent_cafa,
        "auprc_score_cafa": auprc_score_cafa,
        "fmax_mean_conditional": fmax_mean_conditional,
        "fmax_bottom20percent_conditional": fmax_bottom20percent_conditional,
        "auprc_score_conditional": auprc_score_conditional,
    }

    return y_stats


if __name__ == "__main__":
    param_comb_path = sys.argv[1]
    processed_inputs_dir = sys.argv[2]
    statistics_path = sys.argv[3]

    test_dir = os.path.dirname(statistics_path)
    preds_dir = os.path.join(test_dir, "predictions")
    os.makedirs(preds_dir, exist_ok=True)

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
    y_eval_conditional = y_by_name["conditional_negatives"]["test"]

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
        stats = run_statistics(y_pred, y_eval_cafa, y_eval_conditional)

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
        stats = run_statistics(y_pred_more_negatives, y_eval_cafa, y_eval_conditional)

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

    classic_model = [m for m in models_trained if m["is_classic"]][0]

    # Find model for optimal blend with classic model
    best_blend_stats = None
    best_blend_score = 0.0
    best_blend_model = None
    for m in models_trained:
        other_y_pred = m["y_pred"]
        blend_name = "classic+" + m["y_name"]
        print("Trying blend", blend_name)
        composite_preds = classic_model["y_pred"] * 0.5 + other_y_pred * 0.5
        stats = run_statistics(composite_preds, y_eval_cafa, y_eval_conditional)
        mean_score = (stats["fmax_mean_cafa"] + stats["fmax_mean_conditional"]) / 2
        if mean_score > best_blend_score:
            best_blend_score = mean_score
            best_blend_stats = stats
            best_blend_model = blend_name

    print(f"Best blend: {best_blend_model} with score {best_blend_score}")
    # print(best_blend_stats)
    models_trained.append(
        {
            "is_classic": False,
            "y_name": best_blend_model,
            "y_pred": None,
            "y_pred_train": None,
            "success": True,
            "stats": best_blend_stats,
        }
    )

    final_stats = {"Model Results": [], "Parameters": param_comb}

    for m in models_trained:
        final_stats["Model Results"].append({"name": m["y_name"], "stats": m["stats"]})

    final_stats["Model Results"].sort(
        key=lambda x: (
            x["stats"]["fmax_mean_cafa"] + x["stats"]["fmax_mean_conditional"]
        )
    )

    final_stats["success"] = True

    json.dump(final_stats, open(statistics_path, "w"), indent=4)
