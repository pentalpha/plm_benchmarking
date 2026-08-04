import sys
import json
import os
import random
from glob import glob
import numpy as np
import polars as pl
from tqdm import tqdm

# from training import train_param_comb
from custom_statistics import (
    get_ia_vector,
    run_statistics,
    calc_normalized_y_pred,
    get_sorting_score,
)
from fuzzy_ml import create_ontology_dictionaries_full, show_y_density
from train_eval import go_ia_dict, train_and_save_preds

url_ou_caminho_obo = "input_data/go-basic.obo"
parents_dict, children_dict, go_sortings = create_ontology_dictionaries_full(
    url_ou_caminho_obo
)

gene_names = [
    "Phylogenetic True",
    "Conditional False",
    "Derivate False",
    "Random Falses",
    "Random Falses Min Perc",
    "Random False Val",
]


def generate_fuzzy_metaparameters(n_combinations: int, try_more=True):
    # [phylo_true_val, cond_false_val, derivate_false_val, add_random_falses, random_falses_min_perc, random_false_val]
    param_combs = []
    """
    param_combs.append(np.random.uniform(0.75, 1.0, n_combinations))
    param_combs.append(np.random.uniform(0.0, 0.45, n_combinations))
    param_combs.append(np.random.uniform(0.0, 0.4, n_combinations))
    param_combs.append(np.random.choice([True, False], n_combinations))
    param_combs.append(np.random.uniform(0.1, 0.35, n_combinations))
    param_combs.append(np.random.uniform(0.0, 0.45, n_combinations))
    """
    param_combs.append(np.random.uniform(0.88, 1.0, n_combinations))
    param_combs.append(np.random.uniform(0.21, 0.44, n_combinations))
    param_combs.append(np.random.uniform(0.06, 0.33, n_combinations))
    param_combs.append(np.random.choice([True, True], n_combinations))
    param_combs.append(np.random.uniform(0.11, 0.42, n_combinations))
    param_combs.append(np.random.uniform(0.06, 0.43, n_combinations))

    param_combs_valid = set()
    for i in range(n_combinations):
        phylo_true_val = param_combs[0][i]
        cond_false_val = param_combs[1][i]
        derivate_false_val = param_combs[2][i]
        add_random_falses = param_combs[3][i]
        random_falses_min_perc = param_combs[4][i]
        random_false_val = param_combs[5][i]
        comb = (
            round(phylo_true_val, 2),
            round(cond_false_val, 2),
            round(derivate_false_val, 2),
            add_random_falses,
            round(random_falses_min_perc, 2),
            round(random_false_val, 2),
        )
        if derivate_false_val <= cond_false_val:
            param_combs_valid.add(comb)
    print(
        f"| INFO | Number of fuzzy value combinations created: {len(param_combs_valid)} out of {n_combinations}"
    )
    while len(param_combs_valid) < n_combinations and try_more:
        next_batch = n_combinations - len(param_combs_valid)
        print(f"| INFO | Generating {next_batch} additional combinations...")
        new_combs = generate_fuzzy_metaparameters(n_combinations, try_more=False)
        actually_new = new_combs - param_combs_valid
        if len(actually_new) > next_batch:
            actually_new = set(random.sample(list(actually_new), next_batch))
        param_combs_valid.update(actually_new)
        print(
            f"| INFO | Total combinations: {len(param_combs_valid)} out of {n_combinations}"
        )
    return param_combs_valid


def save_stats_json_safe(statistics_path: str, stats, ont: str):
    if os.path.exists(statistics_path):
        old_backup_path = statistics_path.replace(".json", ".old.json")
        if os.path.exists(old_backup_path):
            os.remove(old_backup_path)
        os.rename(statistics_path, old_backup_path)
        json.dump(stats, open(statistics_path, "w"), indent=4)

    else:
        json.dump(stats, open(statistics_path, "w"), indent=4)

    owa_metrics = [
        "OWA Weighted Fmax (micro)",
        "OWA Weighted MCC",
        "OWA Weighted AUPRC",
    ]
    cwa_metrics = ["CAFA Weighted Fmax", "CAFA AUPRC"]

    new_lines = []
    for result in stats:
        norm_stats = result["stats"]
        new_line = {"Config Name": None, "Ontology": ont}
        for i, gene_name in enumerate(gene_names):
            new_line[gene_name] = result["comb"][i]
        owa_sum = 0.0
        for owa_met in owa_metrics:
            new_line[owa_met] = norm_stats[owa_met]
            owa_sum += norm_stats[owa_met]
        cwa_sum = 0.0
        for cwa_met in cwa_metrics:
            new_line[cwa_met] = norm_stats[cwa_met]
            cwa_sum += norm_stats[cwa_met]

        new_line["OWA Sort Score"] = owa_sum / len(owa_metrics)
        new_line["CWA Sort Score"] = cwa_sum / len(cwa_metrics)
        new_line["Sort Score"] = norm_stats["Sort Score"]
        new_lines.append(new_line)

    best_owa = sorted(new_lines, key=lambda x: x["OWA Sort Score"], reverse=True)[0]
    best_cwa = sorted(new_lines, key=lambda x: x["CWA Sort Score"], reverse=True)[0]
    best_sort = sorted(new_lines, key=lambda x: x["Sort Score"], reverse=True)[0]

    best_owa["Config Name"] = "Best OWA"
    best_cwa["Config Name"] = "Best CWA"
    best_sort["Config Name"] = "Best Sort"

    df = pl.DataFrame([best_owa, best_cwa, best_sort])
    df.write_csv(statistics_path.replace(".json", ".tsv"), separator="\t")


def update_y_data_with_new_values(
    y: np.ndarray,
    phylo_true_val: float,
    cond_false_val: float,
    derivate_false_val: float,
):
    derivated_false_orig_val = 0.025
    conditional_false_orig_val = 0.15
    phylogenetic_positive_orig_val = 0.9

    y = np.where(y == derivated_false_orig_val, derivate_false_val, y)
    y = np.where(y == conditional_false_orig_val, cond_false_val, y)
    y = np.where(y == phylogenetic_positive_orig_val, phylo_true_val, y)
    return y


if __name__ == "__main__":
    param_comb_path = sys.argv[1]
    processed_inputs_dir = sys.argv[2]
    test_dir = sys.argv[3]
    n_combinations = int(sys.argv[4])
    run_new = True

    statistics_path = os.path.join(test_dir, "statistics.json")

    # preds_dir = os.path.join(test_dir, "predictions")
    os.makedirs(test_dir, exist_ok=True)
    print(processed_inputs_dir)
    targets_tests_path = os.path.dirname(os.path.dirname(processed_inputs_dir))
    print(targets_tests_path)
    targets_name = os.path.basename(targets_tests_path)
    print(targets_name)
    targets_parquet = f"outputs/{targets_name}.parquet"
    print(targets_parquet)
    label_names_path = targets_parquet + ".targets.txt"
    labels = [l.strip() for l in open(label_names_path)]

    ont = "MF"
    if "bp-" in targets_name:
        ont = "BP"
    elif "cc-" in targets_name:
        ont = "CC"

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
    train_y_fuzzy = y_by_name["fuzzy"]["train"]
    test_y_fuzzy = y_by_name["fuzzy"]["test"]

    assert all([go in go_ia_dict for go in labels])
    weights = get_ia_vector(labels, go_ia_dict)

    existing_fuzzy_combs = glob(f"{test_dir}/fuzzy_test=*/")
    existing_fuzzy_combs = [
        comb.split("fuzzy_test=")[-1].split("/")[0].split("_")
        for comb in existing_fuzzy_combs
    ]
    existing_fuzzy_combs = {
        (
            float(comb[0]),
            float(comb[1]),
            float(comb[2]),
            bool(comb[3]),
            float(comb[4]),
            float(comb[5]),
        )
        for comb in existing_fuzzy_combs
    }
    if len(existing_fuzzy_combs) < n_combinations:
        print(f"Already existing combinations: {len(existing_fuzzy_combs)}")
        n_new_combs = n_combinations - len(existing_fuzzy_combs)
        print(f"Generating {n_new_combs} new combinations")
        new_combs = generate_fuzzy_metaparameters(n_new_combs)
        new_combs = {
            (
                float(comb[0]),
                float(comb[1]),
                float(comb[2]),
                bool(comb[3]),
                float(comb[4]),
                float(comb[5]),
            )
            for comb in new_combs
        }
        existing_fuzzy_combs = list(existing_fuzzy_combs) + list(new_combs)
    else:
        print(
            f"All {n_combinations} combinations already exist. No new combinations to generate."
        )
        existing_fuzzy_combs = list(existing_fuzzy_combs)

    models_trained = []

    targets_progress_bar = tqdm(
        existing_fuzzy_combs,
        total=len(existing_fuzzy_combs),
        desc="Training fuzzy models",
    )

    for fuzzy_comb in targets_progress_bar:
        (
            phylo_true_val,
            cond_false_val,
            derivate_false_val,
            add_random_falses,
            random_falses_min_perc,
            random_false_val,
        ) = fuzzy_comb

        print(f"\n\nAttempting combination: {fuzzy_comb}\n\n")

        fuzzy_test_basename = "fuzzy_test=" + "_".join(str(x) for x in fuzzy_comb)
        fuzzy_test_path = os.path.join(test_dir, fuzzy_test_basename)
        os.makedirs(fuzzy_test_path, exist_ok=True)

        test_preds_path = os.path.join(fuzzy_test_path, "fuzzy_raw.npy")
        if os.path.exists(test_preds_path):
            print("Fuzzy raw predictions already exist. Loading them...")
            y_pred = np.load(test_preds_path)
            success = True
        elif run_new:
            print("Fuzzy raw predictions do not exist. Training fuzzy model...")
            # print("\nShowing original train_y_density\n")
            # show_y_density(train_y_fuzzy)
            changed_train_y = update_y_data_with_new_values(
                train_y_fuzzy, phylo_true_val, cond_false_val, derivate_false_val
            )
            changed_test_y = update_y_data_with_new_values(
                test_y_fuzzy, phylo_true_val, cond_false_val, derivate_false_val
            )
            y_data = {
                "train": changed_train_y,
                "test": changed_test_y,
                "is_classic": False,
            }
            # print("\nShowing changed train_y_density\n")
            # show_y_density(changed_train_y)
            y_pred, _, success = train_and_save_preds(
                train_x,
                test_x,
                "fuzzy_raw",
                y_data,
                add_random_falses,
                fuzzy_test_path,
                param_comb,
                neg_min_perc=random_falses_min_perc,
                zero_val=random_false_val,
            )
            if success:
                train_y_path = os.path.join(fuzzy_test_path, "train_y.npy")
                np.save(train_y_path, changed_train_y)
                test_y_path = os.path.join(fuzzy_test_path, "test_y.npy")
                np.save(test_y_path, changed_test_y)
            else:
                print("Failed to train fuzzy model. Skipping...")
                continue
        else:
            success = False
        if success:
            stats_raw = run_statistics(y_pred, y_eval_cafa, y_eval_owa, weights)
            print("Raw stats:", stats_raw)
            y_pred_norm = calc_normalized_y_pred(
                y_pred, labels, parents_dict, children_dict, go_sortings[ont]
            )
            stats_norm = run_statistics(y_pred_norm, y_eval_cafa, y_eval_owa, weights)
            stats_norm["Sort Score"] = get_sorting_score(stats_norm)
            print("Normalized stats:", stats_norm)

            current_best = (
                models_trained[0]["comb"]
                if len(models_trained) > 0
                else list(fuzzy_comb)
            )

            models_trained.append(
                {
                    "comb": list(fuzzy_comb),
                    # "y_pred": y_pred,
                    "more_negatives": add_random_falses,
                    "success": success,
                    "stats": stats_norm,
                    "stats_raw": stats_raw,
                }
            )

            models_trained.sort(key=lambda x: x["stats"]["Sort Score"], reverse=True)

            new_best = models_trained[0]["comb"]
            save_stats_json_safe(statistics_path, models_trained, ont)

            if current_best != new_best:
                print("\n NEW BEST COMBINATION: ", new_best, "\n")

        targets_progress_bar.update(1)
    save_stats_json_safe(statistics_path, models_trained, ont)
