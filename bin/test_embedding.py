from glob import glob
import json
import os
import sys
import subprocess

import polars as pl
import numpy as np

from sklearn.model_selection import train_test_split
import numpy as np
import obonet

from fuzzy_ml import show_y_density
from training import eval_param_comb
from custom_statistics import create_split_mask, apply_split_mask

from sample_metaparameters import COMBINATIONS_DIR

BASE_TESTS_DIR = "outputs/emb_tests"
TEST_SIZE = 0.35
RANDOM_STATE = 42


def load_data(targets_path, feature_descs):

    targets_df = pl.read_parquet(targets_path)
    target_ids = targets_df["id"].to_list()
    id_to_index = {id: i for i, id in enumerate(target_ids)}
    n_proteins = len(target_ids)

    x_cols = []
    for parquet_path, col_names, embname in feature_descs:
        x = [np.nan for _ in range(n_proteins)]
        emb_df = (
            pl.scan_parquet(parquet_path)
            .filter(pl.col("id").is_in(target_ids))
            .collect()
        )
        for row in emb_df.iter_rows(named=True):
            idx = id_to_index[row["id"]]
            x[idx] = np.concatenate([row[c] for c in col_names])
        x = np.asarray(x)

        x_cols.append(x)
    all_features_x = np.concatenate(x_cols, axis=1)
    df_dict = {
        "id": target_ids,
        "X": all_features_x,
    }
    for col_name in targets_df.columns:
        if "y_" in col_name:
            df_dict[col_name] = targets_df[col_name].to_numpy()
    df = pl.DataFrame(df_dict)

    return df


def run_one_test(
    param_comb_path,
    processed_inputs_dir,
    test_dir,
):
    # Save numpy files for faster loadin
    statistics_path = os.path.join(test_dir, "statistics.json")

    # start external script that will do the training and evaluation
    cmd = [
        "python",
        "bin/train_eval.py",
        param_comb_path,
        processed_inputs_dir,
        statistics_path,
    ]
    # print command output on stdout
    print("Running command: ", cmd)
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        return False, {}

    if not os.path.exists(statistics_path):
        return False, {}

    stats = json.load(open(statistics_path))

    # print(json.dumps(stats, indent=4))

    return True, stats


def count_f1_qualities(fmax_values: list):
    ranges = {
        "very_good": (0.8, 1.0),
        "good": (0.6, 0.8),
        "medium": (0.4, 0.6),
        "poor": (0.2, 0.4),
        "very_poor": (-1.0, 0.2),
    }
    counts = {key: 0 for key in ranges.keys()}
    for val in fmax_values:
        for key, (low, high) in ranges.items():
            if low < val <= high:
                counts[key] += 1
                break
    return counts


if __name__ == "__main__":
    # python test_embedding.py <n_tests> <targets_path> [True | False] <[<embedding_path>:col1,col2,...],...>
    n_tests = int(sys.argv[1])
    targets_path = sys.argv[2]
    remake_tests = "True".lower() == sys.argv[3].lower()

    emb_descriptions = sys.argv[4:]

    feature_descs = []
    for x in emb_descriptions:
        parquet_path, col_names = x.split(":")
        col_names = col_names.split(",")
        emb_basename = os.path.basename(parquet_path).replace(".parquet", "")
        embname = emb_basename + "_" + ",".join(col_names)
        feature_descs.append((parquet_path, col_names, embname))

    full_emb_name = "_".join([f[2] for f in feature_descs])

    targets_basename = os.path.basename(targets_path).replace(".parquet", "")
    TESTS_DIR = os.path.join(BASE_TESTS_DIR, targets_basename, full_emb_name)
    os.makedirs(TESTS_DIR, exist_ok=True)

    meta_param_combinations = glob(os.path.join(COMBINATIONS_DIR, "*.json"))
    meta_param_combinations.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    meta_param_combinations = [json.load(open(x)) for x in meta_param_combinations]

    test_df = load_data(targets_path, feature_descs)
    print(test_df)

    # for now, use only first embedding
    X = test_df["X"].to_numpy()

    print(X.shape)

    train_idx, test_idx = create_split_mask(
        split_size=TEST_SIZE, n_elements=X.shape[0], random_state=RANDOM_STATE
    )
    train_x, test_x = apply_split_mask(X, train_idx, test_idx)
    matrix_with_names = [("x", train_x, test_x)]
    for col in test_df.columns:
        if "y_" in col:
            train_y, test_y = apply_split_mask(
                test_df[col].to_numpy(), train_idx, test_idx
            )
            matrix_with_names.append((col, train_y, test_y))
            print("Density of", col, "in train set")
            show_y_density(train_y)
            print("Density of", col, "in test set")
            show_y_density(test_y)

    processed_inputs_dir = TESTS_DIR + "/processed_inputs/"
    os.makedirs(processed_inputs_dir, exist_ok=True)
    for name, train_m, test_m in matrix_with_names:
        np.save(os.path.join(processed_inputs_dir, f"train_{name}.npy"), train_m)
        np.save(os.path.join(processed_inputs_dir, f"test_{name}.npy"), test_m)

    successful_tests = []
    failed_tests = []

    next_test = 0

    while len(successful_tests) < n_tests:
        progress_text = f"{len(successful_tests)} / {n_tests}"
        print(progress_text, flush=True)
        param_comb = meta_param_combinations[next_test]
        current_test_dir = os.path.join(TESTS_DIR, str(next_test))
        os.makedirs(current_test_dir, exist_ok=True)
        param_comb_save_path = os.path.join(current_test_dir, "param_comb.json")
        statistics_path = os.path.join(current_test_dir, "statistics.json")
        system_error_flag = os.path.join(current_test_dir, "system_error.flag")
        stdout_file = os.path.join(current_test_dir, "stdout.txt")
        stderr_file = os.path.join(current_test_dir, "stderr.txt")
        required_stats = ["Model Results", "success"]

        previously_run = os.path.exists(param_comb_save_path)
        successful_previuosly = False
        if previously_run:
            if os.path.exists(statistics_path):
                stats = json.load(open(statistics_path))
                successful_previuosly = all(key in stats for key in required_stats)
                if successful_previuosly:
                    successful_previuosly &= stats["success"]
                    successful_previuosly &= not os.path.exists(system_error_flag)
            else:
                successful_previuosly = False

        if previously_run and remake_tests and not successful_previuosly:
            run_this = True
        elif successful_previuosly:
            stats = json.load(open(statistics_path))
            successful_tests.append((param_comb, stats))
            next_test += 1
            run_this = False
            print(
                f"{current_test_dir}: previously_run={previously_run}, successful_previuosly={successful_previuosly}, run_this={run_this}"
            )
            continue
        else:
            run_this = True

        print(
            f"{current_test_dir}: previously_run={previously_run}, successful_previuosly={successful_previuosly}, run_this={run_this}"
        )

        with open(param_comb_save_path, "w") as f:
            json.dump(param_comb, f, indent=4)

        try:
            # Temporary redirection of stdout and stderr to files
            """old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = open(stdout_file, "w")
            sys.stderr = open(stderr_file, "w")"""

            success, stats = run_one_test(
                param_comb_save_path,
                processed_inputs_dir,
                current_test_dir,
            )

            """sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr"""

        except Exception as e:
            # Restore stdout and stderr
            """sys.stdout.close()
            sys.stderr.close()
            sys.stdout = old_stdout
            sys.stderr = old_stderr"""
            with open(system_error_flag, "w") as f:
                f.write(str(e))
            print(f"Error in param combination {param_comb}: {e}")
            success = False
            fmax_fuzzy = None
            fmax_cafa = None

        if success:
            stats_dict = json.load(open(statistics_path))
            if stats_dict["success"]:
                successful_tests.append((param_comb, stats_dict))
            else:
                failed_tests.append(param_comb)
        else:
            failed_tests.append(param_comb)
        next_test += 1

    successful_tests.sort(
        key=lambda x: (
            round(
                x[1]["fmax_col_mean_cafa_composite"],
                4,
            ),
            round(
                x[1]["fmax_col_mean_cafa"] + x[1]["fmax_col_mean_cafa_more_negatives"],
                3,
            ),
            round(
                x[1]["fmax_col_mean_nan"] + x[1]["fmax_col_mean_nan_more_negatives"], 3
            ),
        ),
        reverse=True,
    )

    top_params, top_test = successful_tests[0]
    top_test["successful_tests"] = len(successful_tests)
    top_test["failed_tests"] = len(failed_tests)
    top_test["fmax_counts_cafa"] = count_f1_qualities(top_test["fmax_all_values_cafa"])
    top_test["params"] = top_params

    # print(json.dumps(top_test, indent=4))
    with open(os.path.join(TESTS_DIR, "results.json"), "w") as f:
        json.dump(top_test, f, indent=4)
    print("Results saved to ", os.path.join(TESTS_DIR, "results.json"))
