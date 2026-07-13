import sys
import glob
import os
import json
from cupy import full
import polars as pl
import numpy as np
from tqdm import tqdm

from custom_statistics import (
    get_ia_vector,
    ia_adapted_metric,
    faster_fmax_weighted,
    faster_fmax_weighted_nan,
    calc_normalized_y_pred,
    nan_macro_average_precision,
    mcc_bycol_weighted_masked,
)
from fuzzy_ml import create_ontology_dictionaries_full
from train_eval import run_statistics

output_path = sys.argv[1]
recalc_metrics = True
result_dirs = sys.argv[2:]

print(sys.argv[0], output_path, recalc_metrics, sys.argv[3], "...")

eval_metrics = {
    "fmax_mean_cafa": "CAFA Fmax Macro",
    "fmax_mean_conditional": "Conditional Fmax Macro",
    "fmax_bottom20percent_cafa": "CAFA Fmax Macro (least frequent 20%)",
    "fmax_bottom20percent_conditional": "Conditional Fmax Macro (least frequent 20%)",
    "auprc_score_cafa": "AUPRC",
    "auprc_score_conditional": "Conditional AUPRC",
}

"""
"OWA Weighted Fmax"
"OWA Weighted Fmax (micro)"
"OWA Weighted MCC"
"OWA Weighted MCC (micro)"
"OWA Weighted AUPRC"
"OWA Weighted Fmax (lowest 20%)"
"Fmax (lowest 20%)"
"CAFA Weighted Fmax"
"CAFA Weighted Fmax (lowest 20%)"
"CAFA Fmax Macro"
"CAFA AUPRC"
"""
metric_weights_for_sorting = {
    "OWA Weighted Fmax (micro)": 2,
    "OWA Weighted MCC (micro)": 1,
    "OWA Weighted AUPRC": 1,
    "CAFA Weighted Fmax": 2.5,
    "CAFA AUPRC": 1.5,
}

url_ou_caminho_obo = "input_data/go-basic.obo"
parents_dict, children_dict, go_sortings = create_ontology_dictionaries_full(
    url_ou_caminho_obo
)
go_ia_path = "input_data/go_ia.tsv"
go_ia_dict = {
    l.strip().split("\t")[0]: float(l.strip().split("\t")[1]) for l in open(go_ia_path)
}

labels_by_ont = {}
test_y_by_ont = {}


def tuples_str_to_dict(tuples_str: str) -> dict:
    # Remove parentheses and split by comma
    tuples_list = tuples_str[1:-1].split("), (")
    # Convert each tuple to key-value pair
    result = {}
    for t in tuples_list:
        # remove first ( and last )
        t = t.rstrip(")").lstrip("(")
        key, val = t.split(", ")
        key = key.strip("'")
        val = val.strip("'")
        try:
            # val is str representing float or int
            if "." in val:
                val = float(val)
            else:
                val = int(val)
        except:
            # attempt to convert "True" -> True
            if val.lower() == "true":
                val = True
            elif val.lower() == "false":
                val = False
            else:
                # its str, leave it as is
                pass
        result[key] = val
    return result


def get_sorting_score(results: dict) -> float:
    total_score = 0
    for metric_name, metric_value in results.items():
        if metric_name not in metric_weights_for_sorting:
            continue
        total_score += metric_value * metric_weights_for_sorting[metric_name]
    total_score = total_score / sum(metric_weights_for_sorting.values())
    return total_score


def get_y_tests_for_target_ont(input_tests_dir, ont: str):
    if ont in test_y_by_ont:
        return test_y_by_ont[ont]
    else:
        test_y_by_ont[ont] = {}
        test_y_paths = glob.glob(f"{input_tests_dir}/test_y*npy")
        for test_path in test_y_paths:
            name = os.path.basename(test_path)
            name = name.replace("test_y", "").replace(".npy", "").strip().strip("_")
            test_y_by_ont[ont][name] = np.load(test_path)
        return test_y_by_ont[ont]


def calc_metrics(stats_path: str, model_results: dict, ont: str):
    test_dir = os.path.dirname(stats_path)
    params_set_name = os.path.basename(test_dir)
    predictions_dir = f"{test_dir}/predictions"
    input_tests_dir = os.path.dirname(test_dir)
    processed_inputs_dir = f"{input_tests_dir}/processed_inputs"
    tests_for_target_path = os.path.dirname(input_tests_dir)
    targets_name = os.path.basename(tests_for_target_path)
    targets_parquet = f"outputs/{targets_name}.parquet"

    """print(f"test_dir={test_dir}")
    print(f"params_set_name={params_set_name}")
    print(f"predictions_dir={predictions_dir}")
    print(f"input_tests_dir={input_tests_dir}")
    print(f"processed_inputs_dir={processed_inputs_dir}")
    print(f"tests_for_target_path={tests_for_target_path}")
    print(f"targets_name={targets_name}")
    print(f"targets_parquet={targets_parquet}")"""

    if not ont in labels_by_ont:
        label_names_path = targets_parquet + ".targets.txt"
        labels = [l.strip() for l in open(label_names_path)]
        labels_by_ont[ont] = labels
    go_names = labels_by_ont[ont]
    assert all([go in go_ia_dict for go in go_names])
    weights = get_ia_vector(go_names, go_ia_dict)
    y_tests = get_y_tests_for_target_ont(processed_inputs_dir, ont)

    y_pred_nps = glob.glob(f"{predictions_dir}/*.npy")
    y_pred_nps = [
        p
        for p in y_pred_nps
        if not ".train" in p and p.count("+") <= 1 and not "ont_normalized" in p
    ]

    y_preds = []
    for y_pred_path in y_pred_nps:
        name = os.path.basename(y_pred_path).replace(".npy", "")
        y_preds.append((name, y_pred_path, np.load(y_pred_path)))

    # print("Preds found:", y_preds)

    test_y_classic = y_tests["classic"]
    test_y_owa = y_tests["open_world_assumption"]

    y_tests = {k: v for k, v in y_tests.items() if k != "open_world_assumption"}

    for model_name, y_pred_path, y_pred_no_norm in y_preds:
        normalized_path = y_pred_path.replace(".npy", ".ont_normalized.npy")
        # if not os.path.exists(normalized_path):
        y_pred = calc_normalized_y_pred(
            y_pred_no_norm,
            go_names,
            parents_dict,
            children_dict,
            go_sortings[ont],
            verbose=False,
        )
        # print(model_name, ont, params_set_name)
        if not model_name in model_results:
            model_results[model_name] = {}
        if not params_set_name in model_results[model_name]:
            model_results[model_name][params_set_name] = {
                "MF": None,
                "BP": None,
                "CC": None,
            }

        y_stats_default = run_statistics(y_pred, test_y_classic, test_y_owa, weights)

        stats_pretty = {
            (
                eval_metrics[metric_raw_name]
                if metric_raw_name in eval_metrics
                else metric_raw_name
            ): val
            for metric_raw_name, val in y_stats_default.items()
        }
        stats_pretty["Sort Score"] = get_sorting_score(stats_pretty)

        print(
            model_name,
            stats_pretty["Sort Score"],
            stats_pretty["CAFA Weighted Fmax"],
            stats_pretty["OWA Weighted Fmax"],
            stats_pretty["OWA Weighted MCC"],
            stats_pretty["CAFA AUPRC"],
            stats_pretty["OWA Weighted AUPRC"],
        )
        # print("CAFA Fmax W", cafa_fmax)
        model_results[model_name][params_set_name][ont] = stats_pretty


def save_model_results_csv(model_results, output_path):
    short_col_list = [
        "Model Name",
        "Parameters Set ID",
        "Mean Sort Score",
        "MF - Sort Score",
        "BP - Sort Score",
        "CC - Sort Score",
        "MF - CAFA Weighted Fmax",
        "MF - CAFA Weighted Fmax (Conditional)",
        "MF - Fmax (lowest 20%)",
        "MF - AUPRC",
        "BP - CAFA Weighted Fmax",
        "BP - CAFA Weighted Fmax (Conditional)",
        "BP - Fmax (lowest 20%)",
        "BP - AUPRC",
        "CC - CAFA Weighted Fmax",
        "CC - CAFA Weighted Fmax (Conditional)",
        "CC - Fmax (lowest 20%)",
        "CC - AUPRC",
        "Params",
    ]

    for model_name, param_combs in model_results.items():
        for param_comb_key, ont_stats in param_combs.items():
            has_ont = [ont for ont in ont_stats if ont_stats[ont] is not None]
            # print(model_name, "with", param_comb_key, "has", has_ont)

    model_best_params = []

    for model_name, param_combs in model_results.items():
        combs_and_results = []
        for param_comb_uniqkey, ont_stats in param_combs.items():
            new_item = {
                "Model Name": model_name,
                "Parameters Set ID": param_comb_uniqkey,
            }
            # Ontologies for which we have results
            tested_onts = []
            for ont in ["MF", "BP", "CC"]:
                if ont_stats[ont] is not None:
                    tested_onts.append(ont)
            key_scores = [ont_stats[ont]["Sort Score"] for ont in tested_onts]

            mean_key_score = sum(key_scores) / 3
            new_item["Mean Sort Score"] = mean_key_score
            for ont in tested_onts:
                for metric_name, val in ont_stats[ont].items():
                    new_item[ont + " - " + metric_name] = val

            combs_and_results.append(new_item)

        combs_and_results.sort(key=lambda x: x["Mean Sort Score"], reverse=True)
        best_comb = combs_and_results[0]
        comb_index = best_comb["Parameters Set ID"]
        comb_path = f"outputs/metaparameters/comb_{comb_index}.json"
        # best_comb["Params"] = json.dumps(tuples_str_to_dict(best_comb["Params"]))
        best_comb["Params"] = json.dumps(json.load(open(comb_path)))

        model_best_params.append(best_comb)

    model_best_params.sort(key=lambda x: x["Mean Sort Score"], reverse=True)

    # for l in model_best_params:
    #    print(l)
    df = pl.DataFrame(model_best_params)
    df.write_csv(output_path, separator="\t")

    short_col_list = [c for c in short_col_list if c in df.columns]

    df_short = df.select(short_col_list)
    df_short.write_csv(output_path.replace(".tsv", ".short.tsv"), separator="\t")


model_results = {}

progress_bar = tqdm(result_dirs, desc="Processing predictions")
new_results_calculated = 0
for dir_path in progress_bar:
    ont = "MF"
    if "bp-" in dir_path:
        ont = "BP"
    elif "cc-" in dir_path:
        ont = "CC"
    param_comb_uniqkey = os.path.basename(dir_path)
    stats_path = os.path.join(dir_path, "statistics.json")
    calc_metrics(stats_path, model_results, ont)
    new_results_calculated += 1

    if new_results_calculated % 10 == 0:
        save_model_results_csv(model_results, output_path)

save_model_results_csv(model_results, output_path)
