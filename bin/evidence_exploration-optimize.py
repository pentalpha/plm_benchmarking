import sys
import random
from glob import glob
import os
import json

from tqdm import tqdm
import numpy as np
import polars as pl
import pandas as pd

from pddb_lib.gene_ontology import (ONTOLOGIES_SHORT, CWA_DATASET_NAME, OWA_DATASET_NAME, EVIDENCE_REP_STRATEGIES, 
    calc_normalized_y_pred, create_ontology_dictionaries_full)
from pddb_lib.parsing import load_data_optimized, smart_str_parsing
from pddb_lib.sample_metaparameters import generate_for_genelist, GENE_NAMES
from pddb_lib.sample_metaparameters import update_y_data_with_new_values2 as update_y_data_with_new_values
from pddb_lib.training import add_random_false_values
from pddb_lib.training import train_and_pred_failsafe as train_and_pred
from pddb_lib.custom_statistics import run_statistics, get_sorting_score

N_COMBINATIONS = 50

'''
python bin\evidence_exploration-optimize.py N_TARGETS MIN_ANNOTATIONS MAX_TRAIN_PROTEINS \
    Y_DATASET_NAME USE_RANDOM_NEGATIVE_SAMPLING TEST_DIR <[<embedding_path>:col1,col2,...],...> 
'''
if __name__ == "__main__":
    go_ia_path = "input_data/go_ia.tsv"

    n_targets = int(sys.argv[1])
    min_annotations = int(sys.argv[2])
    max_train_proteins = int(sys.argv[3]) #downsampling after loading
    y_dataset_name = sys.argv[4] #Evidence rep. strategy
    use_random_negative_sampling = sys.argv[5].lower() == "true"
    use_soft_labeling = "soft" in y_dataset_name
    test_dir = sys.argv[6]
    feature_descs = sys.argv[7:]

    parents_dict, children_dict, go_sortings = create_ontology_dictionaries_full("input_data/go-basic.obo")

    go_ia_dict = {}
    for rawline in open(go_ia_path, "r"):
        goid, ia = rawline.strip().split("\t")
        go_ia_dict[goid] = float(ia)

    alg_name = y_dataset_name
    if use_random_negative_sampling:
        alg_name += "+rns"

    sampling_prefix = f"outputs/n_ont_target={n_targets}-min_proteins={min_annotations}"
    test_path = f"{sampling_prefix}.test_set.txt"
    train_path = f"{sampling_prefix}.train_set.txt"

    datasets_by_ont = {
        ont: {
            "train_y_path": f"{sampling_prefix}-evi_exp-{ont}-train_y.parquet",
            "test_y_path": f"{sampling_prefix}-evi_exp-{ont}-test_y.parquet",
        }
        for ont in ONTOLOGIES_SHORT.keys()
    }
    for ont, datasets_dict in datasets_by_ont.items():
        targets_path = datasets_dict["train_y_path"] + ".targets.txt"
        targets = [line.strip() for line in open(targets_path)]
        datasets_dict["targets"] = targets

    y_to_load = [f"y_{CWA_DATASET_NAME}", f"y_{OWA_DATASET_NAME}", f"y_{y_dataset_name}"]

    test_ids = [line.strip() for line in open(test_path)]
    train_ids = [line.strip() for line in open(train_path)]

    if len(train_ids) > max_train_proteins:
        print(f"Sampling to {max_train_proteins} proteins...")
        sampling_path = train_path.replace(".txt", f".sampled_{max_train_proteins}.txt")
        if not os.path.exists(sampling_path):
            #seet fixed seed and sample
            random.seed(42)
            train_ids = random.sample(train_ids, max_train_proteins)
            #train_ids = sample_train_proteins_by_inf(train_ids, y, targets, go_ia_dict, max_train_proteins)

            with open(sampling_path, "w") as f:
                for gene_id in train_ids:
                    f.write(f"{gene_id}\n")
        train_ids = [line.strip() for line in open(sampling_path)]
    
    for ont, datasets in datasets_by_ont.items():
        print("Loading", ont)
        train_df = load_data_optimized(datasets["train_y_path"], feature_descs, train_ids, y_to_load)
        test_df = load_data_optimized(datasets["test_y_path"], feature_descs, test_ids, y_to_load)
        datasets["train_df"] = train_df
        datasets["test_df"] = test_df

    #Generate parameters options
    
    genenames = GENE_NAMES[alg_name]
    param_options = generate_for_genelist(N_COMBINATIONS, genenames, try_more=True) #TODO: find already existing combinations and only make more when needed
    
    existing_param_combs = glob(f"{test_dir}/parameters=*/")
    existing_param_combs = [
        comb.split("parameters=")[-1].split("/")[0].split("_")
        for comb in existing_param_combs
    ]
    existing_param_combs = {tuple([smart_str_parsing(x) for x in comb])
        for comb in existing_param_combs
    }
    if len(existing_param_combs) < N_COMBINATIONS:
        print(f"Already existing combinations: {len(existing_param_combs)}")
        n_new_combs = N_COMBINATIONS - len(existing_param_combs)
        print(f"Generating {n_new_combs} new combinations")
        new_combs = generate_for_genelist(n_new_combs, genenames, try_more=False)
        new_combs = {tuple([smart_str_parsing(x) for x in comb])
            for comb in new_combs}
        existing_param_combs = list(existing_param_combs) + list(new_combs)
    else:
        print(
            f"All {N_COMBINATIONS} combinations already exist. No new combinations to generate."
        )
        existing_param_combs = list(existing_param_combs)
    
    #Train over ones not used yet
    models_trained = []

    targets_progress_bar = tqdm(
        existing_param_combs,
        total=len(existing_param_combs),
        desc="Training models",
    )

    for param_comb in targets_progress_bar:
        params_dict = dict(zip(genenames, param_comb))

        print(f"\n\nAttempting combination: {params_dict}\n\n")

        test_basename = "parameters=" + "_".join(str(x) for x in param_comb)
        test_path = os.path.join(test_dir, test_basename)
        os.makedirs(test_path, exist_ok=True)

        test_preds_path = os.path.join(test_path, "y_pred.parquet")
        if os.path.exists(test_preds_path):
            print("Raw predictions already exist. Loading them...")
            y_preds = pl.read_parquet(test_preds_path)
            success = True
        else:
            y_preds = {}
            for ont, dataset_dict in datasets_by_ont.items():
                print(f"training on {ont} ontology")
                train_df = dataset_dict["train_df"]
                train_x = train_df["X"].to_numpy()
                train_y = train_df[y_dataset_name].to_numpy()
                
                test_df = dataset_dict["test_df"]
                test_x = test_df["X"].to_numpy()
                test_y = test_df[y_dataset_name].to_numpy()

                if use_soft_labeling:
                    train_y = update_y_data_with_new_values(
                        train_y, params_dict
                    )
                    test_y = update_y_data_with_new_values(
                        test_y, params_dict
                    )
                
                if use_random_negative_sampling:
                    zero_val = 0.0
                    if "Random False Val" in params_dict:
                        zero_val = params_dict["Random False Val"]
                    train_y, already_had_not_perc = add_random_false_values(train_y, 
                        target_min_zeros = params_dict["Random Falses Min Perc"], zero_val=zero_val)
                else:
                    already_had_not_perc = None
                y_pred = train_and_pred(train_x, train_y, test_x, test_y, 
                    params_dict)
                y_preds[ont] = y_pred
            y_preds = pl.DataFrame(y_preds)
            y_preds.write_parquet(test_preds_path)
            success = True

        eval_metrics = {}

        for ont, datasets_dict in datasets_by_ont.items():
            y_pred = y_preds[ont].to_numpy()
            y_test_cwa = datasets_dict['test_df'][CWA_DATASET_NAME].to_numpy()
            y_test_owa = datasets_dict['test_df'][OWA_DATASET_NAME].to_numpy()
            labels = datasets_dict["targets"]
            weights = np.array([go_ia_dict.get(t, 0) for t in labels])

            y_pred_norm = calc_normalized_y_pred(
                y_pred, labels, parents_dict, children_dict, go_sortings[ont]
            )
            stats_norm = run_statistics(y_pred_norm, y_test_cwa, y_test_owa, weights)
            stats_norm["Sort Score"] = get_sorting_score(stats_norm)
            print("Normalized stats:", stats_norm)

            for metric_name, metric_val in stats_norm.items():
                eval_metrics[f"{ont.upper()} - {metric_name}"] = metric_val

        eval_metrics["parameters"] = json.dumps(params_dict, ensure_ascii=False)

        models_trained.append(eval_metrics)

    #Store results
    all_results_df = pd.DataFrame(models_trained)
    all_results_df.to_csv(f"{test_dir}/all_results.tsv", sep="\t", index=False)
    print("\n\nResults stored in", f"{test_dir}/all_results.tsv")