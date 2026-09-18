import sys
import random
import subprocess
from glob import glob
import os
import json
from datetime import datetime

from tqdm import tqdm
import numpy as np
import polars as pl
import pandas as pd

from pddb_lib.gene_ontology import (ONTOLOGIES_SHORT, CWA_DATASET_NAME, OWA_DATASET_NAME, EVIDENCE_REP_STRATEGIES, 
    calc_normalized_y_pred, create_ontology_dictionaries_full)
from pddb_lib.parsing import load_data_optimized, smart_str_parsing
from pddb_lib.sample_metaparameters import generate_for_genelist, GENE_NAMES
from pddb_lib.sample_metaparameters import update_y_data_with_new_values2 as update_y_data_with_new_values
from pddb_lib.manipulate_y import add_random_false_values, show_y_density
from pddb_lib.training import train_and_pred, reduce_train_negatives_to
from pddb_lib.optimization_utils import go_ia_path, run_eval

#Script to perform manual optimization of a certain model for a certain ontology

ONT="cc"
N_COMBINATIONS=40
N_TARGETS=32
MIN_ANNOTATIONS=120
MAX_TRAIN_PROTEINS=40000
BASE_RESULTS_DIR=f"outputs/remote_results/plm_benchmarking_{ONT}"
EMBS_DIR="/home/pita/fs/data/dimension_db/release_2"

metaparameters = {
    "max_depth": 9, 
    "min_data_in_leaf": 1, 
    "min_gain_to_split": 0.1, 
    "max_bin": 128, 
    "lr": 0.05, 
    "lambda_l2": 25.0, 
    "gd_steps": 1, 
    "colsample": 0.05, 
    "subsample": 0.75, 
    "ntrees": 5000, 
    "es": 300
}

def train_and_pred_on_proc(ont,test_ids, train_x, train_y, test_x, test_y, 
        metaparameters, model_dir, test_preds_path):
    
    test_ids_path = os.path.join(model_dir, f"{ont}-test_ids.txt")
    train_x_path = os.path.join(model_dir, f"{ont}-train_x.npy")
    train_y_path = os.path.join(model_dir, f"{ont}-train_y.npy")
    test_x_path = os.path.join(model_dir, f"{ont}-test_x.npy")
    test_y_path = os.path.join(model_dir, f"{ont}-test_y.npy")
    y_pred_path = os.path.join(model_dir, f"{ont}-y_pred.npy")
    log_path = os.path.join(model_dir, f"{ont}-train.log")
    params_path = os.path.join(model_dir, "params.json")

    files_to_save = [test_ids_path, train_x_path, train_y_path, test_x_path, test_y_path, y_pred_path]

    open(test_ids_path, 'w').write('\n'.join(test_ids))
    np.save(train_x_path, train_x)
    np.save(train_y_path, train_y)
    np.save(test_x_path, test_x)
    np.save(test_y_path, test_y)
    json.dump(metaparameters, open(params_path, 'w'))

    #Run train command
    cmd = ["python", "-u", "bin/plm_benchmarking-train.py", model_dir, ont, ">", f"{model_dir}/{ont}-train.log"]    
    
    print("Launching cmd:", ' '.join(cmd))
    
    with open(log_path, "w") as log_file:
        # subprocess.run waits for the command to complete by default
        try:
            subprocess.run(
                cmd, 
                stdout=log_file, 
                stderr=subprocess.STDOUT, # Merges stderr into the stdout stream
                check=True # Raises an exception immediately if the script fails
            )
            y_pred = np.load(y_pred_path)
            y_preds_ont = pl.DataFrame({"id": test_ids, ont: y_pred})
            y_preds_ont.write_parquet(test_preds_path)
        except Exception as e:
            print(f"Error training and predicting on {ont}. y_pred not created or invalid: {e}")
            
            for p in files_to_save:
                if os.path.exists(p):
                    os.remove(p)
            raise e
    
    for p in files_to_save:
        if os.path.exists(p):
            os.remove(p)

if __name__ == "__main__":
    
    n_targets = N_TARGETS
    min_annotations = MIN_ANNOTATIONS
    max_train_proteins = MAX_TRAIN_PROTEINS #downsampling after loading
    test_dir = BASE_RESULTS_DIR
    embs_parquet_prefix = EMBS_DIR+f"/emb.ankh3_large_nlu"

    benchmarking_dir = os.path.dirname(test_dir)

    outputs_dir = "outputs"
    params_path = "outputs/evi_exploration_results.json"
    y_dataset_name = "soft" #Evidence rep. strategy
    params_dicts = json.load(open(params_path))[y_dataset_name]
    use_random_negative_sampling = False
    use_soft_labeling = True
    feature_cols = ['parti']
    feature_descs = [f"{embs_parquet_prefix}_parti.parquet:parti"]

    print(f"n_targets={n_targets}")
    print(f"min_annotations={min_annotations}")
    print(f"max_train_proteins={max_train_proteins}")
    print(f"y_dataset_name={y_dataset_name}")
    print(f"use_random_negative_sampling={use_random_negative_sampling}")
    print(f"use_soft_labeling={use_soft_labeling}")
    print(f"outputs_dir={outputs_dir}")

    uses_nan = not 'classic' in y_dataset_name
    
    parents_dict, children_dict, go_sortings = create_ontology_dictionaries_full("input_data/go-basic.obo")

    go_ia_dict = {}
    for rawline in open(go_ia_path, "r"):
        goid, ia = rawline.strip().split("\t")
        go_ia_dict[goid] = float(ia)

    alg_name = y_dataset_name
    if use_random_negative_sampling:
        alg_name += "+rns"

    sampling_prefix = f"{outputs_dir}/n_ont_target={n_targets}-min_proteins={min_annotations}"
    test_path = f"{sampling_prefix}.test_set.txt"
    train_path = f"{sampling_prefix}.train_set.txt"
    train_y_path = f"{sampling_prefix}-evi_exp-{ONT}-train_y.parquet"
    targets_path = train_y_path + ".targets.txt"
    targets = [line.strip() for line in open(targets_path)]
    datasets = {
        "train_y_path": train_y_path,
        "test_y_path": f"{sampling_prefix}-evi_exp-{ONT}-test_y.parquet",
        "targets": targets
    }

    y_to_load = [f"y_{CWA_DATASET_NAME}", f"y_{OWA_DATASET_NAME}"]
    if not f"y_{y_dataset_name}" in y_to_load:
        y_to_load.append(f"y_{y_dataset_name}")

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
        train_path = sampling_path
        
    os.makedirs(test_dir, exist_ok=True)
    assert os.path.exists(test_dir)

    train_x_np_path = f"{test_dir}/train_x_max={max_train_proteins}.npy"
    train_y_np_path = f"{test_dir}/train_y_max={max_train_proteins}.npy"
    test_x_np_path = f"{test_dir}/test_x_max={max_train_proteins}.npy"
    test_y_np_path = f"{test_dir}/test_y_max={max_train_proteins}.npy"

    input_paths = [train_x_np_path, train_y_np_path, test_x_np_path, test_y_np_path]
    
    if not all([os.path.exists[p] for p in input_paths]):
        print("Loading", ONT)
        cols = y_to_load
        train_df = load_data_optimized(datasets["train_y_path"], feature_descs, 
            train_ids, cols)
        test_df = load_data_optimized(datasets["test_y_path"], feature_descs, 
            test_ids, cols)
        datasets["train_df"] = train_df
        datasets["test_df"] = test_df
        dataset_dict = datasets
        targets = dataset_dict["targets"]
        params_dict = params_dicts[ONT.upper()]
        print(f"training on {ONT} ontology")
        train_df = dataset_dict["train_df"]
        y_name = "y_"+y_dataset_name
        train_x = train_df["X"].to_numpy()
        train_y = train_df[y_name].to_numpy()
        
        test_df = dataset_dict["test_df"]
        test_x = test_df["X"].to_numpy()
        test_y = test_df[y_name].to_numpy()
        test_ids = test_df["id"].to_list()

        train_y = update_y_data_with_new_values(
            train_y, params_dict
        )
        test_y = update_y_data_with_new_values(
            test_y, params_dict
        )
        
        train_x = np.ascontiguousarray(train_x, dtype=np.float32).copy()
        train_y = np.ascontiguousarray(train_y, dtype=np.float32).copy()
        test_x = np.ascontiguousarray(test_x, dtype=np.float32).copy()
        test_y = np.ascontiguousarray(test_y, dtype=np.float32).copy()

        #Make sure seed is always 1337
        np.random.seed(1337)
        random.seed(1337)

        train_x, train_y = reduce_train_negatives_to(train_x, train_y, target_ratio=0.15, 
                                                use_nan=uses_nan)
    else:
        train_x = np.load(train_x_np_path)
        train_y = np.load(train_y_np_path)
        test_x = np.load(test_x_np_path)
        test_y = np.load(test_y_np_path)
    
    test_uniq_id = f"{datetime.now().strftime('%m_%d_%H_%M_%S')}" 
    model_dir = os.path.join(test_dir, f"model_{test_uniq_id}")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    test_preds_basepath = os.path.join(model_dir, "y_pred")
    test_preds_path = test_preds_basepath + "." + ONT + ".parquet"

    y_preds = {}
    eval_metrics = {}
        
    try:
        train_and_pred_on_proc(ONT, test_ids, train_x, train_y, test_x, test_y, 
            metaparameters, model_dir, test_preds_path)
        y_preds_ont = pl.read_parquet(test_preds_path)
    except Exception as e:
        print(f"Error training on {ont}: {e}")
        y_preds_ont = None
    if y_preds_ont is not None:
        datasets_dict = {ONT: datasets_by_ont[ont]}
        new_eval_metrics = run_eval(datasets_dict, y_preds_ont, metaparameters, go_ia_dict, 
                parents_dict, children_dict, go_sortings)
        print(new_eval_metrics)
        for key, val in new_eval_metrics.items():
            eval_metrics[key] = val
    else:
        print(f"Training failed. Skipping evaluation for {ont}.")
    print(eval_metrics)
    if len(eval_metrics) > 0:
        with open(f"{model_dir}/results_eval.json", "w") as f:
            json.dump(eval_metrics, f, indent=4)
        models_trained.append(eval_metrics)
    
    #Store results
    all_results_df = pd.DataFrame(models_trained)
    all_results_df.to_csv(f"{test_dir}/optimized_results_{n_combinations}.tsv", sep="\t", index=False)
    print("\n\nResults stored in", f"{test_dir}/optimized_results_{n_combinations}.tsv") 