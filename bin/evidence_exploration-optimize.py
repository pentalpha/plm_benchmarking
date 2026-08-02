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
from pddb_lib.manipulate_y import add_random_false_values, show_y_density
from pddb_lib.training import train_and_pred, reduce_train_negatives_to
from pddb_lib.optimization_utils import go_ia_path, run_eval

'''
python bin\evidence_exploration-optimize.py N_TARGETS MIN_ANNOTATIONS MAX_TRAIN_PROTEINS Y_DATASET_NAME USE_RANDOM_NEGATIVE_SAMPLING TEST_DIR <[<embedding_path>:col1,col2,...],...> 
'''

if __name__ == "__main__":

    n_targets = int(sys.argv[1])
    min_annotations = int(sys.argv[2])
    max_train_proteins = int(sys.argv[3]) #downsampling after loading
    y_dataset_name = sys.argv[4] #Evidence rep. strategy
    use_random_negative_sampling = sys.argv[5].lower() == "true"
    use_soft_labeling = "soft" in y_dataset_name
    test_dir = sys.argv[6]
    n_combinations = int(sys.argv[7])
    outputs_dir = 'outputs/'
    feature_descs = sys.argv[8:]

    print(f"n_targets={n_targets}")
    print(f"min_annotations={min_annotations}")
    print(f"max_train_proteins={max_train_proteins}")
    print(f"y_dataset_name={y_dataset_name}")
    print(f"use_random_negative_sampling={use_random_negative_sampling}")
    print(f"use_soft_labeling={use_soft_labeling}")
    print(f"test_dir={test_dir}")
    print(f"outputs_dir={outputs_dir}")
    print(f"feature_descs={feature_descs}")

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
    
    

    #Generate parameters options
    
    genenames = GENE_NAMES[alg_name]
    param_options = generate_for_genelist(n_combinations, genenames, try_more=True) #TODO: find already existing combinations and only make more when needed
    
    existing_param_combs = glob(f"{test_dir}/parameters=*/")
    existing_param_combs = [
        comb.split("parameters=")[-1].split("/")[0].split("_")
        for comb in existing_param_combs
    ]
    existing_param_combs = {tuple([smart_str_parsing(x) for x in comb])
        for comb in existing_param_combs
    }
    if len(existing_param_combs) < n_combinations:
        print(f"Already existing combinations: {len(existing_param_combs)}")
        n_new_combs = n_combinations - len(existing_param_combs)
        print(f"Generating {n_new_combs} new combinations")
        new_combs = generate_for_genelist(n_new_combs, genenames, try_more=False)
        new_combs = {tuple([smart_str_parsing(x) for x in comb])
            for comb in new_combs}
        existing_param_combs = list(existing_param_combs) + list(new_combs)
    else:
        print(
            f"All {n_combinations} combinations already exist. No new combinations to generate."
        )
        existing_param_combs = list(existing_param_combs)
    
    print("Parameters: ", genenames)
    for i, comb in enumerate(existing_param_combs):
        print(i, f"{len(comb)} values:", comb)

    for ont, datasets in datasets_by_ont.items():
        print("Loading", ont)
        train_df = load_data_optimized(datasets["train_y_path"], feature_descs, train_ids, y_to_load)
        test_df = load_data_optimized(datasets["test_y_path"], feature_descs, test_ids, y_to_load)
        datasets["train_df"] = train_df
        datasets["test_df"] = test_df

    #Train over ones not used yet
    models_trained = []

    targets_progress_bar = tqdm(
        existing_param_combs,
        total=len(existing_param_combs),
        desc="Training models",
    )

    os.makedirs(test_dir, exist_ok=True)

    for param_comb in targets_progress_bar:
        params_dict = dict(zip(genenames, param_comb))

        print(f"\n\nAttempting combination: {params_dict}\n\n")

        test_basename = "parameters=" + "_".join(str(x) for x in param_comb)
        test_path = os.path.join(test_dir, test_basename)
        os.makedirs(test_path, exist_ok=True)

        assert os.path.exists(test_path)

        test_preds_path = os.path.join(test_path, "y_pred.parquet")
        if os.path.exists(test_preds_path):
            print("Raw predictions already exist. Loading them...")
            try:
                y_preds = pl.read_parquet(test_preds_path)
                
            except Exception as e:
                print(f"Error loading predictions: {e}")
                success = False
                # Remove bad file
                os.remove(test_preds_path)
                y_preds = None

            if y_preds is not None:
                eval_metrics = run_eval(datasets_by_ont, y_preds, params_dict, go_ia_dict, 
                    parents_dict, children_dict, go_sortings)
                if eval_metrics is not None:
                    models_trained.append(eval_metrics)
                    success = True
                else:
                    os.remove(test_preds_path)
                    success = False
        else:
            y_preds = {}
            for ont, dataset_dict in datasets_by_ont.items():
                print(f"training on {ont} ontology")
                train_df = dataset_dict["train_df"]
                train_x = train_df["X"].to_numpy()
                train_y = train_df["y_"+y_dataset_name].to_numpy()
                
                test_df = dataset_dict["test_df"]
                test_x = test_df["X"].to_numpy()
                test_y = test_df["y_"+y_dataset_name].to_numpy()
                test_ids = test_df["id"].to_list()

                if use_soft_labeling:
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
                
                if use_random_negative_sampling:
                    zero_val = 0.0
                    if "Random False Val" in params_dict:
                        zero_val = params_dict["Random False Val"]
                    minperc = params_dict["Random Falses Min Perc"]*100
                    print(f"Adding {minperc}% RNS with value {zero_val}")
                    show_y_density(train_y)
                    train_y, added_nots = add_random_false_values(train_y, 
                        target_min_zeros = params_dict["Random Falses Min Perc"], zero_val=zero_val)
                    already_had_not_perc = not added_nots
                    print(f"Already had {minperc}% nots? {already_had_not_perc}")
                    show_y_density(train_y)

                else:
                    already_had_not_perc = None
                y_pred = train_and_pred(train_x, train_y, test_x, test_y, 
                    params_dict, uses_nan)
                y_preds["id"] = test_ids
                y_preds[ont] = y_pred
                #if already_had_not_perc:
                #    y_preds[ont+" - exception"] = f"Nots already >= {minperc}%"
            y_preds = pl.DataFrame(y_preds)
            y_preds.write_parquet(test_preds_path)
            eval_metrics = run_eval(datasets_by_ont, y_preds, params_dict, go_ia_dict, 
                    parents_dict, children_dict, go_sortings)
            models_trained.append(eval_metrics)
            success = True
        
        if success:
            #Store results
            all_results_df = pd.DataFrame(models_trained)
            all_results_df.to_csv(f"{test_dir}/all_results.tsv", sep="\t", index=False)
            print("\n\nResults stored in", f"{test_dir}/all_results.tsv")   