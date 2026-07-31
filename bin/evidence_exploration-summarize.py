import sys
import json
import os
from glob import glob

import polars as pl
import pandas as pd

from pddb_lib.gene_ontology import (ONTOLOGIES_SHORT, CWA_DATASET_NAME, OWA_DATASET_NAME, EVIDENCE_REP_STRATEGIES, 
    calc_normalized_y_pred, create_ontology_dictionaries_full)

'''
python bin\evidence_exploration-
'''
if __name__ == "__main__":
    go_ia_path = "input_data/go_ia.tsv"

    n_targets = int(sys.argv[1])
    min_annotations = int(sys.argv[2])
    max_train_proteins = int(sys.argv[3]) #downsampling after loading
    test_dir = sys.argv[4]

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

    test_ids = [line.strip() for line in open(test_path)]
    train_ids = [line.strip() for line in open(train_path)]

    result_lines = []
    for alg_dir in glob(f"{test_dir}/*_test"):
        alg_name = os.path.basename(alg_dir).replace('_test', '').replace('_', ' ').title().replace('Rns', '+ RNS')
        results_tsv = f"{alg_dir}/all_results.tsv"
        if os.path.exists(results_tsv):
            alg_df = pd.read_csv(results_tsv, sep="\t")
            for _, row in alg_df.iterrows():
                for ont in ['MF', 'CC', "BP"]:
                    if f"{ont} - exception" in row.keys():
                        print('Exception in training:', row['parameters'], row[f"{ont} - exception"])
                    else:
                        new_row = {
                            "Algorithm": alg_name,
                            "Ontology": ont,
                        }
                        for key, value in row.items():
                            if f"{ont} - " in key:
                                stat_key = key.replace(f"{ont} - ", "")
                                new_row[stat_key] = value
                            
                        new_row['Parameters'] = row['parameters']
                        result_lines.append(new_row)
        else:
            print("Results not found for", alg_name)
    
    results_df = pd.DataFrame(result_lines)
    results_df.to_csv(f"{test_dir}/all_results.tsv", sep="\t", index=False)

    best_param_lines = []
    for group_tp, group_df in results_df.groupby(['Algorithm', 'Ontology']):
        alg, ont = group_tp

        group_df.sort_values(by=['Sort Score'], inplace=True)
        #Get first line as dict:
        best_line = group_df.iloc[0].to_dict()
        best_param_lines.append(best_line)
    
    best_param_lines_df = pd.DataFrame(best_param_lines)
    best_param_lines_df.sort_values(by=['Sort Score'], ascending=False, inplace=True)
    new_cols_order = ["Algorithm", "Ontology", "Sort Score", "OWA Weighted Fmax (micro)", "OWA Weighted MCC (micro)", "OWA Weighted AUPRC", "CAFA Weighted Fmax", "CAFA AUPRC", "Parameters"]
    best_param_lines_df = best_param_lines_df[new_cols_order]
    best_param_lines_df.to_csv(f"{test_dir}/best_results.tsv", sep="\t", index=False)
    
        
    

        
    