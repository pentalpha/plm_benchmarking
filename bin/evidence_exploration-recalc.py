import sys
import json
import os
from glob import glob

import polars as pl
import pandas as pd
from tqdm import tqdm

from pddb_lib.gene_ontology import (ONTOLOGIES_SHORT, CWA_DATASET_NAME, OWA_DATASET_NAME, EVIDENCE_REP_STRATEGIES, 
    calc_normalized_y_pred, create_ontology_dictionaries_full)
from pddb_lib.parsing import smart_str_parsing
from pddb_lib.sample_metaparameters import GENE_NAMES
from pddb_lib.optimization_utils import go_ia_path, run_eval
from pddb_lib.parsing import load_data_optimized

'''
python bin\evidence_exploration-
'''
if __name__ == "__main__":
    n_targets = int(sys.argv[1])
    min_annotations = int(sys.argv[2])
    max_train_proteins = int(sys.argv[3]) #downsampling after loading
    outputs_dir = 'outputs/'
    test_dir = sys.argv[4]
    feature_descs = sys.argv[5:]

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

    test_ids = [line.strip() for line in open(test_path)]
    train_ids = [line.strip() for line in open(train_path)]

    go_ia_dict = {}
    for rawline in open(go_ia_path, "r"):
        goid, ia = rawline.strip().split("\t")
        go_ia_dict[goid] = float(ia)

    parents_dict, children_dict, go_sortings = create_ontology_dictionaries_full("input_data/go-basic.obo")
    y_to_load = [f"y_{CWA_DATASET_NAME}", f"y_{OWA_DATASET_NAME}"]
    for ont, datasets in datasets_by_ont.items():
        #TODO: load y_soft just to calculate percentages of different evidence types, positives, negatives and NaNs
        print("Loading", ont)
        #train_df = load_data_optimized(datasets["train_y_path"], feature_descs, train_ids, y_to_load)
        test_df = load_data_optimized(datasets["test_y_path"], feature_descs, test_ids, y_to_load)
        #datasets["train_df"] = train_df
        datasets["test_df"] = test_df

    result_lines = []
    for alg_dir in glob(f"{test_dir}/*_test"):
        models_trained = []
        alg_name1 = os.path.basename(alg_dir).replace('_test', '')
        genenames = GENE_NAMES[alg_name1.replace('_rns', '')]
        alg_name = alg_name1.replace('_', ' ').title().replace('Rns', '+ RNS')
        #results_tsv = f"{alg_dir}/all_results.tsv"
        pred_parquets = glob(f"{alg_dir}/parameters=*/y_pred.parquet")

        print(f"Loading {len(pred_parquets)} prediction files of {alg_name} from {alg_dir}")

        for pred_path in tqdm(pred_parquets):
            comb = os.path.dirname(pred_path).split("parameters=")[-1]
            param_comb = tuple([smart_str_parsing(x) for x in comb.split("_")])
            params_dict = dict(zip(genenames, param_comb))
            y_preds = pl.read_parquet(pred_path)
            eval_metrics = run_eval(datasets_by_ont, y_preds, 
                params_dict, go_ia_dict, parents_dict, children_dict, go_sortings)
            if eval_metrics is not None:
                models_trained.append(eval_metrics)
            else:
                print(f"Error running eval for {pred_path}")
    
        
        for row in models_trained:
            for ont in ['MF', 'CC', "BP"]:
                #if f"{ont} - exception" in row.keys():
                #    print('Exception in training:', row['parameters'], row[f"{ont} - exception"])
                #else:
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
    
        results_df = pd.DataFrame(result_lines)
        results_df["OWA Score"] = (results_df["OWA Inverse-Weighted Fmax"] + results_df["OWA Weighted MCC (micro)"] + results_df["OWA Weighted AUPRC"])/3
        results_df["CWA Score"] = (results_df["CAFA Weighted Fmax"] + results_df["CAFA AUPRC"])/2
        results_df.to_csv(f"{test_dir}/all_results.tsv", sep="\t", index=False)

        best_param_lines = []
        for group_tp, group_df in results_df.groupby(['Algorithm', 'Ontology']):
            alg, ont = group_tp

            group_df.sort_values(by=['Sort Score'], inplace=True)
            #Get first line as dict:
            best_line = group_df.iloc[0].to_dict()
            n_tests = len(group_df)
            best_line['N Tests'] = n_tests
            best_param_lines.append(best_line)
        
        best_param_lines_df = pd.DataFrame(best_param_lines)
        best_param_lines_df.sort_values(by=['Sort Score'], ascending=False, inplace=True)
        new_cols_order = ["Algorithm", "Ontology", "N Tests", "Sort Score", "OWA Score", "CWA Score", 
            "OWA Weighted Fmax (micro)", "OWA Fmax (micro)", "OWA Inverse-Weighted Fmax", 
            "OWA Weighted MCC (micro)", "OWA Weighted AUPRC", 
            "CAFA Weighted Fmax", "CAFA AUPRC", "Parameters"]
        best_param_lines_df = best_param_lines_df[new_cols_order]
        best_param_lines_df.to_csv(f"{test_dir}/best_results.tsv", sep="\t", index=False)

        simple_df = best_param_lines_df[['Algorithm', 'Ontology', 'OWA Score', 'CWA Score', 'Parameters']]
        simple_df.to_csv(f"{test_dir}/best_results_simple.tsv", sep="\t", index=False)
    
        
    

        
    