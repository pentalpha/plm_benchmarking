import sys
import json
import os
from glob import glob

import polars as pl
import pandas as pd
from tqdm import tqdm
import numpy as np

from pddb_lib.gene_ontology import (ONTOLOGIES_SHORT, CWA_DATASET_NAME, OWA_DATASET_NAME, EVIDENCE_REP_STRATEGIES, 
    calc_normalized_y_pred, create_ontology_dictionaries_full)
from pddb_lib.parsing import smart_str_parsing
from pddb_lib.sample_metaparameters import GENE_NAMES
from pddb_lib.custom_statistics import metric_weights_for_sorting
from pddb_lib.optimization_utils import go_ia_path, run_eval
from pddb_lib.parsing import load_data_optimized
from pddb_lib.manipulate_y import calc_y_density

def load_to_check_density(y_path, values_col='y_soft'):
    df = pl.read_parquet(y_path)
    y = df[values_col].to_numpy()
    density_by_val = calc_y_density(y)

    return density_by_val

def translate_soft_label(x):
    trans_dict = {
        '0.0': 'Experimental False',
        '0.01': 'Curated False',
        '0.025': 'Phylogenetic False',
        '0.05': "Derived False",
        '0.15': 'Conditional False',
        '0.8': 'Curated True',
        '0.9': 'Phylogenetic True',
        '1.0': 'Experimental True',
        'NaN': "Unknown",
        'Nan': "Unknown",
    }

    if x in trans_dict:
        return trans_dict[x]
    else:
        return x
        

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
    densities = {'mf': {},'cc': {},'bp': {}}  
    for ont, datasets in datasets_by_ont.items():
        #TODO: load y_soft just to calculate percentages of different evidence types, positives, negatives and NaNs
        print("Loading", ont)
        #train_df = load_data_optimized(datasets["train_y_path"], feature_descs, train_ids, y_to_load)
        test_df = load_data_optimized(datasets["test_y_path"], feature_descs, test_ids, y_to_load)
        datasets["test_df"] = test_df

        train_densities = load_to_check_density(datasets["train_y_path"])
        test_densities = load_to_check_density(datasets["test_y_path"])
        #datasets["train_df"] = train_df
        densities[ont.lower()]['train'] = train_densities
        densities[ont.lower()]['test'] = test_densities
    
    density_rows_raw = []
    for ont, data in densities.items():
        for set_name, totals in data.items():
            for val, total in totals.items():
                density_rows_raw.append({
                    "Valores": str(val),
                    "Ontologia": ont,
                    "Conjunto": set_name,
                    "Frequência": total,
                    "Falso": val < 0.5 and val == val if "0." in str(val) else False,
                })
    
    df_raw = pd.DataFrame(density_rows_raw)
    print(df_raw)
    density_rows = []
    falses_by_ont = {}
    totals_by_ont = {}
    for conjunto_valor_tp, group_df in df_raw.groupby(['Conjunto', 'Valores', 'Falso']):
        conjunto, valor, is_false = conjunto_valor_tp
        print(conjunto, valor, is_false)
        onts = list(group_df['Ontologia'].unique())
        onts = sorted(onts)
        new_row = {
            "Evidence Type": conjunto.title()+ ' - ' + valor.title(),
        }
        for ont in onts:
            new_row[ont] = int(group_df[group_df["Ontologia"] == ont]["Frequência"].sum())
            if new_row[ont] != new_row[ont]:
                new_row[ont] = 0
        
            if is_false:
                if ont not in falses_by_ont:
                    falses_by_ont[ont] = {'train': 0, 'test': 0}
                falses_by_ont[ont][conjunto] += new_row[ont]
            
            if ont not in totals_by_ont:
                totals_by_ont[ont] = {'train': 0, 'test': 0}
            if valor.lower() == 'total with evidence' or valor.lower() == 'nan':
                totals_by_ont[ont][conjunto] += new_row[ont]

        density_rows.append(new_row)
    density_rows.append({"Evidence Type": "Train"})
    density_rows.append({"Evidence Type": "Test"})
    density_rows.sort(key = lambda x: x["Evidence Type"])
    for row in density_rows:
        if ' - ' in row["Evidence Type"]:
            parts = row["Evidence Type"].split(' - ')
            row["Evidence Type"] = '\t' + translate_soft_label(parts[1])
        if 'mf' in row:
            row['MF'] = row['mf']
            del row['mf']
        if 'cc' in row:
            row['CC'] = row['cc']
            del row['cc']
        if 'bp' in row:
            row['BP'] = row['bp']
            del row['bp']
    
    df_density = pd.DataFrame(density_rows)
    print(df_density)
    df_density.to_csv(test_dir + "/density.csv", index=False)

    false_perc_by_ont = {ont: {'train': falses_by_ont[ont]['train'] / totals_by_ont[ont]['train'], 
                                'test': falses_by_ont[ont]['test'] / totals_by_ont[ont]['test']} 
                            for ont in falses_by_ont.keys()}
    print("falses_by_ont:", json.dumps(falses_by_ont, indent=2))
    print("totals_by_ont:", json.dumps(totals_by_ont, indent=2))
    print("false_perc_by_ont:", json.dumps(false_perc_by_ont, indent=2))

    #density_rows_raw.sort(key=lambda x: (x["Conjunto"], x["Ontologia"], x["Valores"]))
    #density_rows_raw.to_pandas().to_csv(f"{test_dir}/density.tsv", sep="\t", index=False)

    y_tests_np = {}

    print('Pre-parsing y test matrices')
    for ont, datasets_dict in datasets_by_ont.items():
        y_test_cwa = datasets_dict['test_df']["y_"+CWA_DATASET_NAME].to_numpy()
        y_test_owa = datasets_dict['test_df']["y_"+OWA_DATASET_NAME].to_numpy()
        y_tests_np[ont] = {
            "y_test_cwa": y_test_cwa,
            "y_test_owa": y_test_owa,
            "y_test_ids": datasets_dict['test_df']["id"].to_list(),
            'targets': datasets_dict["targets"]
        }

    result_lines = []
    for alg_dir in glob(f"{test_dir}/*_test"):
        models_trained = []
        alg_name1 = os.path.basename(alg_dir).replace('_test', '')
        genenames = GENE_NAMES[alg_name1.replace('_rns', '+rns')]
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
                params_dict, go_ia_dict, parents_dict, children_dict, go_sortings, y_tests_np=y_tests_np)
            eval_metrics['Random Falses Min Perc'] = np.nan
            if "Random Falses Min Perc" in params_dict:
                eval_metrics['Random Falses Min Perc'] = params_dict['Random Falses Min Perc']
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
                ont_train_falses = false_perc_by_ont[ont.lower()]['train']
                if 'Random Falses Min Perc' in row:
                    if ont_train_falses >= float(row['Random Falses Min Perc']):
                        print(f"Not using random falses for {alg_name} - {ont}: {row['Random Falses Min Perc']} <= {ont_train_falses}")
                    else:
                        print(f"Using random falses for {alg_name} - {ont}: {row['Random Falses Min Perc']} > {ont_train_falses}")
                        result_lines.append(new_row)
                else:
                    print("No RNS")
                    result_lines.append(new_row)
    
        results_df = pd.DataFrame(result_lines)
        owa_metrics = [k for k, w in metric_weights_for_sorting.items() if 'OWA' in k]
        cafa_metrics = [k for k, w in metric_weights_for_sorting.items() if 'CAFA' in k]
        results_df["OWA Score"] = results_df[owa_metrics].mean(axis=1)
        results_df["CWA Score"] = results_df[cafa_metrics].mean(axis=1)
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
        new_cols_order = ["Algorithm", "Ontology", "N Tests", "OWA Score", "CWA Score"] + list(metric_weights_for_sorting.keys()) + ["Parameters"]
        best_param_lines_df = best_param_lines_df[new_cols_order]
        best_param_lines_df.to_csv(f"{test_dir}/best_results.tsv", sep="\t", index=False)

        simple_df = best_param_lines_df[['Algorithm', 'Ontology', 'OWA Score', 'CWA Score', 'Parameters']]
        simple_df.to_csv(f"{test_dir}/best_results_simple.tsv", sep="\t", index=False)
    
        
    

        
    