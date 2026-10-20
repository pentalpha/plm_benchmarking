import sys
import os
from glob import glob
import json

import pandas as pd
from tqdm import tqdm

from pddb_lib.parsing import smart_str_parsing
from pddb_lib.sample_metaparameters import GENE_NAMES



def parse_test_id(val) -> float:
    """Safely parse timestamp/test order, handling None and NaN values."""
    if val is None or pd.isna(val):
        return -1.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return -1.0

def find_datetime(row, param_datetimes):
    alg_name = row["Algorithm"]
    params_dict_str1 = row["Parameters"]

    if alg_name in param_datetimes and params_dict_str1 in param_datetimes[alg_name]:
        return param_datetimes[alg_name][params_dict_str1]
    else:
        print(f"Could not find datetime for {alg_name} with parameters {params_dict_str1}")
        return None

if __name__ == "__main__":
    evi_exp_res_dir = sys.argv[1]
    all_results_path = os.path.join(evi_exp_res_dir, "all_results.tsv")

    df = pd.read_csv(all_results_path, sep="\t")

    param_datetimes = {}
    for alg_dir in glob(f"{evi_exp_res_dir}/*_test"):
        models_trained = []
        alg_name1 = os.path.basename(alg_dir).replace('_test', '')
        genenames = GENE_NAMES[alg_name1.replace('_rns', '+rns')]
        alg_name = alg_name1.replace('_', ' ').title().replace('Rns', '+ RNS')
        #results_tsv = f"{alg_dir}/all_results.tsv"
        pred_parquets = glob(f"{alg_dir}/parameters=*/y_pred.parquet")

        if not alg_name in param_datetimes:
            param_datetimes[alg_name] = {}

        print(f"Loading {len(pred_parquets)} prediction files of {alg_name} from {alg_dir}")

        for pred_path in tqdm(pred_parquets):
            comb = os.path.dirname(pred_path).split("parameters=")[-1]
            param_comb = tuple([smart_str_parsing(x) for x in comb.split("_")])
            params_dict = dict(zip(genenames, param_comb))
            params_str = json.dumps(params_dict, ensure_ascii=False)
            param_datetimes[alg_name][params_str] = os.path.getmtime(pred_path)
    
    param_orders = {}
    for alg_name, datetimes in param_datetimes.items():
        sorted_keys = sorted(datetimes.keys(), key=lambda x: datetimes[x])
        key_to_order = {k: i for i, k in enumerate(sorted_keys)}
        param_orders[alg_name] = key_to_order
    
    df['datetime'] = df.apply(find_datetime, axis=1, param_datetimes=param_datetimes)
    df['test_order'] = df.apply(find_datetime, axis=1, param_datetimes=param_orders)
    df = df.sort_values(by=['Algorithm', 'Ontology', 'test_order'])

    df["Sort Score"] = (df["OWA Score"] + df["CWA Score"]) / 2
    
    # Save modified df
    df.to_csv(all_results_path.replace(".tsv", "_wtimes.tsv"), sep="\t", index=False)
    print(f"Saved modified df to {all_results_path.replace(".tsv", "_wtimes.tsv")}")

    