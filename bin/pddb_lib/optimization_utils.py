from glob import glob
import json

import numpy as np

from pddb_lib.gene_ontology import CWA_DATASET_NAME, OWA_DATASET_NAME, calc_normalized_y_pred
from pddb_lib.custom_statistics import run_statistics, get_sorting_score

'''
python bin\evidence_exploration-optimize.py N_TARGETS MIN_ANNOTATIONS MAX_TRAIN_PROTEINS Y_DATASET_NAME USE_RANDOM_NEGATIVE_SAMPLING TEST_DIR <[<embedding_path>:col1,col2,...],...> 
'''

def align_predictions_and_test(y_pred, y_pred_ids, y_test_cwa, y_test_owa, y_test_ids):
    """
    Finds the intersection of IDs and reorders the prediction and test matrices
    so that they have the exact same samples in the exact same order.
    """
    # Find common IDs to prevent KeyErrors
    common_ids = set(y_pred_ids).intersection(set(y_test_ids))

    common_ids_perc = len(common_ids) / len(y_pred_ids)
    if common_ids_perc < 0.9:
        #raise ValueError(f"Common IDs percentage is only {common_ids_perc}")
        return None, None, None, None
    
    # Keep the reference order based on y_test_ids
    aligned_ids = [uid for uid in y_test_ids if uid in common_ids]
    
    # Map IDs to their original row indices for $O(1)$ lookups
    pred_id_to_idx = {uid: idx for idx, uid in enumerate(y_pred_ids)}
    test_id_to_idx = {uid: idx for idx, uid in enumerate(y_test_ids)}
    
    # Get the correctly ordered indices
    pred_indices = [pred_id_to_idx[uid] for uid in aligned_ids]
    test_indices = [test_id_to_idx[uid] for uid in aligned_ids]
    
    # Reorder and filter the numpy arrays
    aligned_y_pred = y_pred[pred_indices]
    aligned_y_test_cwa = y_test_cwa[test_indices]
    aligned_y_test_owa = y_test_owa[test_indices]
    
    return aligned_y_pred, aligned_y_test_cwa, aligned_y_test_owa, aligned_ids

def run_eval(datasets_by_ont, y_preds, params_dict, go_ia_dict, parents_dict, children_dict, go_sortings,
            y_tests_np = None):
    eval_metrics = {}

    for ont, datasets_dict in datasets_by_ont.items():
        y_pred = y_preds[ont].to_numpy()
        y_pred_ids = y_preds["id"].to_list()
        
        if ont == 'deeploc':
            y_test_cwa = datasets_dict['test_df']["y"].to_numpy()
            y_test_owa = datasets_dict['test_df']["y"].to_numpy()
            y_test_ids = datasets_dict['test_df']["id"].to_list()
            labels = datasets_dict["targets"]
            weights = np.array([1.0 for t in labels])
        else:
            if y_tests_np:
                y_test_cwa = y_tests_np[ont]["y_test_cwa"]
                y_test_owa = y_tests_np[ont]["y_test_owa"]
                y_test_ids = y_tests_np[ont]["y_test_ids"]
                labels = y_tests_np[ont]["targets"]
            else:
                y_test_cwa = datasets_dict['test_df']["y_"+CWA_DATASET_NAME].to_numpy()
                y_test_owa = datasets_dict['test_df']["y_"+OWA_DATASET_NAME].to_numpy()
                y_test_ids = datasets_dict['test_df']["id"].to_list()
                labels = datasets_dict["targets"]
            weights = np.array([go_ia_dict.get(t, 0) for t in labels])
        print("Labels:", labels)
        print("Weights:", weights)
        all_equal_ids = set(y_pred_ids) == set(y_test_ids)

        #y_pred, y_test_cwa, y_test_owa, aligned_ids = align_predictions_and_test(
        #    y_pred, y_pred_ids, y_test_cwa, y_test_owa, y_test_ids
        #)
        if not all_equal_ids:
            return None
        if ont != 'deeploc':
            y_pred_norm = calc_normalized_y_pred(
                y_pred, labels, parents_dict, children_dict, go_sortings[ont.upper()]
            )
        else:
            y_pred_norm = y_pred
        stats_norm = run_statistics(y_pred_norm, y_test_cwa, y_test_owa, weights)
        stats_norm["Sort Score"] = get_sorting_score(stats_norm)
        #print("Normalized stats:", stats_norm)

        for metric_name, metric_val in stats_norm.items():
            eval_metrics[f"{ont.upper()} - {metric_name}"] = metric_val            

    eval_metrics["parameters"] = json.dumps(params_dict, ensure_ascii=False)
    return eval_metrics

go_ia_path = "input_data/go_ia.tsv"