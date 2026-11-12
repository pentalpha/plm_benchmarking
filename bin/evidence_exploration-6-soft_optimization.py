#!/usr/bin/env python3
import sys
import random
from glob import glob
import os
import json
import argparse

from tqdm import tqdm
import numpy as np
import polars as pl
import pandas as pd
import cupy as cp
from cupy.cuda.memory import OutOfMemoryError

# Ensure local library import works regardless of CWD
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from pddb_lib.gene_ontology import (
    ONTOLOGIES_SHORT,
    CWA_DATASET_NAME,
    OWA_DATASET_NAME,
    EVIDENCE_REP_STRATEGIES,
    calc_normalized_y_pred,
    create_ontology_dictionaries_full,
)
from pddb_lib.parsing import load_data_optimized, smart_str_parsing
from pddb_lib.manipulate_y import add_random_false_values, show_y_density
from pddb_lib.training import train_and_pred, reduce_train_negatives_to
from pddb_lib.optimization_utils import go_ia_path, run_eval

SOFT_LABEL_NAMES = [
    "phylo",
    "curated",
    "conditional_not",
    "Random False Val",
    "curated_not",
    "derived_not",
    "phylo_not",
]

ORIGINAL_PLACEHOLDER_VALS = {
    "phylo": 0.9,
    "curated": 0.8,
    "conditional_not": 0.15,
    "derived_not": 0.05,
    "phylo_not": 0.025,
    "curated_not": 0.01,
}


def load_best_parameters_by_ont(input_path: str):
    """
    Parses the input TSV/CSV and extracts the best parameters for each ontology (MF, BP, CC)
    specifically for the 'Soft + RNS' strategy.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    sep = "\t" if input_path.endswith(".tsv") else ","
    df = pd.read_csv(input_path, sep=sep)

    required_cols = ["Algorithm", "Ontology", "Parameters"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Input file {input_path} missing required column '{col}'. Available: {df.columns.tolist()}"
            )

    # Filter for Soft Labeling + RNS
    mask = df["Algorithm"].str.contains("soft", case=False, na=False) & df[
        "Algorithm"
    ].str.contains("rns", case=False, na=False)
    filtered_df = df[mask].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No rows matching 'Soft + RNS' found in {input_path}. Please check the Algorithm column."
        )

    # Sort descending by Sort Score if available
    if "Sort Score" in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by="Sort Score", ascending=False)

    best_params_by_ont = {}
    for ont in ["mf", "bp", "cc"]:
        ont_rows = filtered_df[filtered_df["Ontology"].str.lower() == ont]
        if ont_rows.empty:
            raise ValueError(
                f"No row found for ontology '{ont.upper()}' under 'Soft + RNS' in {input_path}"
            )

        best_row = ont_rows.iloc[0]
        params_raw = best_row["Parameters"]
        if isinstance(params_raw, str):
            params_dict = json.loads(params_raw)
        else:
            params_dict = dict(params_raw)

        best_params_by_ont[ont] = params_dict
        print(f"\n[INFO] Best Soft + RNS parameters for {ont.upper()}:")
        print(f"  Sort Score: {best_row.get('Sort Score', 'N/A')}")
        print(f"  Parameters: {params_dict}")

    return best_params_by_ont


def generate_soft_combinations(
    n_combinations: int,
    seed: int = 42,
    existing_combs: set = None,
    include_defaults: list = None,
):
    """
    Generates combinations of soft label values in [0.01, 0.99] rounded to 2 decimals.
    Supports resuming by keeping already generated/existing combinations.
    """
    n_combs_totalrand = round(n_combinations/2)
    n_combs_negs_below_05 = n_combinations - n_combs_totalrand
    full_range = (0.0, 1.0)
    neg_range = (0.00, 0.45)
    pos_range = (0.55, 1.0)

    totalrand_ranges = [full_range] * len(SOFT_LABEL_NAMES)
    b5_ranges = [neg_range 
        if ('_not' in param_name or 'false' in param_name.lower()) 
        else pos_range 
        for param_name in SOFT_LABEL_NAMES]
    
    if existing_combs is None:
        existing_combs = set()

    combs = set(existing_combs)
    b5_combs = []

    # Optionally seed with reference combinations (e.g. previous top ones) if starting fresh
    if include_defaults:
        for default_comb in include_defaults:
            b5_combs.append(tuple(round(float(x), 2) for x in default_comb))

    np.random.seed(seed)
    random.seed(seed)

    while len(b5_combs) < n_combs_negs_below_05:
        b5_comb = tuple(
            round(random.uniform(bottom, top), 2) 
            for bottom, top in b5_ranges
        )
        if b5_comb in combs:
            continue
        b5_combs.append(b5_comb)
    
    totalrand_combs = []
    while len(totalrand_combs) < n_combs_totalrand:
        tr_comb = tuple(
            round(random.uniform(bottom, top), 2) 
            for bottom, top in totalrand_ranges
        )
        #print("range:", totalrand_ranges)
        #print("comb:", tr_comb)
        if tr_comb in combs:
            continue
        totalrand_combs.append(tr_comb)
    
    all_combs = b5_combs + totalrand_combs

    return all_combs


def apply_soft_labels_with_masks(
    y_pristine: np.ndarray, masks: dict, soft_labels_dict: dict
) -> np.ndarray:
    """
    Safely replace soft labels in y_pristine using precomputed boolean masks.
    Avoids sequential replacement collisions and preserves untouched original data.
    """
    y = y_pristine.copy()
    for param_name, mask in masks.items():
        if param_name in soft_labels_dict:
            y[mask] = float(soft_labels_dict[param_name])
    return y


def parse_cli_args():
    """
    Flexible argument parser supporting both standard CLI flags and positional arguments.
    """
    # Detect if positional style is passed (e.g., python script.py 32 120 80000 input.tsv test_dir 100 ...)
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        n_targets = int(sys.argv[1])
        min_annotations = int(sys.argv[2])
        max_train_proteins = int(sys.argv[3])
        input_file = sys.argv[4]
        test_dir = sys.argv[5]
        n_combinations = int(sys.argv[6])
        features_desc = sys.argv[7:] if len(sys.argv) > 7 else ["input_data/emb.ankh_base_mean.parquet:mean"]
        return argparse.Namespace(
            n_targets=n_targets,
            min_annotations=min_annotations,
            max_train_proteins=max_train_proteins,
            input_file=input_file,
            test_dir=test_dir,
            n_combinations=n_combinations,
            features_desc=features_desc,
            seed=42,
        )

    parser = argparse.ArgumentParser(
        description="Soft Label Optimization Experiment: Explores ~100 combinations of soft labels in [0.01, 0.99] using best GBDT+RNS parameters per ontology."
    )
    parser.add_argument(
        "pos_input_file",
        nargs="?",
        default=None,
        help="Positional input TSV/CSV containing top performance parameters",
    )
    parser.add_argument(
        "pos_test_dir",
        nargs="?",
        default=None,
        help="Positional output directory for results",
    )
    parser.add_argument(
        "pos_n_combinations",
        nargs="?",
        type=int,
        default=None,
        help="Positional number of combinations to evaluate",
    )

    parser.add_argument(
        "--input_file",
        "-i",
        type=str,
        default="outputs/remote_results/evidence_exploration_npad/evi_rep-top_performance.tsv",
        help="Path to TSV/CSV with best parameters (default: outputs/remote_results/evidence_exploration_npad/evi_rep-top_performance.tsv)",
    )
    parser.add_argument(
        "--test_dir",
        "-o",
        type=str,
        default="outputs/evidence_exploration_npad/soft_optimization_test",
        help="Output directory for experiment results (default: outputs/evidence_exploration_npad/soft_optimization_test)",
    )
    parser.add_argument(
        "--n_combinations",
        "-n",
        type=int,
        default=100,
        help="Number of soft label combinations to evaluate (default: 100)",
    )
    parser.add_argument(
        "--n_targets",
        type=int,
        default=32,
        help="Number of targets per ontology (default: 32)",
    )
    parser.add_argument(
        "--min_annotations",
        type=int,
        default=120,
        help="Minimum annotations threshold (default: 120)",
    )
    parser.add_argument(
        "--max_train_proteins",
        type=int,
        default=80000,
        help="Maximum train proteins downsampling threshold (default: 80000)",
    )
    parser.add_argument(
        "--features_desc",
        nargs="+",
        default=["input_data/emb.ankh_base_mean.parquet:mean"],
        help="Feature descriptor(s) for embeddings (default: input_data/emb.ankh_base_mean.parquet:mean)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for combination generation (default: 42)",
    )

    args = parser.parse_args()

    # Apply positional overrides if supplied
    if args.pos_input_file is not None:
        args.input_file = args.pos_input_file
    if args.pos_test_dir is not None:
        args.test_dir = args.pos_test_dir
    if args.pos_n_combinations is not None:
        args.n_combinations = args.pos_n_combinations

    return args


def summarize_results(models_trained: list, test_dir: str):
    """
    Saves and summarizes all evaluation metrics into standard TSV files.
    """
    if not models_trained:
        return

    # 1. Raw evaluation results table
    raw_df = pd.DataFrame(models_trained)
    raw_path = os.path.join(test_dir, "all_results_raw.tsv")
    raw_df.to_csv(raw_path, sep="\t", index=False)

    # 2. Unrolled results table per Ontology + Overall
    unrolled_rows = []
    for entry in models_trained:
        soft_labels = json.loads(entry["soft_labels"])
        
        # Calculate overall across MF, BP, CC
        sort_scores = []
        owa_scores = []
        cwa_scores = []

        for ont in ["MF", "BP", "CC"]:
            sort_sc = entry.get(f"{ont} - Sort Score", np.nan)
            
            # OWA Score: average of OWA metrics
            owa_fmax = entry.get(f"{ont} - OWA Fmax (Inverse-Weighted)", entry.get(f"{ont} - OWA Weighted Fmax (micro)", np.nan))
            owa_mcc = entry.get(f"{ont} - OWA Weighted MCC", np.nan)
            owa_auprc = entry.get(f"{ont} - OWA Weighted AUPRC", np.nan)
            owa_sc = np.nanmean([owa_fmax, owa_mcc, owa_auprc])

            # CWA Score: average of CWA metrics
            cwa_fmax = entry.get(f"{ont} - CAFA Weighted Fmax", np.nan)
            cwa_mcc = entry.get(f"{ont} - CAFA Weighted MCC", np.nan)
            cwa_auprc = entry.get(f"{ont} - CAFA AUPRC", np.nan)
            cwa_sc = np.nanmean([cwa_fmax, cwa_auprc])

            sort_scores.append(sort_sc)
            owa_scores.append(owa_sc)
            cwa_scores.append(cwa_sc)

            row = {
                "Algorithm": "Soft + RNS",
                "Ontology": ont,
                "Sort Score": sort_sc,
                "OWA Score": owa_sc,
                "CWA Score": cwa_sc,
                "OWA Fmax (Inverse-Weighted)": owa_fmax,
                "OWA Weighted MCC": owa_mcc,
                "OWA Weighted AUPRC": owa_auprc,
                "CAFA Weighted Fmax": cwa_fmax,
                "CAFA Weighted MCC": cwa_mcc,
                "CAFA AUPRC": cwa_auprc,
                **soft_labels,
                "Parameters": entry.get("parameters", ""),
            }
            unrolled_rows.append(row)

        # Append Overall row for this combination
        overall_sort = np.nanmean(sort_scores)
        overall_owa = np.nanmean(owa_scores)
        overall_cwa = np.nanmean(cwa_scores)

        unrolled_rows.append({
            "Algorithm": "Soft + RNS",
            "Ontology": "Overall",
            "Sort Score": overall_sort,
            "OWA Score": overall_owa,
            "CWA Score": overall_cwa,
            "OWA Fmax (Inverse-Weighted)": np.nan,
            "OWA Weighted MCC": np.nan,
            "OWA Weighted AUPRC": np.nan,
            "CAFA Weighted Fmax": np.nan,
            "CAFA Weighted MCC": np.nan,
            "CAFA AUPRC": np.nan,
            **soft_labels,
            "Parameters": entry.get("parameters", ""),
        })

    results_df = pd.DataFrame(unrolled_rows)
    all_results_path = os.path.join(test_dir, "all_results.tsv")
    results_df.to_csv(all_results_path, sep="\t", index=False)

    # 3. Best overall rankings
    overall_df = results_df[results_df["Ontology"] == "Overall"].sort_values(
        by="Sort Score", ascending=False
    ).copy()
    overall_df["Rank"] = range(1, len(overall_df) + 1)
    cols_order = ["Rank", "Sort Score", "OWA Score", "CWA Score"] + SOFT_LABEL_NAMES + ["Parameters"]
    overall_df = overall_df[[c for c in cols_order if c in overall_df.columns]]
    best_overall_path = os.path.join(test_dir, "best_results_overall.tsv")
    overall_df.to_csv(best_overall_path, sep="\t", index=False)

    print(f"\n[SUMMARY] Saved updated results:")
    print(f"  - {all_results_path}")
    print(f"  - {best_overall_path}")
    if not overall_df.empty:
        print("\nTop 5 Overall Configurations:")
        print(overall_df.head(5).to_string(index=False))


def main():
    args = parse_cli_args()

    print("=" * 70)
    print("  SOFT LABEL OPTIMIZATION EXPERIMENT")
    print("=" * 70)
    print(f"Input Parameters File : {args.input_file}")
    print(f"Test Output Directory : {args.test_dir}")
    print(f"Target Combinations   : {args.n_combinations}")
    print(f"N Targets per Ont     : {args.n_targets}")
    print(f"Min Annotations       : {args.min_annotations}")
    print(f"Max Train Proteins    : {args.max_train_proteins}")
    print(f"Feature Descriptors   : {args.features_desc}")
    print(f"Soft Label Range      : [0.01, 0.99]")
    print("=" * 70)

    # 1. Load best parameters for each ontology
    best_params_by_ont = load_best_parameters_by_ont(args.input_file)

    # 2. Check and discover existing combinations in test_dir
    os.makedirs(args.test_dir, exist_ok=True)
    existing_folders = glob(f"{args.test_dir}/parameters=*/")
    existing_combs = set()
    for folder in existing_folders:
        raw_name = folder.rstrip("/").split("parameters=")[-1]
        parts = raw_name.split("_")
        if len(parts) == len(SOFT_LABEL_NAMES):
            try:
                comb_tuple = tuple(round(float(x), 2) for x in parts)
                existing_combs.add(comb_tuple)
            except ValueError:
                pass

    print(f"\n[INFO] Found {len(existing_combs)} existing combinations in {args.test_dir}")

    # Extract default reference soft labels from the input TSV
    reference_defaults = []
    for ont in ["mf", "bp", "cc"]:
        ref_params = best_params_by_ont[ont]
        ref_comb = tuple(ref_params.get(name, 0.5) for name in SOFT_LABEL_NAMES)
        reference_defaults.append(ref_comb)
    
    param_combs_list_path = f"{args.test_dir}/param_combs.json"
    if os.path.exists(param_combs_list_path):
        with open(param_combs_list_path, 'r') as f:
            d = json.load(f)
        all_param_combs = d['combs']
    else:
        all_param_combs = generate_soft_combinations(
            n_combinations=args.n_combinations,
            seed=args.seed,
            include_defaults=reference_defaults,
        )
        with open(param_combs_list_path, 'w') as f:
            json.dump({'label_names': SOFT_LABEL_NAMES, 'combs': all_param_combs}, f, indent=4)
    print(f"[INFO] Total combinations scheduled: {len(all_param_combs)}")

    # 4. Load Ontologies dictionaries & IA weights
    obo_path = "input_data/go-basic.obo"
    if not os.path.exists(obo_path):
        raise FileNotFoundError(f"Gene ontology file not found: {obo_path}")
    parents_dict, children_dict, go_sortings = create_ontology_dictionaries_full(obo_path)

    go_ia_dict = {}
    if not os.path.exists(go_ia_path):
        raise FileNotFoundError(f"GO IA weights file not found: {go_ia_path}")
    for rawline in open(go_ia_path, "r"):
        goid, ia = rawline.strip().split("\t")
        go_ia_dict[goid] = float(ia)

    # 5. Locate existing pre-generated datasets
    outputs_dir = "outputs"
    sampling_prefix = f"{outputs_dir}/n_ont_target={args.n_targets}-min_proteins={args.min_annotations}"
    test_path = f"{sampling_prefix}.test_set.txt"
    train_path = f"{sampling_prefix}.train_set.txt"

    if not os.path.exists(test_path) or not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Pre-generated dataset splits not found at {sampling_prefix}. "
            f"Per instructions, existing dataset must be present."
        )

    test_ids = [line.strip() for line in open(test_path)]
    train_ids = [line.strip() for line in open(train_path)]

    if len(train_ids) > args.max_train_proteins:
        print(f"[INFO] Downsampling train set to {args.max_train_proteins} proteins...")
        sampling_path = train_path.replace(".txt", f".sampled_{args.max_train_proteins}.txt")
        if not os.path.exists(sampling_path):
            random.seed(42)
            train_ids = random.sample(train_ids, args.max_train_proteins)
            with open(sampling_path, "w") as f:
                for gene_id in train_ids:
                    f.write(f"{gene_id}\n")
        train_ids = [line.strip() for line in open(sampling_path)]

    datasets_by_ont = {
        ont: {
            "train_y_path": f"{sampling_prefix}-evi_exp-{ont}-train_y.parquet",
            "test_y_path": f"{sampling_prefix}-evi_exp-{ont}-test_y.parquet",
        }
        for ont in ONTOLOGIES_SHORT.keys()
    }

    for ont, datasets_dict in datasets_by_ont.items():
        targets_path = datasets_dict["train_y_path"] + ".targets.txt"
        if not os.path.exists(targets_path):
            raise FileNotFoundError(f"Targets file not found: {targets_path}")
        targets = [line.strip() for line in open(targets_path)]
        datasets_dict["targets"] = targets

    # 6. Load X features and pristine y matrices into memory once
    y_to_load = [f"y_{CWA_DATASET_NAME}", f"y_{OWA_DATASET_NAME}", "y_soft"]

    print("\n[INFO] Loading pre-generated datasets into memory...")
    for ont, datasets in datasets_by_ont.items():
        print(f"  Loading {ont.upper()}...")
        train_df = load_data_optimized(datasets["train_y_path"], args.features_desc, train_ids, y_to_load)
        test_df = load_data_optimized(datasets["test_y_path"], args.features_desc, test_ids, y_to_load)
        datasets["train_df"] = train_df
        datasets["test_df"] = test_df

        # Store pristine arrays and feature vectors
        datasets["train_x"] = np.ascontiguousarray(train_df["X"].to_numpy(), dtype=np.float32)
        datasets["test_x"] = np.ascontiguousarray(test_df["X"].to_numpy(), dtype=np.float32)
        datasets["test_ids"] = test_df["id"].to_list()

        train_y_pristine = np.ascontiguousarray(train_df["y_soft"].to_numpy(), dtype=np.float32)
        test_y_pristine = np.ascontiguousarray(test_df["y_soft"].to_numpy(), dtype=np.float32)

        datasets["train_y_pristine"] = train_y_pristine
        datasets["test_y_pristine"] = test_y_pristine

        # Precompute safe boolean masks against the untouched pristine arrays
        datasets["train_masks"] = {
            name: np.isclose(train_y_pristine, val)
            for name, val in ORIGINAL_PLACEHOLDER_VALS.items()
        }
        datasets["test_masks"] = {
            name: np.isclose(test_y_pristine, val)
            for name, val in ORIGINAL_PLACEHOLDER_VALS.items()
        }

    # 7. Training and Evaluation Loop
    models_trained = []
    progress_bar = tqdm(all_param_combs, desc="Optimizing Soft Labels", unit="comb")

    for param_comb in progress_bar:
        soft_labels_dict = dict(zip(SOFT_LABEL_NAMES, param_comb))
        comb_str = "_".join(f"{x:.2f}" for x in param_comb)
        test_basename = f"parameters={comb_str}"
        test_path = os.path.join(args.test_dir, test_basename)
        os.makedirs(test_path, exist_ok=True)

        test_preds_path = os.path.join(test_path, "y_pred.parquet")
        params_info_path = os.path.join(test_path, "params.json")

        success = False

        if os.path.exists(test_preds_path):
            print(f"\n[INFO] Predictions already exist for {test_basename}. Loading...")
            try:
                y_preds = pl.read_parquet(test_preds_path)
                eval_metrics = run_eval(
                    datasets_by_ont,
                    y_preds,
                    soft_labels_dict,
                    go_ia_dict,
                    parents_dict,
                    children_dict,
                    go_sortings,
                )
                if eval_metrics is not None:
                    eval_metrics["soft_labels"] = json.dumps(soft_labels_dict)
                    for k, v in soft_labels_dict.items():
                        eval_metrics[k] = v
                    models_trained.append(eval_metrics)
                    success = True
                else:
                    print(f"[WARN] Evaluation failed for {test_preds_path}, re-running...")
                    os.remove(test_preds_path)
            except Exception as e:
                print(f"[WARN] Error reading {test_preds_path}: {e}, re-running...")
                if os.path.exists(test_preds_path):
                    os.remove(test_preds_path)

        if not success:
            print(f"\n[RUN] Training soft label combination: {soft_labels_dict}")
            y_preds = {}

            full_params_per_ont = {}

            for ont, dataset_dict in datasets_by_ont.items():
                best_ont_params = best_params_by_ont[ont]

                # Merge ontology-specific best GBDT+RNS params with the current soft labels
                ont_params = {**best_ont_params, **soft_labels_dict}
                full_params_per_ont[ont] = ont_params

                # 1. Apply safe soft label transformation
                train_y = apply_soft_labels_with_masks(
                    dataset_dict["train_y_pristine"],
                    dataset_dict["train_masks"],
                    soft_labels_dict,
                )
                test_y = apply_soft_labels_with_masks(
                    dataset_dict["test_y_pristine"],
                    dataset_dict["test_masks"],
                    soft_labels_dict,
                )

                train_x = dataset_dict["train_x"].copy()
                test_x = dataset_dict["test_x"].copy()

                # 2. Reset deterministic seed to 1337 matching previous experiments
                np.random.seed(1337)
                random.seed(1337)

                # 3. Negative reduction
                train_x, train_y = reduce_train_negatives_to(
                    train_x, train_y, target_ratio=0.15, use_nan=True
                )

                # 4. Random Negative Sampling (RNS)
                rns_perc = ont_params.get("Random Falses Min Perc")
                rns_val = soft_labels_dict["Random False Val"]
                print(f"  [{ont.upper()}] Adding {rns_perc*100:.1f}% RNS with value {rns_val}")
                train_y, added_nots = add_random_false_values(
                    train_y, target_min_zeros=rns_perc, zero_val=rns_val
                )

                # 5. Train with Py-Boost using ontology's best hyperparameters
                try:
                    y_pred = train_and_pred(
                        train_x, train_y, test_x, test_y, ont_params, has_nan=True
                    )
                    y_preds["id"] = dataset_dict["test_ids"]
                    y_preds[ont] = y_pred
                except OutOfMemoryError:
                    print(f"[OOM] Out of memory error for {test_basename}, cancelling...")
                    y_pred = None
                    y_preds["id"] = dataset_dict["test_ids"]

                # Free GPU memory after each ontology
                cp.get_default_memory_pool().free_all_blocks()

            # Save predictions
            if len(y_preds.keys()) > 1:
                y_preds_df = pl.DataFrame(y_preds)
                y_preds_df.write_parquet(test_preds_path)

                # Save detailed parameter log
                with open(params_info_path, "w") as f:
                    json.dump(
                        {
                            "soft_labels": soft_labels_dict,
                            "ontology_parameters": full_params_per_ont,
                        },
                        f,
                        indent=4,
                    )

                # Evaluate predictions
                eval_metrics = run_eval(
                    datasets_by_ont,
                    y_preds_df,
                    soft_labels_dict,
                    go_ia_dict,
                    parents_dict,
                    children_dict,
                    go_sortings,
                )

                if eval_metrics is not None:
                    eval_metrics["soft_labels"] = json.dumps(soft_labels_dict)
                    for k, v in soft_labels_dict.items():
                        eval_metrics[k] = v
                    models_trained.append(eval_metrics)
                    success = True

        if success:
            summarize_results(models_trained, args.test_dir)

    print("\n" + "=" * 70)
    print("  SOFT LABEL OPTIMIZATION EXPERIMENT COMPLETED")
    print("=" * 70)
    summarize_results(models_trained, args.test_dir)


if __name__ == "__main__":
    main()
