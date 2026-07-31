import sys
import os
from collections import defaultdict

import polars as pl
import pandas as pd
from tqdm import tqdm
import numpy as np
import obonet

from pddb_lib.gene_ontology import (EVIDENCE_REP_STRATEGIES, 
    EVIDENCE_GROUP_IMPORTANCE_SEQUENCE, create_ontology_dictionaries,
    create_ontology_dictionaries_full)
from pddb_lib.training import (
    find_conditional_zeros,
    #find_conditional_zeros_inverse,
    show_y_density,
)
from pddb_lib.sample_traintest import split_traintest

url_ou_caminho_obo = "input_data/go-basic.obo"
parents_dict, children_dict = create_ontology_dictionaries(url_ou_caminho_obo)
parents_dict_full, children_dict_full, go_sortings = create_ontology_dictionaries_full(
    url_ou_caminho_obo)

def make_dataset_for_ont(ont_name, n_targets, min_annotations, sampling_prefix, 
                        test_ids, train_ids, targets_by_ont,
                        used_uniprots_set):
    annots_parquet_path = "input_data/go." + ont_name + ".parquet"
    mf_parquet = pl.read_parquet(annots_parquet_path)
    goids = targets_by_ont[ont_name]
    goid_set = set(goids)
    annots_parquet = pl.read_parquet(annots_parquet_path)
    annots_parquet = annots_parquet.filter(pl.col("id").is_in(used_uniprots_set))

    conflicts_dict = {}
    repetitions_dict = {}
    new_annots_dict = {}
    new_der_neg = 0
    repeated_der_neg = 0
    conflicting_der_neg = 0
    exp_negatives = 0
    has_derivated_neg = set()
    has_no_derivated_neg = set()

    not_copy = ["conditional_not", "comp", "comp_not", "iea", "iea_not"]
    to_ignore = ["comp", "comp_not", "iea", "iea_not"]
    evicols_to_copy = [c for c in EVIDENCE_GROUP_IMPORTANCE_SEQUENCE if c not in not_copy]
    evicols_to_use = [c for c in EVIDENCE_GROUP_IMPORTANCE_SEQUENCE if c not in to_ignore]

    uniprot2go = {}
    for row in tqdm(annots_parquet.iter_rows(named=True), desc="Listing uniprot annotations"):
        new_row = {col_name: set(row[col_name]) for col_name in evicols_to_copy}

        cols_present = [c for c in evicols_to_copy if c in new_row]
        cols_present = [c for c in cols_present if len(new_row[c]) > 0]

        for col_name in cols_present:
            current_goids = new_row[col_name]
            if '_not' in col_name:
                #Get descendants
                expansion_dict = children_dict_full
            else:
                expansion_dict = parents_dict_full
            
            new_ids = set()
            for goid in current_goids:
                if goid in expansion_dict:
                    new_ids.update(expansion_dict[goid])
                new_ids.add(goid)
            new_row[col_name] = new_ids
        
        for i, col_name in enumerate(cols_present):
            if i != len(cols_present) - 1:
                next_cols = cols_present[i+1:]
                for next_col in next_cols:
                    intersection = new_row[col_name] & new_row[next_col]
                    if len(intersection) > 0:
                        #If both have '_not' or both dont have 'not', its a repeat
                        #if one has not and the other doesnt, it's a conflict
                        is_repeat = ((col_name.endswith('_not') and next_col.endswith('_not')) or
                                     (not col_name.endswith('_not') and not next_col.endswith('_not')))
                        event_name = col_name + " found in " + next_col
                        event_dict = repetitions_dict if is_repeat else conflicts_dict
                        if event_name not in event_dict:
                            event_dict[event_name] = 0
                        event_dict[event_name] += len(intersection)
                        new_row[next_col] -= intersection
        
        exp_negatives += len(new_row["exp_not"])

        all_confirmed = new_row["exp"] | new_row["phylo"] | new_row["curated"]
        all_nots = new_row["exp_not"] | new_row["phylo_not"] | new_row["curated_not"] | new_row["derived_not"]

        new_row["conditional_not"] = find_conditional_zeros(
            all_confirmed,
            all_nots,
            children_dict,
        )
        
        new_der_neg += len(new_row["derived_not"])
        if len(new_row["derived_not"]) > 0:
            has_derivated_neg.add(row["id"])
        else:
            has_no_derivated_neg.add(row["id"])
        
        for col_name, annots in new_row.items():
            if len(annots) > 0:
                if not col_name in new_annots_dict:
                    new_annots_dict[col_name] = 0
                new_annots_dict[col_name] += len(annots)
        
        new_row["annotation_count"] = len(
            all_confirmed | all_nots
        )
        uniprot2go[row["id"]] = new_row

    print(f"Exp negatives: {exp_negatives}")
    print(f"New derivated negatives: {new_der_neg}")
    print(f"Repeated derivated negatives: {repeated_der_neg}")
    print(f"Conflicting derivated negatives: {conflicting_der_neg}")
    print(f"Number of uniprots with derivated negatives: {len(has_derivated_neg)}")
    print(f"Number of uniprots with no derivated negatives: {len(has_no_derivated_neg)}")

    new_derived_nots = new_annots_dict["derived_not"]
    conflicting_der_nots = 0
    for conflict_name, count in conflicts_dict.items():
        if "found in derived_not" in conflict_name:
            conflicting_der_nots += count
    repetitions_of_der_nots = 0
    for rep_name, count in repetitions_dict.items():
        if "found in derived_not" in rep_name:
            repetitions_of_der_nots += count
    
    total_der_nots = conflicting_der_nots + new_der_neg + repetitions_of_der_nots
    derivated_negatives_confidence = conflicting_der_neg / total_der_nots
    print(f"Total of derived_not including conflicts: {total_der_nots}")
    print(f"Number of new derived_not: {new_derived_nots}")
    print(f"Number of conflicting_der_nots: {conflicting_der_nots}")
    print(f"Number of repetitions_of_der_nots: {repetitions_of_der_nots}")
    print(f"Derivated negatives confidence: {derivated_negatives_confidence}")

    splits = [("train", train_ids), ("test", test_ids)]

    dfs = []

    for split_name, split_ids in splits:
        #local_ids = [x for x in uniprot2go.keys() if x in split_ids]
        local_ids = split_ids
        n_proteins = len(split_ids)
        df_dict = {
            "id": split_ids,
        }

        for strat_name, strat_vals in EVIDENCE_REP_STRATEGIES.items():
            print(f"Creating {strat_name} y matrix for {split_name}")
            if "classic" in strat_name:
                y = np.zeros((n_proteins, len(goids)))
            else:
                y = np.full((n_proteins, len(goids)), np.nan)

            for i, uniprot in tqdm(enumerate(split_ids), desc="Populating y matrix"):
                if uniprot in uniprot2go:
                    protein_annots_by_type = uniprot2go[uniprot]

                    for j, goid in enumerate(goids):
                        annot_type = None
                        for t in evicols_to_use:
                            if goid in protein_annots_by_type[t]:
                                annot_type = t
                                break
                        if annot_type is not None:
                            val = strat_vals[annot_type]
                            if val is not None:
                                y[i, j] = val

            df_dict["y_" + strat_name] = y

            print(y)
            print(y.shape)

            print(f"Counting positive and negative density:")
            show_y_density(y)
            # Look for samples with zero annotations
            zero_annotations = np.where(np.sum(y != np.nan, axis=1) == 0)[0]
            print(f"Number of samples with zero annotations: {len(zero_annotations)}")

        output_parquet_path = f"{sampling_prefix}-evi_exp-{ont_name}-{split_name}_y.parquet"
        targets_list_path = output_parquet_path + ".targets.txt"

        df = pl.DataFrame(df_dict)
        df.write_parquet(output_parquet_path)

        with open(targets_list_path, "w") as f:
            for goid in goids:
                f.write(goid + "\n")

if __name__ == "__main__":
    go_ia_path = "input_data/go_ia.tsv"
    strategy = "multi-strategy"
    # python bin/make_dataset-evidence_exploration.py [ont_name:str] [max_proteins:int] [n_targets:int] [min_annotations:int]

    n_targets = int(sys.argv[1])
    min_annotations = int(sys.argv[2])

    sampling_prefix = f"outputs/n_ont_target={n_targets}-min_proteins={min_annotations}"
    test_path = f"{sampling_prefix}.test_set.txt"
    train_path = f"{sampling_prefix}.train_set.txt"
    deeploc_targets_path = sampling_prefix + ".deeploc.targets.txt"

    required_paths = [test_path, train_path, deeploc_targets_path]

    if not all([os.path.exists(path) for path in required_paths]):
        split_traintest(n_targets, min_annotations)

    targets_by_ont = {
        ont: [go.strip() for go in open(sampling_prefix + "." + ont + ".targets.txt")]
        for ont in ["mf", "cc", "bp"]
    }
    

    test_ids = [l.strip() for l in open(test_path)]
    train_ids = [l.strip() for l in open(train_path)]

    used_uniprots_set = set(test_ids) | set(train_ids)

    for ont in ["mf", "cc", "bp"]:
        make_dataset_for_ont(ont, n_targets, min_annotations, sampling_prefix, 
            test_ids, train_ids, targets_by_ont, used_uniprots_set)