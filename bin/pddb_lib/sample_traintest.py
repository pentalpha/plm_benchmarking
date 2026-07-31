import sys
from collections import defaultdict
import random
from collections import Counter

import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

from pddb_lib.gene_ontology import (
    create_ontology_dictionaries_full
)

# python bin/sample_traintest.py [n_targets:int] [min_annotations:int]
VAL_PERC = 0.35
go_ia_path = "input_data/go_ia.tsv"
mf_path = "input_data/go.mf.parquet"
cc_path = "input_data/go.cc.parquet"
bp_path = "input_data/go.bp.parquet"
deepgo_path = "input_data/deeploc.parquet"
url_ou_caminho_obo = "input_data/go-basic.obo"

def count_annots_by_goid(ont_parquet, evi_cols, parents_dict, children_dict):
    annots_by_goid = {}
    uniprot_to_goids = defaultdict(set)

    bar = tqdm(total=ont_parquet.height, desc="Expanding uniprot annotations to count")
    for row in ont_parquet.iter_rows(named=True):
        uniprot = row["id"]
        all_gos = set()
        for evi_col in evi_cols:
            items = row[evi_col]
            if len(items) > 0:
                for goid in items:
                    if '_not' in evi_col:
                        expansion_dict = children_dict
                    else:
                        expansion_dict = parents_dict
                    
                    new_ids = set()
                    for goid in [x for x in items if x in expansion_dict]:
                        new_ids.update(expansion_dict[goid])
                    
                    if len(new_ids) > 0:
                        all_gos.update(new_ids)
                    all_gos.add(goid)
            
        uniprot_to_goids[uniprot] = list(all_gos)
        bar.update(1)
    bar.close()

    bar = tqdm(total=len(uniprot_to_goids), desc="Counting annotations per GO")
    for uniprot, goids in uniprot_to_goids.items():
        for goid in goids:
            if not goid in annots_by_goid:
                annots_by_goid[goid] = 0
            annots_by_goid[goid] += 1
        bar.update(1)
    bar.close()
    return annots_by_goid, uniprot_to_goids

def split_traintest(n_ont_target, min_annotations):
    output_prefix = f"outputs/n_ont_target={n_ont_target}-min_proteins={min_annotations}"
    parents_dict, children_dict, _ = create_ontology_dictionaries_full(url_ou_caminho_obo)
    go_ia_dict = {}
    for rawline in open(go_ia_path, "r"):
        goid, ia = rawline.strip().split("\t")
        go_ia_dict[goid] = float(ia)
    print(f"Loaded Inf. Accretion for {len(go_ia_dict)} GO terms")
    mean_ia = sum(go_ia_dict.values()) / len(go_ia_dict)
    print(f"Mean IA: {mean_ia}")

    evi_cols = ["exp", "phylo", "curated"]
    evi_cols += [e + "_not" for e in evi_cols] + ["derived_not"]
    
    # uniprots_with_interpro = pl.read_parquet(emb_intepro_path)["id"].to_list()
    # uniprots_with_taxid = pl.read_parquet(emb_taxid_path)["id"].to_list()
    # both_embs = list(set(uniprots_with_interpro) & set(uniprots_with_taxid))
    # print(f"Number of common uniprots with both embeddings: {len(both_embs)}")

    go_counts = {}
    go_by_uniprot = defaultdict(set)
    for parquet_path, ont_name in [
        (mf_path, "mf"),
        (cc_path, "cc"),
        (bp_path, "bp"),
    ]:
        print("Loading", parquet_path)
        annots_df = pl.read_parquet(parquet_path)
        counts_dict, uniprot_annots = count_annots_by_goid(annots_df, evi_cols, 
            parents_dict, children_dict)
        go_counts[ont_name] = counts_dict
        for uniprot, goids in uniprot_annots.items():
            go_by_uniprot[uniprot].update(goids)

    targets_by_ontology = {}
    for ont, goid_counts in go_counts.items():
        print(f"Processing {ont} ontology")
        print(f"{len(goid_counts)} GO terms")
        goids_sorted = sorted(goid_counts.items(), key=lambda x: x[1], reverse=True)
        goids_sorted = [
            {"goid": goid, "annotation_count": count}
            for goid, count in goids_sorted
            if count > min_annotations
        ]

        print(
            f"Number of GOIDs with more than {min_annotations} annotations: {len(goids_sorted)}"
        )

        goid_counts_df = pd.DataFrame(goids_sorted)
        counts_path = f"outputs/{ont}.sorted_goids.tsv"
        goid_counts_df.to_csv(counts_path, sep="\t", index=False)

        frequent_goids = goid_counts_df["goid"].to_list()
        ias = [go_ia_dict.get(goid, 0.0) for goid in frequent_goids]
        non_zero_ias = [ia for ia in ias if ia > 0]
        print(f"Number of GO terms with IA > 0: {len(non_zero_ias)}")
        print(f"Mean IA of GO terms with IA > 0: {sum(non_zero_ias) / len(non_zero_ias)}")
        goid_counts_df["ia"] = ias
        print(goid_counts_df.head())
        before = len(goid_counts_df)
        goid_counts_df = goid_counts_df[goid_counts_df["ia"] > 0]
        print(f"Removed {before - len(goid_counts_df)} GO terms with IA = 0",
            f"Remaining: {len(goid_counts_df)}")
        before = len(goid_counts_df)
        goid_counts_df = goid_counts_df[
            goid_counts_df["annotation_count"] > min_annotations
        ]
        print(f"Removed {before - len(goid_counts_df)} GO terms with less than {min_annotations} annotations",
            f"Remaining: {len(goid_counts_df)}")
        # Multiply frequency with ia
        goid_counts_df["weighted_count"] = (
            goid_counts_df["annotation_count"] * goid_counts_df["ia"]
        )
        print(goid_counts_df.head())
        goid_counts_df = goid_counts_df.sort_values("weighted_count", ascending=False)
        print(goid_counts_df.head())
        # Select targets
        all_goids = goid_counts_df["goid"].to_list()
        indices = np.linspace(0, len(all_goids) - 1, num=n_ont_target, dtype=int)
        goids = [all_goids[i] for i in indices]
        print(goids)
        goid_counts_subsample = goid_counts_df.iloc[indices]
        print(goid_counts_subsample)

        go_ids_set = set(goids)
        targets_by_ontology[ont] = goids
        targets_path = output_prefix + "." + ont + ".targets.txt"

        with open(targets_path, "w") as f:
            for goid in goids:
                f.write(goid + "\n")

    deeploc_df = pl.read_parquet(deepgo_path)
    deeploc_targets = [c for c in deeploc_df.columns if c != "id"]
    deeploc_targets_path = output_prefix + ".deeploc.targets.txt"
    with open(deeploc_targets_path, "w") as f:
        for goid in deeploc_targets:
            f.write(goid + "\n")

    print(f"DeepLoc targets: {len(deeploc_targets)}")

    uniprots_with_relevant_annotations = set()
    deeploc_proteins = deeploc_df["id"].to_list()
    uniprots_with_relevant_annotations.update(deeploc_proteins)

    all_go_targets = set()
    for ont in targets_by_ontology:
        all_go_targets.update(targets_by_ontology[ont])

    print(f"Starting with {len(uniprots_with_relevant_annotations)} from deeploc")

    for uniprot, goids in go_by_uniprot.items():
        if any([x in all_go_targets for x in goids]):
            uniprots_with_relevant_annotations.add(uniprot)

    print(f"With annotations on GO targets: {len(uniprots_with_relevant_annotations)}")

    shuffled_list = list(uniprots_with_relevant_annotations)
    n_validation = int(len(shuffled_list) * VAL_PERC)

    not_in_validation = [1, 2, 3]
    while len(not_in_validation) > 0:
        min_count = round(min_annotations * VAL_PERC * 0.75)
        print("Shuffling ...")
        random.shuffle(shuffled_list)
        random.shuffle(shuffled_list)
        validation_proteins = shuffled_list[:n_validation]
        train_proteins = shuffled_list[n_validation:]

        all_val_annots = []
        for uniprot in validation_proteins:
            from_targets = go_by_uniprot[uniprot] & all_go_targets
            if len(from_targets) > 0:
                all_val_annots.extend(list(from_targets))
        val_counts = Counter(all_val_annots)
        not_in_validation = [
            goid for goid, count in val_counts.items() if count < min_count
        ]
        print(
            f"GOs with freq less than {min_count} in validation annotations: {len(not_in_validation)}"
        )


    test_path = f"{output_prefix}.test_set.txt"
    with open(test_path, "w") as f:
        for uniprot in validation_proteins:
            f.write(uniprot + "\n")

    train_path = f"{output_prefix}.train_set.txt"
    with open(train_path, "w") as f:
        for uniprot in train_proteins:
            f.write(uniprot + "\n")

    print(f"Validation proteins: {len(validation_proteins)}")
    print(f"Train proteins: {len(train_proteins)}")
    print(f"Total proteins: {len(shuffled_list)}")

def sample_train_proteins_by_inf(ids, y, targets, go_ia_dict, 
        max_proteins):
    #Protein weight = sum(IAs)
    #Reverse sort by weight
    #Take firsts
    weights_vec = [go_ia_dict[go] for go in targets]
    not_nan = ~np.isnan(y)
    weighted_not_nan = not_nan * weights_vec
    row_sums = weighted_not_nan.sum(axis=1)
    weights_by_id = {uniprot: row_sums[index]
        for index, uniprot in enumerate(ids)}

    ids_sorted = sorted(ids, 
        key = lambda uniprot: (weights_by_id[uniprot], uniprot), reverse=True)
    ids_sampled = ids_sorted[:max_proteins]

    return ids_sampled

if __name__ == "__main__":
    n_ont_target = int(sys.argv[1])
    min_annotations = int(sys.argv[2])
    split_traintest(n_ont_target, min_annotations)

