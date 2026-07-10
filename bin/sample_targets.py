import sys
import polars as pl
import pandas as pd
from tqdm import tqdm
import numpy as np
import obonet

from fuzzy_ml import (
    create_ontology_dictionaries,
    find_conditional_zeros,
    find_conditional_zeros_inverse,
    show_y_density,
)

go_ia_path = "input_data/go_ia.tsv"

# python bin/sample_targets.py [annots_parquet_path:str] [max_proteins:int] [n_targets:int] [min_annotations:int]

mf_path = sys.argv[1]
max_proteins = int(sys.argv[2])
n_targets = int(sys.argv[3])
min_annotations = int(sys.argv[4])
output_path = sys.argv[5]
strategy = "multi-strategy"
if len(sys.argv) > 6:
    strategy = sys.argv[6]

targets_list_path = output_path + ".targets.txt"
url_ou_caminho_obo = "input_data/go-basic.obo"

# uniprots_with_interpro = pl.read_parquet(emb_intepro_path)["id"].to_list()
# uniprots_with_taxid = pl.read_parquet(emb_taxid_path)["id"].to_list()
# both_embs = list(set(uniprots_with_interpro) & set(uniprots_with_taxid))
# print(f"Number of common uniprots with both embeddings: {len(both_embs)}")

mf_parquet = pl.read_parquet(mf_path)

parents_dict, children_dict = create_ontology_dictionaries(url_ou_caminho_obo)

uniprot2go = {}
for row in tqdm(mf_parquet.iter_rows(named=True), desc="Listing uniprot annotations"):
    new_row = {
        "exp": set(row["exp"]),
        "phylo": set(row["phylo"]),
        "negative": set(row["negative"]),
    }
    new_row["conditional_negative"] = find_conditional_zeros(
        new_row["exp"],
        new_row["negative"],
        children_dict,
    )
    new_row["conditional_negative_phylo"] = find_conditional_zeros(
        new_row["phylo"],
        new_row["negative"],
        children_dict,
    )
    new_row["conditional_negative_upwards"] = find_conditional_zeros_inverse(
        new_row["negative"],
        new_row["exp"] | new_row["phylo"],
        parents_dict,
    )
    new_row["annotation_count"] = len(
        new_row["exp"] | new_row["phylo"] | new_row["negative"]
    )
    uniprot2go[row["id"]] = new_row

filling_strategies = {
    "classic": {
        "exp": 1.0,
        "phylo": 1.0,
        "negative": None,
        "conditional_negative": None,
        "conditional_negative_phylo": None,
        "conditional_negative_upwards": None,
    },
    "conditional_negatives": {
        "exp": 1.0,
        "phylo": 1.0,
        "negative": None,
        "conditional_negative": 0.0,
        "conditional_negative_phylo": 0.0,
        "conditional_negative_upwards": None,
    },
    "fuzzy": {
        "exp": 1.0,
        "phylo": 0.9,
        "negative": 0.0,
        "conditional_negative": 0.15,
        "conditional_negative_phylo": 0.15,
        "conditional_negative_upwards": None,
    },
    "fuzzier": {
        "exp": 1.0,
        "phylo": 0.9,
        "negative": 0.0,
        "conditional_negative": 0.15,
        "conditional_negative_phylo": 0.1,
        "conditional_negative_upwards": None,
    },
}

print(filling_strategies.keys())

# make conditional upwards versions
for str_name in list(filling_strategies.keys()):
    str_vals = filling_strategies[str_name]
    if str_vals["conditional_negative"] is not None:
        new_str = {k: v for k, v in str_vals.items()}
        new_str["conditional_negative_upwards"] = 0.15
        new_str_name = str_name + "_upwards"
        filling_strategies[new_str_name] = new_str

print(filling_strategies.keys())

if strategy != "multi-strategy":
    if strategy in filling_strategies:
        filling_strategies = {strategy: filling_strategies[strategy]}
    else:
        raise ValueError(f"Strategy {strategy} not found.")

print(filling_strategies.keys())

annotated_uniprots = [
    x for x in uniprot2go.keys() if uniprot2go[x]["annotation_count"] > 0
]
annotated_uniprots = sorted(
    annotated_uniprots, key=lambda x: uniprot2go[x]["annotation_count"], reverse=True
)

annots_by_goid = {}

exps = mf_parquet["exp"].to_list()
exps = [x for x in exps if len(x) > 0]
phylos = mf_parquet["phylo"].to_list()
phylos = [x for x in phylos if len(x) > 0]
negatives = mf_parquet["negative"].to_list()
negatives = [x for x in negatives if len(x) > 0]
for goid_lists in [exps, phylos, negatives]:
    for goids in tqdm(goid_lists, desc="Counting annotations"):
        for goid in goids:
            if goid in annots_by_goid:
                annots_by_goid[goid] += 1
            else:
                annots_by_goid[goid] = 1

goids_sorted = sorted(annots_by_goid.items(), key=lambda x: x[1], reverse=True)
goids_sorted = [
    (goid, count) for goid, count in goids_sorted if count > min_annotations
]

print(
    f"Number of GOIDs with more than {min_annotations} annotations: {len(goids_sorted)}"
)

with open("outputs/sorted_goids.tsv", "w") as f:
    f.write("goid\tannotation_count\n")
    for goid, count in goids_sorted:
        f.write(goid + "\t" + str(count) + "\n")

goid_counts = pd.read_csv("outputs/sorted_goids.tsv", sep="\t")
print(goid_counts.head())
go_ia_dict = {}
for rawline in open(go_ia_path, "r"):
    goid, ia = rawline.strip().split("\t")
    go_ia_dict[goid] = float(ia)
# Select targets
goid_counts["ia"] = goid_counts["goid"].apply(lambda x: go_ia_dict.get(x, 0.0))
goid_counts = goid_counts[goid_counts["ia"] > 0]
goid_counts = goid_counts[goid_counts["annotation_count"] > min_annotations]
# Multiply frequency with ia
goid_counts["weighted_count"] = goid_counts["annotation_count"] * goid_counts["ia"]
print(goid_counts.head())
goid_counts = goid_counts.sort_values("weighted_count", ascending=False)
print(goid_counts.head())
# Select targets
all_goids = goid_counts["goid"].to_list()
indices = np.linspace(0, len(all_goids) - 1, num=n_targets, dtype=int)
goids = [all_goids[i] for i in indices]
print(goids)
goid_counts_subsample = goid_counts.iloc[indices]
print(goid_counts_subsample)

go_ids_set = set(goids)

print(f"Number of uniprots with annotations: {len(annotated_uniprots)}")

# If more than max_proteins uniprots, take top max_proteins
max_samples = max_proteins
if len(annotated_uniprots) > max_samples:
    annotated_uniprots = annotated_uniprots[:max_samples]

df_dict = {
    "id": annotated_uniprots,
}
annot_types_by_relevance = [
    "exp",
    "phylo",
    "negative",
    "conditional_negative",
    "conditional_negative_phylo",
    "conditional_negative_upwards",
]

print(filling_strategies.keys())
for str_name, str_vals in filling_strategies.items():
    print(f"Creating {str_name} y matrix")
    if "classic" in str_name:
        y = np.zeros((len(annotated_uniprots), len(goids)))
    else:
        y = np.full((len(annotated_uniprots), len(goids)), np.nan)

    for i, uniprot in tqdm(enumerate(annotated_uniprots), desc="Populating y matrix"):
        protein_annots_by_type = uniprot2go[uniprot]

        for j, goid in enumerate(goids):

            annot_type = None
            for t in annot_types_by_relevance:
                if goid in protein_annots_by_type[t]:
                    annot_type = t
                    break
            if annot_type is not None:
                val = str_vals[annot_type]
                if val is not None:
                    y[i, j] = val

    df_dict["y_" + str_name] = y

    print(y)
    print(y.shape)

    print(f"Counting positive and negative density:")
    show_y_density(y)

    # Look for samples with zero annotations
    zero_annotations = np.where(np.sum(y != np.nan, axis=1) == 0)[0]
    print(f"Number of samples with zero annotations: {len(zero_annotations)}")

df = pl.DataFrame(df_dict)
df.write_parquet(output_path)

with open(targets_list_path, "w") as f:
    for goid in goids:
        f.write(goid + "\n")
