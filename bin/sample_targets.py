import sys
import polars as pl
from collections import defaultdict
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
derivated_negatives_tsv = "input_data/vesztrocy_and_dessimoz_2020-derived_negatives.tsv"
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

all_ont_goids = [goid for goid in annots_by_goid.keys()]
# an	GO_ID	infer_from_an	type	fam	uniprot
derivated_negatives_df = pd.read_csv(derivated_negatives_tsv, sep="\t")
derivated_negatives_df = derivated_negatives_df[derivated_negatives_df["type"] == -1]


# fill up GO_ID (only digits) so that 140096 -> GO:0140096 and 977 -> GO:0000977
def add_zeros(goid, min_len: int = 7):
    return "GO:" + str(goid).zfill(min_len)


derivated_negatives_df["GO_ID"] = derivated_negatives_df["GO_ID"].apply(add_zeros)
derivated_negatives_df = derivated_negatives_df[
    derivated_negatives_df["GO_ID"].isin(all_ont_goids)
]
print(derivated_negatives_df.head())
derivated_neg_annots = defaultdict(set)
derivated_negatives_df = derivated_negatives_df[["uniprot", "GO_ID"]]
# Convert to tuples:
uniprot_and_goid = derivated_negatives_df.itertuples(index=False, name=None)
for uniprot, goid in tqdm(uniprot_and_goid, desc="Listing derivated negatives"):
    derivated_neg_annots[uniprot].add(goid)
    annots_by_goid[goid] += 1

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


new_der_neg = 0
repeated_der_neg = 0
conflicting_der_neg = 0
exp_negatives = 0
has_derivated_neg = set()
has_no_derivated_neg = set()

uniprot2go = {}
for row in tqdm(mf_parquet.iter_rows(named=True), desc="Listing uniprot annotations"):
    new_row = {
        "exp": set(row["exp"]),
        "phylo": set(row["phylo"]),
        "negative": set(row["negative"]),
    }
    exp_negatives += len(new_row["negative"])
    # add derivated negatives to negatives
    if row["id"] in derivated_neg_annots:
        # non_conflicting_der_neg = (
        #    derivated_neg_annots[row["id"]] - new_row["exp"] - new_row["phylo"]
        # )
        # n_conflicting_der_neg = len(derivated_neg_annots[row["id"]]) - len(
        #    non_conflicting_der_neg
        # )
        # repeated = derivated_neg_annots[row["id"]] & new_row["negative"]
        # new_non_conflicting = non_conflicting_der_neg - new_row["negative"]

        # new_der_neg += len(new_non_conflicting)
        # repeated_der_neg += len(repeated)
        # conflicting_der_neg += n_conflicting_der_neg
        # has_derivated_neg.add(row["id"])

        new_expanded = set()
        for goid in derivated_neg_annots[row["id"]]:
            if goid in children_dict:
                new_expanded.update(children_dict[goid])
        new_expanded.update(derivated_neg_annots[row["id"]])

        non_conflicting_der_neg = new_expanded - new_row["exp"] - new_row["phylo"]
        n_conflicting_der_neg = len(new_expanded) - len(non_conflicting_der_neg)
        repeated = new_expanded & new_row["negative"]
        new_non_conflicting = non_conflicting_der_neg - new_row["negative"]

        repeated_der_neg += len(repeated)
        conflicting_der_neg += n_conflicting_der_neg
        if len(new_non_conflicting) > 0:
            new_der_neg += len(new_non_conflicting)
            has_derivated_neg.add(row["id"])

            new_row["derivated_negative"] = new_non_conflicting
        else:
            new_row["derivated_negative"] = set()
            has_no_derivated_neg.add(row["id"])
    else:
        new_row["derivated_negative"] = set()
        has_no_derivated_neg.add(row["id"])

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
        new_row["exp"]
        | new_row["phylo"]
        | new_row["negative"]
        | new_row["derivated_negative"]
    )
    uniprot2go[row["id"]] = new_row

print(f"Exp negatives: {exp_negatives}")
print(f"New derivated negatives: {new_der_neg}")
print(f"Repeated derivated negatives: {repeated_der_neg}")
print(f"Conflicting derivated negatives: {conflicting_der_neg}")
print(f"Number of uniprots with derivated negatives: {len(has_derivated_neg)}")
print(f"Number of uniprots with no derivated negatives: {len(has_no_derivated_neg)}")

derivated_negatives_confidence = conflicting_der_neg / (
    new_der_neg + conflicting_der_neg + repeated_der_neg
)
print(f"Derivated negatives confidence: {derivated_negatives_confidence}")

filling_strategies = {
    "classic": {
        "exp": 1.0,
        "phylo": 1.0,
        "negative": None,
        "derivated_negative": None,
        "conditional_negative": None,
        "conditional_negative_phylo": None,
        "conditional_negative_upwards": None,
    },
    "open_world_assumption": {
        "exp": 1.0,
        "phylo": 1.0,
        "negative": 0.0,
        "derivated_negative": 0.0,
        "conditional_negative": None,
        "conditional_negative_phylo": None,
        "conditional_negative_upwards": None,
    },
    "conditional_negatives": {
        "exp": 1.0,
        "phylo": 1.0,
        "negative": 0.0,
        "derivated_negative": 0.0,
        "conditional_negative": 0.0,
        "conditional_negative_phylo": 0.0,
        "conditional_negative_upwards": None,
    },
    "fuzzy": {
        "exp": 1.0,
        "phylo": 0.9,
        "negative": 0.0,
        "derivated_negative": 0.025,
        "conditional_negative": 0.15,
        "conditional_negative_phylo": 0.15,
        "conditional_negative_upwards": None,
    },
}

print(filling_strategies.keys())

"""# make conditional upwards versions
for str_name in list(filling_strategies.keys()):
    str_vals = filling_strategies[str_name]
    if str_vals["conditional_negative"] is not None:
        new_str = {k: v for k, v in str_vals.items()}
        new_str["conditional_negative_upwards"] = 0.15
        new_str_name = str_name + "_upwards"
        filling_strategies[new_str_name] = new_str"""

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


print(f"Number of uniprots with annotations: {len(annotated_uniprots)}")

# If more than max_proteins uniprots, take top max_proteins
max_samples = max_proteins
if len(annotated_uniprots) > max_samples:
    annotated_uniprots = annotated_uniprots[:max_samples]

    new_has_derivated_neg = has_derivated_neg & set(annotated_uniprots)
    new_has_no_derivated_neg = has_no_derivated_neg & set(annotated_uniprots)

    print(f"Number of uniprots with derivated negatives: {len(new_has_derivated_neg)}")
    print(
        f"Number of uniprots with no derivated negatives: {len(new_has_no_derivated_neg)}"
    )
    perc_derivated_neg = len(new_has_derivated_neg) / max_samples
    perc_no_derivated_neg = len(new_has_no_derivated_neg) / max_samples

    print(f"Percentage of uniprots with derivated negatives: {perc_derivated_neg}")
    print(
        f"Percentage of uniprots with no derivated negatives: {perc_no_derivated_neg}"
    )

df_dict = {
    "id": annotated_uniprots,
}
annot_types_by_relevance = [
    "exp",
    "phylo",
    "negative",
    "derivated_negative",
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

    has_derivated_neg_local = 0
    has_derivated_neg_local_in_labelset = 0
    has_no_derivated_neg_local = 0

    for i, uniprot in tqdm(enumerate(annotated_uniprots), desc="Populating y matrix"):
        protein_annots_by_type = uniprot2go[uniprot]
        if len(protein_annots_by_type["derivated_negative"]) > 0:
            has_derivated_neg_local += 1
            subset = set(goids) & protein_annots_by_type["derivated_negative"]
            if len(subset) > 0:
                has_derivated_neg_local_in_labelset += 1

        if len(protein_annots_by_type["derivated_negative"]) > 0:
            has_no_derivated_neg_local += 1

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

    print(f"Number of uniprots with derivated negatives: {has_derivated_neg_local}")
    print(
        f"Number of uniprots with derivated negatives in labelset: {has_derivated_neg_local_in_labelset}"
    )
    print(
        f"Number of uniprots with no derivated negatives: {has_no_derivated_neg_local}"
    )
    perc1 = has_derivated_neg_local_in_labelset / len(annotated_uniprots)
    perc2 = has_no_derivated_neg_local / len(annotated_uniprots)
    print(f"Percentage of uniprots with derivated negatives in labelset: {perc1}")
    print(f"Percentage of uniprots with no derivated negatives: {perc2}")

    # Look for samples with zero annotations
    zero_annotations = np.where(np.sum(y != np.nan, axis=1) == 0)[0]
    print(f"Number of samples with zero annotations: {len(zero_annotations)}")

df = pl.DataFrame(df_dict)
df.write_parquet(output_path)

with open(targets_list_path, "w") as f:
    for goid in goids:
        f.write(goid + "\n")
