import polars as pl
from tqdm import tqdm
import numpy as np

emb_intepro_path = "data/emb.interpro_autoencoded.parquet"
emb_taxid_path = "data/emb.taxid_autoencoded.parquet"

mf_path = "data/go.mf.parquet"

uniprots_with_interpro = pl.read_parquet(emb_intepro_path)["id"].to_list()
uniprots_with_taxid = pl.read_parquet(emb_taxid_path)["id"].to_list()

both_embs = list(set(uniprots_with_interpro) & set(uniprots_with_taxid))

print(f"Number of common uniprots with both embeddings: {len(both_embs)}")

goid_counts = pl.read_csv("data/sorted_goids.tsv", separator="\t")
# Select 32 targets
goids = goid_counts["goid"].to_list()[:24]
print(goids)

go_ids_set = set(goids)

# List uniprots that have annotations for these goids
mf_parquet = pl.read_parquet(mf_path)
mf_parquet = mf_parquet.filter(pl.col("id").is_in(both_embs))

annotated_uniprots = []
uniprot_positives = {}
uniprot_negatives = {}
for row in tqdm(mf_parquet.iter_rows(named=True), desc="Listing uniprots"):
    all_goids = set(row["exp"] + row["phylo"] + row["negative"])
    if len(all_goids & go_ids_set) > 0:
        annotated_uniprots.append(row["id"])
        uniprot_positives[row["id"]] = row["exp"] + row["phylo"]
        uniprot_negatives[row["id"]] = row["negative"]

print(f"Number of uniprots with annotations: {len(annotated_uniprots)}")

# Sort annotated_uniprots by most annotations
annotated_uniprots.sort(
    key=lambda x: len(uniprot_positives[x]) + len(uniprot_negatives[x]), reverse=True
)

# If more than 30000 uniprots, take top 30000
if len(annotated_uniprots) > 30000:
    annotated_uniprots = annotated_uniprots[:30000]

# Create y matrix
y = np.full((len(annotated_uniprots), len(goids)), np.nan)
for i, uniprot in tqdm(enumerate(annotated_uniprots), desc="Populating y matrix"):
    for j, goid in enumerate(goids):
        if goid in uniprot_positives[uniprot]:
            y[i, j] = 1.0
        elif goid in uniprot_negatives[uniprot]:
            y[i, j] = 0.0

print(y)
print(y.shape)

print(f"Counting positive and negative density:")
n_pos = np.sum(y == 1.0)
n_neg = np.sum(y == 0.0)
n_nan = np.sum(np.isnan(y))

pos_perc = n_pos / (n_pos + n_neg + n_nan)
neg_perc = n_neg / (n_pos + n_neg + n_nan)
nan_perc = n_nan / (n_pos + n_neg + n_nan)

print(f"Positive: {n_pos} ({pos_perc:.2%})")
print(f"Negative: {n_neg} ({neg_perc:.2%})")
print(f"NaN: {n_nan} ({nan_perc:.2%})")

uniprot_to_interpro_emb = {}
uniprot_to_taxid_emb = {}
interpro_parquet = pl.read_parquet(emb_intepro_path)
taxid_parquet = pl.read_parquet(emb_taxid_path)

for row in tqdm(
    interpro_parquet.iter_rows(named=True),
    desc="Mapping uniprots to interpro embeddings",
):
    uniprot_to_interpro_emb[row["id"]] = row["emb"]

for row in tqdm(
    taxid_parquet.iter_rows(named=True), desc="Mapping uniprots to taxid embeddings"
):
    uniprot_to_taxid_emb[row["id"]] = row["emb"]

X = []
for uniprot in tqdm(annotated_uniprots, desc="Concatenating embeddings"):
    X.append(
        np.concatenate(
            [uniprot_to_interpro_emb[uniprot], uniprot_to_taxid_emb[uniprot]]
        )
    )

X = np.asarray(X)
df = pl.DataFrame({"id": annotated_uniprots, "emb": X, "targets": y})
df.write_parquet("data/xy_mf.parquet")
