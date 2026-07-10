import polars as pl
from tqdm import tqdm

emb_intepro_path = "data/emb.interpro_autoencoded.parquet"
emb_taxid_path = "data/emb.taxid_autoencoded.parquet"

mf_path = "data/go.mf.parquet"

uniprots_with_interpro = pl.read_parquet(emb_intepro_path)["id"].to_list()
uniprots_with_taxid = pl.read_parquet(emb_taxid_path)["id"].to_list()

both_embs = list(set(uniprots_with_interpro) & set(uniprots_with_taxid))

print(f"Number of common uniprots with both embeddings: {len(both_embs)}")
annots_by_goid = {}
mf_parquet = pl.read_parquet(mf_path)
mf_parquet = mf_parquet.filter(pl.col("id").is_in(both_embs))
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
goids_sorted = [(goid, count) for goid, count in goids_sorted if count > 40]

print(f"Number of GOIDs with more than 40 annotations: {len(goids_sorted)}")

with open("data/sorted_goids.tsv", "w") as f:
    f.write("goid\tannotation_count\n")
    for goid, count in goids_sorted:
        f.write(goid + "\t" + str(count) + "\n")
