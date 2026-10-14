import sys
import os

import polars as pl
from tqdm import tqdm
import numpy as np

def make_dataset_for_deeploc(n_targets, min_annotations, sampling_prefix, 
                        test_ids, train_ids, deeploc_targets_path,
                        used_uniprots_set):
    annots_parquet_path = "input_data/deeploc.parquet"
    deeploc_df = pl.read_parquet(annots_parquet_path)
    locs = []
    for rawline in open(deeploc_targets_path):
        locs.append(rawline.strip())
    locs_set = set(locs)
    annots_parquet = deeploc_df.filter(pl.col("id").is_in(used_uniprots_set))

    uniprot2go = {}
    for row in tqdm(annots_parquet.iter_rows(named=True), desc="Listing uniprot annotations"):
        bool_vec = [row[loc] for loc in locs]
        uniprot2go[row['id']] = [1.0 if b else 0.0 for b in bool_vec]
    
    ids_with_deeploc = set(uniprot2go.keys())

    train_ids = [x for x in train_ids if x in ids_with_deeploc]
    test_ids = [x for x in test_ids if x in ids_with_deeploc]

    splits = [("train", train_ids), 
        ("test", test_ids)]

    dfs = []

    for split_name, split_ids in splits:
        #local_ids = [x for x in uniprot2go.keys() if x in split_ids]
        local_ids = split_ids
        n_proteins = len(split_ids)
        lines = [np.array(uniprot2go[uniprot]) for uniprot in split_ids]
        
        df_dict = {
            "id": split_ids,
            "y": np.asarray(lines)
        }

        output_parquet_path = f"{sampling_prefix}-evi_exp-deeploc-{split_name}_y.parquet"
        targets_list_path = output_parquet_path + ".targets.txt"

        df = pl.DataFrame(df_dict)
        df.write_parquet(output_parquet_path)

        with open(targets_list_path, "w") as f:
            for loc in locs:
                f.write(loc + "\n")

if __name__ == "__main__":
    n_targets = int(sys.argv[1])
    min_annotations = int(sys.argv[2])
    output_dir = 'outputs/'

    sampling_prefix = f"{output_dir}/n_ont_target={n_targets}-min_proteins={min_annotations}"
    test_path = f"{sampling_prefix}.test_set.txt"
    train_path = f"{sampling_prefix}.train_set.txt"
    deeploc_targets_path = sampling_prefix + ".deeploc.targets.txt"

    required_paths = [test_path, train_path, deeploc_targets_path]

    assert all([os.path.exists(path) for path in required_paths])
    
    test_ids = [l.strip() for l in open(test_path)]
    train_ids = [l.strip() for l in open(train_path)]

    used_uniprots_set = set(test_ids) | set(train_ids)

    make_dataset_for_deeploc(n_targets, min_annotations, sampling_prefix, 
        test_ids, train_ids, deeploc_targets_path, used_uniprots_set)