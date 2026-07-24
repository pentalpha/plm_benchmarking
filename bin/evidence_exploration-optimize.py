import sys

from pddb_lib.gene_ontology import ONTOLOGIES_SHORT
from pddb_lib.parsing import load_data_optimized

if __name__ == "__main__":
    go_ia_path = "input_data/go_ia.tsv"

    n_targets = int(sys.argv[1])
    min_annotations = int(sys.argv[2])
    max_train_proteins = int(sys.argv[3]) #downsampling after loading
    y_dataset_name = sys.argv[4] #Evidence rep. strategy
    test_dir = sys.argv[5]
    feature_descs = sys.argv[6:]

    sampling_prefix = f"outputs/n_ont_target={n_targets}-min_proteins={min_annotations}"
    test_path = f"{sampling_prefix}.test_set.txt"
    train_path = f"{sampling_prefix}.train_set.txt"

    datasets_by_ont = {
        ont: {
            "train_y": f"{sampling_prefix}-evi_exp-{ont}-train_y.parquet",
            "test_y": f"{sampling_prefix}-evi_exp-{ont}-test_y.parquet",
        }
        for ont in ONTOLOGIES_SHORT.keys()
    }

    #Sample and save as np, if not done it yet
    #Generate parameters options
    #Train over ones not used yet
    #Store results