N_PROT=24020
N_LABELS=42
MIN_ANNOTS=40
N_TESTS=20

fuzzy_parquet_name="outputs/mf-fuzzy-$N_PROT-proteins-$N_LABELS-labels-min$MIN_ANNOTS.parquet"
classic_parquet_name="outputs/mf-classic-$N_PROT-proteins-$N_LABELS-labels-min$MIN_ANNOTS.parquet"
u900_parquet_name="outputs/mf-u900-$N_PROT-proteins-$N_LABELS-labels-min$MIN_ANNOTS.parquet"

python bin/sample_targets.py input_data/go.mf.parquet $N_PROT $N_LABELS $MIN_ANNOTS \
    $classic_parquet_name \
    classic > ${classic_parquet_name}.log

python bin/sample_targets.py input_data/go.mf.parquet $N_PROT $N_LABELS $MIN_ANNOTS \
    $fuzzy_parquet_name \
    fuzzy > ${fuzzy_parquet_name}.log

python bin/sample_targets.py input_data/go.mf.parquet $N_PROT $N_LABELS $MIN_ANNOTS \
    $u900_parquet_name \
    u900 > ${u900_parquet_name}.log

# python test_embedding.py <n_tests> <targets_path> [True | False] <[<embedding_path>:col1,col2,...],...>

python bin/test_embedding.py $N_TESTS $fuzzy_parquet_name True input_data/emb.interpro_autoencoded.parquet:emb input_data/emb.taxid_autoencoded.parquet:emb
python bin/test_embedding.py $N_TESTS $u900_parquet_name True input_data/emb.interpro_autoencoded.parquet:emb input_data/emb.taxid_autoencoded.parquet:emb
python bin/test_embedding.py $N_TESTS $classic_parquet_name True input_data/emb.interpro_autoencoded.parquet:emb input_data/emb.taxid_autoencoded.parquet:emb
