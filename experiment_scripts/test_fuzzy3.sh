N_PROT=26000
N_LABELS=42
MIN_ANNOTS=120
N_TESTS=16

parquet_name="outputs/mf-mix-$N_PROT-proteins-$N_LABELS-labels-min$MIN_ANNOTS.parquet"
parquet_name_cc="outputs/cc-mix-$N_PROT-proteins-$N_LABELS-labels-min$MIN_ANNOTS.parquet"
parquet_name_bp="outputs/bp-mix-$N_PROT-proteins-$N_LABELS-labels-min$MIN_ANNOTS.parquet"


#python bin/sample_targets.py input_data/go.mf.parquet $N_PROT $N_LABELS $MIN_ANNOTS \
#    $parquet_name > $parquet_name.log 2>&1
#python bin/sample_targets.py input_data/go.cc.parquet $N_PROT $N_LABELS $MIN_ANNOTS \
#    $parquet_name_cc > $parquet_name_cc.log 2>&1
#python bin/sample_targets.py input_data/go.bp.parquet $N_PROT $N_LABELS $MIN_ANNOTS \
#    $parquet_name_bp > $parquet_name_bp.log 2>&1

# python test_embedding.py <n_tests> <targets_path> [True | False] <[<embedding_path>:col1,col2,...],...>

#python bin/test_embedding.py $N_TESTS $parquet_name True input_data/emb.interpro_autoencoded.parquet:emb input_data/emb.taxid_autoencoded.parquet:emb
#python bin/test_embedding.py $N_TESTS $parquet_name_cc True input_data/emb.interpro_autoencoded.parquet:emb input_data/emb.taxid_autoencoded.parquet:emb
python bin/test_embedding.py $N_TESTS $parquet_name_bp True input_data/emb.interpro_autoencoded.parquet:emb input_data/emb.taxid_autoencoded.parquet:emb
