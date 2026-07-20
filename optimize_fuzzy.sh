N_TESTS=250
gbdt_params=outputs/metaparameters/comb_2.json

mkdir -p outputs/fuzzy_optimization_results-all_onts

inputs_dir_cc=outputs/emb_tests/cc-mix-12000-proteins-42-labels-min120/emb.ankh_base_mean/processed_inputs
test_name_cc=outputs/emb_tests/cc-mix-12000-proteins-42-labels-min120/emb.ankh_base_mean/fuzzy_optimization
python bin/optimize_fuzzy.py $gbdt_params $inputs_dir_cc $test_name_cc $N_TESTS

inputs_dir=outputs/emb_tests/mf-mix-12000-proteins-42-labels-min120/emb.ankh_base_mean/processed_inputs
test_name=outputs/emb_tests/mf-mix-12000-proteins-42-labels-min120/emb.ankh_base_mean/fuzzy_optimization
python bin/optimize_fuzzy.py $gbdt_params $inputs_dir $test_name $N_TESTS

inputs_dir_bp=outputs/emb_tests/bp-mix-12000-proteins-42-labels-min120/emb.ankh_base_mean/processed_inputs
test_name_bp=outputs/emb_tests/bp-mix-12000-proteins-42-labels-min120/emb.ankh_base_mean/fuzzy_optimization
python bin/optimize_fuzzy.py $gbdt_params $inputs_dir_bp $test_name_bp $N_TESTS

cp $test_name/statistics.json outputs/fuzzy_optimization_results-all_onts/mf_statistics.json
cp $test_name/statistics.tsv outputs/fuzzy_optimization_results-all_onts/mf_statistics.tsv
cp $test_name_bp/statistics.json "outputs/fuzzy_optimization_results-all_onts/bp_statistics.json"
cp $test_name_bp/statistics.tsv outputs/fuzzy_optimization_results-all_onts/bp_statistics.tsv
cp $test_name_cc/statistics.json outputs/fuzzy_optimization_results-all_onts/cc_statistics.json
cp $test_name_cc/statistics.tsv outputs/fuzzy_optimization_results-all_onts/cc_statistics.tsv

cd outputs/fuzzy_optimization_results-all_onts/
awk 'FNR==1 && NR!=1 {next} {print}' *_statistics.tsv > combined_statistics.tsv
cd ../..
