#!/bin/bash
# ==============================================================================
# Final Experiment: Broad Soft Label Optimization
# Explores ~100 combinations of soft label values in [0.01, 0.99]
# using Soft Labeling + RNS with the best parameters per ontology (MF, BP, CC)
# ==============================================================================

N_COMBINATIONS=100
N_TARGETS=32
MIN_ANNOTATIONS=120
MAX_TRAIN_PROTEINS=80000
FEATURES_DESC="input_data/emb.ankh_base_mean.parquet:mean"
INPUT_TSV="outputs/remote_results/evidence_exploration_npad/evi_rep-top_performance.tsv"
RESULTS_DIR="outputs/evidence_exploration_npad/soft_optimization_test"

# Initialize conda and activate the Py-Boost environment
eval "$(conda shell.bash hook)"
conda activate pyboost

python -u bin/evidence_exploration-6-soft_optimization.py \
    --input_file "$INPUT_TSV" \
    --test_dir "$RESULTS_DIR" \
    --n_combinations "$N_COMBINATIONS" \
    --n_targets "$N_TARGETS" \
    --min_annotations "$MIN_ANNOTATIONS" \
    --max_train_proteins "$MAX_TRAIN_PROTEINS" \
    --features_desc "$FEATURES_DESC"
