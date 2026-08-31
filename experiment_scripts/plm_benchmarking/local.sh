#!/bin/bash
N_TARGETS=32
MIN_ANNOTATIONS=120
MAX_TRAIN_PROTEINS=80000
BASE_RESULTS_DIR="outputs/plm_benchmarking"

# Initialize conda and activate the Py-Boost environment
conda activate pyboost

#python bin/plm_benchmarking-train.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
#    $BASE_RESULTS_DIR/e1_300 input_data/emb.e1_300
python bin/plm_benchmarking-train.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    $BASE_RESULTS_DIR/e1_600 input_data/emb.e1_600
#python bin/plm_benchmarking-train.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
#    $BASE_RESULTS_DIR/ankh_base input_data/emb.ankh_base
#python bin/plm_benchmarking-train.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
#    $BASE_RESULTS_DIR/ankh_large input_data/emb.ankh_large