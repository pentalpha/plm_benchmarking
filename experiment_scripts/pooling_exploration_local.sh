#!/bin/bash
N_TARGETS=32
MIN_ANNOTATIONS=120
MAX_TRAIN_PROTEINS=80000
BASE_FEATURE_NAME="input_data/emb.ankh_base_incomplete_"
BASE_RESULTS_DIR="outputs/pooling_exploration"

# Initialize conda and activate the Py-Boost environment
conda activate pyboost

#python bin/pooling_exploration-make_dataset.py $N_TARGETS $MIN_ANNOTATIONS

python bin/pooling_exploration-train.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    $BASE_RESULTS_DIR/parti_pooling_test "${BASE_FEATURE_NAME}parti.parquet:parti"
python bin/pooling_exploration-train.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    $BASE_RESULTS_DIR/mean_pooling_test "${BASE_FEATURE_NAME}mean.parquet:mean"
python bin/pooling_exploration-train.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    $BASE_RESULTS_DIR/softmax_pooling_test "${BASE_FEATURE_NAME}softmax.parquet:softmax"
python bin/pooling_exploration-train.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    $BASE_RESULTS_DIR/norm_pooling_test "${BASE_FEATURE_NAME}norm.parquet:norm"
python bin/pooling_exploration-train.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    $BASE_RESULTS_DIR/max_pooling_test "${BASE_FEATURE_NAME}max.parquet:max"

python bin/pooling_exploration-train.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    $BASE_RESULTS_DIR/parti_pooling_test "${BASE_FEATURE_NAME}parti.parquet:parti" "${BASE_FEATURE_NAME}max.parquet:max" "${BASE_FEATURE_NAME}std.parquet:std"
python bin/pooling_exploration-train.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    $BASE_RESULTS_DIR/mean_pooling_test "${BASE_FEATURE_NAME}mean.parquet:mean" "${BASE_FEATURE_NAME}max.parquet:max" "${BASE_FEATURE_NAME}std.parquet:std"
python bin/pooling_exploration-train.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    $BASE_RESULTS_DIR/softmax_pooling_test "${BASE_FEATURE_NAME}softmax.parquet:softmax" "${BASE_FEATURE_NAME}max.parquet:max" "${BASE_FEATURE_NAME}std.parquet:std"
python bin/pooling_exploration-train.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    $BASE_RESULTS_DIR/norm_pooling_test "${BASE_FEATURE_NAME}norm.parquet:norm" "${BASE_FEATURE_NAME}max.parquet:max" "${BASE_FEATURE_NAME}std.parquet:std"
