#!/bin/bash

N_TARGETS = 16
MIN_ANNOTATIONS = 360
MAX_TRAIN_PROTEINS = 6400
FEATURES_DESC = "input_data/emb.ankh_base.parquet:mean"
BASE_RESULTS_DIR = "outputs/evidence_exploration_small_test"

#Make datasets
python bin/evidence_exploration-make_dataset.py $N_TARGETS $MIN_ANNOTATIONS

#Arguments: bin/evidence_exploration-optimize.py N_TARGETS MIN_ANNOTATIONS MAX_TRAIN_PROTEINS \
#   Y_DATASET_NAME USE_RANDOM_NEGATIVE_SAMPLING TEST_DIR <[<embedding_path>:col1,col2,...],...> 
#Launch for classic
python bin/evidence_exploration-optimize.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    y_classic False $BASE_RESULTS_DIR/classic_test "$FEATURES_DESC"

#Conditional Negatives (y_conditional_negatives)
python bin/evidence_exploration-optimize.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    y_conditional_negatives False $BASE_RESULTS_DIR/conditional_negatives_test "$FEATURES_DESC"

#Conditional Negatives + RNS (y_conditional_negatives and USE_RANDOM_NEGATIVE_SAMPLING=True)
python bin/evidence_exploration-optimize.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    y_conditional_negatives True $BASE_RESULTS_DIR/conditional_negatives_rns_test "$FEATURES_DESC"

#Soft Labeling (y_soft)
python bin/evidence_exploration-optimize.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    y_soft False $BASE_RESULTS_DIR/soft_test "$FEATURES_DESC"

#Soft Labeling + RNS (y_soft and USE_RANDOM_NEGATIVE_SAMPLING=True)
python bin/evidence_exploration-optimize.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    y_soft True $BASE_RESULTS_DIR/soft_rns_test "$FEATURES_DESC"