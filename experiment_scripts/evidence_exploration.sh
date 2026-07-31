#!/bin/bash
N_COMBINATIONS=20
N_TARGETS=16
MIN_ANNOTATIONS=360
MAX_TRAIN_PROTEINS=6400
FEATURES_DESC="input_data/emb.ankh_base.parquet:mean"
BASE_RESULTS_DIR="outputs/evidence_exploration_small_test"

#Make datasets
#python bin/evidence_exploration-make_dataset.py $N_TARGETS $MIN_ANNOTATIONS || return 1

#Arguments: bin/evidence_exploration-optimize.py N_TARGETS MIN_ANNOTATIONS MAX_TRAIN_PROTEINS \
#   Y_DATASET_NAME USE_RANDOM_NEGATIVE_SAMPLING TEST_DIR <[<embedding_path>:col1,col2,...],...> 

#Conditional Negatives + RNS (y_conditional_negatives and USE_RANDOM_NEGATIVE_SAMPLING=True)
python bin/evidence_exploration-optimize.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    conditional_negatives True $BASE_RESULTS_DIR/conditional_negatives_rns_test $N_COMBINATIONS "$FEATURES_DESC"

#Launch for classic
#python bin/evidence_exploration-optimize.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
#    classic False $BASE_RESULTS_DIR/classic_test $N_COMBINATIONS "$FEATURES_DESC" || return 1

#Conditional Negatives (y_conditional_negatives)
#python bin/evidence_exploration-optimize.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
#    conditional_negatives False $BASE_RESULTS_DIR/conditional_negatives_test $N_COMBINATIONS "$FEATURES_DESC"


#Soft Labeling (y_soft)
#python bin/evidence_exploration-optimize.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
#    soft False $BASE_RESULTS_DIR/soft_test $N_COMBINATIONS "$FEATURES_DESC"

#Soft Labeling + RNS (y_soft and USE_RANDOM_NEGATIVE_SAMPLING=True)
#python bin/evidence_exploration-optimize.py $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
#    soft True $BASE_RESULTS_DIR/soft_rns_test $N_COMBINATIONS "$FEATURES_DESC"

#shutdown -h now