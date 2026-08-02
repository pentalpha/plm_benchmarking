#!/bin/bash
N_COMBINATIONS=5
N_TARGETS=64
MIN_ANNOTATIONS=120
MAX_TRAIN_PROTEINS=24000
FEATURES_DESC="input_data/emb.ankh_base.parquet:mean"
BASE_RESULTS_DIR="outputs/evidence_exploration_npadtest"

# Create a directory for Slurm log files
mkdir -p logs

# ---------------------------------------------------------
# Step 1: Run Dataset Maker (CPU) on the current node
# ---------------------------------------------------------
echo "Generating dataset (CPU)..."
eval "$(conda shell.bash hook)"
conda activate pyboost

python -u bin/evidence_exploration-make_dataset.py $N_TARGETS $MIN_ANNOTATIONS
if [ $? -ne 0 ]; then
    echo "Error: Dataset creation failed. Aborting GPU job submissions."
    exit 1
fi

echo "Dataset ready! Submitting GPU jobs to Slurm..."

# ---------------------------------------------------------
# Step 2: Define Configurations and Submit GPU Jobs
# ---------------------------------------------------------
# Format: "STRATEGY USE_RNS OUT_DIR"
declare -a configs=(
    "conditional_negatives True $BASE_RESULTS_DIR/conditional_negatives_rns_test"
    "classic False $BASE_RESULTS_DIR/classic_test"
    "conditional_negatives False $BASE_RESULTS_DIR/conditional_negatives_test"
    "soft False $BASE_RESULTS_DIR/soft_test"
    "soft True $BASE_RESULTS_DIR/soft_rns_test"
)

# Loop over the configurations and submit a job for each
for config in "${configs[@]}"; do
    # Read the configuration into variables
    read -r STRATEGY USE_RNS OUT_DIR <<< "$config"
    
    # Create a dynamic, identifiable job name
    JOB_NAME="pb_${STRATEGY}_rns${USE_RNS}"
    
    # Submit to Slurm
    sbatch --job-name="$JOB_NAME" experiment_scripts/evidence_exploration-gpu_template.slurm \
        $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
        $STRATEGY $USE_RNS $OUT_DIR $N_COMBINATIONS "$FEATURES_DESC"
        
    echo "Submitted -> $JOB_NAME"
done

echo "All 5 experiments have been queued successfully!"