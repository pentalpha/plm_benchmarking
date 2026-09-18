#!/bin/bash
N_COMBINATIONS=40
N_TARGETS=32
MIN_ANNOTATIONS=120
MAX_TRAIN_PROTEINS=80000
BASE_RESULTS_DIR="outputs/remote_results/plm_benchmarking"
EMBS_DIR="/home/pita/fs/data/dimension_db/release_2"
LIGHT_TEMPLATE_PATH="experiment_scripts/plm_benchmarking/gpu_template_local.slurm"
# Create a directory for Slurm log files
mkdir -p logs

#conda run -n pyboost python -u bin/plm_benchmarking-prepare.py $N_COMBINATIONS

# ---------------------------------------------------------
# Define MODELS and Submit GPU Jobs
# ---------------------------------------------------------
declare -a configs=(
    #"amplify_350 $LIGHT_TEMPLATE_PATH"
    #"ankh3_large_nlu $LIGHT_TEMPLATE_PATH"
    "esmc_600 $LIGHT_TEMPLATE_PATH"
    #"amplify_120 $LIGHT_TEMPLATE_PATH"
    #"e1_150 $LIGHT_TEMPLATE_PATH"
    #"ankh3_large $LIGHT_TEMPLATE_PATH"
    #"ankh_base $LIGHT_TEMPLATE_PATH"
    #"ankh_large $LIGHT_TEMPLATE_PATH"
    #"esmc_300 $LIGHT_TEMPLATE_PATH"
    #"e1_600 $LIGHT_TEMPLATE_PATH"
    #"ankh2_large $HEAVY_TEMPLATE_PATH"
    #"esm2_650 $LIGHT_TEMPLATE_PATH"
    #"esm2_150 $LIGHT_TEMPLATE_PATH"
    #"e1_300 $LIGHT_TEMPLATE_PATH"
)

# Loop over the configurations, run a job for each and wait for each completion
for config in "${configs[@]}"; do
    # Read the configuration into variables
    read -r MODEL_NAME TEMPLATE_PATH <<< "$config"
    
    # Create a dynamic, identifiable job name
    JOB_NAME="${MODEL_NAME}"
    PARQUET_PREFIX="$EMBS_DIR/emb.$MODEL_NAME"
    OUT_DIR="$BASE_RESULTS_DIR/$MODEL_NAME"
    RESULT_TSV_PATH="$OUT_DIR/optimized_results_${N_COMBINATIONS}.tsv"

    #Submit only if the result file doesn't exist
    if [ -f "$RESULT_TSV_PATH" ]; then
        echo "Result file already exists: $RESULT_TSV_PATH. Skipping job submission."
        continue
    fi

    #Create result dir
    mkdir -p "$OUT_DIR"

    echo "Running: $JOB_NAME | Output Dir: $OUT_DIR | Parquet Prefix: $PARQUET_PREFIX"
    echo "Template Slurm script: $TEMPLATE_PATH"
    
    # Submit to Slurm
    bash $TEMPLATE_PATH \
        $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
        $PARQUET_PREFIX $OUT_DIR $N_COMBINATIONS
        
    echo "Done -> $JOB_NAME"
done

echo "All experiments have been runned successfully!"