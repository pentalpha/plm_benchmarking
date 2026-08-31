#!/bin/bash
N_COMBINATIONS=8
N_TARGETS=32
MIN_ANNOTATIONS=120
MAX_TRAIN_PROTEINS=80000
BASE_RESULTS_DIR="outputs/plm_benchmarking"
EMBS_DIR="/home/pdaasobrinho/scratch/release_2"
TEMPLATE_PATH="experiment_scripts/plm_benchmarking/gpu_template.slurm"
# Create a directory for Slurm log files
mkdir -p logs

conda run -n pyboost python -u bin/plm_benchmarking-prepare.py $N_COMBINATIONS

# ---------------------------------------------------------
# Define MODELS and Submit GPU Jobs
# ---------------------------------------------------------
declare -a configs=(
    #"esm2_150"
    "ankh_base"
    "e1_300"
    "esm2_650"
    "ankh_large"
    "e1_600"
    "ankh2_large"
    "ankh3_large"
)

# Loop over the configurations and submit a job for each
for config in "${configs[@]}"; do
    # Read the configuration into variables
    read -r MODEL_NAME <<< "$config"
    
    # Create a dynamic, identifiable job name
    JOB_NAME="pb_${MODEL_NAME}"
    PARQUET_PREFIX="$EMBS_DIR/emb.$MODEL_NAME"
    OUT_DIR="$BASE_RESULTS_DIR/$MODEL_NAME"
    RESULT_TSV_PATH="$OUT_DIR/optimized_results.tsv"

    #Submit only if the result file doesn't exist
    if [ -f "$RESULT_TSV_PATH" ]; then
        echo "Result file already exists: $RESULT_TSV_PATH. Skipping job submission."
        continue
    fi

    #Create result dir
    mkdir -p "$OUT_DIR"

    echo "Submitting job: $JOB_NAME | Output Dir: $OUT_DIR | Parquet Prefix: $PARQUET_PREFIX"
    echo "Template Slurm script: $TEMPLATE_PATH"
    
    # Submit to Slurm
    sbatch --job-name="$JOB_NAME" $TEMPLATE_PATH \
        $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
        $PARQUET_PREFIX $OUT_DIR $N_COMBINATIONS
        
    echo "Submitted -> $JOB_NAME"
done

echo "All experiments have been queued successfully!"