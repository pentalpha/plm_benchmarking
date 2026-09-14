#!/bin/bash
#SBATCH --partition=gpu-8-v100  
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=42G    
#SBATCH --time=10:00:00          
#SBATCH --job-name=plm_bench
#SBATCH --output=logs/%x_%j.out

# 1. Carrega o módulo CUDA DENTRO do nó computacional (GPU)
module load compilers/nvidia/cuda/12.6

# Initialize conda and activate the Py-Boost environment
eval "$(conda shell.bash hook)"
conda activate pyboost

# ==========================================================
# DIAGNÓSTICO DO NÓ COMPUTACIONAL (Impresso no arquivo .out)
# ==========================================================
echo "=== Diagnóstico de Ambiente no Nó Computacional ==="
echo "Nó alocado: $HOSTNAME"
nvcc --version
nvidia-smi
echo "==================================================="

# Arguments passed by the launcher script
N_TARGETS=$1
MIN_ANNOTATIONS=$2
MAX_TRAIN_PROTEINS=$3
PARQUET_PREFIX=$4
OUT_DIR=$5
N_COMBINATIONS=$6

echo "=========================================================="
echo "Starting GPU Experiment: $PARQUET_PREFIX"
echo "Output Directory: $OUT_DIR"
echo "Number of Combinations: $N_COMBINATIONS"
echo "=========================================================="

python -u bin/plm_benchmarking-optimize.py \
    $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    $OUT_DIR $PARQUET_PREFIX $N_COMBINATIONS