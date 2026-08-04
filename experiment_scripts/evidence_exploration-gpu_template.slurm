#!/bin/bash
#SBATCH --partition=gpu-8-v100  
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8      # Pede 16 núcleos de processamento
#SBATCH --mem=20G    
#SBATCH --time=12:00:00          
#SBATCH --job-name=pb_optimize
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
echo "==================================================="

# Arguments passed by the launcher script
N_TARGETS=$1
MIN_ANNOTATIONS=$2
MAX_TRAIN_PROTEINS=$3
STRATEGY=$4
USE_RNS=$5
OUT_DIR=$6
N_COMBINATIONS=$7
FEATURES_DESC=$8

echo "=========================================================="
echo "Starting GPU Experiment: $STRATEGY | RNS: $USE_RNS"
echo "Output Directory: $OUT_DIR"
echo "=========================================================="

# Use python -u to unbuffer output (ensures Slurm logs update in real-time)
python -u bin/evidence_exploration-optimize.py \
    $N_TARGETS $MIN_ANNOTATIONS $MAX_TRAIN_PROTEINS \
    $STRATEGY $USE_RNS $OUT_DIR $N_COMBINATIONS "$FEATURES_DESC"