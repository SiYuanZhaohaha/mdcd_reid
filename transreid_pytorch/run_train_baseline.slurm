#!/bin/bash
# Train the TransReID-SSL baseline.
#
# Example:
#   DATA_ROOT=/path/to/data PRETRAIN_PATH=/path/to/vit_small_cfs_lup.pth sbatch run_train1.slurm

#SBATCH --job-name=mdcd_base
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/baseline-%j.out

set -e

DATASET=${DATASET:-mixed_market1501}
CONFIG=${CONFIG:-configs/market/vit_small.yml}
DATA_ROOT=${DATA_ROOT:-../data}
PRETRAIN_PATH=${PRETRAIN_PATH:-../pretrained/vit_small_cfs_lup.pth}
OUTPUT_DIR=${OUTPUT_DIR:-./logs/baseline/${DATASET}}
DEVICE_ID=${DEVICE_ID:-0}

mkdir -p logs/slurm

echo "Start baseline training at $(date)"
echo "Dataset: ${DATASET}"

python train.py --config_file "${CONFIG}" \
    MODEL.DEVICE_ID "('${DEVICE_ID}')" \
    DATASETS.NAMES "${DATASET}" \
    DATASETS.ROOT_DIR "${DATA_ROOT}" \
    MODEL.PRETRAIN_PATH "${PRETRAIN_PATH}" \
    OUTPUT_DIR "${OUTPUT_DIR}"

echo "End baseline training at $(date)"
