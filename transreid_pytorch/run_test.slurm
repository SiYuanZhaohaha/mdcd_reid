#!/bin/bash
# Evaluate a trained checkpoint.
#
# Example:
#   DATA_ROOT=/path/to/data TEST_WEIGHT=/path/to/checkpoint.pth sbatch run_test.slurm

#SBATCH --job-name=mdcd_eval
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/eval-%j.out

set -e

DATASET=${DATASET:-mixed_market1501}
CONFIG=${CONFIG:-configs/market/vit_small.yml}
DATA_ROOT=${DATA_ROOT:-../data}
TEST_WEIGHT=${TEST_WEIGHT:-../pretrained/checkpoint.pth}
OUTPUT_DIR=${OUTPUT_DIR:-./logs/eval/${DATASET}}
DEVICE_ID=${DEVICE_ID:-0}

mkdir -p logs/slurm

echo "Start evaluation at $(date)"
echo "Dataset: ${DATASET}"

python test.py --config_file "${CONFIG}" \
    MODEL.DEVICE_ID "('${DEVICE_ID}')" \
    DATASETS.NAMES "${DATASET}" \
    DATASETS.ROOT_DIR "${DATA_ROOT}" \
    TEST.WEIGHT "${TEST_WEIGHT}" \
    OUTPUT_DIR "${OUTPUT_DIR}"

echo "End evaluation at $(date)"
