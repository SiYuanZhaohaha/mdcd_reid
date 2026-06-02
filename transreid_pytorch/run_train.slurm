#!/bin/bash
# Train MDCD with the TransReID-SSL branch.
#
# Example:
#   DATA_ROOT=/path/to/data TEACHER_WEIGHT=/path/to/teacher.pth STUDENT_WEIGHT=/path/to/student.pth sbatch run_train.slurm

#SBATCH --job-name=mdcd_train
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm/mdcd-%j.out

set -e

DATASET=${DATASET:-mixed_market1501}
CONFIG=${CONFIG:-configs/market/vit_small.yml}
DATA_ROOT=${DATA_ROOT:-../data}
TEACHER_WEIGHT=${TEACHER_WEIGHT:-../pretrained/teacher_checkpoint.pth}
STUDENT_WEIGHT=${STUDENT_WEIGHT:-../pretrained/student_checkpoint.pth}
OUTPUT_DIR=${OUTPUT_DIR:-./logs/mdcd/${DATASET}}
DEVICE_ID=${DEVICE_ID:-0}

mkdir -p logs/slurm

echo "Start MDCD training at $(date)"
echo "Dataset: ${DATASET}"

python traincd2.py --config_file "${CONFIG}" \
    MODEL.NAME "teacher_OT" \
    MODEL.DEVICE_ID "('${DEVICE_ID}')" \
    DATASETS.NAMES "${DATASET}" \
    DATASETS.ROOT_DIR "${DATA_ROOT}" \
    OUTPUT_DIR "${OUTPUT_DIR}" \
    MODEL.TEACHER_WEIGHT "${TEACHER_WEIGHT}" \
    MODEL.STUDENT_WEIGHT "${STUDENT_WEIGHT}" \
    MODEL.OT_LOSS_WEIGHT 1.0 \
    MODEL.N_ITERS 100 \
    MODEL.FGW_WEIGHT 0.1 \
    MODEL.OT_BLUR 0.8 \
    MODEL.EMA 0.1 \
    MODEL.interpolated_WEIGHT 10.0 \
    SOLVER.BASE_LR 0.0004 \
    SOLVER.MAX_EPOCHS 120 \
    SOLVER.EVAL_PERIOD 121

echo "End MDCD training at $(date)"
