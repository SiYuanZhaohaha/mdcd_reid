#!/bin/bash
# Local examples without Slurm.
#
# Usage:
#   bash run.sh baseline
#   bash run.sh mdcd
#   bash run.sh eval
#   bash run.sh fisher

set -e

MODE=${1:-baseline}
DATASET=${DATASET:-mixed_market1501}
CONFIG=${CONFIG:-configs/market/vit_small.yml}
DATA_ROOT=${DATA_ROOT:-../data}
PRETRAIN_PATH=${PRETRAIN_PATH:-../pretrained/vit_small_cfs_lup.pth}
TEACHER_WEIGHT=${TEACHER_WEIGHT:-../pretrained/teacher_checkpoint.pth}
STUDENT_WEIGHT=${STUDENT_WEIGHT:-../pretrained/student_checkpoint.pth}
TEST_WEIGHT=${TEST_WEIGHT:-../pretrained/checkpoint.pth}
DEVICE_ID=${DEVICE_ID:-0}

case "${MODE}" in
  baseline)
    python train.py --config_file "${CONFIG}" \
      MODEL.DEVICE_ID "('${DEVICE_ID}')" \
      DATASETS.NAMES "${DATASET}" \
      DATASETS.ROOT_DIR "${DATA_ROOT}" \
      MODEL.PRETRAIN_PATH "${PRETRAIN_PATH}" \
      OUTPUT_DIR "./logs/baseline/${DATASET}"
    ;;
  mdcd)
    python traincd2.py --config_file "${CONFIG}" \
      MODEL.NAME "teacher_OT" \
      MODEL.DEVICE_ID "('${DEVICE_ID}')" \
      DATASETS.NAMES "${DATASET}" \
      DATASETS.ROOT_DIR "${DATA_ROOT}" \
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
      OUTPUT_DIR "./logs/mdcd/${DATASET}"
    ;;
  eval)
    python test.py --config_file "${CONFIG}" \
      MODEL.DEVICE_ID "('${DEVICE_ID}')" \
      DATASETS.NAMES "${DATASET}" \
      DATASETS.ROOT_DIR "${DATA_ROOT}" \
      TEST.WEIGHT "${TEST_WEIGHT}" \
      OUTPUT_DIR "./logs/eval/${DATASET}"
    ;;
  fisher)
    python test11.py --config_file "${CONFIG}" \
      MODEL.DEVICE_ID "('${DEVICE_ID}')" \
      DATASETS.NAMES "${DATASET}" \
      DATASETS.ROOT_DIR "${DATA_ROOT}" \
      TEST.WEIGHT "${TEST_WEIGHT}"
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Use one of: baseline, mdcd, eval, fisher"
    exit 1
    ;;
esac
