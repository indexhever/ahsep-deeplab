#!/bin/bash
set -e
WORK_DIR=$(pwd)
DATASET_DIR="datasets"
HSARAH_FOLDER="hsarah"
EXP_FOLDER="exp/train_on_trainval_set_700_batchnorm_correctlabel_eval_test_train_2classes"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${HSARAH_FOLDER}/${EXP_FOLDER}/train"

# Export the trained checkpoint.
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph.pb"

python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=6 \
  --atrous_rates=12 \
  --atrous_rates=18 \
  --output_stride=16 \
  --decoder_output_stride=4 \
  --num_classes=21 \
  --crop_size=513 \
  --crop_size=513 \
  --inference_scales=1.0 \
  --num_classes=2