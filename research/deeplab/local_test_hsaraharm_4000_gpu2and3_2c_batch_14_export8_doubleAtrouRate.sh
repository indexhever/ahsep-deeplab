#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# This script is used to run local test on HSARAH VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

# Exit immediately if a command exits with a non-zero status.
set -e

# Move one-level up to tensorflow/models/research directory.
cd ..

# Update PYTHONPATH.
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# Set up the working environment.
CURRENT_DIR=$(pwd)
WORK_DIR="${CURRENT_DIR}/deeplab"

# Run model_test first to make sure the PYTHONPATH is correctly set.
#python "${WORK_DIR}"/model_test.py -v

# Go to datasets folder and download HSARAH VOC 2012 segmentation dataset.
DATASET_DIR="datasets"
cd "${WORK_DIR}/${DATASET_DIR}"

# Go back to original directory.
cd "${CURRENT_DIR}"

# Set up the working directori
HSARAH_FOLDER="hsaraharm_4000"
EXP_FOLDER="exp/train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes_batch_14_sem_mirrors"
INIT_FOLDER="${WORK_DIR}/${DATASET_DIR}/${HSARAH_FOLDER}/init_models"
TRAIN_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${HSARAH_FOLDER}/${EXP_FOLDER}/train"
EVAL_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${HSARAH_FOLDER}/${EXP_FOLDER}/eval"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${HSARAH_FOLDER}/${EXP_FOLDER}/vis"
EXP_NEW="exp/train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes_batch_14_sem_mirrors_with_hsarahtrain_and_hsaraharm_img"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${HSARAH_FOLDER}/${EXP_NEW}/vis"
VIS_LOGDIR="${WORK_DIR}/${DATASET_DIR}/${HSARAH_FOLDER}/${EXP_FOLDER}/vis"
EXPORT_DIR="${WORK_DIR}/${DATASET_DIR}/${HSARAH_FOLDER}/${EXP_FOLDER}/export"

# Copy locally the trained checkpoint as the initial checkpoint.
TF_INIT_ROOT="http://download.tensorflow.org/models"
TF_INIT_CKPT="deeplabv3_pascal_train_aug_2018_01_04.tar.gz"

HSARAH_DATASET="${WORK_DIR}/${DATASET_DIR}/${HSARAH_FOLDER}/tfrecord"

# Train 10 iterations.
NUM_ITERATIONS=30000
#NUM_ITERATIONS=700

# Export the trained checkpoint.
EXPORT_OUTPU_STRIDE=8
CKPT_PATH="${TRAIN_LOGDIR}/model.ckpt-${NUM_ITERATIONS}"
EXPORT_PATH="${EXPORT_DIR}/frozen_inference_graph_${EXPORT_OUTPU_STRIDE}_doubleAtrou.pb"

python "${WORK_DIR}"/export_model.py \
  --logtostderr \
  --checkpoint_path="${CKPT_PATH}" \
  --export_path="${EXPORT_PATH}" \
  --model_variant="xception_65" \
  --atrous_rates=12 \
  --atrous_rates=24 \
  --atrous_rates=32 \
  --output_stride=${EXPORT_OUTPU_STRIDE} \
  --decoder_output_stride=4 \
  --num_classes=2 \
  --crop_size=513 \
  --crop_size=513 \
  --inference_scales=1.0


# Run inference with the exported checkpoint.
# Please refer to the provided deeplab_demo.ipynb for an example.
