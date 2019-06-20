#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

python generateConfMatrixAndAcc_handdb.py --model_folder "train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes_batch_14_sem_mirrors" --base_name "val" --run_acc "True"
python generateConfMatrixAndAcc_handdb.py --model_folder "train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes_batch_14_sem_mirrors_2" --base_name "val" --run_acc "True"
python generateConfMatrixAndAcc_handdb.py --model_folder "train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes_batch_14_sem_mirrors_3" --base_name "val" --run_acc "True"
python generateConfMatrixAndAcc_handdb.py --model_folder "train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes_batch_14_sem_mirrors_4" --base_name "val" --run_acc "True"
python generateConfMatrixAndAcc_handdb.py --model_folder "train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes_batch_14_sem_mirrors_5" --base_name "val" --run_acc "True"

python generateConfMatrixAndAcc_handdb.py --model_folder "train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes_14batch_mobilenet" --base_name "val" --run_acc "True"
python generateConfMatrixAndAcc_handdb.py --model_folder "train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes_14batch_mobilenet_2" --base_name "val" --run_acc "True"
python generateConfMatrixAndAcc_handdb.py --model_folder "train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes_14batch_mobilenet_3" --base_name "val" --run_acc "True"
python generateConfMatrixAndAcc_handdb.py --model_folder "train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes_14batch_mobilenet_4" --base_name "val" --run_acc "True"
python generateConfMatrixAndAcc_handdb.py --model_folder "train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes_14batch_mobilenet_5" --base_name "val" --run_acc "True"

python generateConfMatrixAndAcc_handdb.py --model_folder "train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes" --base_name "val" --run_acc "True"
python generateConfMatrixAndAcc_handdb.py --model_folder "train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes_2" --base_name "val" --run_acc "True"
python generateConfMatrixAndAcc_handdb.py --model_folder "train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes_3" --base_name "val" --run_acc "True"
python generateConfMatrixAndAcc_handdb.py --model_folder "train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes_4" --base_name "val" --run_acc "True"
python generateConfMatrixAndAcc_handdb.py --model_folder "train_on_trainval_set_30000_batchnorm_correctlabel_eval_test_train_2classes_5" --base_name "val" --run_acc "True"