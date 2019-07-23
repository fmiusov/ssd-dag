#!/bin/bash

cd ../code

python train.py \
  --pipeline_config_path="sagemaker_mobilenet_v1_ssd_retrain.config" \
  --output_dir="ckpt" \
  --num_train_steps="5000" \
  --num_eval_steps="100"
