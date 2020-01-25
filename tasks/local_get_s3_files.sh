#!/bin/bash

#
# this script is more for testing locally
# assumes you have AWS CLI setup and you have credentials
# 
# it will pull from s3 so you can test your training script 
# on a local machine
#

# reference - the Jupyter TrainModel.ipynb

# run this from the ssd-dag/tasks directory

# GLOBAL environments
# -hint- compare to notebook:  TrainModel_Step1_Local - cell:  Global Constants
S3_TFRECORDS_PATH="s3://cfa-eadatasciencesb-sagemaker/datasets/cfa_products/tfrecords/"
TFRECORDS_TARBALL="20190718_tfrecords.tar.gz"

S3_MODEL_PATH="s3://cfa-eadatasciencesb-sagemaker/trained-models/tensorflow_mobilenet/"
BASE_MODEL_FOLDER="20180718_coco14_mobilenet_v1_ssd300_quantized"
CFA_MODEL_FOLDER="20190718_cfa_prod_mobilenet_v1_ssd300/"

# assumes you are in tasks directory
cd ../code

# --- data:  tfrecords ---
echo ${S3_TFRECORDS_PATH}${TFRECORDS_TARBALL}
# you must have a code/tfrecords directory and .gitkeep should be there
aws s3 cp ${S3_TFRECORDS_PATH}${TFRECORDS_TARBALL} tfrecords

tar -xvf tfrecords/${TFRECORDS_TARBALL} --strip=1 -C tfrecords

# move to subdirectories
rm tfrecords/train/*.tfrecord* -f
rm tfrecords/val/*.tfrecord*   -f

mv tfrecords/train*.* tfrecords/train
mv tfrecords/val*.* tfrecords/val

#- what is this?
#! rm code/tfrecords/$TFRECORDS_TARBALL
rm tfrecords/${TFRECORDS_TARBALL}

# --- ckpt (checkpoint) that you are training on TOP OF - aka xfer learning  ---
rm ckpt/*.*
echo ${S3_MODEL_PATH}${BASE_MODEL_FOLDER}
aws s3 cp ${S3_MODEL_PATH}${BASE_MODEL_FOLDER} ckpt --recursive