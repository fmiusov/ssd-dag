#!/bin/bash

#
# this script is more for testing locally
# assumes you have AWS CLI setup and you have credentials
# 
# it will pull from s3 so you can test your training script 
# on a local machine
#

# reference - the Jupyter TrainModel.ipynb

# GLOBAL environments
S3_TFRECORDS_PATH="s3://cfaanalyticsresearch-sagemaker/datasets/cfa_products/tfrecords/"
TFRECORDS_TARBALL="20190718_tfrecords.tar.gz"

S3_MODEL_PATH="s3://cfaanalyticsresearch-sagemaker/trained-models/tensorflow_mobilenet/"
MODEL_FOLDER="20190718_cfa_prod_mobilenet_v1_ssd300/"

cd $HOME/projects/ssd-dag/code

# --- data:  tfrecords ---
echo ${S3_TFRECORDS_PATH}${TFRECORDS_TARBALL}
# you must have a code/tfrecords directory and .gitkeep should be there
aws s3 cp ${S3_TFRECORDS_PATH}${TFRECORDS_TARBALL} tfrecords

tar -xvf tfrecords/${TFRECORDS_TARBALL} --strip=1 -C tfrecords
rm tfrecords/${TFRECORDS_TARBALL}

# --- model ---
echo ${S3_MODEL_PATH}${MODEL_FOLDER}
aws s3 cp ${S3_MODEL_PATH}${MODEL_FOLDER} model --recursive