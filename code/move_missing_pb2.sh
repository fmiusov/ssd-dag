#!/bin/bash

# extract the tarball of missing software and copy the *_pb2.py files
# to projects/models/reearch/object_detection/protos/

MODEL_RESEARCH="${HOME}/projects/models/research"
MISSING_TARGET_PATH="${MODEL_RESEARCH}/object_detection/protos/"

tar -xvf missing_pb2.tar.gz
mv missing_pb2/*_pb2.py ${MISSING_TARGET_PATH}

rm -r missing_pb2
 
