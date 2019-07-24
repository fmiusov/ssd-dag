#!/bin/bash 

#
#  this will install the tensorflow/models repo
#  - you need this for the actual model and a bunch of utilities
#  - it will also compmile the protobufs
#
cd ../code
# clone the repo into code/models
git clone https://github.com/tensorflow/models.git

# get the protobuf compiler
# ref: https://developers.google.com/protocol-buffers/docs/pythontutorial
wget -O protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip
unzip protobuf.zip
# compile the proto
./bin/protoc object_detection/protos/*.proto --python_out=.

# clean up
rm protobuf.zip
rm -r bin

