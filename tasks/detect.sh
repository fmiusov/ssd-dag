LOCAL_NEW_IMAGES='../data/new_jpeg_images'
MODEL_NAME='tf_lite'
MODEL_PATH='../model'
MODEL_NAME='tf_lite'
MODEL_FILENAME='output_tflite_graph.tflite'

MODEL_FULL_PATH='../model/output_tflite_graph.tflite'
LABEL_MAP_FILENAME='cfa_prod_label_map.pbtext'
LABEL_MAP_FULL_PATH='../model/cfa_prod_label_map.pbtxt'
ANNOTATION_PATH='../data/annotations'

WORKON_HOME=~/.virtualenvs
. /usr/local/bin/virtualenvwrapper.sh

# connect to the tensorflow model code
# set-up: git clone https://github.com/tensorflow/models.git
MODEL_RESEARCH="${HOME}/projects/models/research"
export PYTHONPATH=$PYTHONPATH:${MODEL_RESEARCH}

workon tgpu113

python ../code/detect.py --image_dir ${LOCAL_NEW_IMAGES} --model_name ${MODEL_NAME} \
	 --model_path ${MODEL_FULL_PATH} --label_map_path ${LABEL_MAP_FULL_PATH} \
	 --display no --annotation_dir ${ANNOTATION_PATH}