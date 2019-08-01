import argparse
from os import path
import logging
import sys
import time

import cv2
import numpy as np
from numpy import array

import os
from os import listdir
from os.path import isfile, join, splitext

from utils.utils import  Models, load_image_into_numpy_array
from utils.label_map_util import get_label_map_dict

# from object_detector_detection_api import ObjectDetectorDetectionAPI
# from object_detector_detection_api_lite import ObjectDetectorLite

from tflite_interpreter import get_tflite_interpreterer
from display import inference_to_image
from annotation import inference_to_xml


from PIL import Image

BATCH_SIZE = 32
PROBABILITY_THRESHOLD = 0.6    # only display objects with a 0.6+ probability

# TODO
# logging - not print
# add prob to the printed bbox
# directory
# image
# stream
# display mode
# make this work with SageMaker Jupyter Notebook

logging.basicConfig(
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt=' %I:%M:%S ',
    level="INFO"
)
logger = logging.getLogger('detector')
logging.getLogger().setLevel(logging.INFO)

basepath = path.dirname(__file__)

image_config = {"hand" : 1}

def get_image_list(image_dir):
    '''
    input:  full directory path with jpg files
    output: list of full jpg image paths
    '''
    dir_list = listdir(image_dir)
    image_list = list()
    for f in dir_list:
        full_path = join(image_dir, f)
        if isfile(full_path) and splitext(f)[1].lower() == '.jpg':
            image_list.append(full_path)
    image_count = len(image_list)
    logger.info("image directory: {}".format(image_count))
    return image_list, image_count

def send_image_to_model(logger, image_filepath, preprocessed_image, interpreter):
    # model inference
    start_time = time.time()
    # input (image) is (1,300,300,3) - shaped like a batch of size 1
    interpreter.set_tensor(input_details[0]['index'], preprocessed_image)    # model input is a batch of images
    interpreter.invoke()        # this invokes the model, creating output data

    # model has created an inference from a batch of data
    # - the model creates output like a batch of size = 1
    # - size must be 1, so simplify the shape by taking first row only
    #   [0] at the end effectivtly means bbox (1,10,4) becomes (10,4)
    bbox_data = interpreter.get_tensor(output_details[0]['index'])[0]
    class_data = interpreter.get_tensor(output_details[1]['index'])[0]
    prob_data = interpreter.get_tensor(output_details[2]['index'])[0]

    finish_time = time.time()
    logger.info("time spent: {:.4f}".format(finish_time - start_time))

    return bbox_data, class_data, prob_data



if __name__ == '__main__':
    # initiate the parser
    parser = argparse.ArgumentParser(prog='test_models.py')

    # add arguments

    parser.add_argument("--image_dir", type=str, required=True,
                        help="directory to 1 - n images")                   
    parser.add_argument("--model_name", type=Models.from_string,
                        required=True, choices=list(Models),
                        help="name of detection model: {}".format(
                        list(Models)))
    parser.add_argument("--model_path", type=str, required=True,
                        help="path to model frozen graph *.pb file")
    parser.add_argument("--label_map_path", type=str, required=True,
                        help="path to label map *.pbtext file")
    parser.add_argument("--display", type=str, required=False,
                        default='gtk', help='gtk to display on gui')
    parser.add_argument("--annotation_dir", type=str, required=False,
                        help="dir for annotations")

    # read arguments from the command line
    args = parser.parse_args()

    for k, v in vars(args).items():
        logger.info('Arguments. {}: {}'.format(k, v))

    # read the label map
    label_dict = get_label_map_dict(args.label_map_path, 'id')
    logger.info("label dict: {}".format(label_dict))

    # this value is validated by the argument definition
    if args.model_name == Models.tf_lite:
        logger.info('TF Lite Model loading...')
        interpreter = get_tflite_interpreterer(logger, args.model_path)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        model_input_shape = input_shape = input_details[0]['shape']   # (batch, h, w, channels)
        model_image_dim = (model_input_shape[1], model_input_shape[2])    # model - image dimension
        model_input_dim = (1, model_input_shape[1], model_input_shape[2], 3) # model - batch of images dimensions
        logger.info("Model Input Dimension: {}".format(model_input_dim))
    elif args.model_name == Models.edge_tpu:
        from edgetpu.detection.engine import DetectionEngine
        logger.info('Edge TPU Model loading...')
        engine = DetectionEngine(args.model_path)
        model_image_dim = (300,300)
        model_input_dim = (1,300,300,3)



        
    if args.image_dir != None:
        image_list, image_count = get_image_list(args.image_dir)
        for i in range(image_count):
                image_filepath = image_list[i]
                logger.info("Image: {}".format(image_filepath))
            

                if args.model_name == Models.tf_lite:
                    model_type = 'tf_lite'
                    image = load_image_into_numpy_array(image_filepath)                                 # image to numpy array
                    resized_image = cv2.resize(image, model_image_dim, interpolation = cv2.INTER_AREA)  # resized to 300x300xRGB
                    reshaped_image = np.reshape(resized_image, model_input_dim)                         # reshape for model (1,300,300,3)
                    bbox_array, class_id_array, prob_array = send_image_to_model(logger, image_filepath, reshaped_image, interpreter)
                elif args.model_name == Models.edge_tpu:
                    model_type = 'edge_tpu'    # we'll need this to process the output differently
                    # cv2_img = cv2.cvtColor(reshaped_image, cv2.COLOR_BGR2RGB)
                    # pil_image = Image.fromarray(cv2_img)         # expecting a PIL image
                    # #cv2_image = load_image_into_numpy_array('/home/jay/Downloads/parrot.jpg')
                    # print ("cv2 image shape:", cv2_image.shape)
                    pil_image = Image.open(image_filepath)
                    # returns class:  DetectionCandidate
                    start_time = time.time()
                    ans = engine.DetectWithImage(pil_image, threshold=0.05, keep_aspect_ratio=True, relative_coord=False, top_k=10)
                    finish_time = time.time()
                    logger.info("time spent: {:.4f}".format(finish_time - start_time))
                    bbox_list = []
                    class_id_list = []
                    prob_list = []
                    for i,obj in enumerate(ans):
                        box = obj.bounding_box.flatten().tolist()
                        bbox_list.append(box)
                        class_id_list.append(obj.label_id)
                        prob_list.append(obj.score)
                    bbox_array = array(bbox_list)
                    class_id_array = array(class_id_list)
                    prob_array = array(prob_list)

                inference_image, orig_image_dim, detected_objects = inference_to_image(model_type, logger, 
                    image_filepath,
                    bbox_array, class_id_array, prob_array, 
                    model_input_dim, label_dict, PROBABILITY_THRESHOLD)

                if args.display == 'gtk':
                    cv2.namedWindow('Object Detect')
                    image = cv2.cvtColor(inference_image, cv2.COLOR_RGB2BGR)
                    cv2.imshow('Object Detect', image)
                    cv2.waitKey(0)
                elif args.display == 'None':
                    pass
                else:
                    pass
                if args.annotation_dir != None:
                    print ("detected objects:", detected_objects)
                    image_basename = os.path.basename(image_filepath)
                    annotation_xml = inference_to_xml(args.image_dir, image_basename, orig_image_dim, detected_objects, args.annotation_dir)
                    


