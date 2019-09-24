#
#  scratch work
#    assumes you have a model running on TensorFlow Serving
#
"""Test program to send an image to a MobileNet SSD model that is running on
TensorFlow Serving
"""

import base64
import os
import sys
import requests
import tensorflow as tf

# without this PATH append
# it won't find nets - in the /slim directory
code_dir = os.getcwd()  
jpeg_images_dir = os.path.expanduser('~/projects/ssd-dag/data/jpeg_images')
tf_serving_repo = os.path.expanduser('~/projects/serving')
sys.path.append(tf_serving_repo)


# GLOBALS
SERVER_URL = SERVER_URL = 'http://localhost:8501/v1/models/cfa_prod:predict'
SAMPLE_IMAGE = '20190710_variety_1562781001.jpg'


def main():
    print ("-- TensorFlow Serving - MobileNet Inference --")

    # read in the sample image & shape the numpy array
    img_path = os.path.join(jpeg_images_dir, SAMPLE_IMAGE)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=[300, 300])   # PIL image
    x = tf.keras.preprocessing.image.img_to_array(img)  # numpy array (300,300,3)
    x32 = tf.keras.applications.mobilenet.preprocess_input(x[tf.newaxis,...])  # numpy (1,300,300,3) float32
    print (type(x32), x32.shape)

    # encode the array
      # Compose a JSON Predict request (send JPEG image in base64).
    jpeg_bytes = base64.b64encode(x32)
    jpeg_string = jpeg_bytes.decode('utf-8')
    print ("jpeg_bytes:", type(jpeg_bytes))
    print (jpeg_bytes[:50])
    print ("jpeg_striing:", type(jpeg_string))
    print (jpeg_string[:50])

    predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_string

      # Send few actual requests and report average latency.
    total_time = 0
    num_requests = 10
    for _ in range(num_requests):
        response = requests.post(SERVER_URL, data=predict_request)
        response.raise_for_status()
        total_time += response.elapsed.total_seconds()
        prediction = response.json()['predictions'][0]

    print('Prediction class: {}, avg latency: {} ms'.format(
        prediction['classes'], (total_time*1000)/num_requests))

if __name__ == '__main__':
  main()