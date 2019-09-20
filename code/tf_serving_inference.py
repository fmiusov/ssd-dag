#
#  scratch work
#    assumes you have a model running on TensorFlow Serving
#
"""Test program to send an image to a MobileNet SSD model that is running on
TensorFlow Serving
"""

import base64

# without this PATH append
# it won't find nets - in the /slim directory
cwd = os.getcwd()
tf_serving_repo = os.path.expanduser('~/projects/serving')
models = os.path.join(cwd, 'models/research/')
slim = os.path.join(cwd, 'models/research/slim')
sys.path.append(models)
sys.path.append(slim)
import requests

# GLOBALS
TEST_IMAGE = '/home'

def main():
    print ("hello world")


if __name__ == '__main__':
  main()