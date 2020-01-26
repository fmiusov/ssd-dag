# Using Docker

## install Docker
version 19.03
## Install NVIDIA Docker
nvidia-docker2  
test  
## Host
### prerequisites
#### CUDA
you need CUDA > = 10.0 with a good driver.   Docker image will do the rest.  You don't need cuDNN.  
#### Python
you don't really need Anaconda or a virtual environent, the Docker will serve as the virutal environment.  

1. Update your AWS Credentials
2. cd /tasks
3. Get Data - you can go to /tasks and use local_get_s3_files.sh
4. Get Model Software: bash install_tf_models.sh
5. cd ~/projects

docker run --runtime=nvidia -it -w /ssd-dag -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)"  grilledclub/cuda100-tf114:20200124 bash

in docker

cd /mnt/ssd-dag/code
pip install tensorflow-gpu==1.14
