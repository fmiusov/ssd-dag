# ssd-dag

## setup

you need a virtual environment (conda on SageMaker, I use virtualenvwrapper on my laptop)

- tensorflow
- opencv-python
- pillow

There is a requirements.txt file


## git clone

### tensorflow/models

you need tensorflow/models.  this is a collection of models and helper functions (supplements tensorflow) 
`git clone https://github.com/tensorflow/models.git`

#### TO USE THIS - you need to add these environment variables:

(they are already in existing scripts)   
`MODEL_RESEARCH="${OBJ_DET_DIR}/../models/research"`
`export PYTHONPATH=$PYTHONPATH:${MODEL_RESEARCH}`
