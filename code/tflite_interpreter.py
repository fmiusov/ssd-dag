import numpy as np
import tensorflow as tf

def get_tflite_interpreterer(logger, model_path="converted_model.tflite"):

        # Load Tensorflow model into memory
        logger.info('Loading Model: {}'.format(model_path))

        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path)
        interpreter.allocate_tensors()
        logger.info('Loaded Model')

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logger.info('Input Details: {}'.format(input_details))
        logger.info('Output Details: {}'.format(output_details))
        # Test model on random input data.
        # self.input_shape = self.input_details[0]['shape']
        #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        #interpreter.set_tensor(input_details[0]['index'], input_data)
        return interpreter