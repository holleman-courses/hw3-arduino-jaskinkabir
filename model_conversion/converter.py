import tensorflow as tf
import numpy as np

path = "/home/jaskin/Intro_IoT_ML/hw3-arduino-jaskinkabir/model_conversion/sin_predictor.h5"



model = tf.keras.models.load_model(path)
# conver model to have 8-bit (signed) integer inputs and outputs and well as 8b signed integer weights

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

rng = tf.random.Generator.from_seed(0)
def representative_dataset():
    for _ in range(100):
        yield [*rng.uniform((1, 7), -1, 1)]
        
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_qaunt_model = converter.convert()

 

# Save the TFLite model
with open("sin_predictor.tflite", "wb") as f:
    f.write(tflite_qaunt_model)
