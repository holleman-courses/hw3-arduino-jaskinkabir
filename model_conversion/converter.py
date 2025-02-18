import tensorflow as tf

path = "/home/jaskin/Intro_IoT_ML/hw3-arduino-jaskinkabir/model_conversion/sin_predictor.h5"



model = tf.keras.models.load_model(path)
# conver model to have 8-bit (signed) integer inputs and outputs and well as 8b signed integer weights

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("sin_predictor.tflite", "wb") as f:
    f.write(tflite_model)
