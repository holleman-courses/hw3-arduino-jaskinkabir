import tensorflow as tf
import numpy as np

path = "D:\School\Intro_IOT_ML\hw3-arduino-jaskinkabir\model_conversion\sin_predictor.h5"


print(f"Using TF Version: {tf.version.VERSION}")

def representative_dataset():
    # Generate synthetic sine wave data matching training distribution
    for _ in range(100):
        freq = np.random.uniform(0.02, 0.2)
        phase = np.random.uniform(0, 2*np.pi)
        sample = np.sin(2 * np.pi * freq * np.arange(7) + phase).astype(np.float32)
        yield [sample.reshape(1, 7)]  # Shape: [1,7]

data = np.array(list(representative_dataset()))

model = tf.keras.models.load_model(path)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()
# Save the TFLite model
with open("sin_predictor.tflite", "wb") as f:
    f.write(tflite_quant_model)
# Test in Python
interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Get quantization parameters
input_scale = input_details[0]['quantization'][0]
input_zero_point = input_details[0]['quantization'][1]

print(f"Input Scale: {input_scale} | Input Zero Point: {input_zero_point}")
print(f"Output Scale: {output_details[0]['quantization'][0]} | Output Zero Point: {output_details[0]['quantization'][1]}")

# Scale and quantize input
raw_input = data[0].reshape(1, 7)
print("Raw Input:", raw_input)
quantized_input = raw_input / input_scale + input_zero_point
input_data = quantized_input.astype(np.int8)
print("Quantized Input:", input_data)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print("Quantized Prediction:", output)

# Dequantize output
output_scale = output_details[0]['quantization'][0]
output_zero_point = output_details[0]['quantization'][1]
dequantized_output = (output.astype(np.float32) - output_zero_point) * output_scale
print("Dequantized Prediction:", dequantized_output)

size = 0
for tensor in interpreter.get_tensor_details():
    match tensor['dtype']:
        case np.int8:
            size += np.prod(tensor['shape'])
        case np.float32:
            size += np.prod(tensor['shape']) * 4
        case np.int32:
            size += np.prod(tensor['shape']) * 4
        case _:
            raise ValueError(f"Unsupported data type: {tensor['dtype']}")
    
            

print(f"Arena Size: {size} bytes")