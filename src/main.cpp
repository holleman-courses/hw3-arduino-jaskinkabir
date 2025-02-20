#include <Arduino.h>

#include "TensorFlowLite.h"
#include "model_data.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"




#define INPUT_BUFFER_SIZE 64
#define OUTPUT_BUFFER_SIZE 64
#define INT_ARRAY_SIZE 8

namespace {
  
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor * output = nullptr;
  tflite::ErrorReporter* error_reporter = nullptr;

  constexpr int kTensorArenaSize = 1300;
  uint8_t tensor_arena[kTensorArenaSize];
  }  // namespace

// put function declarations here:
int string_to_array(char *in_str, int8_t *int_array);
void print_int_array(int8_t *int_array, int array_len);


char received_char = (char)NULL;              
int chars_avail = 0;                    // input present on terminal
char out_str_buff[OUTPUT_BUFFER_SIZE];  // strings to print to terminal
char in_str_buff[INPUT_BUFFER_SIZE];    // stores input from terminal
int8_t input_array[INT_ARRAY_SIZE];        // array of integers input by user

int in_buff_idx=0; // tracks current input location in input buffer
int array_length=0;
int array_sum=0;

void setup() {
  // put your setup code here, to run once:
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  //Serial.begin(9600);
  delay(3000);
  Serial.println("Starting Up");
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;
  Serial.println("Error reporter loaded");
  model = tflite::GetModel(sin_predictor_tflite);
  Serial.println("Model loaded");
  Serial.print("Expected schema version: ");
  Serial.println(TFLITE_SCHEMA_VERSION);
  Serial.print("Model version: ");
  Serial.println(model->version());

  Serial.println("Creating op resolver");


  static tflite::MicroMutableOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddFullyConnected();  // Dense layer

  micro_op_resolver.AddRelu();            // ReLU activation
  micro_op_resolver.AddQuantize();        // Often needed for quantized models
  micro_op_resolver.AddDequantize();      // If input/output requires it
  micro_op_resolver.AddAdd();
  micro_op_resolver.AddMul();

  

  Serial.println("Initializing interpreter");
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  Serial.println("Interpreter loaded");

  

  
  input = interpreter->input(0);
  
  Serial.print("Input scale: ");
  Serial.println(input->params.scale);
  Serial.print("Input zero-point: ");
  Serial.println(input->params.zero_point);
  Serial.print("Input type: ");
  Serial.println(input->type);  // Should be kTfLiteInt8 (9)
  int8_t test_input[7] = {91,-31,-123,-90,33,123,86};
  input->data.int8 = test_input;
  
  Serial.println("Input set");
  
    Serial.println("Allocating tensors");
  
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      Serial.println("AllocateTensors failed!");
      Serial.print("Error code: ");
      Serial.println(allocate_status);
      while (1); // Halt
    }
    
    Serial.println("Tensors allocated");
  
  int t0 = millis();
  Serial.println("Test Project waking up");
  
  int t1 = millis();
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }
  int t2 = millis();
  
  int t_print = t1 - t0;
  int t_infer = t2 - t1;

  output = interpreter->output(0);

  sprintf(out_str_buff, "Predicted Output: %d Expected Output: -38", output->data.int8[0]);
  sprintf(out_str_buff, "Printing Time: %d Inference Time: %d", t_print, t_infer);

  memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE*sizeof(char)); 
}



void loop() {
  Serial.println("Enter Seven Integers separated by commas: "); 
  // put your main code here, to run repeatedly:

  // check if characters are avialble on the terminal input
  chars_avail = Serial.available(); 
  if (chars_avail > 0) {
    received_char = Serial.read(); // get the typed character and 
    Serial.print(received_char);   // echo to the terminal

    in_str_buff[in_buff_idx++] = received_char; // add it to the buffer
    if (received_char == 13) { // 13 decimal = newline character
      // user hit 'enter', so we'll process the line.
      Serial.print("About to process line: ");
      Serial.println(in_str_buff);

      // Process and print out the array
      array_length = string_to_array(in_str_buff, input_array);
      sprintf(out_str_buff, "Read in  %d integers: ", array_length);
      if (array_length != 7) {
        Serial.print("Error: Not enough inputs!");
        return;
      }

      Serial.print(out_str_buff);
      print_int_array(input_array, array_length);
      

      
      input->data.int8 = input_array;

      if (kTfLiteOk != interpreter->Invoke()) {
        TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
      }
      TfLiteTensor * output = interpreter->output(0);

      sprintf(out_str_buff, "Predicted Output: %d", output->data.int8[0]);

      
      // Now clear the input buffer and reset the index to 0
      memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE*sizeof(char)); 
      in_buff_idx = 0;
    }
    else if (in_buff_idx >= INPUT_BUFFER_SIZE) {
      memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE*sizeof(char)); 
      in_buff_idx = 0;
    }    
  }
}

int string_to_array(char *in_str, int8_t *int_array) {
  int num_integers=0;
  char *token = strtok(in_str, ",");
  
  while (token != NULL) {
    int_array[num_integers++] = (int8_t)atoi(token);
    token = strtok(NULL, ",");
    if (num_integers >= INT_ARRAY_SIZE) {
      break;
    }
  }
  
  return num_integers;
}

void print_int_array(int8_t *int_array, int array_len) {
  int curr_pos = 0; // track where in the output buffer we're writing

  sprintf(out_str_buff, "Integers: [");
  curr_pos = strlen(out_str_buff); // so the next write adds to the end
  for(int i=0;i<array_len;i++) {
    // sprintf returns number of char's written. use it to update current position
    curr_pos += sprintf(out_str_buff+curr_pos, "%d, ", int_array[i]);
  }
  sprintf(out_str_buff+curr_pos, "]\r\n");
  Serial.print(out_str_buff);
}
