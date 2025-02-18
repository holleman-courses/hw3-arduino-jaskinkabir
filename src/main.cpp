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
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  
  // In order to use optimized tensorflow lite kernels, a signed int8_t quantized
  // model is preferred over the legacy unsigned model format. This means that
  // throughout this project, input images must be converted from unisgned to
  // signed format. The easiest and quickest way to convert from unsigned to
  // signed 8-bit integers is to subtract 128 from the unsigned value to get a
  // signed value.
  
  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 136 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
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
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(sin_predictor_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
  }

  static tflite::MicroMutableOpResolver<1> micro_op_resolver;
  micro_op_resolver.AddFullyConnected();

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
  }
  
  input = interpreter->input(0);
  int8_t test_input[7] = {91,-31,-123,-90,33,123,86};
  input->data.int8 = test_input;


  delay(5000);
  // Arduino does not have a stdout, so printf does not work easily
  // So to print fixed messages (without variables), use 
  // Serial.println() (appends new-line)  or Serial.print() (no added new-line)
  int t0 = millis();
  Serial.println("Test Project waking up");
  int t1 = millis();
  if (kTfLiteOk != interpreter->Invoke()) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
  }
  int t2 = millis();
  int t_print = t1 - t0;
  int t_infer = t2 - t1;

  TfLiteTensor * output = interpreter->output(0);

  sprintf(out_str_buff, "Predicted Output: %d", output[0]);
  sprintf(out_str_buff, "Printing Time: %d Inference Time: %d", t_print, t_infer);

  memset(in_str_buff, (char)0, INPUT_BUFFER_SIZE*sizeof(char)); 
}



void loop() {
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

      sprintf(out_str_buff, "Predicted Output: %d", output[0]);

      
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
