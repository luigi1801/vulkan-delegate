#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/delegates/vulkan/vulkan_delegate.h"
#include "tensorflow/lite/delegates/vulkan/vulkan_delegate_adaptor.h"

#include "tensorflow/lite/c/common.h"
#include <iostream>

void linearExample(){
  std::cout << "Loading Linear model: y = 2x-1" << std::endl;
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile("linear.tflite");

  if(!model){
    std::cout << "Failed to mmap model" << std::endl;
    exit(0);
  }

  std::cout << "Creating Interpreter" << std::endl;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

  std::cout << "Creating Delegate" << std::endl;
  VulkanDelegateOptions params = TfLiteVulkanOptionsDefault();
  auto* delegate_ptr = TfLiteVulkanDelegateCreate(&params);

  tflite::Interpreter::TfLiteDelegatePtr delegate(delegate_ptr,
  [](TfLiteDelegate* delegate) {
      TfLiteVulkanDelegateDelete(delegate);
  });

  std::cout << "Modifying Graph with delegate" << std::endl;
  interpreter->ModifyGraphWithDelegate(delegate.get());
  interpreter->AllocateTensors();

  std::cout << "Set input: x = 10" << std::endl;
  float* input = interpreter->typed_input_tensor<float>(0);
  // Dummy input for testing
  *input = 10.0;

  std::cout << "Invoke Inference" << std::endl;
  interpreter->Invoke();

  float* output = interpreter->typed_output_tensor<float>(0);

  std::cout << "Output: y = " << *output << std::endl;
};

void conv2DExample(){
  std::cout << "Loading Conv2D model" << std::endl;
  std::unique_ptr<tflite::FlatBufferModel> model = 
       tflite::FlatBufferModel::BuildFromFile("conv2d_2l_1k.tflite");
  //std::unique_ptr<tflite::FlatBufferModel> model = 
  //     tflite::FlatBufferModel::BuildFromFile("mobilenet_v1_1.0_224.tflite");

  if(!model){
    std::cout << "Failed to mmap model" << std::endl;
    exit(0);
  }

  std::cout << "Creating Interpreter" << std::endl;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

  std::cout << "Creating Delegate" << std::endl;
  VulkanDelegateOptions params = TfLiteVulkanOptionsDefault();
  auto* delegate_ptr = TfLiteVulkanDelegateCreate(&params);

  tflite::Interpreter::TfLiteDelegatePtr delegate(delegate_ptr,
  [](TfLiteDelegate* delegate) {
     TfLiteVulkanDelegateDelete(delegate);
  });

  std::cout << "Modifying Graph with delegate" << std::endl;
  interpreter->ModifyGraphWithDelegate(delegate.get());
  interpreter->AllocateTensors();

  std::cout << "Set input" << std::endl;
  int inputSize = 16;
  float* input = interpreter->typed_input_tensor<float>(0);
  // Dummy input for testing
  for(int i = 0; i< inputSize*inputSize; i++){
    *(input+i) = ((float) i);    
  }

  std::cout << "Invoke Inference" << std::endl;
  interpreter->Invoke();

  std::cout << "Output" << std::endl;
  int outputSize = 12;
  int amplifier = 1;
  float* output = interpreter->typed_output_tensor<float>(0);

  for(int i = 0; i  < outputSize; i++){
    for(int j = 0; j  < outputSize; j++){
      float val = *(output+(i*outputSize+j));
      std::cout << amplifier*val;
      if(j != outputSize-1){
        std::cout << ", ";
      }
    }
    std::cout << std::endl;
  }
  /*
  for(int i = 0; i  < 14; i++){
    for(int j = 0; j  < 14; j++){
      float val = *(output+(i*14+j)*2);
      std::cout << 4*val;
      if(j != 13){
        std::cout << ", ";
      }
    }
    std::cout << std::endl;
  }

  for(int i = 0; i  < 14; i++){
    for(int j = 0; j  < 14; j++){
      float val = *(output+(i*14+j)*2+1);
      std::cout << 2*val;
      if(j != 13){
        std::cout << ", ";
      }
    }
    std::cout << std::endl;
  }*/
};

int main(int argc, char* argv[])
{
    conv2DExample();

    return 0;
}
