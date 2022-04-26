#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/delegates/vulkan-delegate/vulkan_delegate.h"
#include "tensorflow/lite/delegates/vulkan-delegate/vulkan_delegate_adaptor.h"

#include "tensorflow/lite/c/common.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <unistd.h>
#include <cstdlib>
#include <ctime>

std::string LOG = "";

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
       tflite::FlatBufferModel::BuildFromFile("conv2d_test.tflite");
  //std::unique_ptr<tflite::FlatBufferModel> model = 
  //     tflite::FlatBufferModel::BuildFromFile("conv2d_2l_1k.tflite");

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
  /*
  float* input = interpreter->typed_input_tensor<float>(0);
  for(int i = 0; i< 1024; i++){
    *(input+i) = i/512-1;
  }
  */
  int inputSize = 14;
  float* input = interpreter->typed_input_tensor<float>(0);
  // Dummy input for testing
  for(int i = 0; i< inputSize*inputSize; i++){
    for(int x = 0; x < 512; x++){
      int dir = 512*i+x;
      *(input+dir) = ((float) dir)/512/7-1;    
    }
  }


  //for(int i = 0; i< inputSize*inputSize; i++){
  //  *(input+3*i) = ((float) i)*6/255;    
  //  *(input+3*i+1) = ((float) 3);    
  //  *(input+3*i+2) = ((float) 255-i)/255;    
  //}

  std::cout << "Invoke Inference" << std::endl;
  interpreter->Invoke();

  std::cout << "Output" << std::endl;
  float* output = interpreter->typed_output_tensor<float>(0);
  int outputSize = 14;
  int outputDepth = 512;
  int amplifier = 1;
  //for(int i=0;i<1001; i++){
  //  std::cout << "Class: " << i << " = " << output[i] <<std::endl;
  //  //if(output[i]> maxVal)
  //  //{
  //  //  maxVal = output[i];
  //  //  maxIdx = i;
  //  //}
  //}
  std::ofstream myfile;
  myfile.open ("example2.txt");
  for(int k = 0; k  < outputDepth; k++){
    myfile << std::endl;
    myfile << "Out: "<< k << std::endl;
    for(int i = 0; i  < outputSize; i++){
      for(int j = 0; j  < outputSize; j++){
        float val = *(output+(i*outputSize+j)*outputDepth+k);
        myfile << std::setprecision(6) << amplifier*val;
        if(j != outputSize-1){
          myfile << ", ";
        }
      }
      myfile << std::endl;
    }
  }
  myfile.close();
};


void ReadFileStr(std::string FileName, std::vector<std::string>& list){
  // Load list of images to be processed
  std::ifstream file(FileName);
  if (!file)
    throw "Unable to open the available image list file ";
  for (std::string s; !getline(file, s).fail();)
  {
    list.emplace_back(s);
  }
  file.close();
};

void ReadFileInts(std::string FileName, std::vector<int>& list){
  // Load list of images to be processed
  std::ifstream file(FileName);
  if (!file)
    throw "Unable to open the available image list file ";
  for (std::string s; !getline(file, s).fail();)
  {
    list.emplace_back(stoi(s));
  }
  file.close();
};

void mobilenetTime(){
  std::vector<std::string> availableImageList;
  ReadFileStr("timeTest/image_list.txt", availableImageList);
    
  //Loading Model, Delegate, interpreter and graph  
  std::unique_ptr<tflite::FlatBufferModel> model = 
       tflite::FlatBufferModel::BuildFromFile("mobilenet_v1_1.0_224.tflite");

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

  //std::cout << "Set input" << std::endl;
  float* input = interpreter->typed_input_tensor<float>(0);
  int xS = 224, yS = 224, zS = 3;
  int sizebuff = xS*yS*zS;
  std::vector<uint8_t>buffInt(sizebuff, 0);

  auto path = "timeTest/" + availableImageList[0];
  std::ifstream inFile(path, std::ios::in | std::ios::binary);
  if (!inFile) throw "Failed to open image data " + path;
  inFile.read(reinterpret_cast<char*>(buffInt.data()), sizebuff);
  inFile.close();

  for (int i = 0; i< sizebuff ;i++) {
    auto val = buffInt[i];
    input[i] = (val / 255.0 - 0.5) * 2.0;
  }
  auto start = std::chrono::steady_clock::now();
  for(int i =0;i<100;i++){  
    interpreter->Invoke();
  }  
  auto end = std::chrono::steady_clock::now();
  std::cout << "Elapsed time in nanoseconds: "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()
        << " ns" << std::endl;
 
  std::cout << "Elapsed time in microseconds: "
      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
      << " µs" << std::endl;
 
  std::cout << "Elapsed time in milliseconds: "
      << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
     << " ms" << std::endl;
  
};

void mobilenetAcc(){
  std::vector<std::string> availableImageList;
  std::vector<int> availableImageListRes;
  ReadFileStr("dataset/image_list.txt", availableImageList);
  ReadFileInts("dataset/image_list_res.txt", availableImageListRes);
    
  //Loading Model, Delegate, interpreter and graph  
  std::unique_ptr<tflite::FlatBufferModel> model = 
       tflite::FlatBufferModel::BuildFromFile("mobilenet_v1_1.0_224.tflite");

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

  //std::cout << "Set input" << std::endl;
  float* input = interpreter->typed_input_tensor<float>(0);
  float* output = interpreter->typed_output_tensor<float>(0);
  int xS = 224, yS = 224, zS = 3;
  int sizebuff = xS*yS*zS;
  std::vector<uint8_t>buffInt(sizebuff, 0);
  int matchCounter = 0;

  for(int j=0;j<availableImageList.size(); j++)
  {
    auto path = "dataset/" + availableImageList[j];
    std::ifstream inFile(path, std::ios::in | std::ios::binary);
    if (!inFile) throw "Failed to open image data " + path;
    inFile.read(reinterpret_cast<char*>(buffInt.data()), sizebuff);
    inFile.close();

    for (int i = 0; i< sizebuff ;i++) {
      auto val = buffInt[i];
      input[i] = (val / 255.0 - 0.5) * 2.0;
    }
    
    interpreter->Invoke();

    float maxVal = 0;
    float maxIdx = 0;
    for(int i=0;i<1001; i++){
      if(output[i]> maxVal)
      {
        maxVal = output[i];
        maxIdx = i-1;
      }
    }
    std::cout <<availableImageList[j] << " "<< maxIdx; /*<< " (" << maxVal << ")" */ ;    
    if(maxIdx == availableImageListRes[j]){
      matchCounter++;
      std::cout <<" (TRUE)";   
    }else{
      std::cout <<" (FALSE)";   
    }
    std::cout <<std::endl;  
  }
  std::cout <<"Total matches = " << matchCounter << "("<<((float)matchCounter*100/500)<<"%)\n";   
  
};


int dummyModelTime(bool modGraph, std::string modelName, std::vector<uint8_t>& inImage, int input_size,
                   std::vector<float>& outputVect, int output_size){
  int ITERATIONS = 100;
    
  //Loading Model, Delegate, interpreter and graph  
  std::unique_ptr<tflite::FlatBufferModel> model = 
       tflite::FlatBufferModel::BuildFromFile(modelName.c_str());

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
  if(modGraph) interpreter->ModifyGraphWithDelegate(delegate.get());
  interpreter->AllocateTensors();

  //std::cout << "Set input" << std::endl;
  float* input = interpreter->typed_input_tensor<float>(0);
  float* output = interpreter->typed_output_tensor<float>(0);

  for (int i = 0; i< input_size ;i++) {
    auto val = inImage[i];
    input[i] = (val / 255.0 - 0.5) * 2.0;
  }
  auto start = std::chrono::steady_clock::now();
  for(int i =0;i<ITERATIONS;i++){  
    interpreter->Invoke();
  }  
  auto end = std::chrono::steady_clock::now();

  for(int i=0;i<output_size*output_size; i++){
    outputVect[i] = output[i];
  }
  return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/ITERATIONS;
};

bool compareVector(int vectSize, std::vector<float>& vect1, std::vector<float>& vect2){
  float error = 2e-5;
  for(int i =0;i<vectSize*vectSize; i++)
    if (vect1[i]-vect2[i] > error || vect2[i]-vect1[i] > error)
      return false;
  return true;
}

void dummyModelsTest(){
  srand(time(NULL));
  std::vector<std::string> availableImageList;
  int xS = 56, yS = 56, zS = 512;
  int sizebuff = xS*yS*zS;
  std::vector<uint8_t> buffInt(sizebuff, 0);
  for(int i = 0; i<sizebuff; i++)
    buffInt[i] = static_cast<uint8_t>(rand()%255);

//  ReadFileStr("timeTest/image_list.txt", availableImageList);
//  auto path = "timeTest/" + availableImageList[0];
//  std::ifstream inFile(path, std::ios::in | std::ios::binary);
//  if (!inFile) throw "Failed to open image data " + path;
//  inFile.read(reinterpret_cast<char*>(buffInt.data()), sizebuff);
//  inFile.close();

  std::vector<std::string> availableModelsList;
  ReadFileStr("models/models_list_512.txt", availableModelsList);

  int output_size = 56;
  for(std::string modelName:availableModelsList){
    output_size -= 2;
    LOG += "\n####################\nExecuting model:" + modelName + "\n";    
    std::cout << std::endl<< "####################" << std::endl<< "Executing model: " << modelName.c_str() << std::endl;
    std::vector<float> output_tf(output_size*output_size);
    std::vector<float> output_vulkan(output_size*output_size);
    int Time1 = dummyModelTime(false, modelName, buffInt, sizebuff, output_tf, output_size);
    int Time2 = dummyModelTime(true, modelName, buffInt, sizebuff, output_vulkan, output_size);
    LOG += std::string("")+
      std::to_string(float(Time1)/10e6)+
      std::string(" \t")+
      std::to_string(float(Time2)/10e6)+
      std::string(" \n");
    //std::cout << output_tf[i]-output_vulkan[i]<< std::endl;
    //if (compareVector(output_size, output_tf,output_vulkan))
    //  LOG += "Both vectors are equal\n";
    //else
    //  LOG += "BOTH  VECTORS ARE DIFFERENT\n";

  }
}


int main(int argc, char* argv[])
{
  dummyModelsTest();
  std::cout<<LOG.c_str();
    //dummyModelTime();
    //mobilenetTime();
    //conv2DExample();
    return 0;
}
