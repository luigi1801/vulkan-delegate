#include "vulkan_delegate_kernel.h"

#include <memory>
#include <numeric>
#include <iostream>
#include <iomanip>


template <typename T>
static void
printArray(const T array, size_t size)
{
  assert(array.size() == size * size);
  for (size_t y=0; y<size; ++y) {
    std::cout << "| ";
    for (size_t x=0; x<size; ++x) {
      std::cout << std::setw(10) << std::setprecision(3)
        << array[x + size*y] << ' ';
    }
    std::cout << "|\n";
  }
}

namespace tflite {
namespace vulkan {

TfLiteStatus VulkanKernel::Init(TfLiteContext* context,
                                const TfLiteDelegateParams* params) {

  VulkanConv2D_Control control = {0};
  DelegatedNodesConv2D.resize(params->nodes_to_replace->size);
  for(int i = 0; i<params->nodes_to_replace->size; i++){
    TfLiteNode* delegated_node = nullptr;
    TfLiteRegistration* delegated_node_registration = nullptr;
    TF_LITE_ENSURE_EQ(
      context,
      context->GetNodeAndRegistration(context, params->nodes_to_replace->data[i], &delegated_node,
                                       &delegated_node_registration),
      kTfLiteOk);
    DelegatedNodesConv2D[i].inputTensorIdx = delegated_node->inputs->data[0];
    DelegatedNodesConv2D[i].kernelTensorIdx = delegated_node->inputs->data[1];
    DelegatedNodesConv2D[i].biasTensorIdx = delegated_node->inputs->data[2];
    DelegatedNodesConv2D[i].outputTensorIdx = delegated_node->outputs->data[0];    

    TfLiteConvParams convParams = *((TfLiteConvParams*)delegated_node->builtin_data);
    control.Padding = kTfLitePaddingValid == convParams.padding ? 0 : 1;
    control.stride_h = convParams.stride_height;
  }
  vulkanPrimitive = vulkanPrimitivesFact_->GetPrimitive(Vulkan_Conv2d, control.AllBits);

  std::cout<<"INITIALIZATION\n";
  Id++;
  std::cout<<"Id: " << Id << "\n";
  std::cout<<"######### Nodes #######\n";
  std::cout<<"-> Replaced:" << params->nodes_to_replace->size << "\n";
  for(int i = 0; i<params->nodes_to_replace->size; i++){
    TfLiteNode* delegated_node = nullptr;
    TfLiteRegistration* delegated_node_registration = nullptr;
    TF_LITE_ENSURE_EQ(
      context,
      context->GetNodeAndRegistration(context, params->nodes_to_replace->data[i], &delegated_node,
                                       &delegated_node_registration),
      kTfLiteOk);

    std::cout << "-> Node: "<< i << "\n";
    std::cout<<"-->Inputs Size:" << delegated_node->inputs->size << "\n";
    for(int j = 0; j < delegated_node->inputs->size; j++){
      std::cout << "--->data["<< j << "]: "<<delegated_node->inputs->data[j] << "\n";
    }
    std::cout<<"-->Output Size:" << delegated_node->outputs->size << "\n";
    for(int j = 0; j < delegated_node->outputs->size; j++){
      std::cout << "--->data["<< j << "]: "<<delegated_node->outputs->data[j] << "\n";
    }
    TfLiteConvParams convParams = *((TfLiteConvParams*)delegated_node->builtin_data);
    std::cout<<"padding: " << (int)convParams.padding << "\n";
    std::cout<<"stride_width: " << convParams.stride_width << "\n";
  }

  std::cout<<params->input_tensors->size<< "\n";
  std::cout<<params->output_tensors->size<< "\n";
  
  return kTfLiteOk;
}

TfLiteStatus VulkanKernel::Prepare(TfLiteContext* context, TfLiteNode* node) {
  if (1 < DelegatedNodesConv2D.size()){
    intermediateTensors.resize(DelegatedNodesConv2D.size()-1);
    int intTensorId = 0;
    for(auto nodeIt = DelegatedNodesConv2D.begin(); nodeIt<DelegatedNodesConv2D.end()-1;  nodeIt++, intTensorId++){    
      TfLiteTensor& outputTensor = context->tensors[nodeIt->outputTensorIdx];
      int outputSize = outputTensor.dims->data[1];
      intermediateTensors[intTensorId].resize(outputSize*outputSize);
    }
  }

  std::cout<<"\n\nPREPARATION\n";
  std::cout<<"Id: " << Id << "\n";

  std::cout<<"######### Tensors #######\n";
  std::cout<<"-> Size:" << context->tensors_size << "\n";
  //for(int i = 0; i<context->tensors_size; i++){
  //  std::cout << "->Tensor: "<< i << "\n";
  //  std::cout << "-->name: "<<(context->tensors[i].name) << "\n";
  //  std::cout << "-->AllocType: " <<(context->tensors[i].allocation_type) << "\n";
  //}
  std::cout<<"######## Inputs ########\n";
  std::cout<<"-> Inputs Size:" << node->inputs->size << "\n";
  //for(int i = 0; i < node->inputs->size; i++){
  //  std::cout << "-->data["<< i << "]: "<<node->inputs->data[i] << "\n";
  //}
  std::cout<<"######## Outputs ########\n";
  std::cout<<"-> Outputs Size:" << node->outputs->size << "\n";
  //for(int i = 0; i < node->outputs->size; i++){
  //  std::cout << "-->data["<< i << "]: "<<node->outputs->data[i] << "\n";
  //}

  return kTfLiteOk;
}

TfLiteStatus VulkanKernel::Eval(TfLiteContext* context, TfLiteNode* node) {
  int intTensorId = 0;
  std::cout<<"\n\nEVALUATION\n";
  for(auto nodeIt = DelegatedNodesConv2D.begin(); nodeIt<DelegatedNodesConv2D.end();  nodeIt++, intTensorId++){    

    std::vector<float*> inputs(1);
    std::vector<float*> weights(2);
    std::vector<float*> outputs(1);

    TfLiteTensor& inputTensor = context->tensors[nodeIt->inputTensorIdx];
    TfLiteTensor& kernelTensor = context->tensors[nodeIt->kernelTensorIdx];
    TfLiteTensor& biasTensor = context->tensors[nodeIt->biasTensorIdx];
    TfLiteTensor& outputTensor = context->tensors[nodeIt->outputTensorIdx];
    if(nodeIt == DelegatedNodesConv2D.begin()){
      inputs[0]  = reinterpret_cast<float*>(inputTensor.data.data);
    }
    else{
      inputs[0]  = intermediateTensors[intTensorId-1].data();
    }
    weights[0] = reinterpret_cast<float*>(kernelTensor.data.data);
    weights[1] = reinterpret_cast<float*>(biasTensor.data.data);
    if(nodeIt+1 == DelegatedNodesConv2D.end()){
      outputs[0] = reinterpret_cast<float*>(outputTensor.data.data);
    }else
    {
      outputs[0] = intermediateTensors[intTensorId].data();
    }
    std::cout<<"-> Input Size: "  <<inputTensor.dims->data[0]<<"x"<<inputTensor.dims->data[1]<<"x"<<inputTensor.dims->data[2]<<"x"<<inputTensor.dims->data[3] << "\n";
    std::cout<<"-> Kernel Size: " <<kernelTensor.dims->data[0]<<"x"<<kernelTensor.dims->data[1]<<"x"<<kernelTensor.dims->data[2]<<"x"<<kernelTensor.dims->data[3] << "\n";
    std::cout<<"-> Bias Size: " <<biasTensor.dims->data[0] << "\n";
    std::cout<<"-> output Size: " <<outputTensor.dims->data[0]<<"x"<<outputTensor.dims->data[1]<<"x"<<outputTensor.dims->data[2]<<"x"<<outputTensor.dims->data[3] << "\n";
    std::vector<MemDims> inputsDims = {{1, inputTensor.dims->data[1], inputTensor.dims->data[1], inputTensor.dims->data[3]}};
    std::vector<MemDims> weightsDims = {
      {kernelTensor.dims->data[0], kernelTensor.dims->data[1], kernelTensor.dims->data[1], kernelTensor.dims->data[3]},
      {1,1,1,biasTensor.dims->data[0]}};
    std::vector<MemDims> outputsDims = {{1, outputTensor.dims->data[1], outputTensor.dims->data[1], outputTensor.dims->data[3]}};

    //VulkanConvolution2D* vulkanConv = (static_cast<VulkanConvolution2D*>(vulkanPrimitive.get()));
    vulkanPrimitive->Init(inputs, inputsDims, weights, weightsDims, outputs, outputsDims);
    vulkanPrimitive->Process();
  }
  /*
  std::cout<<"\n\nEVALUATION\n";
  std::cout<<"Id: " << Id << "\n";
  TfLiteTensor& inputTensor_t = context->tensors[3];
  TfLiteTensor& kernelTensor_t = context->tensors[2];
  TfLiteTensor& outputTensor_t = context->tensors[0];

  float* inputData_t = reinterpret_cast<float*>(inputTensor_t.data.data);
  float* kernelData_t = reinterpret_cast<float*>(kernelTensor_t.data.data);
  float* outputData_t = reinterpret_cast<float*>(outputTensor_t.data.data);
  int inputSize_t = inputTensor_t.dims->data[1];
  int kernelSize_t = kernelTensor_t.dims->data[1];
  int OutputSize_t = outputTensor_t.dims->data[1];
  
  std::cout<<"######### Tensors #######\n";
  std::cout<<"-> Size:" << context->tensors_size << "\n";
  for(int i = 0; i<context->tensors_size; i++){
    std::cout << "->Tensor: "<< i << "\n";
    std::cout << "-->name: "<<(context->tensors[i].name) << "\n";
    std::cout << "-->AllocType: " <<(context->tensors[i].allocation_type) << "\n";
  }
  std::cout<<"######## Inputs ########\n";
  std::cout<<"-> Inputs Size:" << node->inputs->size << "\n";
  for(int i = 0; i < node->inputs->size; i++){
    std::cout << "-->data["<< i << "]: "<<node->inputs->data[i] << "\n";
  }
  std::cout<<"######## Outputs ########\n";
  std::cout<<"-> Outputs Size:" << node->outputs->size << "\n";
  for(int i = 0; i < node->outputs->size; i++){
    std::cout << "-->data["<< i << "]: "<<node->outputs->data[i] << "\n";
  }
  std::cout<<"######## intermediates ########\n";
  std::cout<<"-> intermediates Size:" << node->intermediates->size << "\n";
  for(int i = 0; i < node->intermediates->size; i++){
    std::cout << "-->data["<< i << "]: "<<node->intermediates->data[i] << "\n";
  }
  
  std::cout<<inputSize_t<<"\n";
  std::cout<<kernelSize_t<<"\n";
  std::cout<<OutputSize_t<<"\n";
*/
  return kTfLiteOk;
}

}  // namespace vulkan
}  // namespace tflite

