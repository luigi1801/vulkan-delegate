#ifndef TENSORFLOW_LITE_DELEGATES_VULKAN_VULKAN_DELEGATE_KERNEL_H_
#define TENSORFLOW_LITE_DELEGATES_VULKAN_VULKAN_DELEGATE_KERNEL_H_

#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "vulkan_delegate.h"

namespace tflite {
namespace vulkan {

class VulkanKernel : public SimpleDelegateKernelInterface {
 public:
  explicit VulkanKernel(const VulkanDelegateOptions& options, std::shared_ptr<VulkanPrimitivesFactory> vulkanPrimitivesFact)
      : options_(options), vulkanPrimitivesFact_(vulkanPrimitivesFact) {}

  TfLiteStatus Init(TfLiteContext* context,
                    const TfLiteDelegateParams* params) override;

  TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) override;

  TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) override;

 private:
  typedef struct Conv2D_Node {
    int inputTensorIdx = 0;
    int outputTensorIdx = 0;
    int kernelTensorIdx = 0;

    float* inputData = nullptr;
    float* kernelData = nullptr;
    float* outputData = nullptr;
    int inputSize = 0;
    int kernelSize = 0;
    int OutputSize = 0;
  } Conv2D_Node;
  std::vector<Conv2D_Node> DelegatedNodesConv2D;
  std::vector<std::vector<float>> intermediateTensors;

////// Version 3: Graphs with several nodes, even continuos
//  typedef struct Conv2D_Node {
//    int inputTensorIdx = 0;
//    int outputTensorIdx = 0;
//    int kernelTensorIdx = 0;
//
//    float* inputData = nullptr;
//    float* kernelData = nullptr;
//    float* outputData = nullptr;
//    int inputSize = 0;
//    int kernelSize = 0;
//    int OutputSize = 0;
//  } Conv2D_Node;
//  std::vector<Conv2D_Node> DelegatedNodesConv2D;
//  std::vector<std::vector<float>> intermediateTensors;

////// Version 2: Graphs with several nodes, not continuous
//  typedef struct Conv2D_Node {
//    int inputTensorIdx = 0;
//    int outputTensorIdx = 0;
//    int kernelTensorIdx = 0;
//  } Conv2D_Node;
//  std::vector<Conv2D_Node> DelegatedNodesConv2D;

  const VulkanDelegateOptions options_;
  std::shared_ptr<VulkanPrimitivesFactory> vulkanPrimitivesFact_ = nullptr;
  std::unique_ptr<VulkanPrimitive> vulkanPrimitive = nullptr;
  int Id = 0;
};

}  // namespace vulkan
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_VULKAN_VULKAN_DELEGATE_KERNEL_H_
