#pragma once
#ifndef TENSORFLOW_LITE_DELEGATES_VULKAN_VULKAN_DELEGATE_H
#define TENSORFLOW_LITE_DELEGATES_VULKAN_VULKAN_DELEGATE_H_

#include "vulkan-primitives/src/vulkan_primitives_factory.h"
#include "tensorflow/lite/delegates/utils/simple_delegate.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/builtin_op_data.h"

#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct {
  // Allowed ops to delegate.
  int allowed_builtin_code;
  // Report error during init.
  bool error_during_init;
  // Report error during prepare.
  bool error_during_prepare;
  // Report error during invoke.
  bool error_during_invoke;
} VulkanDelegateOptions;

#ifdef __cplusplus
}
#endif  // __cplusplus

namespace tflite {
namespace vulkan {

class VulkanDelegate : public SimpleDelegateInterface {
 public:
  explicit VulkanDelegate(const VulkanDelegateOptions& options)
      : options_(options) {
        vulkanPrimitivesFact = std::make_shared<VulkanPrimitivesFactory>();
      }
  virtual bool IsNodeSupportedByDelegate(
      const TfLiteRegistration* registration, const TfLiteNode* node,
      TfLiteContext* context) const override {
    bool retVal = false;
    switch(registration-> builtin_code){   
      case kTfLiteBuiltinConv2d: {
        TfLiteConvParams convParams = *((TfLiteConvParams*)node->builtin_data);
        if(convParams.activation == kTfLiteActRelu6)
        retVal = true;
        break;
      } 
      //case kTfLiteBuiltinDepthwiseConv2d: {
      //  retVal = true;
      //  break;
      //}
    }
    return retVal;
  }

  virtual const char* Name() const override { return "Vulkan Delegate"; }

  virtual SimpleDelegateInterface::Options DelegateOptions() const override {
    return SimpleDelegateInterface::Options();
  }

  virtual std::unique_ptr<SimpleDelegateKernelInterface>
  CreateDelegateKernelInterface() override;
  virtual TfLiteStatus Initialize(TfLiteContext* context) override;

 private:
  const VulkanDelegateOptions options_;
  std::shared_ptr<VulkanPrimitivesFactory> vulkanPrimitivesFact = nullptr;
};

}  // namespace vulkan
}  // namespace tflite


#endif  // TENSORFLOW_LITE_DELEGATES_VULKAN_VULKAN_DELEGATE_H_
