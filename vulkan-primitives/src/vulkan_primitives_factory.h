#pragma once

#include "vulkan_primitives.h"
//#include "vulkan_convolution_k1x1.h"
#include "vulkan_convolution.h"
#include <memory>
#include <vulkan/vulkan.hpp>

enum PrimitiveType {
  Vulkan_Conv2d,
  Vulkan_Conv2d_k1x1,
  Vulkan_Last
};

class VulkanPrimitivesFactory
{
 public:
  VulkanPrimitivesFactory();
  ~VulkanPrimitivesFactory();

  std::unique_ptr<VulkanPrimitive> GetPrimitive(PrimitiveType type, 
                                                uint32_t control);

 private:
  vk::Instance instance;
  vk::PhysicalDevice physicalDevice;
  int queueFamilyIndex = -1;
  vk::Device device;
};

