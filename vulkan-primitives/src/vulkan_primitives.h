#pragma once
#include <vulkan/vulkan.hpp>
#include <fstream>

struct MemDims{
  uint32_t Batch = 1;
  uint32_t Height = 1;
  uint32_t Width = 1;
  uint32_t Depth = 1;
};

class VulkanPrimitive
{

 public:

  virtual ~VulkanPrimitive();
  virtual void Init(std::vector<float*> inputs, std::vector<MemDims> inputsDims, 
                    std::vector<float*> weights, std::vector<MemDims> weightsDims, 
                    std::vector<float*> outputs, std::vector<MemDims> outputsDims) = 0;
  virtual void Process() = 0;

 protected:
  vk::PhysicalDevice* p_physicalDevice;
  int m_queueFamilyIndex = -1;
  vk::Device* p_device;

  uint32_t  outputSize;
  uint32_t  workGroupSize;

  vk::UniqueCommandPool commandPool;
  std::vector<vk::UniqueCommandBuffer> m_commandBuffers;
  vk::UniqueDescriptorSetLayout m_descriptorSetLayout;
  vk::UniqueDescriptorPool m_descriptorPool;
  vk::UniqueShaderModule m_shaderModule;
  std::vector<vk::DescriptorSet> m_descriptorSet;
    
  vk::UniquePipelineLayout m_pipelineLayout;
  vk::UniquePipeline m_pipeline;

  float* m_output = nullptr;

  vk::UniqueShaderModule loadShader(const char *fileName);
};
