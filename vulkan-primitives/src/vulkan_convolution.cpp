#undef VULKAN_HPP_DISABLE_ENHANCED_MODE

#include <vulkan/vulkan.hpp>

#include "vulkan_primitives.h"
#include "vulkan_convolution.h"
//#include "cross_correlation.hpp"

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <getopt.h>
#include <iomanip>

#include "vkutils.hpp"
namespace vku = vkutils;

struct Resources {
  vku::Resource input;
  vku::Resource kernel;
  vku::Resource output;
};

VulkanConvolution2D::VulkanConvolution2D()
{
  std::cout<<"Creating Instance of Convolution"<<std::endl;
}

VulkanConvolution2D::VulkanConvolution2D(vk::PhysicalDevice* physicalDevice, 
                                         int queueFamilyIndex, 
                                         vk::Device* device, 
                                         VulkanConv2D_Control primitiveControl)
    : control(primitiveControl)
{
  std::cout<<"Creating Instance of Convolution with vulkan"<<std::endl;
  // VkPhysicalDeviceProperties pProperties;//physicalDevice->getPhysicalDeviceProperties();
  // vkGetPhysicalDeviceProperties(*physicalDevice, &pProperties);
  // 
  // std::cout<<"deviceName: "<<pProperties.deviceName<<std::endl;
  // std::cout<<"queueFamilyIndex: "<<queueFamilyIndex<<std::endl;

  p_physicalDevice = physicalDevice;
  m_queueFamilyIndex = queueFamilyIndex;
  p_device = device;

  // Create command pool
  commandPool = p_device->createCommandPoolUnique(vk::CommandPoolCreateInfo(
      vk::CommandPoolCreateFlagBits::eResetCommandBuffer, m_queueFamilyIndex));

  // Create command buffer
  m_commandBuffers = p_device->allocateCommandBuffersUnique(
      {*commandPool, vk::CommandBufferLevel::ePrimary, 1});

  // Create Set Layout descriptor
  const std::array<vk::DescriptorSetLayoutBinding, 3> setLayoutBindings{
      {
      // binding,                       type,count,                          flags
       {0, vk::DescriptorType::eStorageBuffer, 1, 
        vk::ShaderStageFlagBits::eCompute},
       {1, vk::DescriptorType::eStorageBuffer, 1, 
        vk::ShaderStageFlagBits::eCompute},
       {2, vk::DescriptorType::eStorageBuffer, 1, 
        vk::ShaderStageFlagBits::eCompute}}};

  m_descriptorSetLayout = p_device->createDescriptorSetLayoutUnique(
      {vk::DescriptorSetLayoutCreateFlags(),
       static_cast<uint32_t>(setLayoutBindings.size()),
       setLayoutBindings.data()});

  // Create PoolDescriptor
  std::array<vk::DescriptorPoolSize, 1> poolSizes{
      {{vk::DescriptorType::eStorageBuffer, 3}}};

  m_descriptorPool = p_device->createDescriptorPoolUnique(
      {vk::DescriptorPoolCreateFlags(), 1,
       static_cast<uint32_t>(poolSizes.size()), poolSizes.data()});

  // Create Pipeline Layout
  m_pipelineLayout = p_device->createPipelineLayoutUnique(
      {vk::PipelineLayoutCreateFlags(), 1, &*m_descriptorSetLayout});

  // Load shader
  m_shaderModule = loadShader("shaders/cross_correlation.comp.spv");

  //Create Descriptor Sets
  m_descriptorSet = p_device->allocateDescriptorSets(
      {*m_descriptorPool, 1, &*m_descriptorSetLayout});
  std::cout<<"Finished creating Instance of Convolution with vulkan"
           <<std::endl;
}

VulkanConvolution2D::~VulkanConvolution2D()
{
  std::cout<<"Destroy Vulkan Convolution"<<std::endl;
}

void VulkanConvolution2D::Init()
{
  std::cout<<"Initializing Vulkan Convolution"<<std::endl;
}

void VulkanConvolution2D::Init(std::vector<float>& input, uint32_t inputSize, 
                                std::vector<float>& kernel, uint32_t kernelSize, 
                                std::vector<float>& output)
{
  uint32_t computedOutputSize = ComputeOutputSize(inputSize, kernelSize);
  output.resize(computedOutputSize*computedOutputSize);
  Init(input.data(), inputSize, kernel.data(), kernelSize, output.data());
}

void VulkanConvolution2D::Init(float* input, uint32_t inputSize, 
                                float* kernel, uint32_t kernelSize, 
                                float* output)
{
  std::cout<<"Initializing Vulkan Convolution"<<std::endl;
  ComputeRealSizes(inputSize, kernelSize);
  m_output = output;

  workGroupSize = 16;
  // Pass SSBO size via specialization constant
  struct SpecializationData {
    uint32_t kernelSize;
    uint32_t outputSize;
    uint32_t workGroupSize;
  };

  const std::array<vk::SpecializationMapEntry, 3> specializationMapEntries{
      {{0, offsetof(SpecializationData, kernelSize),   
        sizeof(SpecializationData::kernelSize)},
       {1, offsetof(SpecializationData, outputSize),   
        sizeof(SpecializationData::outputSize)},
       {2, offsetof(SpecializationData, workGroupSize),
        sizeof(SpecializationData::workGroupSize)}}};

  const SpecializationData specializationData{kernelSize, outputSize,
                                              workGroupSize};

  const vk::SpecializationInfo specializationInfo{
      static_cast<uint32_t>(specializationMapEntries.size()),
      specializationMapEntries.data(), sizeof(SpecializationData),
      &specializationData};

  vk::PipelineShaderStageCreateInfo shaderStage;
  shaderStage.stage = vk::ShaderStageFlagBits::eCompute;
  shaderStage.module = *m_shaderModule;
  shaderStage.pName = "main";
  shaderStage.pSpecializationInfo = &specializationInfo;
    
  vk::ComputePipelineCreateInfo computePipelineCreateInfo;
  computePipelineCreateInfo.setLayout(*m_pipelineLayout);
  computePipelineCreateInfo.stage = shaderStage;

  m_pipeline 
      = p_device->createComputePipelineUnique(nullptr, computePipelineCreateInfo);

  createResource(vk::BufferUsageFlagBits::eStorageBuffer,
                 vk::MemoryPropertyFlagBits::eHostVisible,
                 inputSize*inputSize*sizeof(float));
  createResource(vk::BufferUsageFlagBits::eStorageBuffer,
                 vk::MemoryPropertyFlagBits::eHostVisible,
                 kernelSize*kernelSize*sizeof(float));
  createResource(vk::BufferUsageFlagBits::eStorageBuffer,
                 vk::MemoryPropertyFlagBits::eHostVisible,
                 outputSize*outputSize*sizeof(float));

  vku::copyToDeviceMemory(*p_device, kernel, kernelSize*kernelSize, m_newResources[1].memory);
  copyInputToDeviceMemory(input, inputSize);

  const vk::DescriptorBufferInfo inputBufferDescriptor = { 
      m_newResources[0].buffer, 0, VK_WHOLE_SIZE };
  const vk::DescriptorBufferInfo kernelBufferDescriptor = { 
      m_newResources[1].buffer, 0, VK_WHOLE_SIZE };
  const vk::DescriptorBufferInfo outputBufferDescriptor = { 
      m_newResources[2].buffer, 0, VK_WHOLE_SIZE };

  const std::array<vk::WriteDescriptorSet, 3> computeWriteDescriptorSets{
      {{m_descriptorSet[0], 0, 0, 1, vk::DescriptorType::eStorageBuffer, 
        nullptr, &inputBufferDescriptor},
       {m_descriptorSet[0], 1, 0, 1, vk::DescriptorType::eStorageBuffer, 
        nullptr, &kernelBufferDescriptor},
       {m_descriptorSet[0], 2, 0, 1, vk::DescriptorType::eStorageBuffer, 
        nullptr, &outputBufferDescriptor}}};

  p_device->updateDescriptorSets(computeWriteDescriptorSets, nullptr);

  std::cout<<"Finished initializing Vulkan Convolution"<<std::endl;
  SetCommandBuffer(*m_commandBuffers[0]);
}

void VulkanConvolution2D::SetCommandBuffer(vk::CommandBuffer& commandBuffer)
{
  vk::CommandBufferBeginInfo info;
  //info.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
  commandBuffer.begin(info);

  vk::BufferMemoryBarrier inputBufferBarrier;
  inputBufferBarrier.buffer = m_newResources[0].buffer;
  inputBufferBarrier.size = VK_WHOLE_SIZE;
  inputBufferBarrier.srcAccessMask = vk::AccessFlagBits::eHostWrite;
  inputBufferBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
  inputBufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  inputBufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

  vk::BufferMemoryBarrier kernelBufferBarrier;
  kernelBufferBarrier.buffer = m_newResources[1].buffer;
  kernelBufferBarrier.size = VK_WHOLE_SIZE;
  kernelBufferBarrier.srcAccessMask = vk::AccessFlagBits::eHostWrite;
  kernelBufferBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
  kernelBufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  kernelBufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eHost,
                                vk::PipelineStageFlagBits::eComputeShader,
                                vk::DependencyFlags(), {}, 
                                {inputBufferBarrier, kernelBufferBarrier}, {});

  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *m_pipeline);
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   *m_pipelineLayout, 0, {m_descriptorSet}, {});

  const auto numWorkgroups = (outputSize - 1)/workGroupSize + 1; // round up
  commandBuffer.dispatch(numWorkgroups, numWorkgroups, 1);

  vk::BufferMemoryBarrier outputBufferBarrier;
  outputBufferBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
  outputBufferBarrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
  outputBufferBarrier.buffer = m_newResources[2].buffer;
  outputBufferBarrier.size = VK_WHOLE_SIZE;
  outputBufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  outputBufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eHost,
                                vk::DependencyFlags(), {}, 
                                {outputBufferBarrier} , {});

  commandBuffer.end();
}

void VulkanConvolution2D::createResource(vk::BufferUsageFlags useage, vk::MemoryPropertyFlags memFlags,
    vk::DeviceSize size)
{
  NewResource newRes;

  // Create the buffer handle
  vk::BufferCreateInfo bufferCreateInfo;
  bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive; // default
  bufferCreateInfo.usage = useage;
  bufferCreateInfo.size = size;

  newRes.buffer = p_device->createBuffer(bufferCreateInfo);

  auto deviceMemoryProperties = p_physicalDevice->getMemoryProperties();
  auto memReqs = p_device->getBufferMemoryRequirements(newRes.buffer);

  vk::MemoryAllocateInfo memAlloc;
  memAlloc.allocationSize = memReqs.size;

  auto memoryTypeIndex = vku::findProperties(deviceMemoryProperties,
                                             memReqs.memoryTypeBits, memFlags);
  if (!memoryTypeIndex) return; //throw Error("no memory found");
  memAlloc.memoryTypeIndex = memoryTypeIndex.value();

  newRes.memory = p_device->allocateMemory(memAlloc, nullptr);

  p_device->bindBufferMemory(newRes.buffer, newRes.memory, 0);

  m_newResources.push_back(newRes);
}

void VulkanConvolution2D::Process() 
{
  std::cout << "Processing a Convolution"<<std::endl;

  vk::Queue queue = p_device->getQueue(m_queueFamilyIndex, 0);
    
  const vk::PipelineStageFlags waitStageMask(vk::PipelineStageFlagBits::eHost);

  vk::SubmitInfo computeSubmitInfo;
  computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
  computeSubmitInfo.commandBufferCount = 1;
  computeSubmitInfo.pCommandBuffers = &*m_commandBuffers[0];

  auto fence = p_device->createFenceUnique({});
  queue.submit({computeSubmitInfo}, *fence);
  p_device->waitForFences( {*fence}, true, std::numeric_limits<uint64_t>::max());
    
  vku::copyFromDeviceMemory(*p_device, m_newResources[2].memory, m_output, outputSize*outputSize);
}

void VulkanConvolution2D::ComputeRealSizes(uint32_t& inputSize, uint32_t kernelSize) 
{
  if(1==control.Padding){
    std::cout << "Padding Same\n";
    outputSize = inputSize;
    uint32_t padding = kernelSize-1;
    paddingTop = padding/2;
    paddingBottom = padding-paddingTop;
    inputSize += (padding);
    std::cout << paddingTop << std::endl;
    std::cout << paddingBottom << std::endl;
  }else {
    std::cout << "Padding Valid\n";
    outputSize = inputSize - (kernelSize-1);
  }
}

uint32_t VulkanConvolution2D::ComputeOutputSize(uint32_t inputSize, uint32_t kernelSize) 
{
  uint32_t computedOutputSize = 0;
  if(1==control.Padding){
    computedOutputSize = inputSize;
  } else{
    computedOutputSize = inputSize - (kernelSize-1);
  }
  return computedOutputSize;
}

void VulkanConvolution2D::copyInputToDeviceMemory(float* input, uint32_t inputSize){
  uint32_t inputFlatSize = inputSize*inputSize;

  if(1==control.Padding){
    std::cout << "Padding Input\n";
    uint32_t origInputSize = inputSize-paddingTop-paddingBottom;
    const auto bufferSize = origInputSize * sizeof(float);

    std::vector<float> paddedInput(inputFlatSize, 0);
    float* paddedInput_p = paddedInput.data();
    
    for(uint32_t row = 0; row < origInputSize; row++){
      int paddedOffset = (row+paddingTop)*inputSize + paddingTop;
      int inputOffset = (row)*origInputSize;
      std::memcpy(paddedInput_p + paddedOffset, input + inputOffset, bufferSize);
    }
    
    vku::copyToDeviceMemory(*p_device, paddedInput_p, inputFlatSize, m_newResources[0].memory);
    std::cout << "Input Padded\n";
  } else{
    vku::copyToDeviceMemory(*p_device, input, inputFlatSize, m_newResources[0].memory);
  }
  
}
