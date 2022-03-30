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
#include <unistd.h>
#include "vkutils.hpp"
#include <math.h> 

namespace vku = vkutils;
unsigned int microsecond = 1000000;

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
  std::cout<<"Creating Instance of Convolution 2D SIMPLE with vulkan"<<std::endl;
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
  //m_commandBuffers = p_device->allocateCommandBuffersUnique(
  //    {*commandPool, vk::CommandBufferLevel::ePrimary, 1});

  // Create Set Layout descriptor
  const std::array<vk::DescriptorSetLayoutBinding, 4> setLayoutBindings{
      {
      // binding,                       type,count,                          flags
       {0, vk::DescriptorType::eStorageBuffer, 1, 
        vk::ShaderStageFlagBits::eCompute},
       {1, vk::DescriptorType::eStorageBuffer, 1, 
        vk::ShaderStageFlagBits::eCompute},
       {2, vk::DescriptorType::eStorageBuffer, 1, 
        vk::ShaderStageFlagBits::eCompute},
       {3, vk::DescriptorType::eStorageBuffer, 1, 
        vk::ShaderStageFlagBits::eCompute}}};

  m_descriptorSetLayout = p_device->createDescriptorSetLayoutUnique(
      {vk::DescriptorSetLayoutCreateFlags(),
       static_cast<uint32_t>(setLayoutBindings.size()),
       setLayoutBindings.data()});

  // Create PoolDescriptor
  std::array<vk::DescriptorPoolSize, 1> poolSizes{
      {{vk::DescriptorType::eStorageBuffer, 4}}};

  m_descriptorPool = p_device->createDescriptorPoolUnique(
      {vk::DescriptorPoolCreateFlags(), 1,
       static_cast<uint32_t>(poolSizes.size()), poolSizes.data()});

  // Create Pipeline Layout
  m_pipelineLayout = p_device->createPipelineLayoutUnique(
      {vk::PipelineLayoutCreateFlags(), 1, &*m_descriptorSetLayout});

  // Load shader
  //m_shaderModule = loadShader("shaders/cross_correlation_strided.comp.spv");
  m_shaderModule = loadShader("shaders/cross_correlation_strided_depthed_batched_biased.comp.spv");

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

void VulkanConvolution2D::Init(std::vector<MemDims> inputsDims, 
                               std::vector<MemDims> weightsDims,                               
                               std::vector<MemDims> outputsDims){
  //if(m_newResources.size()!=0) return;
  inputDepth = weightsDims[0].Depth;
  outputDepth = weightsDims[0].Batch;
  inputSize = inputsDims[0].Height;
  kernelSize = weightsDims[0].Height;
  ComputeRealSizes(inputSize, kernelSize);

  std::cout<<"Initializing Vulkan Convolution"<<std::endl;

  workGroupSizeZ = outputDepth > 64 ? 64 : outputDepth;
  workGroupSize = sqrt(1024/workGroupSizeZ);
  

  m_specializationMapEntries = {
      {{0, offsetof(SpecializationData, inputSize),   
        sizeof(SpecializationData::inputSize)},
       {1, offsetof(SpecializationData, inputDepth),   
        sizeof(SpecializationData::inputDepth)},
       {2, offsetof(SpecializationData, kernelSize),   
        sizeof(SpecializationData::kernelSize)},
       {3, offsetof(SpecializationData, kernelOffset),   
        sizeof(SpecializationData::kernelOffset)},
       {4, offsetof(SpecializationData, outputSize),   
        sizeof(SpecializationData::outputSize)},
       {5, offsetof(SpecializationData, outputDepth),   
        sizeof(SpecializationData::outputDepth)},
       {6, offsetof(SpecializationData, workGroupSize),
        sizeof(SpecializationData::workGroupSize)},
       {7, offsetof(SpecializationData, workGroupSizeZ),
        sizeof(SpecializationData::workGroupSizeZ)},
       {8, offsetof(SpecializationData, stride),
        sizeof(SpecializationData::stride)}}};


  uint32_t kernelOffset =  kernelSize*kernelSize*inputDepth;
  m_specializationData = {inputSize, inputDepth, kernelSize, kernelOffset, 
                                              outputSize, outputDepth, workGroupSize, workGroupSizeZ, stride};
  
  //std::cout<<
  m_specializationInfo = {
      static_cast<uint32_t>(m_specializationMapEntries.size()),
      m_specializationMapEntries.data(), sizeof(SpecializationData),
      &m_specializationData};

  //vk::PipelineShaderStageCreateInfo shaderStage;
  m_shaderStage.stage = vk::ShaderStageFlagBits::eCompute;
  m_shaderStage.module = *m_shaderModule;
  m_shaderStage.pName = "main";
  m_shaderStage.pSpecializationInfo = &m_specializationInfo;
    
  //vk::ComputePipelineCreateInfo computePipelineCreateInfo;
  m_computePipelineCreateInfo.setLayout(*m_pipelineLayout);
  m_computePipelineCreateInfo.stage = m_shaderStage;

  //m_pipeline 
  //    = p_device->createComputePipelineUnique(nullptr, computePipelineCreateInfo);

  createResource(vk::BufferUsageFlagBits::eStorageBuffer,
                 vk::MemoryPropertyFlagBits::eHostVisible,
                 inputSize*inputSize*inputDepth*sizeof(float),
                 vk::AccessFlagBits::eHostWrite,
                 vk::AccessFlagBits::eShaderRead);
  createResource(vk::BufferUsageFlagBits::eStorageBuffer,
                 vk::MemoryPropertyFlagBits::eHostVisible,
                 outputDepth*kernelSize*kernelSize*inputDepth*sizeof(float),
                 vk::AccessFlagBits::eHostWrite,
                 vk::AccessFlagBits::eShaderRead);
  createResource(vk::BufferUsageFlagBits::eStorageBuffer,
                 vk::MemoryPropertyFlagBits::eHostVisible,
                 outputDepth*sizeof(float),
                 vk::AccessFlagBits::eHostWrite,
                 vk::AccessFlagBits::eShaderRead);
  createResource(vk::BufferUsageFlagBits::eTransferDst,
                 vk::MemoryPropertyFlagBits::eHostVisible,
                 outputSize*outputSize*outputDepth*sizeof(float),
                 vk::AccessFlagBits::eShaderWrite,
                 vk::AccessFlagBits::eHostRead);

  //vku::copyToDeviceMemory(*p_device, weights[0], outputDepth*kernelSize*kernelSize*inputDepth, m_newResources[1].memory);
  //vku::copyToDeviceMemory(*p_device, weights[1], outputDepth, m_newResources[2].memory);
  //copyInputToDeviceMemory(inputs[0], inputSize);

  const vk::DescriptorBufferInfo inputBufferDescriptor = { 
      m_newResources[0].buffer, 0, VK_WHOLE_SIZE };
  const vk::DescriptorBufferInfo kernelBufferDescriptor = { 
      m_newResources[1].buffer, 0, VK_WHOLE_SIZE };
  const vk::DescriptorBufferInfo outputBufferDescriptor = { 
      m_newResources[3].buffer, 0, VK_WHOLE_SIZE };
  const vk::DescriptorBufferInfo biasBufferDescriptor = { 
      m_newResources[2].buffer, 0, VK_WHOLE_SIZE };

  const std::array<vk::WriteDescriptorSet, 4> computeWriteDescriptorSets{
      {{m_descriptorSet[0], 0, 0, 1, vk::DescriptorType::eStorageBuffer, 
        nullptr, &inputBufferDescriptor},
       {m_descriptorSet[0], 1, 0, 1, vk::DescriptorType::eStorageBuffer, 
        nullptr, &kernelBufferDescriptor},
       {m_descriptorSet[0], 3, 0, 1, vk::DescriptorType::eStorageBuffer, 
        nullptr, &outputBufferDescriptor},
       {m_descriptorSet[0], 2, 0, 1, vk::DescriptorType::eStorageBuffer, 
        nullptr, &biasBufferDescriptor}}};

  p_device->updateDescriptorSets(computeWriteDescriptorSets, nullptr);

  //std::cout<<"Finished initializing Vulkan Convolution"<<std::endl;
}

void VulkanConvolution2D::Init()
{
  std::cout<<"Initializing Vulkan Convolution"<<std::endl;
}

void VulkanConvolution2D::SetCommandBuffer(vk::CommandBuffer& commandBuffer, vk::Pipeline pipeline)
{
  vk::CommandBufferBeginInfo info;
  //info.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
  commandBuffer.begin(info);

  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eHost,
                                vk::PipelineStageFlagBits::eComputeShader,
                                vk::DependencyFlags(), {}, 
                                {m_newResources[0].barrier, m_newResources[1].barrier, m_newResources[2].barrier}, {});

  commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, pipeline);
  commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
                                   *m_pipelineLayout, 0, {m_descriptorSet}, {});

  const auto numWorkgroups = (outputSize - 1)/workGroupSize + 1; // round up
  const auto numWorkgroupsZ = (outputDepth - 1)/workGroupSizeZ + 1; // round up
  commandBuffer.dispatch(numWorkgroups, numWorkgroups, numWorkgroupsZ);
  //std::cout << "----------------------> Num Work Groups: "<< numWorkgroups << "x"<< numWorkgroups << "x"<< outputDepth<<"="<< numWorkgroups*numWorkgroups*outputDepth << std::endl;
    
  commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
                                vk::PipelineStageFlagBits::eHost,
                                vk::DependencyFlags(), {}, 
                                {m_newResources[3].barrier} , {});

  commandBuffer.end();
}

void VulkanConvolution2D::createResource(vk::BufferUsageFlags useage, vk::MemoryPropertyFlags memFlags,
    vk::DeviceSize size, vk::AccessFlagBits srcAccessMask, vk::AccessFlagBits dstAccessMask)
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

  newRes.barrier.buffer = newRes.buffer;
  newRes.barrier.size = VK_WHOLE_SIZE;
  newRes.barrier.srcAccessMask = srcAccessMask;
  newRes.barrier.dstAccessMask = dstAccessMask;
  newRes.barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  newRes.barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

  m_newResources.push_back(newRes);
}

void VulkanConvolution2D::Process(std::vector<float*> inputs, std::vector<float*> weights, 
                               std::vector<float*> outputs) 
{
  
  std::vector<vk::UniqueCommandBuffer> commandBuffers = p_device->allocateCommandBuffersUnique(
      {*commandPool, vk::CommandBufferLevel::ePrimary, 1});

  vk::Pipeline pipeline 
    = p_device->createComputePipeline(nullptr, m_computePipelineCreateInfo);
  SetCommandBuffer(*commandBuffers[0], pipeline);



  m_output = outputs[0];
  //std::cout << "Processing a Convolution"<<std::endl;
  vku::copyToDeviceMemory(*p_device, weights[0], outputDepth*kernelSize*kernelSize*inputDepth, m_newResources[1].memory);
  vku::copyToDeviceMemory(*p_device, weights[1], outputDepth, m_newResources[2].memory);
  copyInputToDeviceMemory(inputs[0], inputSize);

  vk::Queue queue = p_device->getQueue(m_queueFamilyIndex, 0);
    
  const vk::PipelineStageFlags waitStageMask(vk::PipelineStageFlagBits::eHost);

  vk::SubmitInfo computeSubmitInfo;
  computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
  computeSubmitInfo.commandBufferCount = 1;
  computeSubmitInfo.pCommandBuffers = &*commandBuffers[0];

  auto fence = p_device->createFenceUnique({});
  queue.submit({computeSubmitInfo}, *fence);
  p_device->waitForFences( {*fence}, true, std::numeric_limits<uint64_t>::max());
    
  vku::copyFromDeviceMemory(*p_device, m_newResources[3].memory, m_output, outputSize*outputSize*outputDepth);

  p_device->destroyPipeline( pipeline );
}

void VulkanConvolution2D::ComputeRealSizes(uint32_t& inputSize, uint32_t kernelSize) 
{
  if(1<control.stride_h){
    stride = control.stride_h;
  } else {
    stride = 1;
  }

  if(1==control.Padding){
    //std::cout << "Padding Same\n";
    padding = 0;
    if(0==inputSize%stride)
    {
      padding = kernelSize > stride ? kernelSize - stride : 0;
    }else{
      padding = kernelSize > inputSize%stride ? kernelSize - inputSize%stride : 0;
    }
  }else{
    //std::cout << "Padding Valid\n";
    padding = 0;
  }
  
  inputSize += padding;
  outputSize = (inputSize - kernelSize)/stride + 1;
  std::cout << "Stride: " << stride << std::endl;
  std::cout << "Padding: "<<padding << std::endl;
  std::cout << "inputSize: "<<inputSize << std::endl;
  std::cout << "inputDepth: "<<inputDepth << std::endl;
  std::cout << "outputSize: "<<outputSize << std::endl;
  std::cout << "outputDepth: "<<outputDepth << std::endl;
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
  uint32_t inputFlatSize = (inputSize)*(inputSize)*inputDepth;

  if(1==control.Padding && 0 != padding){
    //std::cout << "Padding Input\n";
    uint32_t origInputSize = inputSize-padding;
    uint32_t paddingTop = padding/2;
    const auto bufferSize = origInputSize *inputDepth* sizeof(float);

    std::vector<float> paddedInput(inputFlatSize, 0);
    float* paddedInput_p = paddedInput.data();
    
    for(uint32_t row = 0; row < origInputSize; row++){
      int paddedOffset = (row+paddingTop)*inputSize + paddingTop;
      int inputOffset = (row)*origInputSize;
      std::memcpy(paddedInput_p + paddedOffset*inputDepth, input + inputOffset*inputDepth, bufferSize);
    }
    
    //for (size_t y=0; y<inputSize; ++y) {
    //  std::cout << "| ";
    //  for (size_t x=0; x<inputSize; ++x) {
    //    std::cout << std::setw(10) << std::setprecision(3)
    //      << paddedInput[x + inputSize*y] << ' ';
    //  }
    //  std::cout << "|\n";
    //}

    vku::copyToDeviceMemory(*p_device, paddedInput_p, inputFlatSize, m_newResources[0].memory);
    //std::cout << "Input Padded\n";
  } else{
    vku::copyToDeviceMemory(*p_device, input, inputFlatSize, m_newResources[0].memory);
  }
  
}
