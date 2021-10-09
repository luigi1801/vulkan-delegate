#pragma once

#undef VULKAN_HPP_DISABLE_ENHANCED_MODE

#include "vkdebug.hpp"
#include "vkutils.hpp"

#include <vulkan/vulkan.hpp>

#include <numeric>
#include <utility>
#include <cmath>

namespace vku = vkutils;

class ValidCrossCorrelation2DVulkan {
  public:
    class Error : public std::runtime_error {
      public:
        using std::runtime_error::runtime_error;
    };

    ValidCrossCorrelation2DVulkan(uint32_t inputSize, uint32_t kernelSize,
        uint32_t workgroupSize=1)
      : inputSize { inputSize },
        kernelSize { kernelSize },
        workgroupSize_m { workgroupSize }

    {
      if (kernelSize > inputSize || kernelSize % 2 != 1) {
        throw Error("invalid kernel size");
      }
      commandBufferSetConvolutionOperation(*commandBuffers[0]);
    }

    void
    run()
    { crossCorrelation(); }

    uint32_t
    outputSize() const
    { return inputSize - (kernelSize-1); }

    uint32_t
    workGroupSize()
    {
      const auto limits = physicalDevice.getProperties().limits;

      const auto sizeMax = std::min<uint32_t>({
          limits.maxComputeWorkGroupSize[0],
          limits.maxComputeWorkGroupSize[1],
          static_cast<uint32_t>(sqrt(limits.maxComputeWorkGroupInvocations))
          });
      const auto countMax = std::min<uint32_t>(limits.maxComputeWorkGroupCount[0],
                                         limits.maxComputeWorkGroupCount[1]);

      const auto sizeMin = std::max<uint32_t>(1, (outputSize() - 1) / countMax + 1);

      if (sizeMax < sizeMin) {
        throw Error("no possible valid work group size");
      }

      return std::min(std::max(sizeMin, workgroupSize_m), sizeMax);
    }

    void
    setKernel(const std::vector<float>& data)
    {
      assert(data.size() == kernelSize*kernelSize);

      vku::copyToDeviceMemory(*device, data, resources.kernel.memory);
    }

    void
    setInput(const std::vector<float>& data)
    {
      assert(data.size() == inputSize*inputSize);

      vku::copyToDeviceMemory(*device, data, resources.input.memory);
    }

    std::vector<float>
    getOuptput()
    {
      std::vector<float> ret(outputSize()*outputSize());
      vku::copyFromDeviceMemory(*device, resources.output.memory, ret);

      return ret;
    }

  private:
    uint32_t inputSize;
    uint32_t kernelSize;
    uint32_t workgroupSize_m;

    vk::UniqueInstance instance = createUniqueInstance();

#ifndef NDEBUG
    vk::UniqueDebugReportCallbackEXT debugReportCallback = createDebugCallback();;
#endif

    vk::PhysicalDevice physicalDevice = pickPhysicalDevice();


    vk::UniqueDevice device = createLogicalDevice(queueFamilyIndex());
    vk::Queue queue = device->getQueue(queueFamilyIndex(), 0);

    vk::UniqueCommandPool commandPool = device->createCommandPoolUnique(
        vk::CommandPoolCreateInfo(
          vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
          queueFamilyIndex()));

    std::vector<vk::UniqueCommandBuffer> commandBuffers
      = device->allocateCommandBuffersUnique({ *commandPool,
          vk::CommandBufferLevel::ePrimary, 1});

    vk::UniqueDescriptorSetLayout descriptorSetLayout
      = createDescriptorSetLayout();

    vk::UniquePipelineLayout pipelineLayout
      = device->createPipelineLayoutUnique({vk::PipelineLayoutCreateFlags(),
          1, &*descriptorSetLayout});

    vk::UniqueShaderModule shaderModule
      = vku::loadShader("shaders/cross_correlation.comp.spv", *device);

    vk::UniquePipeline pipeline = initPipeline();

    std::array<vk::DescriptorPoolSize, 1> poolSizes {{
      {vk::DescriptorType::eStorageBuffer, 3}
    }};

    struct Resources {
      vku::Resource input;
      vku::Resource kernel;
      vku::Resource output;
    };
    Resources resources = {
      {*device, physicalDevice,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible,
        inputSize*inputSize*sizeof(float)},

      {*device, physicalDevice,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible,
        kernelSize*kernelSize*sizeof(float)},

      {*device, physicalDevice,
        vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible,
        outputSize()*outputSize()*sizeof(float)}};

    vk::UniqueDescriptorPool descriptorPool
      = device->createDescriptorPoolUnique({vk::DescriptorPoolCreateFlags(), 1 ,
          static_cast<uint32_t>(poolSizes.size()), poolSizes.data()});

    // not unique since they are deallocated when the pool is destroyed
    std::vector<vk::DescriptorSet> descriptorSet = createDescriptorSets();

    uint32_t
    queueFamilyIndex() const
    {
      return vku::getQueueFamilyIndex(physicalDevice,
          vk::QueueFlagBits::eCompute).value();
    }

    vk::UniqueDescriptorSetLayout
    createDescriptorSetLayout()
    {
      const std::array<vk::DescriptorSetLayoutBinding, 3> setLayoutBindings {{
        // binding,                       type,count,                          flags
        {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}
      }};

      return device->createDescriptorSetLayoutUnique({
          vk::DescriptorSetLayoutCreateFlags(),
          static_cast<uint32_t>(setLayoutBindings.size()),
          setLayoutBindings.data()
          });
    }

    vk::UniquePipeline
    initPipeline()
    {
      // Pass SSBO size via specialization constant
      struct SpecializationData {
        uint32_t kernelSize;
        uint32_t outputSize;
        uint32_t workGroupSize;
      };

      const std::array<vk::SpecializationMapEntry, 3> specializationMapEntries {{
        {0, offsetof(SpecializationData, kernelSize),    sizeof(SpecializationData::kernelSize)},
        {1, offsetof(SpecializationData, outputSize),    sizeof(SpecializationData::outputSize)},
        {2, offsetof(SpecializationData, workGroupSize), sizeof(SpecializationData::workGroupSize)}
      }};

      const SpecializationData specializationData { kernelSize, outputSize(),
        workGroupSize() };

      const vk::SpecializationInfo specializationInfo {
        static_cast<uint32_t>(specializationMapEntries.size()),
        specializationMapEntries.data(), sizeof(SpecializationData),
        &specializationData
      };

      vk::PipelineShaderStageCreateInfo shaderStage;
      shaderStage.stage = vk::ShaderStageFlagBits::eCompute;
      shaderStage.module = *shaderModule;
      shaderStage.pName = "main";
      shaderStage.pSpecializationInfo = &specializationInfo;

      vk::ComputePipelineCreateInfo computePipelineCreateInfo;
      computePipelineCreateInfo.setLayout(*pipelineLayout);
      computePipelineCreateInfo.stage = shaderStage;

      return device->createComputePipelineUnique(nullptr,
          computePipelineCreateInfo);
    }


    void
    updateDescriptorSets(const std::vector<vk::DescriptorSet>& descriptorSet)
    {
      const vk::DescriptorBufferInfo inputBufferDescriptor
        = { resources.input.buffer, 0, VK_WHOLE_SIZE };
      const vk::DescriptorBufferInfo kernelBufferDescriptor
        = { resources.kernel.buffer, 0, VK_WHOLE_SIZE };
      const vk::DescriptorBufferInfo outputBufferDescriptor
        = { resources.output.buffer, 0, VK_WHOLE_SIZE };

      const std::array<vk::WriteDescriptorSet, 3> computeWriteDescriptorSets {{
        {descriptorSet[0], 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &inputBufferDescriptor},
        {descriptorSet[0], 1, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &kernelBufferDescriptor},
        {descriptorSet[0], 2, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &outputBufferDescriptor}
      }};

      device->updateDescriptorSets(computeWriteDescriptorSets, nullptr);
    }

    static vk::UniqueInstance
    createUniqueInstance()
    {
      vk::ApplicationInfo appInfo("Cross-correlation");
      appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
      appInfo.apiVersion = VK_API_VERSION_1_0;

      vk::InstanceCreateInfo createInfo;
      createInfo.pApplicationInfo = &appInfo;

#ifdef NDEBUG
      return vk::createInstanceUnique(createInfo);
#else
      return vku::debug::createInstanceUnique(createInfo);
#endif
    }

#ifndef NDEBUG
    vk::UniqueDebugReportCallbackEXT
    createDebugCallback()
    {
      if (!vku::debug::isValidationLayerSupported())
        return vk::UniqueDebugReportCallbackEXT(); // VK_NULL_HANDLE

      return instance->createDebugReportCallbackEXTUnique(
          { vk::DebugReportFlagBitsEXT::eError
          | vk::DebugReportFlagBitsEXT::eWarning
          | vk::DebugReportFlagBitsEXT::ePerformanceWarning,
          vku::debug::reportCallback });
    }
#endif

    vk::PhysicalDevice
    pickPhysicalDevice()
    {
      const auto devices = instance->enumeratePhysicalDevices();
      const auto device = find_if(devices.begin(), devices.end(),
          [](auto& device) {
          return vku::getQueueFamilyIndex(device, vk::QueueFlagBits::eCompute);
          });
      if (device == devices.end()) throw Error("No suitable GPU found");

      return *device;
    }

    vk::UniqueDevice
    createLogicalDevice(uint32_t queueFamilyIndex)
    {
      vk::DeviceQueueCreateInfo queueCreateInfo;

      queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
      queueCreateInfo.queueCount = 1;
      float queuePriority = 1;
      queueCreateInfo.pQueuePriorities = &queuePriority;

      vk::PhysicalDeviceFeatures deviceFeatures;

      vk::DeviceCreateInfo createInfo;
      createInfo.pQueueCreateInfos = &queueCreateInfo;
      createInfo.queueCreateInfoCount = 1;

      createInfo.pEnabledFeatures = &deviceFeatures;

      return physicalDevice.createDeviceUnique(createInfo);
    }

    std::vector<vk::DescriptorSet>
    createDescriptorSets()
    {
      auto descriptorSet = device->allocateDescriptorSets(
          {*descriptorPool, 1, &*descriptorSetLayout});
      updateDescriptorSets(descriptorSet);

      return descriptorSet;
    }

    void
    commandBufferSetConvolutionOperation(vk::CommandBuffer& commandBuffer)
    {
      vk::CommandBufferBeginInfo info;
      //info.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse;
      commandBuffer.begin(info);

      vk::BufferMemoryBarrier inputBufferBarrier;
      inputBufferBarrier.buffer = resources.input.buffer;
      inputBufferBarrier.size = VK_WHOLE_SIZE;
      inputBufferBarrier.srcAccessMask = vk::AccessFlagBits::eHostWrite;
      inputBufferBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
      inputBufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      inputBufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

      vk::BufferMemoryBarrier kernelBufferBarrier;
      kernelBufferBarrier.buffer = resources.kernel.buffer;
      kernelBufferBarrier.size = VK_WHOLE_SIZE;
      kernelBufferBarrier.srcAccessMask = vk::AccessFlagBits::eHostWrite;
      kernelBufferBarrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;
      kernelBufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      kernelBufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

      commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eHost,
          vk::PipelineStageFlagBits::eComputeShader,
          vk::DependencyFlags(),
          {}, {inputBufferBarrier, kernelBufferBarrier} , {});

      commandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, *pipeline);
      commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute,
          *pipelineLayout, 0, {descriptorSet}, {});

      const auto numWorkgroups = (outputSize() - 1)/workGroupSize() + 1; // round up
      commandBuffer.dispatch(numWorkgroups, numWorkgroups, 1);

      vk::BufferMemoryBarrier outputBufferBarrier;
      outputBufferBarrier.srcAccessMask = vk::AccessFlagBits::eShaderWrite;
      outputBufferBarrier.dstAccessMask = vk::AccessFlagBits::eHostRead;
      outputBufferBarrier.buffer = resources.output.buffer;
      outputBufferBarrier.size = VK_WHOLE_SIZE;
      outputBufferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      outputBufferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

      commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader,
          vk::PipelineStageFlagBits::eHost,
          vk::DependencyFlags(),
          {}, {outputBufferBarrier} , {});

      commandBuffer.end();
    }

    void
    crossCorrelation()
    {
      const vk::PipelineStageFlags waitStageMask(vk::PipelineStageFlagBits::eHost);

      vk::SubmitInfo computeSubmitInfo;
      computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
      computeSubmitInfo.commandBufferCount = 1;
      computeSubmitInfo.pCommandBuffers = &*commandBuffers[0];

      auto fence = device->createFenceUnique({});
      queue.submit({computeSubmitInfo}, *fence);
      device->waitForFences( {*fence}, true, std::numeric_limits<uint64_t>::max());
    }
};
