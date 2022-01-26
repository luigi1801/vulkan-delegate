#undef VULKAN_HPP_DISABLE_ENHANCED_MODE

#include<iostream>
#include "vulkan_primitives_factory.h"

VulkanPrimitivesFactory::VulkanPrimitivesFactory()
{
  std::cout<< "Creating VulkanPrimitivesFactory"<< std::endl;

  // Creating Instance
  //vk::ApplicationInfo appInfo("Primitives-factory");
  //appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  //appInfo.apiVersion = VK_API_VERSION_1_0;

  vk::InstanceCreateInfo instCreateInfo;
  //instCreateInfo.pApplicationInfo = &appInfo;

  instance = vk::createInstance(instCreateInfo);
  // pick PhysicalDevice
  const auto allPhysicaldevices = instance.enumeratePhysicalDevices();
  std::cout<< "Total devices        = " << allPhysicaldevices.size() << "\n";
  int counter = 0;
  if(false){
  for(auto physicalDeviceIt : allPhysicaldevices){
     VkPhysicalDeviceProperties                 properties;
    vkGetPhysicalDeviceProperties(physicalDeviceIt, &properties);

    //std::vector<vk::ExtensionProperties> extensionProperties =
    //  allPhysicaldevices[counter].enumerateDeviceExtensionProperties();
    //auto properties2 = allPhysicaldevices[counter]
    //                     .getProperties2<vk::PhysicalDeviceProperties2,
    //                                     vk::PhysicalDeviceVertexAttributeDivisorPropertiesEXT>();
    ////std::cout<< "properties        = " << (properties==nullptr) << "\n";
    //auto properties = properties2.get<vk::PhysicalDeviceProperties2>().properties;
    std::cout << "\t"
              << "Properties:\n";
    std::cout << "\t\t"
              << "apiVersion        = " /*<< decodeAPIVersion( properties.apiVersion )*/ << "\n";
    std::cout << "\t\t"
              << "driverVersion     = " /*<< decodeDriverVersion( properties.driverVersion, properties.vendorID )*/
              << "\n";
  std::cout << "\t\t"
            << "vendorID          = " /*<< decodeVendorID( properties.vendorID )*/ << "\n";
  std::cout << "\t\t"
            << "deviceID          = " << properties.deviceID << "\n";
//  std::cout << "\t\t"
//            << "deviceType        = " /*<< vk::to_string( properties.deviceType ) */<< "\n";
  std::cout << "\t\t"
            << "deviceName        = " << properties.deviceName << "\n";
   std::cout << "\t\t"
                << "limits:\n";
      std::cout << "\t\t\t"
                << "bufferImageGranularity                          = " << properties.limits.bufferImageGranularity
                << "\n";
      std::cout << "\t\t\t"
                << "discreteQueuePriorities                         = " << properties.limits.discreteQueuePriorities
                << "\n";
      std::cout << "\t\t\t"
                << "framebufferColorSampleCounts                    = "
                /*<< vk::to_string( properties.limits.framebufferColorSampleCounts ) */<< "\n";
      std::cout << "\t\t\t"
                << "framebufferDepthSampleCounts                    = "
                /*<< vk::to_string( properties.limits.framebufferDepthSampleCounts ) */<< "\n";
      std::cout << "\t\t\t"
                << "framebufferNoAttachmentsSampleCounts            = "
                /*<< vk::to_string( properties.limits.framebufferNoAttachmentsSampleCounts ) */<< "\n";
      std::cout << "\t\t\t"
                << "framebufferStencilSampleCounts                  = "
                /*<< vk::to_string( properties.limits.framebufferStencilSampleCounts ) */<< "\n";
      std::cout << "\t\t\t"
                << "lineWidthGranularity                            = " << properties.limits.lineWidthGranularity
                << "\n";
      std::cout << "\t\t\t"
                << "lineWidthRange                                  = "
                << "[" << properties.limits.lineWidthRange[0] << ", " << properties.limits.lineWidthRange[1] << "]"
                << "\n";
      std::cout << "\t\t\t"
                << "maxBoundDescriptorSets                          = " << properties.limits.maxBoundDescriptorSets
                << "\n";
      std::cout << "\t\t\t"
                << "maxClipDistances                                = " << properties.limits.maxClipDistances << "\n";
      std::cout << "\t\t\t"
                << "maxColorAttachments                             = " << properties.limits.maxColorAttachments
                << "\n";
      std::cout << "\t\t\t"
                << "maxCombinedClipAndCullDistances                 = "
                << properties.limits.maxCombinedClipAndCullDistances << "\n";
      std::cout << "\t\t\t"
                << "maxComputeSharedMemorySize                      = " << properties.limits.maxComputeSharedMemorySize
                << "\n";
      std::cout << "\t\t\t"
                << "maxComputeWorkGroupCount                        = "
                << "[" << properties.limits.maxComputeWorkGroupCount[0] << ", "
                << properties.limits.maxComputeWorkGroupCount[1] << ", "
                << properties.limits.maxComputeWorkGroupCount[2] << "]"
                << "\n";
      std::cout << "\t\t\t"
                << "maxComputeWorkGroupInvocations                  = "
                << properties.limits.maxComputeWorkGroupInvocations << "\n";
      std::cout << "\t\t\t"
                << "maxComputeWorkGroupSize                         = "
                << "[" << properties.limits.maxComputeWorkGroupSize[0] << ", "
                << properties.limits.maxComputeWorkGroupSize[1] << ", " << properties.limits.maxComputeWorkGroupSize[2]
                << "]"
                << "\n";
      std::cout << "\t\t\t"
                << "maxCullDistances                                = " << properties.limits.maxCullDistances << "\n";
      std::cout << "\t\t\t"
                << "maxDescriptorSetInputAttachments                = "
                << properties.limits.maxDescriptorSetInputAttachments << "\n";
      std::cout << "\t\t\t"
                << "maxDescriptorSetSampledImages                   = "
                << properties.limits.maxDescriptorSetSampledImages << "\n";
      std::cout << "\t\t\t"
                << "maxDescriptorSetSamplers                        = " << properties.limits.maxDescriptorSetSamplers
                << "\n";
      std::cout << "\t\t\t"
                << "maxDescriptorSetStorageBuffers                  = "
                << properties.limits.maxDescriptorSetStorageBuffers << "\n";
      std::cout << "\t\t\t"
                << "maxDescriptorSetStorageBuffersDynamic           = "
                << properties.limits.maxDescriptorSetStorageBuffersDynamic << "\n";
      std::cout << "\t\t\t"
                << "maxDescriptorSetStorageImages                   = "
                << properties.limits.maxDescriptorSetStorageImages << "\n";
      std::cout << "\t\t\t"
                << "maxDescriptorSetUniformBuffers                  = "
                << properties.limits.maxDescriptorSetUniformBuffers << "\n";
      std::cout << "\t\t\t"
                << "maxDescriptorSetUniformBuffersDynamic           = "
                << properties.limits.maxDescriptorSetUniformBuffersDynamic << "\n";
      std::cout << "\t\t\t"
                << "maxDrawIndexedIndexValue                        = " << properties.limits.maxDrawIndexedIndexValue
                << "\n";
      std::cout << "\t\t\t"
                << "maxDrawIndirectCount                            = " << properties.limits.maxDrawIndirectCount
                << "\n";
      std::cout << "\t\t\t"
                << "maxFragmentCombinedOutputResources              = "
                << properties.limits.maxFragmentCombinedOutputResources << "\n";
      std::cout << "\t\t\t"
                << "maxFragmentDualSrcAttachments                   = "
                << properties.limits.maxFragmentDualSrcAttachments << "\n";
      std::cout << "\t\t\t"
                << "maxFragmentInputComponents                      = " << properties.limits.maxFragmentInputComponents
                << "\n";
      std::cout << "\t\t\t"
                << "maxFragmentOutputAttachments                    = "
                << properties.limits.maxFragmentOutputAttachments << "\n";
      std::cout << "\t\t\t"
                << "maxFramebufferHeight                            = " << properties.limits.maxFramebufferHeight
                << "\n";
      std::cout << "\t\t\t"
                << "maxFramebufferLayers                            = " << properties.limits.maxFramebufferLayers
                << "\n";
      std::cout << "\t\t\t"
                << "maxFramebufferWidth                             = " << properties.limits.maxFramebufferWidth
                << "\n";
      std::cout << "\t\t\t"
                << "maxGeometryInputComponents                      = " << properties.limits.maxGeometryInputComponents
                << "\n";
      std::cout << "\t\t\t"
                << "maxGeometryOutputComponents                     = " << properties.limits.maxGeometryOutputComponents
                << "\n";
      std::cout << "\t\t\t"
                << "maxGeometryOutputVertices                       = " << properties.limits.maxGeometryOutputVertices
                << "\n";
      std::cout << "\t\t\t"
                << "maxGeometryShaderInvocations                    = "
                << properties.limits.maxGeometryShaderInvocations << "\n";
      std::cout << "\t\t\t"
                << "maxGeometryTotalOutputComponents                = "
                << properties.limits.maxGeometryTotalOutputComponents << "\n";
      std::cout << "\t\t\t"
                << "maxImageArrayLayers                             = " << properties.limits.maxImageArrayLayers
                << "\n";
      std::cout << "\t\t\t"
                << "maxImageDimension1D                             = " << properties.limits.maxImageDimension1D
                << "\n";
      std::cout << "\t\t\t"
                << "maxImageDimension2D                             = " << properties.limits.maxImageDimension2D
                << "\n";
      std::cout << "\t\t\t"
                << "maxImageDimension3D                             = " << properties.limits.maxImageDimension3D
                << "\n";
      std::cout << "\t\t\t"
                << "maxImageDimensionCube                           = " << properties.limits.maxImageDimensionCube
                << "\n";
      std::cout << "\t\t\t"
                << "maxInterpolationOffset                          = " << properties.limits.maxInterpolationOffset
                << "\n";
      std::cout << "\t\t\t"
                << "maxMemoryAllocationCount                        = " << properties.limits.maxMemoryAllocationCount
                << "\n";
      std::cout << "\t\t\t"
                << "maxPerStageDescriptorInputAttachments           = "
                << properties.limits.maxPerStageDescriptorInputAttachments << "\n";
      std::cout << "\t\t\t"
                << "maxPerStageDescriptorSampledImages              = "
                << properties.limits.maxPerStageDescriptorSampledImages << "\n";
      std::cout << "\t\t\t"
                << "maxPerStageDescriptorSamplers                   = "
                << properties.limits.maxPerStageDescriptorSamplers << "\n";
      std::cout << "\t\t\t"
                << "maxPerStageDescriptorStorageBuffers             = "
                << properties.limits.maxPerStageDescriptorStorageBuffers << "\n";
      std::cout << "\t\t\t"
                << "maxPerStageDescriptorStorageImages              = "
                << properties.limits.maxPerStageDescriptorStorageImages << "\n";
      std::cout << "\t\t\t"
                << "maxPerStageDescriptorUniformBuffers             = "
                << properties.limits.maxPerStageDescriptorUniformBuffers << "\n";
      std::cout << "\t\t\t"
                << "maxPerStageResources                            = " << properties.limits.maxPerStageResources
                << "\n";
      std::cout << "\t\t\t"
                << "maxPushConstantsSize                            = " << properties.limits.maxPushConstantsSize
                << "\n";
      std::cout << "\t\t\t"
                << "maxSampleMaskWords                              = " << properties.limits.maxSampleMaskWords << "\n";
      std::cout << "\t\t\t"
                << "maxSamplerAllocationCount                       = " << properties.limits.maxSamplerAllocationCount
                << "\n";
      std::cout << "\t\t\t"
                << "maxSamplerAnisotropy                            = " << properties.limits.maxSamplerAnisotropy
                << "\n";
      std::cout << "\t\t\t"
                << "maxSamplerLodBias                               = " << properties.limits.maxSamplerLodBias << "\n";
      std::cout << "\t\t\t"
                << "maxStorageBufferRange                           = " << properties.limits.maxStorageBufferRange
                << "\n";
      std::cout << "\t\t\t"
                << "maxTessellationControlPerPatchOutputComponents  = "
                << properties.limits.maxTessellationControlPerPatchOutputComponents << "\n";
      std::cout << "\t\t\t"
                << "maxTessellationControlPerVertexInputComponents  = "
                << properties.limits.maxTessellationControlPerVertexInputComponents << "\n";
      std::cout << "\t\t\t"
                << "maxTessellationControlPerVertexOutputComponents = "
                << properties.limits.maxTessellationControlPerVertexOutputComponents << "\n";
      std::cout << "\t\t\t"
                << "maxTessellationControlTotalOutputComponents     = "
                << properties.limits.maxTessellationControlTotalOutputComponents << "\n";
      std::cout << "\t\t\t"
                << "maxTessellationEvaluationInputComponents        = "
                << properties.limits.maxTessellationEvaluationInputComponents << "\n";
      std::cout << "\t\t\t"
                << "maxTessellationEvaluationOutputComponents       = "
                << properties.limits.maxTessellationEvaluationOutputComponents << "\n";
      std::cout << "\t\t\t"
                << "maxTessellationGenerationLevel                  = "
                << properties.limits.maxTessellationGenerationLevel << "\n";
      std::cout << "\t\t\t"
                << "maxTessellationPatchSize                        = " << properties.limits.maxTessellationPatchSize
                << "\n";
      std::cout << "\t\t\t"
                << "maxTexelBufferElements                          = " << properties.limits.maxTexelBufferElements
                << "\n";
      std::cout << "\t\t\t"
                << "maxTexelGatherOffset                            = " << properties.limits.maxTexelGatherOffset
                << "\n";
      std::cout << "\t\t\t"
                << "maxTexelOffset                                  = " << properties.limits.maxTexelOffset << "\n";
      std::cout << "\t\t\t"
                << "maxUniformBufferRange                           = " << properties.limits.maxUniformBufferRange
                << "\n";
      std::cout << "\t\t\t"
                << "maxVertexInputAttributeOffset                   = "
                << properties.limits.maxVertexInputAttributeOffset << "\n";
      std::cout << "\t\t\t"
                << "maxVertexInputAttributes                        = " << properties.limits.maxVertexInputAttributes
                << "\n";
      std::cout << "\t\t\t"
                << "maxVertexInputBindings                          = " << properties.limits.maxVertexInputBindings
                << "\n";
      std::cout << "\t\t\t"
                << "maxVertexInputBindingStride                     = " << properties.limits.maxVertexInputBindingStride
                << "\n";
      std::cout << "\t\t\t"
                << "maxVertexOutputComponents                       = " << properties.limits.maxVertexOutputComponents
                << "\n";
      std::cout << "\t\t\t"
                << "maxViewportDimensions                           = "
                << "[" << properties.limits.maxViewportDimensions[0] << ", "
                << properties.limits.maxViewportDimensions[1] << "]"
                << "\n";
      std::cout << "\t\t\t"
                << "maxViewports                                    = " << properties.limits.maxViewports << "\n";
      std::cout << "\t\t\t"
                << "minInterpolationOffset                          = " << properties.limits.minInterpolationOffset
                << "\n";
      std::cout << "\t\t\t"
                << "minMemoryMapAlignment                           = " << properties.limits.minMemoryMapAlignment
                << "\n";
      std::cout << "\t\t\t"
                << "minStorageBufferOffsetAlignment                 = "
                << properties.limits.minStorageBufferOffsetAlignment << "\n";
      std::cout << "\t\t\t"
                << "minTexelBufferOffsetAlignment                   = "
                << properties.limits.minTexelBufferOffsetAlignment << "\n";
      std::cout << "\t\t\t"
                << "minTexelGatherOffset                            = " << properties.limits.minTexelGatherOffset
                << "\n";
      std::cout << "\t\t\t"
                << "minTexelOffset                                  = " << properties.limits.minTexelOffset << "\n";
      std::cout << "\t\t\t"
                << "minUniformBufferOffsetAlignment                 = "
                << properties.limits.minUniformBufferOffsetAlignment << "\n";
      std::cout << "\t\t\t"
                << "mipmapPrecisionBits                             = " << properties.limits.mipmapPrecisionBits
                << "\n";
      std::cout << "\t\t\t"
                << "nonCoherentAtomSize                             = " << properties.limits.nonCoherentAtomSize
                << "\n";
      std::cout << "\t\t\t"
                << "optimalBufferCopyOffsetAlignment                = "
                << properties.limits.optimalBufferCopyOffsetAlignment << "\n";
      std::cout << "\t\t\t"
                << "optimalBufferCopyRowPitchAlignment              = "
                << properties.limits.optimalBufferCopyRowPitchAlignment << "\n";
      std::cout << "\t\t\t"
                << "pointSizeGranularity                            = " << properties.limits.pointSizeGranularity
                << "\n";
      std::cout << "\t\t\t"
                << "pointSizeRange                                  = "
                << "[" << properties.limits.pointSizeRange[0] << ", " << properties.limits.pointSizeRange[1] << "]"
                << "\n";
      std::cout << "\t\t\t"
                << "sampledImageColorSampleCounts                   = "
                /*<< vk::to_string( properties.limits.sampledImageColorSampleCounts ) */<< "\n";
      std::cout << "\t\t\t"
                << "sampledImageDepthSampleCounts                   = "
                /*<< vk::to_string( properties.limits.sampledImageDepthSampleCounts ) */<< "\n";
      std::cout << "\t\t\t"
                << "sampledImageIntegerSampleCounts                 = "
                /*<< vk::to_string( properties.limits.sampledImageIntegerSampleCounts ) */<< "\n";
      std::cout << "\t\t\t"
                << "sampledImageStencilSampleCounts                 = "
                /*<< vk::to_string( properties.limits.sampledImageStencilSampleCounts ) */<< "\n";
      std::cout << "\t\t\t"
                << "sparseAddressSpaceSize                          = " << properties.limits.sparseAddressSpaceSize
                << "\n";
      std::cout << "\t\t\t"
                << "standardSampleLocations                         = "
                << !!properties.limits.standardSampleLocations << "\n";
      std::cout << "\t\t\t"
                << "storageImageSampleCounts                        = "
                /*<< vk::to_string( properties.limits.storageImageSampleCounts ) */<< "\n";
      std::cout << "\t\t\t"
                << "strictLines                                     = "
                << !!properties.limits.strictLines << "\n";
      std::cout << "\t\t\t"
                << "subPixelInterpolationOffsetBits                 = "
                << properties.limits.subPixelInterpolationOffsetBits << "\n";
      std::cout << "\t\t\t"
                << "subPixelPrecisionBits                           = " << properties.limits.subPixelPrecisionBits
                << "\n";
      std::cout << "\t\t\t"
                << "subTexelPrecisionBits                           = " << properties.limits.subTexelPrecisionBits
                << "\n";
      std::cout << "\t\t\t"
                << "timestampComputeAndGraphics                     = "
                << !!properties.limits.timestampComputeAndGraphics << "\n";
      std::cout << "\t\t\t"
                << "timestampPeriod                                 = " << properties.limits.timestampPeriod << "\n";
      std::cout << "\t\t\t"
                << "viewportBoundsRange                             = "
                << "[" << properties.limits.viewportBoundsRange[0] << ", " << properties.limits.viewportBoundsRange[1]
                << "]"
                << "\n";
      std::cout << "\t\t\t"
                << "viewportSubPixelBits                            = " << properties.limits.viewportSubPixelBits
                << "\n";
      std::cout << "\t\t"
                << "sparseProperties:\n";
      std::cout << "\t\t\t"
                << "residencyAlignedMipSize                   = "
                << !!properties.sparseProperties.residencyAlignedMipSize << "\n";
      std::cout << "\t\t\t"
                << "residencyNonResidentStrict                = "
                << !!properties.sparseProperties.residencyNonResidentStrict << "\n";
      std::cout << "\t\t\t"
                << "residencyStandard2DBlockShape             = "
                << !!properties.sparseProperties.residencyStandard2DBlockShape << "\n";
      std::cout << "\t\t\t"
                << "residencyStandard2DMultisampleBlockShape  = "
                << !!properties.sparseProperties.residencyStandard2DMultisampleBlockShape << "\n";
      std::cout << "\t\t\t"
                << "residencyStandard3DBlockShape             = "
                << !!properties.sparseProperties.residencyStandard3DBlockShape << "\n";
      std::cout << "\n";//  counter++; break; continue;
  }
  }
  for(auto physicalDeviceIt : allPhysicaldevices)
  {
    auto queueFamilies = physicalDeviceIt.getQueueFamilyProperties();
    auto familiesIt = 
        std::find_if(queueFamilies.begin(), queueFamilies.end(), [](auto& q) {
          return q.queueCount > 0 && q.queueFlags | vk::QueueFlagBits::eCompute;
        });
    if (familiesIt != queueFamilies.end())
    {
           //std::cout<<familiesIt->queueCount<<std::endl;
      queueFamilyIndex = std::distance( queueFamilies.begin(), familiesIt );
      physicalDevice = physicalDeviceIt;
      break;
    } 
  }
  if(-1 == queueFamilyIndex)
  {
    return;
  }

  // Create Logical device
  vk::DeviceQueueCreateInfo queueCreateInfo;
  queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
  queueCreateInfo.queueCount = 1;
  float queuePriority = 1;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  vk::PhysicalDeviceFeatures deviceFeatures;

  vk::DeviceCreateInfo devCreateInfo;
  devCreateInfo.pQueueCreateInfos = &queueCreateInfo;
  devCreateInfo.queueCreateInfoCount = 1;
  devCreateInfo.pEnabledFeatures = &deviceFeatures;

  device = physicalDevice.createDevice(devCreateInfo);
}

VulkanPrimitivesFactory::~VulkanPrimitivesFactory()
{
  std::cout<< "Destroying everything" <<std::endl;
  device.destroy();
  instance.destroy();
}

std::unique_ptr<VulkanPrimitive> VulkanPrimitivesFactory::GetPrimitive(PrimitiveType type, uint32_t control)
{
  std::unique_ptr<VulkanPrimitive> Primitive;
  switch(type){
    case Vulkan_Conv2d:{
      VulkanConv2Dk1x1_Control primitiveControl;
      primitiveControl.AllBits = control;
      Primitive = std::make_unique<VulkanConvolution2Dk1x1>(
          &physicalDevice,queueFamilyIndex,&device, primitiveControl);
      break;
    }
    default:{
      break;
    }
  }
  return std::move(Primitive);
}
