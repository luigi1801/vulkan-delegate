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
  for(auto physicalDeviceIt : allPhysicaldevices){
    std::vector<vk::ExtensionProperties> extensionProperties =
      allPhysicaldevices[counter].enumerateDeviceExtensionProperties();
    auto properties2 = allPhysicaldevices[counter]
                         .getProperties2<vk::PhysicalDeviceProperties2,
                                         vk::PhysicalDeviceVertexAttributeDivisorPropertiesEXT>();
    //std::cout<< "properties        = " << (properties==nullptr) << "\n";
    auto properties = properties2.get<vk::PhysicalDeviceProperties2>().properties;
    std::cout << "\t"
              << "Properties:\n";
    std::cout << "\t\t"
              << "apiVersion        = " ;//<< decodeAPIVersion( properties.apiVersion ) << "\n";
    std::cout << "\t\t"
              << "driverVersion     = " //<< decodeDriverVersion( properties.driverVersion, properties.vendorID )
              << "\n";
    std::cout << "\t\t"
              << "vendorID          = " ;//<< decodeVendorID( properties.vendorID ) << "\n";
    std::cout << "\t\t"
              << "deviceID          = " << properties.deviceID << "\n";
    std::cout << "\t\t"
              << "deviceType        = " << vk::to_string( properties.deviceType ) << "\n";
    std::cout << "\t\t"
              << "deviceName        = " << properties.deviceName << "\n";
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
    counter++; break; continue;

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
      VulkanConv2D_Control primitiveControl;
      primitiveControl.AllBits = control;
      Primitive = std::make_unique<VulkanConvolution2D>(
          &physicalDevice,queueFamilyIndex,&device, primitiveControl);
      break;
    }
    default:{
      break;
    }
  }
  return std::move(Primitive);
}
