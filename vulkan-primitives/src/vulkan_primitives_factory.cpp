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
  std::cout<<std::endl;
  std::cout<< "Executing GetPrimitive" <<std::endl;
  std::unique_ptr<VulkanPrimitive> Primitive;
  switch(type){
    case Vulkan_Conv2d:{
      std::cout<< "Creating Vulkan_Conv2d Primitive" <<std::endl;
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
