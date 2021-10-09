#pragma once

#include <vulkan/vulkan.hpp>

#include <iostream>
#include <experimental/string_view>

VkResult
vkCreateDebugReportCallbackEXT(VkInstance instance,
    const VkDebugReportCallbackCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator, VkDebugReportCallbackEXT* pCallback)
{
  static const auto func = reinterpret_cast<PFN_vkCreateDebugReportCallbackEXT>(
      vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT"));

  if (!func) return VK_ERROR_EXTENSION_NOT_PRESENT;

  return func(instance, pCreateInfo, pAllocator, pCallback);
}

void
vkDestroyDebugReportCallbackEXT(VkInstance instance,
    VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator)
{
  static const auto func = reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(
      vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));

  if (func) {
    func(instance, callback, pAllocator);
  }
}

namespace vkutils {
namespace debug {
  namespace detail {
    const std::array<const char*, 1> validationLayers {{
      "VK_LAYER_LUNARG_standard_validation"
    }};
    const std::array<const char*, 1> validationExtensions {{
      VK_EXT_DEBUG_REPORT_EXTENSION_NAME
    }};
  };
  bool
  isValidationLayerSupported()
  {
    const auto availableLayers = vk::enumerateInstanceLayerProperties();

    return std::all_of(detail::validationLayers.begin(), detail::validationLayers.end(),
        [&](std::experimental::string_view validationLayerName) {
          return availableLayers.end() !=
          std::find_if(availableLayers.begin(), availableLayers.end(),
            [&](auto& layer) {
              return layer.layerName == validationLayerName;
          });
       });
  }

  vk::UniqueInstance
  createInstanceUnique(vk::InstanceCreateInfo createInfo,
      vk::Optional<const vk::AllocationCallbacks> allocator = nullptr)
  {
    if (!isValidationLayerSupported()) {
      return vk::createInstanceUnique(createInfo, allocator);
    }

    std::vector<const char*> extensions(createInfo.enabledExtensionCount);
    std::copy_n(createInfo.ppEnabledExtensionNames, extensions.size(),
        extensions.begin());

    for (const auto& extension : detail::validationExtensions) {
      extensions.push_back(extension);
    }
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    std::vector<const char*> layers(detail::validationLayers.size()
        + createInfo.enabledLayerCount);
    std::copy(detail::validationLayers.begin(), detail::validationLayers.end(), layers.begin());
    std::copy_n(createInfo.ppEnabledLayerNames, createInfo.enabledLayerCount,
        layers.begin() + detail::validationLayers.size());

    createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
    createInfo.ppEnabledLayerNames = layers.data();

    return vk::createInstanceUnique(createInfo, allocator);
  }

  VkBool32
  reportCallback(VkDebugReportFlagsEXT vkflags,
      VkDebugReportObjectTypeEXT objType, uint64_t obj,
      size_t location, int32_t code, const char* layerPrefix,
      const char* msg, void* userData)
  {
    const auto flags = vk::DebugReportFlagsEXT(vk::DebugReportFlagBitsEXT(vkflags));
    const auto objectType = static_cast<vk::DebugReportObjectTypeEXT>(objType);

    const auto type = vk::to_string(flags);

    std::cerr << "[Vulkan Debug Report] "
      << type.substr(1, type.size()-2) << ": [" << layerPrefix << "] ";

    if (code) {
      std::cerr << "Code  " << code << ": ";
    }

    if (objectType != vk::DebugReportObjectTypeEXT::eUnknown) {
      std::cerr << "[object: " << vk::to_string(objectType)
        << " " << std::showbase << std::hex << obj << "]: ";
    }
    std::cerr << msg << std::endl;

    return VK_FALSE;
  }
} // namespace debug
} // namespace vkutils
