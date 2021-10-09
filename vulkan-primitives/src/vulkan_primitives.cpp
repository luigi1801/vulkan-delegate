#include "vulkan_primitives.h"

VulkanPrimitive::~VulkanPrimitive(){}

vk::UniqueShaderModule
VulkanPrimitive::loadShader(const char *fileName){
  std::ifstream file(fileName, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    // throw Error("failed to open file!");
  }
  std::vector<char> buffer(file.tellg());

  file.seekg(0);
  file.read(buffer.data(), buffer.size());
  assert(file.gcount() == static_cast<ssize_t>(buffer.size()));

  file.close();

  return p_device->createShaderModuleUnique(
      {{}, buffer.size(),reinterpret_cast<const uint32_t*>(buffer.data()) },
      nullptr);
};