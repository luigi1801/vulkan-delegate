#pragma once

#include <vulkan/vulkan.hpp>

#include <fstream>
#include <experimental/optional>

namespace vkutils {

class Error : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

// https://www.khronos.org/registry/vulkan/specs/1.0/html/vkspec.html#memory-allocation
// Find a memory in `memoryTypeBitsRequirement` that includes all of `requiredProperties`
std::experimental::optional<uint32_t>
findProperties(const vk::PhysicalDeviceMemoryProperties& pMemoryProperties,
    uint32_t memoryTypeBitsRequirement, vk::MemoryPropertyFlags requiredProperties) {
  const auto memoryCount = pMemoryProperties.memoryTypeCount;

  for (uint32_t memoryIndex = 0; memoryIndex < memoryCount; ++memoryIndex) {
    const uint32_t memoryTypeBits = (1 << memoryIndex);
    const bool isRequiredMemoryType = memoryTypeBitsRequirement & memoryTypeBits;

    const auto properties = pMemoryProperties.memoryTypes[memoryIndex].propertyFlags;
    const auto hasRequiredProperties
      = (properties & requiredProperties) == requiredProperties;

    if (isRequiredMemoryType && hasRequiredProperties)
      return memoryIndex;
  }

  // failed to find memory type
  return {};
}

template <typename T>
void copyToDeviceMemory(const vk::Device& device,
    const std::vector<T>& data, const vk::DeviceMemory memory) {
  const auto bufferSize = data.size() * sizeof(T);

  auto mapped = device.mapMemory(memory, 0, VK_WHOLE_SIZE);
  std::memcpy(mapped, data.data(), bufferSize);

  vk::MappedMemoryRange mappedRange(memory, 0, VK_WHOLE_SIZE);
  device.flushMappedMemoryRanges({mappedRange});

  device.unmapMemory(memory);
}

template <typename T>
void copyToDeviceMemory(const vk::Device& device,
    const T* data, int dataSize, const vk::DeviceMemory memory) {
  const auto bufferSize = dataSize * sizeof(T);

  auto mapped = device.mapMemory(memory, 0, VK_WHOLE_SIZE);
  std::memcpy(mapped, data, bufferSize);

  vk::MappedMemoryRange mappedRange(memory, 0, VK_WHOLE_SIZE);
  device.flushMappedMemoryRanges({mappedRange});

  device.unmapMemory(memory);
}

template <typename T>
void copyFromDeviceMemory(const vk::Device& device,
    const vk::DeviceMemory memory, std::vector<T>& data) {
  const auto bufferSize = data.size() * sizeof(T);

  const auto mapped = device.mapMemory(memory, 0, VK_WHOLE_SIZE);

  std::memcpy(data.data(), mapped, bufferSize);

  vk::MappedMemoryRange mappedRange(memory, 0, VK_WHOLE_SIZE);
  device.invalidateMappedMemoryRanges({mappedRange});

  device.unmapMemory(memory);
}

template <typename T>
void copyFromDeviceMemory(const vk::Device& device,
    const vk::DeviceMemory memory, T* data, int dataSize) {
  const auto bufferSize = dataSize * sizeof(T);

  const auto mapped = device.mapMemory(memory, 0, VK_WHOLE_SIZE);

  std::memcpy(data, mapped, bufferSize);

  vk::MappedMemoryRange mappedRange(memory, 0, VK_WHOLE_SIZE);
  device.invalidateMappedMemoryRanges({mappedRange});

  device.unmapMemory(memory);
}

class Resource {
  public:
    Resource(vk::Device device, vk::PhysicalDevice physicalDevice,
        vk::BufferUsageFlags useage, vk::MemoryPropertyFlags memFlags,
        vk::DeviceSize size)
      : device(device)
    {
      assert(size >= 0);

      // Create the buffer handle
      vk::BufferCreateInfo bufferCreateInfo;
      bufferCreateInfo.sharingMode = vk::SharingMode::eExclusive; // default
      bufferCreateInfo.usage = useage;
      bufferCreateInfo.size = size;

      this->buffer = device.createBuffer(bufferCreateInfo);

      auto deviceMemoryProperties = physicalDevice.getMemoryProperties();
      auto memReqs = device.getBufferMemoryRequirements(this->buffer);

      vk::MemoryAllocateInfo memAlloc;
      memAlloc.allocationSize = memReqs.size;

      auto memoryTypeIndex = findProperties(deviceMemoryProperties,
          memReqs.memoryTypeBits, memFlags);
      if (!memoryTypeIndex) throw Error("no memory found");
      memAlloc.memoryTypeIndex = memoryTypeIndex.value();

      this->memory = device.allocateMemory(memAlloc, nullptr);

      device.bindBufferMemory(this->buffer, memory, 0);
    }

    template <typename T>
    Resource(const vk::Device& device, vk::PhysicalDevice physicalDevice,
        vk::BufferUsageFlags useage, vk::MemoryPropertyFlags memFlags,
        const std::vector<T>& data)
    : Resource(device, physicalDevice, useage, memFlags, data.size() * sizeof(T))
    {
      copyToDeviceMemory(device, data, this->memory);
    }

    Resource()
    { }

    operator bool () const
    { return device; }

    ~Resource()
    {
      if (device) {
        assert(buffer);
        device.destroyBuffer(buffer);

        assert(memory);
        device.freeMemory(memory, nullptr);
      }
    }

    vk::Buffer buffer;
    vk::DeviceMemory memory;
  private:
    vk::Device device;
};

std::vector<char>
readFile(const std::string& filename)
{
  std::ifstream file(filename, std::ios::ate | std::ios::binary);

  if (!file.is_open()) {
    throw Error("failed to open file!");
  }

  std::vector<char> buffer(file.tellg());

  file.seekg(0);
  file.read(buffer.data(), buffer.size());
  assert(file.gcount() == static_cast<ssize_t>(buffer.size()));

  file.close();

  return buffer;
}

vk::UniqueShaderModule
loadShader(const char *fileName, vk::Device& device)
{
  const auto shaderCode = readFile(fileName);

  return device.createShaderModuleUnique(
      { {}, shaderCode.size(),
      reinterpret_cast<const uint32_t*>(shaderCode.data()) },
      nullptr);
}

std::experimental::optional<uint32_t>
getQueueFamilyIndex(const vk::PhysicalDevice& device, vk::QueueFlagBits type)
{
  auto queueFamilies = device.getQueueFamilyProperties();

  auto iter = std::find_if(queueFamilies.begin(), queueFamilies.end(),
      [type](auto& q) {
      return q.queueCount > 0 && q.queueFlags | type;
      });

  if (iter == queueFamilies.end()) return {};

  return iter - queueFamilies.begin();
}

} // namespace vkutils
