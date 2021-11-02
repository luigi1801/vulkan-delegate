#pragma once

typedef union {    
  struct {
    unsigned Padding : 1;  // 0 -> Valid; 1 -> Same
    unsigned stride_h : 8;
    unsigned stride_w : 8;
    unsigned _unused : 15;
  };
  uint32_t AllBits;
} VulkanConv2D_Control;

class VulkanConvolution2D : public VulkanPrimitive
{
 public:
  VulkanConvolution2D();
  VulkanConvolution2D(vk::PhysicalDevice* physicalDevice, 
                      int queueFamilyIndex, vk::Device* device, 
                      VulkanConv2D_Control primitiveControl);
  ~VulkanConvolution2D();

  virtual void Init(std::vector<MemDims> inputsDims, 
                    std::vector<MemDims> weightsDims, 
                    std::vector<MemDims> outputsDims) override;
  void Init();
  virtual void Process(std::vector<float*> inputs,
                       std::vector<float*> weights,
                       std::vector<float*> outputs) override;

 private:

  struct SpecializationData {
    uint32_t inputSize;
    uint32_t inputDepth;
    uint32_t kernelSize;
    uint32_t kernelOffset;
    uint32_t outputSize;
    uint32_t outputDepth;
    uint32_t workGroupSize;
    uint32_t workGroupSizeZ;
    uint32_t stride;
  };
  std::array<vk::SpecializationMapEntry, 9> m_specializationMapEntries;
  SpecializationData m_specializationData;
  vk::SpecializationInfo m_specializationInfo;
  
  VulkanConv2D_Control control;
  struct NewResource {
    vk::Buffer buffer;
    vk::DeviceMemory memory;
    vk::BufferMemoryBarrier barrier;
  };
  std::vector<NewResource> m_newResources;

  uint32_t workGroupSizeZ = 1;
  uint32_t padding = 0;
  uint32_t inputSize = 0;
  uint32_t kernelSize = 0;
  uint32_t inputDepth = 1;
  uint32_t outputDepth = 1;
  uint32_t stride = 0;

  void createResource(vk::BufferUsageFlags useage, vk::MemoryPropertyFlags memFlags,
                      vk::DeviceSize size, vk::AccessFlagBits srcAccessMask, vk::AccessFlagBits dstAccessMask);
  void SetCommandBuffer(vk::CommandBuffer& commandBuffer, vk::Pipeline pipeline);
  uint32_t ComputeOutputSize(uint32_t inputSize, uint32_t kernelSize);
  void ComputeRealSizes(uint32_t& inputSize, uint32_t kernelSize);

  void copyInputToDeviceMemory(float* input, uint32_t inputSize);
};

