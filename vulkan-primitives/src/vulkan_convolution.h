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

  virtual void Init(std::vector<float*> inputs, std::vector<MemDims> inputsDims, 
                    std::vector<float*> weights, std::vector<MemDims> weightsDims, 
                    std::vector<float*> outputs, std::vector<MemDims> outputsDims) override;
  void Init();
  void Init(std::vector<float>& input, uint32_t inputSize, 
            std::vector<float>& kernel, uint32_t kernelSize, 
            std::vector<float>& output);
  void Init(float* input, uint32_t inputSize, float* kernel, 
            uint32_t kernelSize, float* output);
  virtual void Process() override;

 private:
  VulkanConv2D_Control control;
  struct NewResource {
    vk::Buffer buffer;
    vk::DeviceMemory memory;
  };
  std::vector<NewResource> m_newResources;

  uint32_t padding = 0;
  uint32_t inputDepth = 1;
  uint32_t outputDepth = 1;
  uint32_t stride = 0;

  void createResource(vk::BufferUsageFlags useage, vk::MemoryPropertyFlags memFlags,
                      vk::DeviceSize size);
  void SetCommandBuffer(vk::CommandBuffer& commandBuffer);
  uint32_t ComputeOutputSize(uint32_t inputSize, uint32_t kernelSize);
  void ComputeRealSizes(uint32_t& inputSize, uint32_t kernelSize);

  void copyInputToDeviceMemory(float* input, uint32_t inputSize);
};

