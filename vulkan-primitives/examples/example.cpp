#include <memory>
#include "../src/vulkan_primitives_factory.h"
#include <numeric>
#include <iostream>
#include <iomanip>
#include <cmath>

//#include "../../vulkan_compute_convolution/src/cross_correlation.cpp"

template <typename T>
static void
printArray(const T array, size_t size)
{
  assert(array.size() == size * size);
  for (size_t y=0; y<size; ++y) {
    std::cout << "| ";
    for (size_t x=0; x<size; ++x) {
      std::cout << std::setw(10) << std::setprecision(3)
        << array[x + size*y] << ' ';
    }
    std::cout << "|\n";
  }
}

int main()
{

  uint32_t kernelSize     { 3 };
  uint32_t inputSize      { 16 };
  uint32_t outputSize     { 14 };

//    uint32_t workgroupSize { 16 }; 

  std::vector<float> kernel(kernelSize*kernelSize,0);
  MemDims kerDim = {1, kernelSize, kernelSize, 1};
  kernel[4]=1;  // 3x3
  //kernel[5]=1;
  //kernel[12]=1;  //5x5
  std::vector<float> compute_input(inputSize*inputSize);
  MemDims inDim ;
  std::vector<float> output(outputSize*outputSize);
  MemDims outDim = {1, outputSize, outputSize, 1};
    
  std::iota(compute_input.begin(), compute_input.end(), 0);
  //std::iota(kernel.begin(), kernel.end(), 0);

  printArray(compute_input, inputSize);
  printArray(kernel, kernelSize);

  VulkanPrimitivesFactory vpfact;
  VulkanConv2D_Control control = {0};
  control.Padding = 0;
  control.stride_h = 0;
  std::unique_ptr<VulkanPrimitive> vp = vpfact.GetPrimitive(Vulkan_Conv2d, control.AllBits);
  if(nullptr != vp)
  {
    std::vector<float*> inputs(1);
    std::vector<MemDims> inputsDims(1);
    inputs[0] = compute_input.data();
    inputsDims[0] = inDim;

    std::vector<float*> weights(1);
    std::vector<MemDims> weightsDims(1);
    weights[0] = kernel.data();
    weightsDims[0] = kerDim;

    std::vector<float*> outputs(1);
    std::vector<MemDims> outputsDims(1);
    outputs[0] = output.data();
    outputsDims[0] = outDim;

    //VulkanConvolution2D* vc = (static_cast<VulkanConvolution2D*>(vp.get()));
    vp->Init(inputs, inputsDims, weights, weightsDims, outputs, outputsDims);
    //vc->Init(compute_input, inputSize, kernel, kernelSize, output);
    vp->Process();
    printArray(output, outputSize);
  }
};

