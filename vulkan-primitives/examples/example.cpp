#include <memory>
#include "../src/vulkan_primitives_factory.h"
#include <numeric>
#include <iostream>
#include <iomanip>

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
    uint32_t inputSize     { 16 };

//    uint32_t workgroupSize { 16 }; 

    std::vector<float> kernel(kernelSize*kernelSize,0);
    kernel[4]=1;  // 3x3
    //kernel[5]=1;
    //kernel[12]=1;  //5x5
    std::vector<float> compute_input(inputSize*inputSize);
    std::vector<float> output;
    std::vector<float> inputCopy(inputSize*inputSize);
    
    std::iota(compute_input.begin(), compute_input.end(), 0);
    //std::iota(kernel.begin(), kernel.end(), 0);
    std::memcpy(inputCopy.data()+2, compute_input.data()+2, inputSize*inputSize*sizeof(float)/2);

    printArray(compute_input, inputSize);
    printArray(kernel, kernelSize);
    //printArray(inputCopy, inputSize);
    //return 0;
    VulkanPrimitivesFactory vpfact;
    std::unique_ptr<VulkanPrimitive> vp = vpfact.GetPrimitive(Vulkan_Conv2d, 1);
    if(nullptr != vp)
    {
        VulkanConvolution2D* vc = (static_cast<VulkanConvolution2D*>(vp.get()));
        vc->Init(compute_input, inputSize, kernel, kernelSize, output);
        vp->Process();
        //printArray(output, inputSize-kernelSize+1);
        printArray(output, inputSize);
    }


};

