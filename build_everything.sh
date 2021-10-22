#!/bin/bash 

LOCAL_FOLDER=$(pwd)

cd vulkan-primitives/
make
make shader
cd ../../../../../
bazel build -c opt tensorflow/lite/delegates/vulkan-delegate:vulkan_delegate.so
cd $LOCAL_FOLDER
cp vulkan-primitives/shaders/cross_correlation_strided_depthed_batched_biased.comp.spv example/shaders/
cd example
make
cd ..
