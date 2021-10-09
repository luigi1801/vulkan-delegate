#!/bin/bash 

LOCAL_FOLDER=$(pwd)

cd vulkan-primitives/
make
cd ../../../../../
bazel build -c opt tensorflow/lite/delegates/vulkan-delegate:vulkan_delegate.so
cd $LOCAL_FOLDER
cd example
make
cd ..
