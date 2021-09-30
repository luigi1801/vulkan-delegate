cc_import(
  name = "vulkan_primitives_factory",
  hdrs = [
	"vulkan-primitives/src/vulkan_primitives_factory.h", 
	"vulkan-primitives/src/vulkan_primitives.h", 
	"vulkan-primitives/src/vulkan_convolution.h",
  ],
  shared_library = "vulkan-primitives/src/libvulkan_primitives_factory.so",
)

cc_import(
  name = "vulkan",  

  # libvulkan.so is provided by system environment, for example it can be found in PATH.
  # This indicates that Bazel is not responsible for making libvulkan.so available.
  system_provided = 1,
)

cc_library(
    name = "vulkan_delegate",
    srcs = [
        "vulkan_delegate.cc",
        "vulkan_delegate_kernel.cc",
    ],
    hdrs = [
        "vulkan_delegate.h",
        "vulkan_delegate_kernel.h",
    ],
    deps = [
	":vulkan_primitives_factory",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/delegates/utils:simple_delegate",
    ],
)

cc_binary(
    name = "vulkan_delegate.so",
    srcs = [
        "vulkan_delegate_adaptor.h",
        "vulkan_delegate_adaptor.cc",
    ],
    linkshared = 1,
    linkstatic = 0,
    deps = [
	":vulkan_primitives_factory",
        ":vulkan_delegate",
        "//tensorflow/lite/c:common",
        "//tensorflow/lite/tools:command_line_flags",
        "//tensorflow/lite/tools:logging",
    ],
)
