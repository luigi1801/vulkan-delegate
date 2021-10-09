LIB_SRC = src/
LIB_SRC_OUT = $(LIB_SRC)libvulkan_primitives_factory.so

OBJECTS = src/vulkan_primitives_factory.o src/vulkan_primitives.o src/vulkan_convolution.o
SHADER := shaders/cross_correlation_strided.comp.spv
EXAMPLE_PATH = examples/example

CXXFLAGS = -std=c++14 -fPIC
CXXINC = -I/usr/lib/x86_64-linux-gnu/
CXXLIB = -lvulkan
GLS_COMPILER := glslangValidator

src/%.o: src/%.cpp 
	$(CXX) $(CXXFLAGS) -g $(CXXINC) -c $< -o $@ 

src/libvulkan_primitives_factory.so  : $(OBJECTS)
	$(CXX) $(CXXFLAGS) -g -shared $(OBJECTS) -o $@  $(CXXLIB); 
	rm $(OBJECTS)

lib: CXXFLAGS += -Wall -fPIC 
lib: $(LIB_SRC_OUT) 

examples/%: examples/%.cpp lib
	$(CXX) $(CXXFLAGS) -o $@ $< $(LIB_SRC_OUT) $(CXXLIB);
	rm -f examples/*.o

example: $(EXAMPLE_PATH)
	./$(EXAMPLE_PATH)

shaders/%.spv: shaders/%
	$(GLS_COMPILER) -o $@ -V $<

shader: $(SHADER)