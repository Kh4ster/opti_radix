project(template)
cmake_minimum_required(VERSION 3.18)

include(FindCUDAToolkit)

enable_language(CUDA)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

add_compile_options(-W -Wall)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
#set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr --ptxas-options=-v --generate-line-info")

add_library(async_memcpy
  src/to_bench.cu
  src/cuda_tools/host_unique_ptr.cu
  src/cuda_tools/device_unique_ptr.cu)

include_directories(src)

find_package(GTest 1.10 REQUIRED)
find_package(benchmark REQUIRED)

add_library(TestHelpers OBJECT bench/test_helpers.cu)

add_executable(Bench bench/main.cu)

target_link_libraries(Bench async_memcpy GTest::GTest benchmark::benchmark TestHelpers)

#add_test(NAME UTMaxtree COMMAND UTMaxtree)