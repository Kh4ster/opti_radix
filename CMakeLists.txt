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
set(CMAKE_CUDA_FLAGS "--extended-lambda")
#set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr --ptxas-options=-v --generate-line-info")

add_library(async_memcpy
  src/cuda_tools/host_shared_ptr.cu
  src/cuda_tools/device_buffer.cu)

include_directories(src)

find_package(GTest 1.10 REQUIRED)
find_package(benchmark REQUIRED)

add_executable(Bench bench/main.cu)
set_property(TARGET Bench PROPERTY CXX_STANDARD 20)

target_link_libraries(Bench async_memcpy GTest::GTest benchmark::benchmark)

#add_test(NAME UTMaxtree COMMAND UTMaxtree)

# BUILD_TYPE
if (NOT CMAKE_BUILD_TYPE)
    string(REGEX REPLACE ".*/build_" "" BUILD_NAME ${CMAKE_BINARY_DIR})
    # Default BUILD_TYPE is Release
    set(CMAKE_BUILD_TYPE Release)
    # If folder is named "build_debug" BUILD_TYPE is set to: Debug
    if(${BUILD_NAME} STREQUAL "Debug" OR ${BUILD_NAME} STREQUAL "debug")
        set(CMAKE_BUILD_TYPE Debug)
    endif()
    message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
endif()