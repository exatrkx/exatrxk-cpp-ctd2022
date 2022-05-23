cmake_minimum_required(VERSION 3.20.1)
# list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(Exatrkx_Inference VERSION 0.2.0 LANGUAGES C CXX CUDA)


if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA compiler version must be at least 11.0")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.3)
    message(FATAL_ERROR "GCC compiler must be at least 9.3")
endif()


set(supported_archs "60" "62" "70" "72" "75" "80")
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")


option(CUDA_STATIC_RUNTIME "Use CUDA static runtime" OFF)
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)

set(EXATRKX_CXX_FLAGS "")
set(EXATRKX_CUDA_FLAGS "")

if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND EXATRKX_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
endif(CMAKE_COMPILER_IS_GNUCXX)


message("-- Building for GPU_ARCHS = ${CMAKE_CUDA_ARCHITECTURES}")

list(APPEND EXATRKX_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND EXATRKX_CUDA_FLAGS -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable)
list(APPEND EXATRKX_CUDA_FLAGS -Xfatbin=-compress-all)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if (CMAKE_CUDA_LINEINFO)
    list(APPEND EXATRKX_CUDA_FLAGS -lineinfo)
endif(CMAKE_CUDA_LINEINFO)

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    list(APPEND EXATRKX_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif(CMAKE_BUILD_TYPE MATCHES Debug)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_C_FLAGS "-Wno-error=format-truncation") 

find_package(Torch REQUIRED)
find_package(Boost REQUIRED)
find_package(TBB REQUIRED)

find_package(TritonCore CONFIG REQUIRED)
find_package(TritonCommon CONFIG REQUIRED)
find_package(TritonClient CONFIG REQUIRED COMPONENTS TritonClient::triton-client)

add_subdirectory(ExaTrkX)

add_executable(inference inference.cpp)
target_link_libraries(inference PUBLIC ActsPluginExaTrkXTriton tbb)
set_target_properties(inference
PROPERTIES  CXX_STANDARD                        17
            CXX_STANDARD_REQUIRED               ON
)
target_include_directories(
  inference
  PUBLIC
    /workspace/install/include
    /usr/local/cuda/include
)