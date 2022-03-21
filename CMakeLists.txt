cmake_minimum_required(VERSION 3.17)

project(cc-clients LANGUAGES C CXX)

#
# Options
#
option(TRITON_ENABLE_CC_HTTP "Build C++ HTTP client libraries" OFF)
option(TRITON_ENABLE_CC_GRPC "Build C++ GRPC client libraries" OFF)
option(TRITON_ENABLE_PERF_ANALYZER "Enable Performance Analyzer" OFF)
option(TRITON_ENABLE_EXAMPLES "Include examples in build" ON)
option(TRITON_ENABLE_TESTS "Include tests in build" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in libraries" OFF)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)

if(TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER)
  set(TRITON_COMMON_ENABLE_PROTOBUF ON)
  set(TRITON_COMMON_ENABLE_GRPC ON)

  FetchContent_Declare(
    repo-core
    GIT_REPOSITORY https://github.com/triton-inference-server/core.git
    GIT_TAG ${TRITON_CORE_REPO_TAG}
    GIT_SHALLOW ON 
  )
  FetchContent_MakeAvailable(repo-core)
endif() # TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER

FetchContent_MakeAvailable(repo-common)

#
# CUDA
#
if(TRITON_ENABLE_GPU)
  find_package(CUDAToolkit REQUIRED)
endif() # TRITON_ENABLE_GPU

#set("CMAKE_PREFIX_PATH")

#
# libcurl
#
set(CURL_DIR /workspace/build/third-party/curl/lib/cmake/CURL)
if(TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_PERF_ANALYZER)
  find_package(CURL REQUIRED)
  message(STATUS "Using curl ${CURL_VERSION}")
endif() # TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_PERF_ANALYZER

#
# Protobuf
#
set(Protobuf_DIR /workspace/build/third-party/protobuf/lib/cmake/protobuf)
if(TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER)
  set(protobuf_MODULE_COMPATIBLE TRUE CACHE BOOL "protobuf_MODULE_COMPATIBLE" FORCE)
  find_package(Protobuf CONFIG REQUIRED PATHS /workspace/build/third-party/protobuf/cmake/protobuf)
  message(STATUS "Using protobuf ${Protobuf_VERSION}")
  include_directories(${Protobuf_INCLUDE_DIRS})
endif() # TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER

#
# GRPC
#
set(gRPC_DIR /workspace/build/third-party/grpc/lib/cmake/grpc)
if(TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER)
  find_package(gRPC CONFIG REQUIRED)
  message(STATUS "Using gRPC ${gRPC_VERSION}")
  include_directories($<TARGET_PROPERTY:gRPC::grpc,INTERFACE_INCLUDE_DIRECTORIES>)
endif() # TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER

#if(TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER)
#  add_subdirectory(library)
#endif() # TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC OR TRITON_ENABLE_PERF_ANALYZER

if(TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC)
  if(TRITON_ENABLE_EXAMPLES)
    add_subdirectory(src)
  endif() # TRITON_ENABLE_EXAMPLES
endif() # TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC