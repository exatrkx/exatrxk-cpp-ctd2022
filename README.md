# exatrkx-cpp
C++ version of the exa.trkx ML tracking pipeline

$cmake -DCMAKE_INSTALL_PREFIX=`pwd`/install -DTRITON_ENABLE_CC_HTTP=ON -DTRITON_ENABLE_CC_GRPC=ON -DTRITON_ENABLE_PERF_ANALYZER=OFF  -DTRITON_ENABLE_GPU=OFF -DTRITON_ENABLE_EXAMPLES=ON  ..

Add 
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

to build/_deps/repo-common-src/protobuf/CMakeLists.txt

$make