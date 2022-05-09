cmake_minimum_required(VERSION 3.20.1)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

project(Exatrkx_Inference VERSION 0.2.0 LANGUAGES C CXX CUDA)
message(STATUS "CUGRAPH_TESTS: CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")


if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA" AND
   CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA compiler version must be at least 11.0")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.3)
    message(FATAL_ERROR "GCC compiler must be at least 9.3")
endif()


# Remove the following archs from CMAKE_CUDA_ARCHITECTURES that
# cuhornet currently doesn't support
#
# >= 86
set(supported_archs "60" "62" "70" "72" "75" "80")
foreach( arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
    string(REPLACE "-real" "" arch ${arch})
    if( arch IN_LIST supported_archs )
        list(APPEND usable_arch_values ${arch})
    endif()
endforeach()
# Make sure everything but the 'newest' arch
# is marked as `-real` so we only generate PTX for
# arch > 80
list(POP_BACK usable_arch_values latest_arch)
list(TRANSFORM usable_arch_values APPEND "-real")
list(APPEND usable_arch_values ${latest_arch})
set(CMAKE_CUDA_ARCHITECTURES ${usable_arch_values})


option(CUDA_STATIC_RUNTIME "Use CUDA static runtime" OFF)
option(CMAKE_CUDA_LINEINFO "Enable the -lineinfo option for nvcc (useful for cuda-memcheck / profiler" OFF)

set(CUGRAPH_CXX_FLAGS "")
set(CUGRAPH_CUDA_FLAGS "")

if(CMAKE_COMPILER_IS_GNUCXX)
    list(APPEND CUGRAPH_CXX_FLAGS -Werror -Wno-error=deprecated-declarations)
endif(CMAKE_COMPILER_IS_GNUCXX)


message("-- Building for GPU_ARCHS = ${CMAKE_CUDA_ARCHITECTURES}")

list(APPEND CUGRAPH_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND CUGRAPH_CUDA_FLAGS -Xcompiler=-Wall,-Wno-error=sign-compare,-Wno-error=unused-but-set-variable)
list(APPEND CUGRAPH_CUDA_FLAGS -Xfatbin=-compress-all)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if (CMAKE_CUDA_LINEINFO)
    list(APPEND CUGRAPH_CUDA_FLAGS -lineinfo)
endif(CMAKE_CUDA_LINEINFO)

# Debug options
if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Building with debugging flags")
    list(APPEND CUGRAPH_CUDA_FLAGS -G -Xcompiler=-rdynamic)
endif(CMAKE_BUILD_TYPE MATCHES Debug)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_C_FLAGS "-Wno-error=format-truncation") 

find_package(Torch REQUIRED)
find_package(cugraph REQUIRED)
find_package(Boost REQUIRED)

find_package(TritonCore CONFIG REQUIRED)
find_package(TritonCommon CONFIG REQUIRED)
find_package(TritonClient CONFIG REQUIRED COMPONENTS TritonClient::triton-client)

add_subdirectory(ExaTrkX)

add_executable(inference inference.cpp)
target_link_libraries(inference PUBLIC ActsPluginExaTrkXTorch)


add_executable(inference_client inference_client.cpp)
target_link_libraries(inference_client PUBLIC ActsPluginExaTrkXTriton)
target_include_directories(
    inference_client
  PUBLIC
    /workspace/install/include
    /usr/local/cuda/include
)

# target_link_libraries(inference_client
#  PUBLIC
#    ActsPluginExaTrkXTriton
#    /workspace/install/lib/libgrpcclient.so
#    /workspace/build/third-party/protobuf/lib/libprotobuf.a
#  )
# target_include_directories(
#   inference_client
#   PRIVATE
#     /workspace/install/include
# )
# target_link_libraries(inference
#   PUBLIC
# #   /workspace/install/lib/libgrpcclient_static.a
#   # /workspace/install/lib/libgrpcclient.so
#   # /workspace/build/third-party/protobuf/lib/libprotobuf.a
# )
# target_link_libraries(inference 
#   PUBLIC
#     frnn
#     cugraph
#     # ${TORCH_LIBRARIES}
# )


# set_target_properties(inference
# PROPERTIES  BUILD_RPATH    "\$ORIGIN"
#             INSTALL_RPATH  "\$ORIGIN"
#             CXX_STANDARD                        17
#             CXX_STANDARD_REQUIRED               ON
#             CUDA_STANDARD                       17
#             CUDA_STANDARD_REQUIRED              ON
#             INTERFACE_POSITION_INDEPENDENT_CODE ON
# )
# target_compile_options(inference
#     PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
#             "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>"
# )
# target_compile_definitions(inference PUBLIC TRITON_ENABLE_GPU)
# target_compile_definitions(inference PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
# target_compile_definitions(inference PUBLIC CUDA_SEPARABLE_COMPILATION ON)

# target_include_directories(
#   inference
#   PUBLIC
#     /workspace/install/include
#     /usr/local/cuda/include
# )