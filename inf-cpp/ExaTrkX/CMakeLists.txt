add_subdirectory(libFRNN)

add_library(
  ActsPluginExaTrkXTorch SHARED
  src/ExaTrkXTrackFinding.cpp)
set_target_properties(ActsPluginExaTrkXTorch
PROPERTIES  BUILD_RPATH    "\$ORIGIN"
            INSTALL_RPATH  "\$ORIGIN"
            CXX_STANDARD                        17
            CXX_STANDARD_REQUIRED               ON
            CUDA_STANDARD                       17
            CUDA_STANDARD_REQUIRED              ON
            INTERFACE_POSITION_INDEPENDENT_CODE ON
            CUDA_SEPARABLE_COMPILATION          ON
)
target_compile_options(ActsPluginExaTrkXTorch
    PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
            "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>")
target_compile_definitions(ActsPluginExaTrkXTorch PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
target_include_directories(
  ActsPluginExaTrkXTorch
  PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(
  ActsPluginExaTrkXTorch
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>

)
target_link_libraries(
  ActsPluginExaTrkXTorch
  PUBLIC
    ${TORCH_LIBRARIES}
)
target_link_libraries(
  ActsPluginExaTrkXTorch
  PRIVATE
    frnn cugraph ${TORCH_LIBRARIES}
)


#### ActsPluginExaTrkXTriton
set(CMAKE_BUILD_RPATH "")
add_library(ActsPluginExaTrkXTriton
SHARED
  src/ExaTrkXTriton.cpp
  src/ExaTrkXTrackFindingTriton.cpp
)

set_target_properties(ActsPluginExaTrkXTriton
  PROPERTIES  CXX_STANDARD                        17
              CXX_STANDARD_REQUIRED               ON
              CUDA_STANDARD                       17
              CUDA_STANDARD_REQUIRED              ON
              INTERFACE_POSITION_INDEPENDENT_CODE ON
              CUDA_SEPARABLE_COMPILATION          ON
  )
target_compile_options(ActsPluginExaTrkXTriton
  PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${CUGRAPH_CXX_FLAGS}>"
          "$<$<COMPILE_LANGUAGE:CUDA>:${CUGRAPH_CUDA_FLAGS}>")
target_compile_definitions(ActsPluginExaTrkXTriton PUBLIC CUDA_API_PER_THREAD_DEFAULT_STREAM)
target_compile_definitions(ActsPluginExaTrkXTriton PUBLIC TRITON_ENABLE_GPU)
target_include_directories(
    ActsPluginExaTrkXTriton
    PUBLIC 
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_include_directories(
    ActsPluginExaTrkXTriton
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)


target_link_libraries(
  ActsPluginExaTrkXTriton
  PRIVATE
    frnn 
    cugraph 
    ${TORCH_LIBRARIES}
    # /workspace/install/lib/libgrpcclient_static.a
    /workspace/install/lib/libgrpcclient.so
    /workspace/build/third-party/protobuf/lib/libprotobuf.a
)
target_include_directories(
  ActsPluginExaTrkXTriton
  PUBLIC
     /workspace/install/include
)