cmake_minimum_required (VERSION 3.18)
#
# Protobuf
#
set(Protobuf_DIR /workspace/build/third-party/protobuf/lib/cmake/protobuf)

set(protobuf_MODULE_COMPATIBLE TRUE CACHE BOOL "protobuf_MODULE_COMPATIBLE" FORCE)
find_package(Protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${Protobuf_VERSION}")
include_directories(${Protobuf_INCLUDE_DIRS})

#
# simple_grpc_infer_client
#
add_executable(simple_grpc_infer_client simple_grpc_infer_client.cc)
target_link_libraries(
  simple_grpc_infer_client
  PUBLIC
    /workspace/install/lib/libgrpcclient.so
    ${PROTOBUF_LIBRARY}
)
target_include_directories(
  simple_grpc_infer_client
  PRIVATE
    /workspace/install/include
)
install(
  TARGETS simple_grpc_infer_client
  RUNTIME DESTINATION bin
)


#
# simple_http_infer_client
#
#add_executable(simple_http_infer_client simple_http_infer_client.cc)
#target_link_libraries(
#  simple_http_infer_client
#  PUBLIC
#    /workspace/install/lib/libhttpclient.so
#  )
#  target_include_directories(
#    simple_http_infer_client
#    PRIVATE
#      /workspace/install/include
#  )
#  install(
#    TARGETS simple_http_infer_client
#    RUNTIME DESTINATION bin
#)