#
# simple_grpc_infer_client
#
add_executable(simple_grpc_infer_client simple_grpc_infer_client.cc)
target_link_libraries(
  simple_grpc_infer_client
  PUBLIC
  /workspace/install/lib/libgrpcclient.so
  /workspace/build/third-party/protobuf/lib/libprotobuf.a
)
target_include_directories(
  simple_grpc_infer_client
  PUBLIC
  /workspace/install/include
)

#
# simple_http_infer_client
#
add_executable(simple_http_infer_client simple_http_infer_client.cc)
target_link_libraries(
 simple_http_infer_client
 PUBLIC
   /workspace/install/lib/libhttpclient.so
 )
 target_include_directories(
   simple_http_infer_client
   PRIVATE
     /workspace/install/include
 )