#include <istream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <getopt.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>
#include <unistd.h>

// #include "ExaTrkXTrackFindingTriton.hpp"
#include "grpc_client.h"

namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

namespace {

void
ValidateShapeAndDatatype(
    const std::string& name, std::shared_ptr<tc::InferResult> result)
{
  std::vector<int64_t> shape;
  FAIL_IF_ERR(
      result->Shape(name, &shape), "unable to get shape for '" + name + "'");
  // Validate shape
  if ((shape.size() != 2) || (shape[0] != 314) || (shape[1] != 8)) {
    std::cerr << "error: received incorrect shapes for '" << name << "'"
              << std::endl;
    exit(1);
  }
  std::string datatype;
  FAIL_IF_ERR(
      result->Datatype(name, &datatype),
      "unable to get datatype for '" + name + "'");
  // Validate datatype
  if (datatype.compare("FP32") != 0) {
    std::cerr << "error: received incorrect datatype for '" << name
              << "': " << datatype << std::endl;
    exit(1);
  }
}

}  // namespace


// initialize  enviroment...one enviroment per process
// enviroment maintains thread pools and other state info
int main(int argc, char* argv[])
{
    bool verbose = false;
    std::string url("localhost:8001");
    tc::Headers http_headers;
    uint32_t client_timeout = 0;
    const uint32_t lo = 314;
    bool use_ssl = false;
    std::string root_certificates;
    std::string private_key;
    std::string certificate_chain;
    grpc_compression_algorithm compression_algorithm =
        grpc_compression_algorithm::GRPC_COMPRESS_NONE;

    // {name, has_arg, *flag, val}
    static struct option long_options[] = {{"ssl", 0, 0, 0},
                                           {"root-certificates", 1, 0, 1},
                                           {"private-key", 1, 0, 2},
                                           {"certificate-chain", 1, 0, 3}};

    
    // We use a simple model that takes 2 input tensors of 16 integers
    // each and returns 2 output tensors of 16 integers each. One output
    // tensor is the element-wise sum of the inputs and one output is
    // the element-wise difference.
    std::string model_name = "embed";
    std::string model_version = "";

  // Create a InferenceServerGrpcClient instance to communicate with the
  // server using gRPC protocol.
    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    if (use_ssl) {
      tc::SslOptions ssl_options;
      ssl_options.root_certificates = root_certificates;
      ssl_options.private_key = private_key;
      ssl_options.certificate_chain = certificate_chain;
      FAIL_IF_ERR(
          tc::InferenceServerGrpcClient::Create(
              &client, url, verbose, use_ssl, ssl_options),
          "unable to create secure grpc client");
    } else {
      FAIL_IF_ERR(
          tc::InferenceServerGrpcClient::Create(&client, url, verbose),
          "unable to create grpc client");
    }
    
    
    // std::cout << "Building and running a GPU inference engine for Embedding" << std::endl;
    // Infer::Config config({"../datanmodels"});
    // Infer infer(config);
    
    // // read spacepoints table saved in csv
    // std::string input_filepath = "../datanmodels/in_e1000.csv";
    // std::vector<float> input_tensor_values;
    // processInput(input_filepath, input_tensor_values);

    // int numSpacepoints = input_tensor_values.size()/config.spacepointFeatures;
    // std::cout << numSpacepoints << " spacepoints." << std::endl;

    // // <TODO: add real spacepoint ids
    // std::vector<int> spacepoint_ids;
    // for (int i=0; i < numSpacepoints; ++i){
    //     spacepoint_ids.push_back(i);
    // }


    // std::vector<std::vector<int> > track_candidates;
    // infer.getTracks(input_tensor_values, spacepoint_ids, track_candidates);
    // std::cout << track_candidates.size() << " reconstructed tracks." << std::endl;
    return 0;
}