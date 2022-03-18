// Copyright 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <getopt.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include "http_client.h"

namespace tc = triton::client;

#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                           \
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
  //if ((shape.size() != 1) || (shape[0] != 16)) {
    std::cerr << "error: received incorrect shapes for '" << name << "'"
              << shape[0] << " " << shape[1] << std::endl;
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

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-t <client timeout in microseconds>" << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr << "\t-i <none|gzip|deflate>" << std::endl;
  std::cerr << "\t-o <none|gzip|deflate>" << std::endl;
  std::cerr << std::endl;
  std::cerr << "\t--verify-peer" << std::endl;
  std::cerr << "\t--verify-host" << std::endl;
  std::cerr << "\t--ca-certs" << std::endl;
  std::cerr << "\t--cert-file" << std::endl;
  std::cerr << "\t--key-file" << std::endl;
  std::cerr
      << "For -H, header must be 'Header:Value'. May be given multiple times."
      << std::endl
      << "For -i, it sets the compression algorithm used for sending request "
         "body."
      << "For -o, it sets the compression algorithm used for receiving "
         "response body."
      << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
{
  bool verbose = false;
  std::string url("localhost:8000");
  tc::Headers http_headers;
  uint32_t client_timeout = 0;
  const uint32_t lo = 314;
  auto request_compression_algorithm =
      tc::InferenceServerHttpClient::CompressionType::NONE;
  auto response_compression_algorithm =
      tc::InferenceServerHttpClient::CompressionType::NONE;
  long verify_peer = 1;
  long verify_host = 2;
  std::string cacerts;
  std::string certfile;
  std::string keyfile;

  // {name, has_arg, *flag, val}
  static struct option long_options[] = {
      {"verify-peer", 1, 0, 0}, {"verify-host", 1, 0, 1}, {"ca-certs", 1, 0, 2},
      {"cert-file", 1, 0, 3},   {"key-file", 1, 0, 4},    {0, 0, 0, 0}};

  // Parse commandline...
  int opt;
  while ((opt = getopt_long(argc, argv, "vu:t:H:i:o:", long_options, NULL)) !=
         -1) {
    switch (opt) {
      case 0:
        verify_peer = std::atoi(optarg);
        break;
      case 1:
        verify_host = std::atoi(optarg);
        break;
      case 2:
        cacerts = optarg;
        break;
      case 3:
        certfile = optarg;
        break;
      case 4:
        keyfile = optarg;
        break;
      case 'v':
        verbose = true;
        break;
      case 'u':
        url = optarg;
        break;
      case 't':
        client_timeout = std::stoi(optarg);
        break;
      case 'H': {
        std::string arg = optarg;
        std::string header = arg.substr(0, arg.find(":"));
        http_headers[header] = arg.substr(header.size() + 1);
        break;
      }
      case 'i': {
        std::string arg = optarg;
        if (arg == "gzip") {
          request_compression_algorithm =
              tc::InferenceServerHttpClient::CompressionType::GZIP;
        } else if (arg == "deflate") {
          request_compression_algorithm =
              tc::InferenceServerHttpClient::CompressionType::DEFLATE;
        }
        break;
      }
      case 'o': {
        std::string arg = optarg;
        if (arg == "gzip") {
          response_compression_algorithm =
              tc::InferenceServerHttpClient::CompressionType::GZIP;
        } else if (arg == "deflate") {
          response_compression_algorithm =
              tc::InferenceServerHttpClient::CompressionType::DEFLATE;
        }
        break;
      }
      case '?':
        Usage(argv);
        break;
    }
  }
  //client_->setBatchSize(1);
  // We use a simple model that takes 2 input tensors of 16 integers
  // each and returns 2 output tensors of 16 integers each. One output
  // tensor is the element-wise sum of the inputs and one output is
  // the element-wise difference.
  std::string model_name = "embedo"; //simple
  std::string model_version = "";

  srand( (unsigned)time( NULL ) );

  tc::HttpSslOptions ssl_options;
  ssl_options.verify_peer = verify_peer;
  ssl_options.verify_host = verify_host;
  ssl_options.ca_info = cacerts;
  ssl_options.cert = certfile;
  ssl_options.key = keyfile;
  // Create a InferenceServerHttpClient instance to communicate with the
  // server using HTTP protocol.
  std::unique_ptr<tc::InferenceServerHttpClient> client;
  FAIL_IF_ERR(
      tc::InferenceServerHttpClient::Create(&client, url, verbose, ssl_options),
      "unable to create http client");

  // Create the data for the two input tensors. Initialize the first
  // to unique integers and the second to all ones.
  //std::array<std::array<float, lo>, 12> input0_data;
 
  //std::vector<std::vector<float>> input0_data(lo, std::vector<float>(12, 1.0));

  std::fstream is("../data/in_e.csv", std::ios_base::in);
  std::vector<float> input0_data; //(lo*12,1.0);
  float number;
  while (is >> number)
  {
   input0_data.push_back(number);
  }

  for (size_t i = 0; i < lo; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      //input0_data[j+i*12] = (float) rand()/RAND_MAX;
      std::cout << input0_data[j+i*3] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "Size: "<< input0_data.size() << std::endl;
  std::vector<int64_t> shape{lo,3};

  // Initialize the inputs with the data.
  tc::InferInput* input0;
 
  FAIL_IF_ERR(
      tc::InferInput::Create(&input0, "sp_features", shape, "FP32"), //"INPUT__0"
      "unable to get INPUT0");
  std::shared_ptr<tc::InferInput> input0_ptr;
  input0_ptr.reset(input0);

  FAIL_IF_ERR(
      input0_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&input0_data[0]), 
          input0_data.size() * 3 * sizeof(float)),
      "unable to set data for INPUT0");

  // The inference settings. Will be using default for now.
  tc::InferOptions options(model_name);
  options.model_version_ = model_version;
  options.client_timeout_ = client_timeout;

  std::vector<tc::InferInput*> inputs = {input0_ptr.get()};
  // Empty output vector will request data for all the output tensors from
  // the server.
  std::vector<const tc::InferRequestedOutput*> outputs = {};

  tc::InferResult* results;
  FAIL_IF_ERR(
      client->Infer(
          &results, options, inputs, outputs, http_headers, tc::Parameters(),
          request_compression_algorithm, response_compression_algorithm),
      "unable to run model");
  std::shared_ptr<tc::InferResult> results_ptr;
  results_ptr.reset(results);

  // Validate the results...
  ValidateShapeAndDatatype("embedding_output", results_ptr); //"OUTPUT__0"

  // Get pointers to the result returned...
  float* output0_data;
  size_t output0_byte_size;
  FAIL_IF_ERR(
      results_ptr->RawData(
          "embedding_output", (const uint8_t**)&output0_data, &output0_byte_size),
      "unable to get result data for 'OUTPUT0'");
  std::cout << "Output Size: "<< output0_byte_size << std::endl;
  //if (output0_byte_size != 32) {
  //  std::cerr << "error: received incorrect byte size for 'OUTPUT0': "
  //            << output0_byte_size << std::endl;
  //  exit(1);
  //}

    std::cout << std::endl << std::endl;
    for (size_t i = 0; i < lo; ++i) {
      for (size_t j = 0; j < 8; ++j) 
        std::cout << *(output0_data + j + i*8) <<  " ";
      std::cout << std::endl;
    }
  // Get full response
  //std::cout << results_ptr->DebugString() << std::endl;

  tc::InferStat infer_stat;
  client->ClientInferStat(&infer_stat);
  std::cout << "======Client Statistics======" << std::endl;
  std::cout << "completed_request_count " << infer_stat.completed_request_count
            << std::endl;
  std::cout << "cumulative_total_request_time_ns "
            << infer_stat.cumulative_total_request_time_ns << std::endl;
  std::cout << "cumulative_send_time_ns " << infer_stat.cumulative_send_time_ns
            << std::endl;
  std::cout << "cumulative_receive_time_ns "
            << infer_stat.cumulative_receive_time_ns << std::endl;

  std::string model_stat;
  FAIL_IF_ERR(
      client->ModelInferenceStatistics(&model_stat, model_name),
      "unable to get model statistics");
  std::cout << "======Model Statistics======" << std::endl;
  std::cout << model_stat << std::endl;

  std::cout << "PASS : Infer" << std::endl;

  return 0;
}
