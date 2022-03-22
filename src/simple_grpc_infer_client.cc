#include <getopt.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
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

void
Usage(char** argv, const std::string& msg = std::string())
{
  if (!msg.empty()) {
    std::cerr << "error: " << msg << std::endl;
  }

  std::cerr << "Usage: " << argv[0] << " [options]" << std::endl;
  std::cerr << "\t-v" << std::endl;
  std::cerr << "\t-m <model name>" << std::endl;
  std::cerr << "\t-u <URL for inference service>" << std::endl;
  std::cerr << "\t-t <client timeout in microseconds>" << std::endl;
  std::cerr << "\t-H <HTTP header>" << std::endl;
  std::cerr << std::endl;
  std::cerr
      << "For -H, header must be 'Header:Value'. May be given multiple times."
      << std::endl;

  exit(1);
}

}  // namespace

int
main(int argc, char** argv)
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

  // Parse commandline...
  int opt;
  while ((opt = getopt_long(argc, argv, "vu:t:H:C:", long_options, NULL)) !=
         -1) {
    switch (opt) {
      case 0:
        use_ssl = true;
        break;
      case 1:
        root_certificates = optarg;
        break;
      case 2:
        private_key = optarg;
        break;
      case 3:
        certificate_chain = optarg;
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
      case 'C': {
        std::string algorithm_str{optarg};
        if (algorithm_str.compare("deflate") == 0) {
          compression_algorithm =
              grpc_compression_algorithm::GRPC_COMPRESS_DEFLATE;
        } else if (algorithm_str.compare("gzip") == 0) {
          compression_algorithm =
              grpc_compression_algorithm::GRPC_COMPRESS_GZIP;
        } else if (algorithm_str.compare("none") == 0) {
          compression_algorithm =
              grpc_compression_algorithm::GRPC_COMPRESS_NONE;
        } else {
          Usage(
              argv,
              "unsupported compression algorithm specified... only "
              "\'deflate\', "
              "\'gzip\' and \'none\' are supported.");
        }
        break;
      }
      case '?':
        Usage(argv);
        break;
    }
  }

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

  // Create the data for the two input tensors. Initialize the first
  // to unique integers and the second to all ones.
  
  std::fstream is("../data/in_e.csv", std::ios_base::in);
  std::vector<float> input0_data; //(lo*12,1.0);
  float number;
  while (is >> number)
  {
   input0_data.push_back(number);
   std::cout<<number<<std::endl;
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
      tc::InferInput::Create(&input0, "INPUT__0" , shape, "FP32"), //"sp_features"
      "unable to get INPUT0");
  std::shared_ptr<tc::InferInput> input0_ptr;
  input0_ptr.reset(input0);
 
  FAIL_IF_ERR(
      input0_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&input0_data[0]),
          input0_data.size() * sizeof(float)),
      "unable to set data for INPUT0");

  // Generate the outputs to be requested.
  tc::InferRequestedOutput* output0;

  //FAIL_IF_ERR(
  //    tc::InferRequestedOutput::Create(&output0, "OUTPUT__0"), //"embedding_output"
  //    "unable to get embedding_output");
  //std::shared_ptr<tc::InferRequestedOutput> output0_ptr;
  //output0_ptr.reset(output0);

  // The inference settings. Will be using default for now.
  tc::InferOptions options(model_name);
  options.model_version_ = model_version;
  options.client_timeout_ = client_timeout;

  std::vector<tc::InferInput*> inputs = {input0_ptr.get()};
  std::vector<const tc::InferRequestedOutput*> outputs = {}; //output0_ptr.get()

  tc::InferResult* results;
  FAIL_IF_ERR(
      client->Infer(
          &results, options, inputs, outputs, http_headers,
          compression_algorithm),
      "unable to run model");
  std::shared_ptr<tc::InferResult> results_ptr;
  results_ptr.reset(results);

  // Validate the results...
  ValidateShapeAndDatatype("OUTPUT__0", results_ptr);
  
  // Get pointers to the result returned...
  float* output0_data;
  size_t output0_byte_size;
  FAIL_IF_ERR(
      results_ptr->RawData(
          "OUTPUT__0", (const uint8_t**)&output0_data, &output0_byte_size),
      "unable to get result data for 'OUTPUT__0'");
  //if (output0_byte_size != 64) {
  //  std::cerr << "error: received incorrect byte size for 'OUTPUT0': "
  //            << output0_byte_size << std::endl;
  //  exit(1);
  //}

  std::cout << std::endl << std::endl;
    for (size_t i = 0; i < lo; ++i) {
      for (size_t j = 0; j < 8; ++j) 
        std::cout << *(output0_data + j + i*8) <<  " ";
      std::cout << "Asta e:"<<std::endl;
    }

  // Get full response
  std::cout << results_ptr->DebugString() << std::endl;

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

  inference::ModelStatisticsResponse model_stat;
  client->ModelInferenceStatistics(&model_stat, model_name);
  std::cout << "======Model Statistics======" << std::endl;
  std::cout << model_stat.DebugString() << std::endl;


  std::cout << "PASS : Infer" << std::endl;

  return 0;
}