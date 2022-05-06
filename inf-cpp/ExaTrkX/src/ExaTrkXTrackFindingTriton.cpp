#include "ExaTrkXTrackFindingTriton.hpp"

#include "mmio_read.h"
#include "build_edges.hpp"

#include "grpc_client.h"
#include "grpc_service.pb.h"
namespace tc = triton::client;


#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }

ExaTrkXTrackFindingTriton::ExaTrkXTrackFindingTriton(
    const ExaTrkXTrackFindingTriton::Config& config): m_cfg(config)
{
    initTrainedModels();
}

ExaTrkXTrackFindingTriton::~ExaTrkXTrackFindingTriton() {

}

void ExaTrkXTrackFindingTriton::initTrainedModels(){
    // start the communication with the Server?
    bool verbose = false;
    const std::string url("localhost:8001");
    const uint32_t lo = 314;
    bool use_ssl = false;
    std::string root_certificates;
    std::string private_key;
    std::string certificate_chain;


    // We use a simple model that takes 2 input tensors of 16 integers
    // each and returns 2 output tensors of 16 integers each. One output
    // tensor is the element-wise sum of the inputs and one output is
    // the element-wise difference.


    // Create a InferenceServerGrpcClient instance to communicate with the
    // server using gRPC protocol.
    std::unique_ptr<triton::client::InferenceServerGrpcClient> embedClient;
    if (use_ssl) {
        tc::SslOptions ssl_options;
        ssl_options.root_certificates = root_certificates;
        ssl_options.private_key = private_key;
        ssl_options.certificate_chain = certificate_chain;
        tc::Error err = tc::InferenceServerGrpcClient::Create(
            &embedClient, url, verbose, use_ssl, ssl_options);
    } else {
        FAIL_IF_ERR(
            tc::InferenceServerGrpcClient::Create(&embedClient, url, verbose),
            "unable to create grpc client");
         // tc::InferenceServerGrpcClient::Create(&embedClient, url, verbose);
    }
    m_embedClient = std::move(embedClient);
}


// The main function that runs the Exa.TrkX pipeline
// Be care of sharpe corners.
void ExaTrkXTrackFindingTriton::getTracks(
    std::vector<float>& inputValues,
    std::vector<int>& spacepointIDs,
    std::vector<std::vector<int> >& trackCandidates)
{

    // ************
    // Embedding
    // ************

    int64_t numSpacepoints = inputValues.size() / m_cfg.spacepointFeatures;
    std::vector<int64_t> embedInputShape{numSpacepoints, m_cfg.spacepointFeatures};
    std::cout << "My input shape is: " << inputValues.size() << std::endl;
    std::cout << "My embedding shape is: " << embedInputShape[0] << " " << embedInputShape[1] << std::endl;
    // Initialize the inputs with the data.
    tc::InferInput* input0;
    FAIL_IF_ERR(
        tc::InferInput::Create(&input0, "INPUT__0" , embedInputShape, "FP32"), 
        "unable to get INPUT0");

    std::shared_ptr<tc::InferInput> input0_ptr;
    input0_ptr.reset(input0);
    FAIL_IF_ERR(input0_ptr->AppendRaw(
        reinterpret_cast<uint8_t*>(&inputValues[0]), // why uint8?
        inputValues.size() * sizeof(float)), "unable to set data for INPUT0");

    tc::InferRequestedOutput* embedOutput;
    std::string model_name = "embed";
    std::string model_version = "";
    uint32_t client_timeout = 0;
    tc::Headers http_headers;
    grpc_compression_algorithm compression_algorithm =
        grpc_compression_algorithm::GRPC_COMPRESS_NONE;
    
    tc::InferOptions options(model_name);
    options.model_version_ = model_version;
    options.client_timeout_ = client_timeout;

    std::vector<tc::InferInput*> inputs = {input0_ptr.get()};
    std::cout << "Input size: " << inputs.size() << std::endl;
    std::vector<const tc::InferRequestedOutput*> outputs = {}; //output0_ptr.get()

    tc::InferResult* results;
    FAIL_IF_ERR(m_embedClient->Infer(
        &results, options, inputs, outputs, http_headers,
        compression_algorithm), "unable to run Embedding");

    std::shared_ptr<tc::InferResult> results_ptr;
    results_ptr.reset(results);

    float* output0_data;
    size_t output0_byte_size;
    results_ptr->RawData(
          "OUTPUT__0", (const uint8_t**)&output0_data, &output0_byte_size);

    std::cout << "bit size: " << output0_byte_size << std::endl;
    std::cout << "size of outputs: " << sizeof(output0_data) << std::endl;
    std::cout << "entries of outputs: " << sizeof(output0_data) / output0_byte_size << std::endl;

    std::vector<float> embededData;
    for (size_t i = 0; i < numSpacepoints; ++i) {
      for (size_t j = 0; j < m_cfg.embeddingDim; ++j) 
        // std::cout << *(output0_data + j + i*m_cfg.embeddingDim) <<  " ";
        embededData.push_back(*(output0_data + i*m_cfg.embeddingDim));
    }
    std::cout << "after embedding: " << embededData.size() << std::endl;
    trackCandidates.clear();
}