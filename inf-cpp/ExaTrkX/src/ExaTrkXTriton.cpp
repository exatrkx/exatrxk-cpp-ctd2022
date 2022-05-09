#include "ExaTrkXTriton.hpp"

#include <iostream>

namespace tc = triton::client;

ExaTrkXTriton::ExaTrkXTriton(
    const std::string& modelName,
    const std::string& url,
    const std::string& modelVersion,
    uint32_t client_timeout, bool verbose
){
    options_ = std::make_unique<tc::InferOptions>(modelName);
    options_->model_version_ = modelVersion;
    options_->client_timeout_ = client_timeout;

    inputs_.clear();

    // Create a InferenceServerGrpcClient instance to communicate with the
    // server using gRPC protocol.
    std::unique_ptr<triton::client::InferenceServerGrpcClient> tClient;
    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(&tClient, url, verbose),
        "unable to create grpc client");
    m_Client_ = std::move(tClient);
}

bool ExaTrkXTriton::GetOutput(
    const std::string& outputName, std::vector<float>& outputData,
    const std::vector<int64_t>&  outputShape)
{
    if (outputShape.size() != 2) {
        std::cerr << "error: output shape must be 2D" << std::endl;
    }
    tc::Headers http_headers;
    grpc_compression_algorithm compression_algorithm =
        grpc_compression_algorithm::GRPC_COMPRESS_NONE;

    tc::InferResult* results;
    std::vector<const tc::InferRequestedOutput*> outputs = {}; //output0_ptr.get()

    FAIL_IF_ERR(m_Client_->Infer(
        &results, *options_, inputs_, outputs, http_headers,
        compression_algorithm), "unable to run Embedding");

    std::shared_ptr<tc::InferResult> results_ptr;
    results_ptr.reset(results);

    float* output_data;
    size_t output_size;
    results_ptr->RawData(
          outputName, (const uint8_t**)&output_data, &output_size);

    outputData.clear();
    for (size_t i = 0; i < outputShape[0]; ++i) {
      for (size_t j = 0; j < outputShape[1]; ++j) 
        outputData.push_back(*(output_data + i*outputShape[1] + j));
    }

    return true;
}

void ExaTrkXTriton::ClearInput()
{
    /// <TODO: do I have to delete the InferInput object?>
    inputs_.clear();
}