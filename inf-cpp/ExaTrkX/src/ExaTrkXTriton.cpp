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
    std::cout << "In the inference" << std::endl;
    if (outputShape.size() != 2) {
        std::cerr << "error: output shape must be 2D" << std::endl;
    }
    tc::Headers http_headers;
    grpc_compression_algorithm compression_algorithm =
        grpc_compression_algorithm::GRPC_COMPRESS_NONE;

    tc::InferResult* results;
    std::vector<const tc::InferRequestedOutput*> outputs = {};

    std::cout << "prepare to run inference with " << inputs_.size() << " input(s)." << std::endl;
    std::vector<tc::InferInput*> inputs;
    for (auto& input : inputs_) {
        inputs.push_back(input.get());
    }
    FAIL_IF_ERR(m_Client_->Infer(
        &results, *options_, inputs, outputs, http_headers,
        compression_algorithm), "unable to run Embedding");

    std::shared_ptr<tc::InferResult> results_ptr;
    results_ptr.reset(results);

    std::cout << "after run inference" << std::endl;
    float* output_data;
    size_t output_size;
    results_ptr->RawData(
          outputName, (const uint8_t**)&output_data, &output_size);

    std::cout << "before transfer data" << std::endl;
    outputData.clear();
    int64_t output_entries = sizeof(output_data) / sizeof(output_data[0]);
    std::cout << "output_size: " << output_entries << std::endl;

    int64_t num_rows = outputShape[0];
    int64_t num_cols = outputShape[1];
    if (num_rows < 0) {
        num_rows = (int64_t) output_entries / num_cols;
    }
    if (num_cols < 0) {
        num_cols = (int64_t) output_entries / num_rows;
    }


    for (size_t i = 0; i < num_rows; ++i) {
      for (size_t j = 0; j < num_cols; ++j) 
        outputData.push_back(*(output_data + i*num_cols + j));
    }

    return true;
}

void ExaTrkXTriton::ClearInput()
{
  inputs_.clear();
}