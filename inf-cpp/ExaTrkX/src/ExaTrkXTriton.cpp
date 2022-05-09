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
    std::cout << "output_size: " << sizeof(output_data) << std::endl;
    for (size_t i = 0; i < outputShape[0]; ++i) {
      for (size_t j = 0; j < outputShape[1]; ++j) 
        outputData.push_back(*(output_data + i*outputShape[1] + j));
    }

    return true;
}

void ExaTrkXTriton::Inference(
    const std::string& inputName, const std::vector<int64_t>& inputShape,
    std::vector<float>& inputValues, const std::string dataType,
    const std::string outputName, std::vector<float>& outputData,
    const std::vector<int64_t>& outputShape
) {
    tc::InferInput* input0;
    FAIL_IF_ERR(
        tc::InferInput::Create(&input0, inputName, inputShape, dataType), 
        "unable to get"+inputName);

    std::shared_ptr<tc::InferInput> input0_ptr(input0);

    FAIL_IF_ERR(input0_ptr->AppendRaw(
        reinterpret_cast<uint8_t*>(&inputValues[0]), // why uint8?
        inputValues.size() * sizeof(float)), "unable to set data"+inputName);
    
    std::vector<tc::InferInput*> inputs = {input0_ptr.get()};

    tc::Headers http_headers;
    grpc_compression_algorithm compression_algorithm =
        grpc_compression_algorithm::GRPC_COMPRESS_NONE;

    tc::InferResult* results;
    std::vector<const tc::InferRequestedOutput*> outputs = {};

    std::cout << "prepare to run inference with " << inputs_.size() << " input(s)." << std::endl;
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
    std::cout << "output_size: " << sizeof(output_data) << std::endl;
    for (size_t i = 0; i < outputShape[0]; ++i) {
      for (size_t j = 0; j < outputShape[1]; ++j) 
        outputData.push_back(*(output_data + i*outputShape[1] + j));
    }
}

void ExaTrkXTriton::ClearInput()
{
    /// <TODO: do I have to delete the InferInput object?>
    inputs_.clear();
}