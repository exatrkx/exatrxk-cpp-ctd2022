#include "ExaTrkXTriton.hpp"

namespace tc = triton::client;

ExaTrkXTriton::ExaTrkXTriton(std::string modelName)
: options_(modelName)
{};

ExaTrkXTriton::~ExaTrkXTriton() {};

bool ExaTrkXTriton::InitClient(
    std::string url, std::string modelVersion = "",
    uint32_t client_timeout = 0, bool verbose = false) 
{
    options_.model_version_ = modelVersion;
    options_.client_timeout_ = client_timeout;

    // Create a InferenceServerGrpcClient instance to communicate with the
    // server using gRPC protocol.
    std::unique_ptr<triton::client::InferenceServerGrpcClient> tClient;
    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(&tClient, url, verbose),
        "unable to create grpc client");
    m_Client_ = std::move(tClient);

    return true;
}

bool ExaTrkXTriton::PrepareInput(
    const std::string& inputName, const std::vector<int64_t>& inputShape,
    std::vector<float>& inputValues)
{
    tc::InferInput* input0;
    FAIL_IF_ERR(
        tc::InferInput::Create(&input0, inputName, inputShape, "FP32"), 
        "unable to get"+inputName);

    std::shared_ptr<tc::InferInput> input0_ptr;
    input0_ptr.reset(input0);
    FAIL_IF_ERR(input0_ptr->AppendRaw(
        reinterpret_cast<uint8_t*>(&inputValues[0]), // why uint8?
        inputValues.size() * sizeof(float)), "unable to set data for INPUT0");

    inputs_.push_back(input0_ptr.get());

    return true;
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
        &results, options_, inputs_, outputs, http_headers,
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
    inputs_.clear();
}