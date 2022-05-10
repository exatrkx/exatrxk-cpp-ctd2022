#include "ExaTrkXTriton.hpp"

namespace tc = triton::client;

ExaTrkXTriton::ExaTrkXTriton(std::string modelName)
: options_(modelName)
{};

ExaTrkXTriton::~ExaTrkXTriton() {};

bool ExaTrkXTriton::InitClient(std::string url, std::string modelVersion = "", uint32_t client_timeout = 0, bool verbose = false) 
{
    options_.model_version_ = model_version;
    options_.client_timeout_ = client_timeout;

    // Create a InferenceServerGrpcClient instance to communicate with the
    // server using gRPC protocol.
    std::unique_ptr<triton::client::InferenceServerGrpcClient> tClient;
    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(&tClient, url, verbose),
        "unable to create grpc client");
    m_Client = std::move(embedClient);

    return 1
}

bool ExaTrkXTriton::PrepareInput(std::string inputName, std::vector<int64_t> inputShape, std::vector<float>& inputValues)
{
    tc::InferInput* input0;
    FAIL_IF_ERR(
        tc::InferInput::Create(&input0, inputName, inputShape, "FP32"), 
        "unable to get INPUT0");

    std::shared_ptr<tc::InferInput> input0_ptr;
    input0_ptr.reset(input0);
    FAIL_IF_ERR(input0_ptr->AppendRaw(
        reinterpret_cast<uint8_t*>(&inputValues[0]), // why uint8?
        inputValues.size() * sizeof(float)), "unable to set data for INPUT0");

    inputs_.push_back(nput0_ptr.get());

    return 1;
}

bool ExaTrkXTriton::GetOutput(std::string outputName, float* output_data, size_t& output_size)
{
    tc::InferResult* results;
    std::vector<const tc::InferRequestedOutput*> outputs = {}; //output0_ptr.get()
    tc::Headers http_headers;
    grpc_compression_algorithm compression_algorithm =
        grpc_compression_algorithm::GRPC_COMPRESS_NONE;

    FAIL_IF_ERR(m_Client_->Infer(
        &results, options_, inputs_, outputs, http_headers,
        compression_algorithm), "unable to run Embedding");

    std::shared_ptr<tc::InferResult> results_ptr;
    results_ptr.reset(results);

    results_ptr->RawData(
          outputName, (const uint8_t**)&output_data, &output_size);

}

