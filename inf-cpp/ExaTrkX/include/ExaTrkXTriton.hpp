#pragma once

#include "grpc_client.h"
#include "grpc_service.pb.h"

#include <string>
#include <vector>
#include <memory>

namespace tc = triton::client;


#define FAIL_IF_ERR(X, MSG)                                        \
{                                                                \
    tc::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
}

class ExaTrkXTriton {
  public:
    ExaTrkXTriton(const std::string& modelName,
    const std::string& url,
    const std::string& modelVersion="",
    uint32_t client_timeout=0, bool verbose=false);


    ExaTrkXTriton() = delete;
    ExaTrkXTriton(const ExaTrkXTriton&) = delete;
    ExaTrkXTriton& operator=(const ExaTrkXTriton&) = delete;
    ~ExaTrkXTriton() {};

    // currently works only for 1 input; need to extend this to multiple inputs
    template <typename T>
    bool PrepareInput(
      const std::string& inputName, const std::vector<int64_t>& inputShape,
      std::vector<T>& inputValues)
    {
      tc::InferInput* input0;
      FAIL_IF_ERR(
          tc::InferInput::Create(&input0, inputName, inputShape, "FP32"), 
          "unable to get"+inputName);

      std::shared_ptr<tc::InferInput> input0_ptr;
      input0_ptr.reset(input0);
      FAIL_IF_ERR(input0_ptr->AppendRaw(
          reinterpret_cast<uint8_t*>(&inputValues[0]), // why uint8?
          inputValues.size() * sizeof(T)), "unable to set data"+inputName);

      inputs_.push_back(input0_ptr.get());

      return true;
    }

    void ClearInput();

    bool GetOutput(const std::string& outputName,
      std::vector<float>& outputData, const std::vector<int64_t>&  outputShape);


  private:
    std::unique_ptr<tc::InferenceServerGrpcClient> m_Client_;
    std::vector<tc::InferInput*> inputs_;
    std::unique_ptr<tc::InferOptions> options_;
};
