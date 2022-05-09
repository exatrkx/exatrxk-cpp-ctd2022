#pragma once

#include "grpc_client.h"
#include "grpc_service.pb.h"

#include <string>
#include <vector>

namespace triton {
    namespace client {
        class InferenceServerGrpcClient;
    }
}

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
    ExaTrkXTriton(std::string modelName);
    ~ExaTrkXTriton() {};

    bool InitClient(
      std::string url, std::string modelVersion = "",
      uint32_t client_timeout = 0, bool verbose = false);

    // currently works only for 1 input; need to extend this to multiple inputs
    bool PrepareInput(
      const std::string& inputName, const std::vector<int64_t>& inputShape,
      std::vector<float>& inputValues);

    void ClearInput();

    bool GetOutput(const std::string& outputName,
      std::vector<float>& outputData, const std::vector<int64_t>&  outputShape);


  private:
    std::unique_ptr<triton::client::InferenceServerGrpcClient> m_Client_;
    std::vector<triton::client::InferInput*> inputs_;
    triton::client::InferOptions options_;
};
