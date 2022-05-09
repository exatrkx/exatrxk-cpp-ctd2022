#include "ExaTrkXTrackFindingTriton.hpp"
#include "ExaTrkXTriton.hpp"

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
    const ExaTrkXTrackFindingTriton::Config& config, std::string modelName = "embed", std::string url = "localhost:8021"): 
    m_cfg(config),
    triton_(modelName)
{
    triton_.InitClient(url);
}

ExaTrkXTrackFindingTriton::~ExaTrkXTrackFindingTriton() {

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

    triton_.PrepareInput("INPUT_0", embedInputShape, inputValues);

    float* output0_data;
    size_t output0_byte_size;
    triton_.GetOutput("OUTPUT__0", float* output_data, size_t& output_size);

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
