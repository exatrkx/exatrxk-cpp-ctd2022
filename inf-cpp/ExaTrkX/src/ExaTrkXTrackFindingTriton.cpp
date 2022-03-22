#include "ExaTrkXTrackFindingTriton.hpp"

#include <torch/torch.h>
#include <torch/script.h>
using namespace torch::indexing;

#include "mmio_read.h"
#include "build_edges.hpp"

ExaTrkXTrackFindingTriton::ExaTrkXTrackFindingTriton(
    const ExaTrkXTrackFindingTriton::Config& config): m_cfg(config)
{
    initTrainedModels();
}

void ExaTrkXTrackFindingTriton::initTrainedModels(){
    // start the communication with the Server?
}


// The main function that runs the Exa.TrkX pipeline
// Be care of sharpe corners.
void ExaTrkXTrackFindingTriton::getTracks(
    std::vector<float>& inputValues,
    std::vector<int>& spacepointIDs,
    std::vector<std::vector<int> >& trackCandidates)
{


}