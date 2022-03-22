#pragma once

#include <string>
#include <vector>
#include <memory>

#include <torch/torch.h>
#include <torch/script.h>
using namespace torch::indexing;

class ExaTrkXTrackFinding
{
public:
    struct Config{
        std::string modelDir;

        // hyperparameters in the pipeline.
        int64_t spacepointFeatures = 3;
        int embeddingDim = 8;
        float rVal = 1.6;
        int knnVal = 500;
        float filterCut = 0.21;
    };


    ExaTrkXTrackFinding(const Config& config);
    virtual ~ExaTrkXTrackFinding() {}

    void getTracks(
        std::vector<float>& inputValues,
        std::vector<int>& spacepointIDs,
        std::vector<std::vector<int> >& trackCandidates);

    const Config& config() const { return m_cfg; }

private:
    void initTrainedModels();

private:
    Config m_cfg;
    torch::jit::script::Module e_model;
    torch::jit::script::Module f_model;
    torch::jit::script::Module g_model;
};