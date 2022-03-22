#pragma once

#include <string>
#include <vector>
#include <memory>

class ExaTrkXTrackFindingTriton
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


    ExaTrkXTrackFindingTriton(const Config& config);
    virtual ~ExaTrkXTrackFindingTriton() {}

    void getTracks(
        std::vector<float>& inputValues,
        std::vector<int>& spacepointIDs,
        std::vector<std::vector<int> >& trackCandidates);

    const Config& config() const { return m_cfg; }

private:
    void initTrainedModels();

private:
    Config m_cfg;
};