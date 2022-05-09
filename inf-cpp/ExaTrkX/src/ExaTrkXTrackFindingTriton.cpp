#include "ExaTrkXTrackFindingTriton.hpp"


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
    const ExaTrkXTrackFindingTriton::Config& config): m_cfg(config)
{
    bool verbose = false;
    uint32_t client_timeout = 0;
    std::string model_version = "";
    e_client_ = std::make_unique<ExaTrkXTriton>(m_cfg.embedModelName, m_cfg.url, model_version, client_timeout, verbose);
    f_client_ = std::make_unique<ExaTrkXTriton>(m_cfg.filterModelName, m_cfg.url, model_version, client_timeout, verbose);
    g_client_ = std::make_unique<ExaTrkXTriton>(m_cfg.gnnModelName, m_cfg.url, model_version, client_timeout, verbose);
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

    e_client_->ClearInput();
    std::cout <<"prepare inputs" << std::endl;
    e_client_->PrepareInput<float>("INPUT__0", embedInputShape, inputValues);
    std::cout <<"prepare inference" << std::endl;
    std::vector<float> eOutputData;
    std::vector<int64_t> embedOutputShape{numSpacepoints, m_cfg.embeddingDim};
    e_client_->GetOutput("OUTPUT__0", eOutputData, embedOutputShape);

    // e_client_->Inference("INPUT__0", embedInputShape, inputValues, "FP32",
    //     "OUTPUT__0", eOutputData, embedOutputShape);

    std::cout <<"Embedding space of the first SP: ";
    std::copy(eOutputData.begin(), eOutputData.begin() + m_cfg.embeddingDim,
              std::ostream_iterator<float>(std::cout, " "));

    // ************
    // Building Edges
    // ************

    std::vector<int64_t> edgeList;
    buildEdges(
      eOutputData, edgeList, 
      numSpacepoints, m_cfg.embeddingDim, m_cfg.rVal, m_cfg.knnVal);
    int64_t numEdges = edgeList.size() / 2;
    std::cout << "Built " << numEdges<< " edges." << std::endl;


    std::copy(edgeList.begin(), edgeList.begin() + 10,
              std::ostream_iterator<int64_t>(std::cout, " "));
    std::cout << std::endl;
    std::copy(edgeList.begin()+numEdges, edgeList.begin()+numEdges+10,
              std::ostream_iterator<int64_t>(std::cout, " "));
    std::cout << std::endl;    


    // ************
    // Filtering
    // ************
    f_client_->ClearInput();
    /// <TODO: reuse the embedding inputs?>
    f_client_->PrepareInput<float>("f_nodes", embedInputShape, inputValues);
    std::vector<int64_t> fEdgeShape{2, numEdges};
    f_client_->PrepareInput<int64_t>("f_edges", fEdgeShape, edgeList);

    std::vector<float> fOutputData;
    std::vector<int64_t> fOutputShape{numEdges, 1};
    f_client_->GetOutput("f_edge_score", fOutputData, fOutputShape);

    // However, I have to convert those numbers to a score by applying sigmoid!
    // Use torch::tensor
    torch::Tensor edgeListCTen = torch::tensor(edgeList, {torch::kInt64});
    edgeListCTen = edgeListCTen.reshape({2, numEdges});

    torch::Tensor fOutputCTen = torch::tensor(fOutputData, {torch::kFloat32});
    fOutputCTen = fOutputCTen.sigmoid();

    torch::Tensor filterMask = fOutputCTen > m_cfg.filterCut;
    torch::Tensor edgesAfterFCTen = edgeListCTen.index({Slice(), filterMask});

    std::vector<int64_t> edgesAfterFiltering;
    std::copy(
        edgesAfterFCTen.data_ptr<int64_t>(),
        edgesAfterFCTen.data_ptr<int64_t>() + edgesAfterFCTen.numel(),
        std::back_inserter(edgesAfterFiltering));

    int64_t numEdgesAfterF = edgesAfterFiltering.size() / 2;
    std::cout << "After filtering: " << numEdgesAfterF << " edges." << std::endl;

    // ************
    // GNN
    // ************

    g_client_->ClearInput();
    g_client_->PrepareInput<float>("g_nodes", embedInputShape, inputValues);
    std::vector<int64_t> gEdgeShape{2, numEdgesAfterF};
    g_client_->PrepareInput<int64_t>("g_edges", gEdgeShape, edgesAfterFiltering);

    std::vector<float> gOutputData;
    std::vector<int64_t> gOutputShape{numEdgesAfterF, 1};
    g_client_->GetOutput("gnn_edge_score", gOutputData, gOutputShape);

    torch::Tensor gOutputCTen = torch::tensor(gOutputData, {torch::kFloat32});
    gOutputCTen = gOutputCTen.sigmoid();


    // ************
    // Track Labeling with cugraph::connected_components
    // ************
    std::vector<int32_t> rowIndices;
    std::vector<int32_t> colIndices;
    std::vector<float> edgeWeights;
    std::vector<int32_t> trackLabels(numSpacepoints);
    std::copy(
        edgesAfterFiltering.begin(),
        edgesAfterFiltering.begin()+numEdgesAfterF,
        std::back_insert_iterator(rowIndices));
    std::copy(
        edgesAfterFiltering.begin()+numEdgesAfterF,
        edgesAfterFiltering.end(),
        std::back_insert_iterator(colIndices));
    std::copy(
        gOutputCTen.data_ptr<float>(),
        gOutputCTen.data_ptr<float>() + numEdgesAfterF,
        std::back_insert_iterator(edgeWeights));

    weakly_connected_components<int32_t,int32_t,float>(
        rowIndices, colIndices, edgeWeights, trackLabels);

    int idx = 0;
    std::cout << "size of components: " << trackLabels.size() << std::endl;
    if (trackLabels.size() == 0)  return;


    trackCandidates.clear();

    int existTrkIdx = 0;
    // map labeling from MCC to customized track id.
    std::map<int32_t, int32_t> trackLableToIds;

    for(int32_t idx=0; idx < numSpacepoints; ++idx) {
        int32_t trackLabel = trackLabels[idx];
        int spacepointID = spacepointIDs[idx];

        int trkId;
        if(trackLableToIds.find(trackLabel) != trackLableToIds.end()) {
            trkId = trackLableToIds[trackLabel];
            trackCandidates[trkId].push_back(spacepointID);
        } else {
            // a new track, assign the track id
            // and create a vector
            trkId = existTrkIdx;
            trackCandidates.push_back(std::vector<int>{spacepointID});
            trackLableToIds[trackLabel] = trkId;
            existTrkIdx++;
        }
    }    
}
