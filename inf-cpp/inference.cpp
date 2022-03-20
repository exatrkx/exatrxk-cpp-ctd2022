#include <istream>
#include <fstream>
#include <iostream>
#include <sstream>

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <assert.h>
#include <chrono>
#include <algorithm>

#include <torch/torch.h>
#include <torch/script.h>
using namespace torch::indexing;

#include <grid.h>
#include <insert_points.h>
#include <counting_sort.h>
#include <prefix_sum.h>
#include <find_nbrs.h>

#include "cuda.h"
#include "cuda_runtime_api.h"
#include "mmio_read.h"

void saveCsv(torch::Tensor t, const std::string& filepath, const std::string& separator = ",")
{
    t = t.flatten(1).contiguous().cpu();
    float* ptr = t.data_ptr<float>();

    std::ofstream csvHandle(filepath);

    for (size_t i = 0; i < t.sizes()[0]; ++i)
    {
        for (size_t k = 0; k < t.sizes()[1]; ++k)
        {
            csvHandle << *ptr++;

            if (k < (t.sizes()[1] - 1))
            {
                csvHandle << separator;
            }
        }

        csvHandle << "\n";
    }
}

class Infer
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


    Infer(const Config& params);
    ~Infer();

    void getTracks(
        std::vector<float>& input_values,
        std::vector<int>& spacepoint_ids,
        std::vector<std::vector<int> >& track_candidates);

private:
    void initTrainedModels();

    torch::Tensor buildEdges(at::Tensor& embedFeatures, int64_t numSpacepoints)
    {
        torch::Device device(torch::kCUDA);
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);

        int grid_params_size;
        int grid_delta_idx;
        int grid_total_idx;
        int grid_max_res;
        int grid_dim;
        int dim = m_cfg.embeddingDim;
        if (dim >= 3) {
            grid_params_size = 8;
            grid_delta_idx = 3;
            grid_total_idx = 7;
            grid_max_res = 128;
            grid_dim = 3;
        } else {
            throw std::runtime_error("DIM < 3 is not supported for now.\n");
        }


        float cell_size;
        float radius_cell_ratio = 2.0;
        int G = -1;
        int batch_size = 1;
        float rVal = m_cfg.rVal; // radius of nearest neighours
        int kVal = m_cfg.knnVal;  // maximum number of nearest neighbours.
        
        // Set up grid properties
        torch::Tensor grid_min;
        torch::Tensor grid_max;
        torch::Tensor grid_size;

        torch::Tensor embedTensor = embedFeatures.reshape({1, numSpacepoints, m_cfg.embeddingDim});
        torch::Tensor gridParamsCuda = torch::zeros({batch_size, grid_params_size}, device).to(torch::kFloat32);
        torch::Tensor r_tensor = torch::full({batch_size}, rVal, device);
        torch::Tensor lengths = torch::full({batch_size}, numSpacepoints, device);
        
        
        // build the grid
        for(int i=0; i < batch_size; i++) {
            torch::Tensor allPoints = embedTensor.index({i, Slice(None, lengths.index({i}).item().to<long>()), Slice(None, grid_dim)});
            grid_min = std::get<0>(allPoints.min(0));
            grid_max = std::get<0>(allPoints.max(0));
            gridParamsCuda.index_put_({i, Slice(None, grid_delta_idx)}, grid_min);
            
            grid_size = grid_max - grid_min;
            
            cell_size = r_tensor.index({i}).item().to<float>() / radius_cell_ratio;
            
            if (cell_size < (grid_size.min().item().to<float>() / grid_max_res)) {
                cell_size = grid_size.min().item().to<float>() / grid_max_res;
            }
            
            gridParamsCuda.index_put_({i, grid_delta_idx}, 1 / cell_size);
            
            gridParamsCuda.index_put_({i, Slice(1 + grid_delta_idx, grid_total_idx)},
                                    floor(grid_size / cell_size) + 1);
            
            gridParamsCuda.index_put_({i, grid_total_idx}, gridParamsCuda.index({i, Slice(1 + grid_delta_idx, grid_total_idx)}).prod());
            
            if (G < gridParamsCuda.index({i, grid_total_idx}).item().to<int>()) {
                G = gridParamsCuda.index({i, grid_total_idx}).item().to<int>();
            }
        }
        
        torch::Tensor pc_grid_cnt = torch::zeros({batch_size, G}, device).to(torch::kInt32);
        torch::Tensor pc_grid_cell = torch::full({batch_size, numSpacepoints}, -1, device).to(torch::kInt32);
        torch::Tensor pc_grid_idx = torch::full({batch_size, numSpacepoints}, -1, device).to(torch::kInt32);
        
        std::cout << "Inserting points" << std::endl;
        
        // put spacepoints into the grid
        InsertPointsCUDA(embedTensor, lengths.to(torch::kInt64), gridParamsCuda, 
                        pc_grid_cnt, pc_grid_cell, pc_grid_idx, G);
        
        torch::Tensor pc_grid_off = torch::full({batch_size, G}, 0, device).to(torch::kInt32);
        torch::Tensor grid_params = gridParamsCuda.to(torch::kCPU);
        
        std::cout << "Prefix Sum" << std::endl;
        
        for(int i = 0; i < batch_size ; i++) {
            PrefixSumCUDA(pc_grid_cnt.index({i}), grid_params.index({i, grid_total_idx}).item().to<int>(), pc_grid_off.index({i}));
        }
        
        torch::Tensor sorted_points = torch::zeros({batch_size, numSpacepoints, dim}, device).to(torch::kFloat32);
        torch::Tensor sorted_points_idxs = torch::full({batch_size, numSpacepoints}, -1, device).to(torch::kInt32);
        
        CountingSortCUDA(embedTensor, lengths.to(torch::kInt64), pc_grid_cell,
                        pc_grid_idx, pc_grid_off,
                        sorted_points, sorted_points_idxs);
        
        std::cout << "Counting sorted" << std::endl;
        
        // torch::Tensor K_tensor = torch::full({batch_size}, kVal, device); 
        
        std::tuple<at::Tensor, at::Tensor> nbr_output = FindNbrsCUDA(sorted_points, sorted_points,
                                                            lengths.to(torch::kInt64), lengths.to(torch::kInt64),
                                                                pc_grid_off.to(torch::kInt32),
                                                            sorted_points_idxs, sorted_points_idxs,
                                                                gridParamsCuda.to(torch::kFloat32),
                                                            kVal, r_tensor, r_tensor*r_tensor);

        std::cout << "Neigbours to Edges" << std::endl;                             
        torch::Tensor positiveIndices = std::get<0>(nbr_output) >= 0;

        torch::Tensor repeatRange = torch::arange(positiveIndices.size(1), device).repeat({1, positiveIndices.size(2), 1}).transpose(1,2);
        
        torch::Tensor stackedEdges = torch::stack({repeatRange.index({positiveIndices}), std::get<0>(nbr_output).index({positiveIndices})});

        //  Remove self-loops:

        torch::Tensor selfLoopMask = stackedEdges.index({0}) != stackedEdges.index({1});
        stackedEdges = stackedEdges.index({Slice(), selfLoopMask});
        
        // Perform any other post-processing here. E.g. Can remove half of edge list with:
        
        torch::Tensor duplicate_mask = stackedEdges.index({0}) > stackedEdges.index({1});
        stackedEdges = stackedEdges.index({Slice(), duplicate_mask});
        
        // // And randomly flip direction with:
        // torch::Tensor random_cut_keep = torch::randint(2, {stackedEdges.size(1)});
        // torch::Tensor random_cut_flip = 1-random_cut_keep;
        // torch::Tensor keep_edges = stackedEdges.index({Slice(), random_cut_keep.to(torch::kBool)});
        // torch::Tensor flip_edges = stackedEdges.index({Slice(), random_cut_flip.to(torch::kBool)}).flip({0});
        // stackedEdges = torch::cat({keep_edges, flip_edges}, 1);
        stackedEdges = stackedEdges.toType(torch::kInt64).to(torch::kCPU);

        return stackedEdges;
        // std::cout << "copy edges to std::vector" << std::endl;
    }

public:
    Config m_cfg;
    // treat trained models as private properties
private:
    // It seems using points is a must.
    torch::jit::script::Module e_model;
    torch::jit::script::Module f_model;
    torch::jit::script::Module g_model;
};

Infer::Infer(const Infer::Config& config): m_cfg(config){
    initTrainedModels();
}

void Infer::initTrainedModels(){
    std::string l_embedModelPath(m_cfg.modelDir + "/torchscript/embed.pt");
    std::string l_filterModelPath(m_cfg.modelDir + "/torchscript/filter.pt");
    std::string l_gnnModelPath(m_cfg.modelDir + "/torchscript/gnn.pt");
    c10::InferenceMode guard;
    try {   
        e_model = torch::jit::load(l_embedModelPath.c_str());
        e_model.eval();
        f_model = torch::jit::load(l_filterModelPath.c_str());
        f_model.eval();
        g_model = torch::jit::load(l_gnnModelPath.c_str());  
        g_model.eval();
    } catch (const c10::Error& e) {
        throw std::invalid_argument("Failed to load models: " + e.msg()); 
    }
}

Infer::~Infer(){
}

// The main function that runs the Exa.TrkX inference pipeline
// Be care of sharpe corners.

void Infer::getTracks(std::vector<float>& inputValues, std::vector<int>& spacepointIDs,
    std::vector<std::vector<int> >& trackCandidates)
{
    // hardcoded debugging information
    c10::InferenceMode guard(true);
    bool debug = true;
    const std::string embedding_outname = "debug_embedding_outputs.txt";
    const std::string edgelist_outname = "debug_edgelist_outputs.txt";
    const std::string filtering_outname = "debug_filtering_scores.txt";
    torch::Device device(torch::kCUDA);

     // printout the r,phi,z of the first spacepoint
    std::cout <<"First spacepoint information: ";
    std::copy(inputValues.begin(), inputValues.begin() + 3,
              std::ostream_iterator<float>(std::cout, " "));
    std::cout << std::endl;

    // ************
    // Embedding
    // ************

    int64_t numSpacepoints = inputValues.size() / m_cfg.spacepointFeatures;
    std::vector<torch::jit::IValue> eInputTensorJit;
    auto e_opts = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor eLibInputTensor = torch::from_blob(
        inputValues.data(),
        {numSpacepoints, m_cfg.spacepointFeatures},
        e_opts).to(torch::kFloat32);

    eInputTensorJit.push_back(eLibInputTensor.to(device));
    at::Tensor eOutput = e_model.forward(eInputTensorJit).toTensor();
    std::cout <<"Embedding space of libtorch the first SP: \n";
    std::cout << eOutput.slice(/*dim=*/0, /*start=*/0, /*end=*/1) << std::endl;
    std::cout << std::endl;
    saveCsv(eOutput, "lib_debug_embedding_outputs.txt");

    
    // ************
    // Building Edges
    // ************
    torch::Tensor edgeList = buildEdges(eOutput, numSpacepoints);
    int64_t numEdges = edgeList.size(1);
    std::cout << "Built " << edgeList.size(1) << " edges. " <<  edgeList.size(0) << std::endl;
    std::cout << edgeList.slice(1, 0, 5) << std::endl;

    // ************
    // Filtering
    // ************
    std::cout << "Get scores for " << numEdges<< " edges." << std::endl;
    
    std::vector<torch::jit::IValue> fInputTensorJit;
    fInputTensorJit.push_back(eLibInputTensor.to(device));
    fInputTensorJit.push_back(edgeList.to(device));
    at::Tensor fOutput = f_model.forward(fInputTensorJit).toTensor();
    std::cout << "After filtering: " << fOutput.size(0) << " " << fOutput.size(1) << std::endl;
    fOutput.squeeze_();
    fOutput.sigmoid_();

    std::cout << fOutput.slice(/*dim=*/0, /*start=*/0, /*end=*/9) << std::endl;

    torch::Tensor filterMask = fOutput > m_cfg.filterCut;
    torch::Tensor edgesAfterF = edgeList.index({Slice(), filterMask});
    edgesAfterF = edgesAfterF.to(torch::kInt64);
    int64_t numEdgesAfterF = edgesAfterF.size(1);
    std::cout << "After filtering: " << numEdgesAfterF << " edges." << std::endl;
    std::cout << edgesAfterF.slice(1, 0, 5) << std::endl;


    // ************
    // GNN
    // ************    
    // std::vector<torch::jit::IValue> gInputTensorJit;
    // auto g_opts = torch::TensorOptions().dtype(torch::kInt64);
    // gInputTensorJit.push_back(eLibInputTensor.to(device));
    // gInputTensorJit.push_back(edgesAfterF.to(device));
    // auto gOutput = g_model.forward(gInputTensorJit).toTensor()
    // gOutput.sigmoid_();

    at::Tensor gOutput = fOutput.index({filterMask});
    gOutput = gOutput.cpu();
    // torch::Tensor gOutput = torch::rand({numEdgesAfterF});

    std::cout << "GNN scores for " << gOutput.size(0) << " edges." << std::endl;
    std::cout << gOutput.slice(0, 0, 5) << std::endl;
    // ************
    // Track Labeling with cugraph::connected_components
    // ************
    using vertex_t = int32_t;
    std::vector<vertex_t> rowIndices;
    std::vector<vertex_t> colIndices;
    std::vector<float> edgeWeights;
    std::vector<vertex_t> trackLabels(numSpacepoints);
    std::copy(
        edgesAfterF.data_ptr<int64_t>(),
        edgesAfterF.data_ptr<int64_t>()+numEdgesAfterF,
        std::back_insert_iterator(rowIndices));
    std::copy(
        edgesAfterF.data_ptr<int64_t>()+numEdgesAfterF,
        edgesAfterF.data_ptr<int64_t>() + numEdgesAfterF+numEdgesAfterF,
        std::back_insert_iterator(colIndices));
    std::copy(
        gOutput.data_ptr<float>(),
        gOutput.data_ptr<float>() + numEdgesAfterF,
        std::back_insert_iterator(edgeWeights));

    // std::cout << "rows: " << rowIndices.size() << " " << rowIndices[0] << std::endl;
    // std::cout << "column: " << colIndices.size() << " " << colIndices[0] << std::endl;
    // std::cout << "weights: " << edgeWeights.size() << " " << edgeWeights[0] << std::endl;


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

void processInput(std::string file_path, std::vector<float>& input_tensor_values){
    input_tensor_values.clear();

    std::ifstream f (file_path);   /* open file */
    if (!f.is_open()) {     /* validate file open for reading */
        perror (("error while opening file " + file_path).c_str());
    }
    std::string line;                    /* string to hold each line */
    while (getline (f, line)) {         /* read each line */
        std::string val;                     /* string to hold value */
        std::vector<float> row;                /* vector for row of values */
        std::stringstream s (line);          /* stringstream to parse csv */
        while (getline (s, val, ','))   /* for each value */
            row.push_back (stof(val));  /* convert to float, add to row */
        //array.push_back (row);          /* add row to array */
        input_tensor_values.insert (input_tensor_values.end(),row.begin(),row.end());  
    }
    f.close();
}

// initialize  enviroment...one enviroment per process
// enviroment maintains thread pools and other state info
int main(int argc, char* argv[])
{
    std::cout << "Building and running a GPU inference engine for Embedding" << std::endl;
    Infer::Config config({"../datanmodels"});
    Infer infer(config);
    
    // read spacepoints table saved in csv
    std::string input_filepath = "../datanmodels/in_e1000.csv";
    std::vector<float> input_tensor_values;
    processInput(input_filepath, input_tensor_values);

    int numSpacepoints = input_tensor_values.size()/config.spacepointFeatures;
    std::cout << numSpacepoints << " spacepoints." << std::endl;

    // <TODO: add real spacepoint ids
    std::vector<int> spacepoint_ids;
    for (int i=0; i < numSpacepoints; ++i){
        spacepoint_ids.push_back(i);
    }


    std::vector<std::vector<int> > track_candidates;
    infer.getTracks(input_tensor_values, spacepoint_ids, track_candidates);
    std::cout << track_candidates.size() << " reconstructed tracks." << std::endl;
}