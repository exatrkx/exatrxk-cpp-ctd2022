#pragma once 

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

torch::Tensor buildEdges(
    at::Tensor& embedFeatures, int64_t numSpacepoints,
    int dim, float rVal, int kVal
);

void buildEdges(
    std::vector<float>& embedFeatures,
    std::vector<int64_t>& edgeList,
    int64_t numSpacepoints,
    int embeddingDim,    // dimension of embedding space
    float rVal, // radius of the ball
    int kVal    // number of nearest neighbors
);