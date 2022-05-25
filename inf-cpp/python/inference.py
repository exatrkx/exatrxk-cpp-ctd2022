import os
from typing import Any
import time

import numpy as np
import faiss
import torch

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

device = 'cpu'

def build_edges(spatial, r_max, k_max):
    """
    Builds edges for graph based on FAISS for CPU only.
    """
    index_flat = faiss.IndexFlatL2(spatial.shape[1])
    spatial = spatial.cpu().detach().numpy()
    index_flat.add(spatial)
    D, I = index_flat.search(spatial, k_max)

    ind = torch.Tensor.repeat(
        torch.arange(I.shape[0]), (I.shape[1], 1), 1).T
    I = torch.Tensor(I)

    edge_list = torch.stack([ind[D <= r_max**2], I[D <= r_max**2]])

    # Remove self-loops
    edge_list = edge_list[:, edge_list[0] != edge_list[1]]
    
    # remove directional edges
    edge_list = edge_list[:, edge_list[0] > edge_list[1]]
    return edge_list


class Inference:
    def __init__(self, model_dir):
        self.e_model = torch.jit.load(os.path.join(model_dir, 'torchscript/embed.pt'), map_location=device)
        self.f_model = torch.jit.load(os.path.join(model_dir, 'torchscript/filter.pt'), map_location=device)
        self.g_model = torch.jit.load(os.path.join(model_dir, 'torchscript/gnn.pt'), map_location=device)

        self.filter_cut = 0.21
        self.gnn_cut = 0.75
        self.r_val = 1.6
        self.knn_val = 500
        self.time_list = []
    
    def __call__(self, filename, *args: Any, **kwds: Any) -> Any:
        with open(filename, 'rb') as f:
            data = np.loadtxt(f, delimiter=',').astype(np.single)
        
        start_time = time.time()
        data = torch.from_numpy(data).to(device)
        embed_output = self.e_model.forward(data)

        edge_list = build_edges(embed_output, self.r_val, self.knn_val)
        edge_list = edge_list.to(torch.int64)


        filter_output = self.f_model.forward(data, edge_list)
        filter_output = filter_output.squeeze().sigmoid()

        edges_after_filter = edge_list[:, filter_output > self.filter_cut]


        g_output = self.g_model.forward(data, edges_after_filter)
        g_output = g_output.sigmoid()

        edges = edges_after_filter[:, g_output > self.gnn_cut]
        graph = csr_matrix(
            (np.ones(edges.shape[1]), (edges[0], edges[1])), shape=(data.shape[0], data.shape[0]))

        n_components, labels = connected_components(graph, directed=False, return_labels=True)
        self.time_list.append(time.time() - start_time)

        return labels


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Inference')
    add_arg = parser.add_argument
    add_arg('-d', '--data', help='input data', default='../datanmodels/in_e1000.csv')
    add_arg('-m', '--model', help='model directory', default='../datanmodels')
    
    args = parser.parse_args()
    
    infer = Inference(args.model)
    data = args.data
    if os.path.isdir(data):
        for f in os.listdir(data):
            infer(os.path.join(data, f))
    else:
        infer(data)

    print("Average time: {:.4f} for {:} events".format(
        np.sum(infer.time_list)/ len(infer.time_list),
        len(infer.time_list)))