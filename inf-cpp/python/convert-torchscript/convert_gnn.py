import torch

from trained_models_v0.GNN.model.Models.agnn import ResAGNN

device='cuda:0'
ckpt_fname = 'trained_models_v0/GNN/trained/checkpoints/last.ckpt'
checkpoint = torch.load(ckpt_fname, map_location=device)
hparams = checkpoint["hyper_parameters"]
state_dict = checkpoint["state_dict"]

g_model = ResAGNN(hparams).to(device)
g_model.load_state_dict(state_dict)
g_model.eval()

num_spacepoints = 100
spacepoint_features = 3
num_edges = 2000

x = torch.rand(num_spacepoints, spacepoint_features).to(torch.float32)
edge_list = torch.randint(0, 100, (2, num_edges)).to(torch.int64)
input_data = (x, edge_list)
g_script = g_model.to_torchscript(file_path="gnn_script.pt",example_inputs=input_data)

with torch.jit.optimized_execution(True):
    g_script = g_model.to_torchscript()

torch.jit.save(g_script, 'gnn.pt')

