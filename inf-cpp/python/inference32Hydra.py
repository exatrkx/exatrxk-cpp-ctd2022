#!/usr/bin/env python
# coding: utf-8
import os
import sys
import time

import hydra
from omegaconf import DictConfig, OmegaConf

os.environ['TRKXINPUTDIR'] = '/users/PLS0129/ysu0053/trackml/train_500_events/' # better change to your copy of the dataset.
os.environ['TRKXINPUTDIR'] = '/users/PLS0129/ysu0053/trackml/train_500_events/' 
os.environ['TRKXOUTPUTDIR'] = '../run200' # change to your own directory
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np
import pandas as pd

# 3rd party
import torch
from torch_geometric.data import Data
import tensorflow as tf
import sonnet as snt
from graph_nets import utils_tf
import gc
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader

from trackml.dataset import load_event
from exatrkx.src.processing.utils.detector_utils import load_detector
from exatrkx.src.processing.utils.cell_utils import get_one_event

# local import
from exatrkx import LayerlessEmbedding
from exatrkx.src import utils_torch
from exatrkx import VanillaFilter
from exatrkx import SegmentClassifier
import faiss.contrib.torch_utils
# for labeling
import cudf, cugraph
from exatrkx.scripts.tracks_from_gnn import prepare as prepare_labeling
from exatrkx.scripts.tracks_from_gnn import clustering as dbscan_clustering

from trackml.score import _analyze_tracks
frac_reco_matched, frac_truth_matched = 0.5, 0.5 # parameters for track matching
min_hits = 5 # minimum number of hits associated with a particle to define "reconstructable particles"
from pynvml import *
from pynvml.smi import nvidia_smi
#import psutil

global device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    nvsmi = nvidia_smi.getInstance()
    #nvsmi.DeviceQuery('memory.free, memory.used')
    #os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    #print("Memory before models ",nvsmi.DeviceQuery('memory.free, memory.used'))

def load_torch_tf_models(embed_ckpt_dir, filter_ckpt_dir, gnn_ckpt_dir,
                        filter_cut, r_cut, knn_cut):
    ckpt_idx=-1

    # embedding model
    e_ckpt = torch.load(embed_ckpt_dir, map_location=device)
    e_config = e_ckpt['hyper_parameters']
    e_config['clustering'] = 'build_edges'
    e_config['knn_val'] = knn_cut #500
    e_config['r_val'] = r_cut #1.6

    #global e_model
    e_model = LayerlessEmbedding(e_config).to(device)
    e_model.load_state_dict(e_ckpt["state_dict"])
    e_model.eval()
    #print("Memory after e_models ",nvsmi.DeviceQuery('memory.free, memory.used'))
    # filtering model#
    f_ckpt = torch.load(filter_ckpt_dir, map_location=device)
    f_config = f_ckpt['hyper_parameters']
    f_config['train_split'] = [0, 0, 1]
    f_config['filter_cut'] = filter_cut

    #global f_model
    f_model = VanillaFilter(f_config).to(device)
    f_model.load_state_dict(f_ckpt['state_dict'])
    f_model.eval()
    #print("Memory after f_models ",nvsmi.DeviceQuery('memory.free, memory.used'))
    #tensorflow
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    #gnn
    optimizer = snt.optimizers.Adam(0.001)
    model = SegmentClassifier()

    output_dir = gnn_ckpt_dir
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir, max_to_keep=10)
    status = checkpoint.restore(ckpt_manager.checkpoints[ckpt_idx]).expect_partial()
    #print("Memory after gnn models ",nvsmi.DeviceQuery('memory.free, memory.used'))
    return e_model, f_model, model

def load_onnx_tf_models():
    import onnxruntime
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads=16
    sess_options.enable_profiling = False
    if device == 'cuda':
        prov = ['CUDAExecutionProvider']
    else:
        prov = ['CPUExecutionProvider']
    sess_e = onnxruntime.InferenceSession('../datanmodels/e_model.onnx', sess_options, providers=prov)
    sess_f = onnxruntime.InferenceSession('../datanmodels/f_model.onnx', sess_options,providers=prov)
    sess_g = onnxruntime.InferenceSession('ResAGNN_model.onnx', sess_options, providers=prov)
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    #gnn
    optimizer = snt.optimizers.Adam(0.001)
    model = SegmentClassifier()

    output_dir = gnn_ckpt_dir
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=output_dir, max_to_keep=10)
    status = checkpoint.restore(ckpt_manager.checkpoints[ckpt_idx]).expect_partial()
    #print("Memory after gnn models ",nvsmi.DeviceQuery('memory.free, memory.used'))
    return (sess_e, sess_f, sess_g, model)


def get_tracks(predict_tracks_df):
    trkx_groups = predict_tracks_df.groupby(['track_id'])
    all_trk_ids = np.unique(predict_tracks_df.track_id)
    n_trkxs = all_trk_ids.shape[0]
    predict_tracks = [trkx_groups.get_group(all_trk_ids[idx])['hit_id'].to_numpy().tolist() for idx in range(n_trkxs)]
    return predict_tracks

    
def gnn_track_finding(evtid, hid, x, cell_data, e_model, f_model, g_model, graph_cut, dbscan_epsilon=0.25, dbscan_minsamples=2):
    running_time = [evtid]
    gpu_time = 0
    if device == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start = time.time()
    start_cpu = time.process_time()
    if device == 'cuda':        
        starter.record() #gpu
    data = Data(
            hid=torch.from_numpy(hid),
            x=torch.from_numpy(x).float(),
            cell_data=torch.from_numpy(cell_data).float(),
            pin_memory=True).to(device)
    if device == 'cuda':
        ender.record()
        torch.cuda.synchronize()
        gpu_time = starter.elapsed_time(ender)/1000.0
    end = time.time()
    end_cpu = time.process_time()
    running_time.extend(((end - start),(end_cpu - start_cpu), gpu_time))
    #print(nvsmi.DeviceQuery('memory.free, memory.used'))

    # ### Evaluating Embedding
    # Map each hit to the embedding space, return the embeded parameters for each hit
    gpu_time = 0
    start = time.time()
    start_cpu = time.process_time()
    with torch.no_grad():
        if device == 'cuda':
            with torch.cuda.amp.autocast():
                starter.record() #gpu
                spatial = e_model(torch.cat([data.cell_data, data.x], axis=-1)) #.to(device)
                ender.record()
                torch.cuda.synchronize()
                gpu_time = starter.elapsed_time(ender)/1000.0
        else:
            spatial = e_model(torch.cat([data.cell_data, data.x], axis=-1)) #.to(device)
    end = time.time()
    end_cpu = time.process_time()
    running_time.extend(((end - start),(end_cpu - start_cpu), gpu_time))

    # ### From embeddeding space form doublets

    # `r_val = 1.7` and `knn_val = 500` are the hyperparameters to be studied.
    # 
    # * `r_val` defines the radius of the clustering method
    # * `knn_val` defines the number of maximum neighbors in the embedding space

    gpu_time = 0
    start = time.time()
    start_cpu = time.process_time()
    if device == 'cuda':
        starter.record() #gpu
    e_spatial = utils_torch.build_edges(spatial, e_model.hparams['r_val'], e_model.hparams['knn_val']) #spatial.float()
    if device == 'cuda':
        ender.record()
        torch.cuda.synchronize()
        gpu_time = starter.elapsed_time(ender)/1000.0
    end = time.time()
    end_cpu = time.process_time()
    running_time.extend(((end - start),(end_cpu - start_cpu), gpu_time))
    #print("Time for build edges")
    #print(nvsmi.DeviceQuery('memory.free, memory.used'))
    
    R_dist = torch.sqrt(data.x[:,0]**2 + data.x[:,2]**2) # distance away from origin...
    e_spatial = e_spatial[:, (R_dist[e_spatial[0]] <= R_dist[e_spatial[1]])]
    

    #print(nvsmi.DeviceQuery('memory.free, memory.used'))
    gpu_time = 0
    start = time.time()
    start_cpu = time.process_time()
    if device == 'cuda':
        starter.record()
    emb = None # embedding information was not used in the filtering stage.
    batch_size=800000
    f_loader=torch.split(e_spatial, batch_size, 1)

    output_list = []
    for batch_ndx, sample in enumerate(f_loader):
        with torch.no_grad():
            if device == 'cuda':
                with torch.cuda.amp.autocast():
                    output = f_model(torch.cat([data.cell_data, data.x], axis=-1), sample, emb).squeeze()
            else:
                output = f_model(torch.cat([data.cell_data, data.x], axis=-1).to(device), sample, emb).squeeze()
        output_list.append(output)
    output = torch.cat(output_list)
    output = torch.sigmoid(output)
    if device == 'cuda':
        ender.record()
        torch.cuda.synchronize()
        gpu_time = starter.elapsed_time(ender)/1000.0
    end = time.time()
    end_cpu = time.process_time()
    running_time.extend(((end - start),(end_cpu - start_cpu), gpu_time))

    edge_list = e_spatial[:, output > f_model.hparams['filter_cut']]
   
    # ### Form a graph
    # Now moving TensorFlow for GNN inference.

    num_processing_steps_tr = 8
    n_nodes = data.x.shape[0]
    n_edges = edge_list.shape[1]
    nodes = data.x.cpu().numpy().astype(np.float32)
    edges = np.zeros((n_edges, 1), dtype=np.float32)
    senders = edge_list[0].cpu()
    receivers = edge_list[1].cpu()

    input_datadict = {
        "n_node": n_nodes,
        "n_edge": n_edges,
        "nodes": nodes,
        "edges": edges,
        "senders": senders,
        "receivers": receivers,
        "globals": np.array([n_nodes], dtype=np.float32)
    }
    
    #with tf.device('/gpu:1')
    input_graph = utils_tf.data_dicts_to_graphs_tuple([input_datadict])
    gpu_time = 0
    start = time.time()
    start_cpu = time.process_time()
    if device == 'cuda':
        starter.record()
    outputs_gnn = g_model(input_graph, num_processing_steps_tr)
    output_graph = outputs_gnn[-1]
    if device == 'cuda':
        ender.record()
        torch.cuda.synchronize()
        gpu_time = starter.elapsed_time(ender)/1000.0        
    end = time.time()
    end_cpu = time.process_time()
    running_time.extend(((end - start),(end_cpu - start_cpu), gpu_time)) #tensorflow is already syncronized
    #print("Time to apply the GNN: %f cpu: %f" % ((end - start),(end_cpu - start_cpu)))
   
    # ### Track labeling
    gpu_time = 0
    start = time.time()
    start_cpu = time.process_time()
    if device == 'cuda':
        starter.record()
    if device == 'cuda':
        output = tf.identity(tf.squeeze(output_graph.edges)).numpy() #.cpu().numpy()
        hids = data.hid.cpu().numpy()
        cut_edges = hids[edge_list.cpu().numpy()][:,output > graph_cut]
        cut_df = cudf.DataFrame(cut_edges.T)
        G=cugraph.Graph()
        G.from_cudf_edgelist(cut_df,source=0, destination=1, edge_attr=None)
        labels = cugraph.components.connectivity.weakly_connected_components(G)
        predict_tracks_df = labels.to_pandas()
        predict_tracks_df.columns = ["track_id","hit_id"]
    else:
        input_matrix = prepare_labeling(tf.squeeze(output_graph.edges).cpu().numpy(), senders, receivers, n_nodes)
        predict_tracks_df = dbscan_clustering(data.hid.cpu(), input_matrix, dbscan_epsilon, dbscan_minsamples)
    if device == 'cuda':    
        ender.record()
        torch.cuda.synchronize()
        gpu_time = starter.elapsed_time(ender)/1000.0     
    end = time.time()
    end_cpu = time.process_time()
    running_time.extend(((end - start),(end_cpu - start_cpu), gpu_time, e_spatial.shape[1]))
    #print("Time for labelling: %f cpu: %f" % ((end - start),(end_cpu - start_cpu)))
    return predict_tracks_df, running_time

def get_track_efficiency(evtid, hits,particles,tracks):
    tracks = _analyze_tracks(hits, tracks)
    purity_rec = np.true_divide(tracks['major_nhits'], tracks['nhits'])
    purity_maj = np.true_divide(tracks['major_nhits'], tracks['major_particle_nhits'])
    good_track = (frac_reco_matched < purity_rec) & (frac_truth_matched < purity_maj)

    matched_pids = tracks[good_track].major_particle_id.values
    score = tracks['major_weight'][good_track].sum()

    n_recotable_trkx = particles.shape[0]
    n_reco_trkx = tracks.shape[0]
    n_good_recos = np.sum(good_track)
    matched_idx = particles.particle_id.isin(matched_pids).values
    accuracy = [evtid]
    accuracy.extend((n_recotable_trkx,n_reco_trkx,n_good_recos,n_good_recos/n_recotable_trkx,n_good_recos/n_reco_trkx))
    #print("Processed {} events from {}".format(evtid, utils_dir.inputdir))
    #print("Reconstructable tracks:         {}".format(n_recotable_trkx))
    #print("Reconstructed tracks:           {}".format(n_reco_trkx))
    #print("Reconstructable tracks Matched: {}".format(n_good_recos))
    #print("Tracking efficiency:            {:.4f}".format(n_good_recos/n_recotable_trkx))
    #print("Tracking purity:               {:.4f}".format(n_good_recos/n_reco_trkx))
    return accuracy
    

def run_one_event(evtid,input_dir,detector_orig, detector_proc, e_model, f_model, g_model, graph_cut):
    event_file = os.path.join(input_dir, 'event{:09}'.format(evtid))
    hits, particles, truth = load_event(event_file, parts=['hits', 'particles', 'truth'])

    r = np.sqrt(hits.x**2 + hits.y**2)
    phi = np.arctan2(hits.y, hits.x)
    hits = hits.assign(r=r, phi=phi)
    hits = hits.merge(truth, on='hit_id')
    hits = hits[hits['particle_id'] != 0]
    hits = hits.merge(particles, on='particle_id', how='left')
    hits = hits[hits.nhits >= min_hits]
    particles = particles[particles.nhits >= min_hits]

    angles = get_one_event(event_file, detector_orig, detector_proc)
    hits = hits.merge(angles, on='hit_id')

    cell_features = ['cell_count', 'cell_val', 'leta', 'lphi', 'lx', 'ly', 'lz', 'geta', 'gphi']
    feature_scale = np.array([1000, np.pi, 1000])
    hid = hits['hit_id'].to_numpy()
    x = hits[['r', 'phi', 'z']].to_numpy() / feature_scale
    cell_data = hits[cell_features].to_numpy()

    gpu_time = 0
    if device == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start = time.time()
    start_cpu = time.process_time()
    if device == 'cuda':
        starter.record()
    tracks, r_time = gnn_track_finding(evtid, hid, x, cell_data, e_model, f_model, g_model, graph_cut)
    end = time.time()
    end_cpu = time.process_time()
    if device == 'cuda':
        ender.record()
        torch.cuda.synchronize()
        gpu_time = starter.elapsed_time(ender)/1000.0     
    r_time.extend(((end - start),(end_cpu - start_cpu),gpu_time,hid.shape[0])) 
    acc = get_track_efficiency(evtid,hits,particles,tracks)
    r_time.extend(acc)   
    #print(tracks[0])
    #print(tracks[1])
    return tracks, r_time, acc
 


@hydra.main(config_path="conf", config_name="config")
def my_inference(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    #import argparse
    #parser = argparse.ArgumentParser(description="perform inference")
    #add_arg = parser.add_argument
    #add_arg("event_id", help="event id", type=int)
    #add_arg('--detector-dir', help='detector path',
    #        default='/users/PLS0129/ysu0053/trackml/detectors.csv'
    #)
    #add_arg("--input-dir", help='input directory', default='/users/PLS0129/ysu0053/trackml/train_500_events/' )
    #args = parser.parse_args()
    #my_app()
    #print(torch.cuda.device_count())
    
    #evtid = args.event_id
    #detector_orig, detector_proc = detector()
    #detector_orig, detector_proc = detector()
    detector_orig, detector_proc = load_detector(cfg.paths.detector_dir)
    e_model, f_model, g_model = load_torch_tf_models(cfg.models.embed_ckpt_dir, cfg.models.filter_ckpt_dir, cfg.models.gnn_ckpt_dir,\
                                                    cfg.cuts.filter_cut, cfg.cuts.r_cut,cfg.cuts.knn_cut)
    evtid = 1050
    print("Event:",evtid)
    tracks, r_time, acc = run_one_event(evtid,cfg.paths.input_dir,detector_orig, detector_proc, e_model, f_model, g_model, cfg.cuts.graph_cut)

    time_df = pd.DataFrame()
    acc_df = pd.DataFrame()
    col=['Event_ID', 'Loading', 'Loading CPU', 'Loading GPU', \
         'Embedding','Embedding CPU','Embedding GPU','Build Edge', 'Build Edge CPU', 'Build Edge GPU', \
         'Filtering', 'Filtering CPU', 'Filtering GPU','GNN', 'GNN CPU','GNN GPU', \
         'Labels', 'Labels CPU', 'Labels GPU','edges', 'Total', 'Total CPU', 'Total GPU','hits', 'Event_ID_2',\
         'Reconstructable track', 'Reconstructed tracks','Reconstructable tracks Matched',\
         'Tracking efficiency','Tracking purity']
    col_acc = ['Event_ID','Reconstructable track', 'Reconstructed tracks','Reconstructable tracks Matched',\
         'Tracking efficiency','Tracking purity']

    for evtid in range(1000,1500):
        print("Event:",evtid)
        
        tracks, r_time, acc = run_one_event(evtid,cfg.paths.input_dir,detector_orig, detector_proc, e_model, f_model, g_model, \
                                           cfg.cuts.graph_cut)
        time_df = time_df.append(pd.Series(r_time), ignore_index=True)
        acc_df = acc_df.append(pd.Series(acc), ignore_index=True)
        #print(tracks[0])
        #print(tracks[1])

    time_df.columns = col
    time_df['Event_ID'] = time_df['Event_ID'].astype(int) 
    time_df = time_df.set_index('Event_ID')
    time_df = time_df.drop(['Event_ID_2'], axis=1)
    time_df.to_csv('times_gpu.csv') 
    print("Overall total {:.2f} +/- {:.2f} seconds".format(time_df.Total.mean(), time_df.Total.std()))

    acc_df.columns = col_acc
    acc_df['Event_ID'] = acc_df['Event_ID'].astype(int) 
    acc_df = acc_df.set_index('Event_ID')
    print("Tracking efficiency: {:.2f} +/- {:.2f}".format(acc_df['Tracking efficiency'].mean(), acc_df['Tracking efficiency'].std()))
    print("Tracking purity: {:.2f} +/- {:.2f}".format(acc_df['Tracking purity'].mean(), acc_df['Tracking purity'].std()))
    acc_df.to_csv('accuracy_gpu.csv') 

if __name__ == "__main__":
    my_inference()