#general imports
import multiprocessing.shared_memory
from typing import List, Optional, Tuple, Union, Dict
import os
import os.path as osp
import sys
import time
import json #to implement saving and loading the properties of datasets without having to run the computation multiple times
import tqdm
from enum import Enum
import pickle
import zipfile

#parallelization
import multiprocessing
from multiprocessing.shared_memory import SharedMemory

#pytorch, pytorch geometric
import torch
from torch import Tensor

from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Dataset, download_url
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
import torch_geometric.utils

#numpy
import numpy as np

#scikit-learn
from sklearn import datasets

#import own functionality
import developmentHelpers as helpers
import SP_features as spf
from k_disk_sp_feature_generator import K_Disk_SP_Feature_Generator
from r_s_ring_sp_feature_generator import R_S_Ring_SP_Feature_Generator
from vertex_sp_feature_generator import Vertex_SP_Feature_Generator
from CSL_dataset import CSL_Dataset
from Proximity_dataset import ProximityDataset
import constants

# OGB
import ogb.nodeproppred as ogb_node
from ogb.nodeproppred import PygNodePropPredDataset
import ogb.graphproppred as ogb_graph
from ogb.graphproppred import PygGraphPropPredDataset

#imports to be removed later
import matplotlib.pyplot as plt


#test zone

# Only for testing purposes
def run_mutag():
    path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir, 'data', 'TU')
    mutag_path = osp.join(path, "MUTAG")
    result_mmap_path = 'results.np'
    editmask_mmap_path = 'editmask.np'
    dataset_mutag = TUDataset(root=path, name="MUTAG", use_node_attr=True)
    filename = "sp_properties.json"
    mutag_properties_path = osp.join(mutag_path, filename)

    k = 3
    r = 3
    s = 5

    dataset_write_filename_r_s_ring = f"{r}_{s}_ring_SP_features_MUTAG.svmlight"
    dataset_write_filename_k_disk = f"{k}_disk_SP_features_MUTAG.svmlight"

    sp_gen = K_Disk_SP_Feature_Generator(dataset = dataset_mutag, k = k, node_pred = False, samples = None, dataset_write_path = mutag_path, dataset_write_filename = dataset_write_filename_k_disk, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = mutag_properties_path, write_properties_root_path = mutag_path, write_properties_filename = filename)
    #sp_gen = R_S_Ring_SP_Feature_Generator(dataset = dataset_mutag, r = r, s = s, dataset_write_path = mutag_path, dataset_write_filename = dataset_write_filename_r_s_ring, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = mutag_properties_path, write_properties_root_path = mutag_path, write_properties_filename = filename)

    print('Multi process performance: ')
    ts_multi = time.time_ns()
    sp_gen.generate_features(num_processes = 8, comment = None, log_times=True, dump_times = True, time_summary_path = mutag_path)
    time_multi = (time.time_ns() - ts_multi) / 1_000_000
    print('Multi threaded time: ' + str(time_multi))

def run_products():

    # Testing the OGB data loader compatibility
    path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir, 'data', 'OGB')
    products_path = osp.join(path, "PRODUCTS")
    result_mmap_path = 'results.np'
    editmask_mmap_path = 'editmask.np'
    dataset_products = PygNodePropPredDataset(name = 'ogbn-products', root = products_path)
    filename = "sp_properties.json"
    products_properties_path = None #osp.join(products_path, filename)

    # Necessary to ensure proper function since x is set to None by default
    dataset_products._data.x = torch.zeros((dataset_products[0].num_nodes, 1), dtype = torch.long)

    print(dataset_products[0])

    k = 3
    r = 3
    s = 4
    dataset_write_filename_r_s_ring = f"{r}_{s}_ring_SP_features_PRODUCTS.svmlight"
    dataset_write_filename_k_disk = f"{k}_disk_SP_features_PRODUCTS.svmlight"

    #sp_gen = K_Disk_SP_Feature_Generator(dataset = dataset_products, k = k, dataset_write_path = products_path, dataset_write_filename = dataset_write_filename_k_disk, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = products_properties_path, write_properties_root_path = products_path, write_properties_filename = filename)
    sp_gen = R_S_Ring_SP_Feature_Generator(dataset = dataset_products, r = r, s = s, dataset_write_path = products_path, dataset_write_filename = dataset_write_filename_r_s_ring, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = products_properties_path, write_properties_root_path = products_path, write_properties_filename = filename)

    print('Multi process performance: ')
    ts_multi = time.time_ns()
    sp_gen.generate_features(num_processes = 1, comment = None, log_times=True, dump_times = False, time_summary_path = products_path)
    time_multi = (time.time_ns() - ts_multi) / 1_000_000
    print('Multi threaded time: ' + str(time_multi))

def run_arxiv():
    # Testing the OGB data loader compatibility
    path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir, 'data', 'OGB')
    arxiv_path = osp.join(path, "ARXIV")
    result_mmap_path = 'results.np'
    editmask_mmap_path = 'editmask.np'
    dataset_arxiv = PygNodePropPredDataset(name = 'ogbn-arxiv', root = arxiv_path)
    filename = "sp_properties.json"
    arxiv_properties_path = None #osp.join(products_path, filename)

    # Necessary to ensure proper function since x is set to None by default
    dataset_arxiv._data.x = torch.zeros((dataset_arxiv[0].num_nodes, 1), dtype = torch.long)

    print(dataset_arxiv[0])

    k = 6
    r = 3
    s = 4
    dataset_write_filename_r_s_ring = f"{r}_{s}_ring_SP_features_ARXIV.svmlight"
    dataset_write_filename_k_disk = f"{k}_disk_SP_features_ARXIV.svmlight"

    #sp_gen = K_Disk_SP_Feature_Generator(dataset = dataset_arxiv, k = k, dataset_write_path = arxiv_path, dataset_write_filename = dataset_write_filename_k_disk, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = arxiv_properties_path, write_properties_root_path = arxiv_path, write_properties_filename = filename)
    sp_gen = R_S_Ring_SP_Feature_Generator(dataset = dataset_arxiv, r = r, s = s, dataset_write_path = arxiv_path, dataset_write_filename = dataset_write_filename_r_s_ring, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = arxiv_properties_path, write_properties_root_path = arxiv_path, write_properties_filename = filename)

    print('Multi process performance: ')
    ts_multi = time.time_ns()
    sp_gen.generate_features(num_processes = 1, comment = None, log_times=True, dump_times = False, time_summary_path = arxiv_path)
    time_multi = (time.time_ns() - ts_multi) / 1_000_000
    print('Multi threaded time: ' + str(time_multi))

def run_molhiv():
    # Testing the OGB data loader compatibility
    path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir, 'data', 'OGB')
    molhiv_path = osp.join(path, "MOL_HIV")
    result_mmap_path = 'results.np'
    editmask_mmap_path = 'editmask.np'
    dataset_molhiv = PygGraphPropPredDataset(name = 'ogbg-molhiv', root = molhiv_path)
    filename = "sp_properties.json"
    molhiv_properties_path = osp.join(molhiv_path, filename)
    k = 6

    split_idx = dataset_molhiv.get_idx_split()

    #remove vertex features
    dataset_molhiv._data.x = torch.zeros((dataset_molhiv._data.x.size()[0], 1), dtype = torch.long)
    
    k = 6
    r = 3
    s = 6
    dataset_write_filename_r_s_ring = f"{r}_{s}_ring_SP_features_MOLHIV.svmlight"
    dataset_write_filename_k_disk = f"{k}_disk_SP_features_MOLHIV.svmlight"
    dataset_write_filename_vertex = f"Vertex_SP_features_MOLHIV.svmlight"

    # split_idx["train"]

    # sp_gen = R_S_Ring_SP_Feature_Generator(dataset = dataset_molhiv, r = r, s = s, node_pred = False, samples = None, dataset_write_path = molhiv_path, dataset_write_filename = dataset_write_filename_r_s_ring, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = molhiv_properties_path, write_properties_root_path = molhiv_path, write_properties_filename = filename)
    # sp_gen = K_Disk_SP_Feature_Generator(dataset = dataset_molhiv, k = k, node_pred = False, samples = None, dataset_write_path = molhiv_path, dataset_write_filename = dataset_write_filename_k_disk, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = molhiv_properties_path, write_properties_root_path = molhiv_path, write_properties_filename = filename)
    sp_gen = Vertex_SP_Feature_Generator(dataset = dataset_molhiv, node_pred = False, samples = None, dataset_write_path = molhiv_path, dataset_write_filename = dataset_write_filename_vertex, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = molhiv_properties_path, write_properties_root_path = molhiv_path, write_properties_filename = filename)

    # Debugging purposes
    # graph_id, vertex_id = sp_gen.get_vertex_identifier_from_dataset_idx(83879)
    # graph = dataset_molhiv.get(graph_id)
    # graph = dataset_molhiv.get(0)
    # print(f"Number of vertices in the given graph: {graph.num_nodes}")
    # helpers.drawGraph(graph = graph)
    # graph = dataset_molhiv.get(1)
    # print(f"Number of vertices in the given graph: {graph.num_nodes}")
    # helpers.drawGraph(graph = graph, figure_count=2)
    # graph = dataset_molhiv.get(3)
    # print(f"Number of vertices in the given graph: {graph.num_nodes}")
    # helpers.drawGraph(graph = graph, figure_count=3)
    # plt.show()

    # vector_buffer_size = 16_384
    # vector_buffer_size = 4096

    # chunksize = 128

    graph_mode = True

    print('Multi process performance: ')
    ts_multi = time.time_ns()
    # sp_gen.generate_features(num_processes = 8, chunksize = 512, vector_buffer_size = 16_384, comment = None, log_times = False, dump_times = False, time_summary_path = molhiv_path)
    sp_gen.generate_features(num_processes = 8, chunksize = 512, vector_buffer_size = 16_384, comment = None, log_times = False, dump_times = False, time_summary_path = molhiv_path, graph_mode = graph_mode)
    time_multi = (time.time_ns() - ts_multi) / 1_000_000
    print('Multi threaded time: ' + str(time_multi))

def run_exp():
    raise NotImplementedError

def run_csl():

    path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir, 'data', 'CSL')
    csl_path = osp.join(path, 'CSL_dataset')
    result_mmap_path = 'results.np'
    editmask_mmap_path = 'editmask.np'
    dataset_csl = CSL_Dataset(root = csl_path)

    filename = "properties.json"
    csl_properties_path = None # osp.join(csl_path, filename)
    k = 6

    # split_idx = dataset_csl.get_idx_split()
    
    k = 6
    r = 3
    s = 6
    dataset_write_filename_r_s_ring = f"{r}_{s}_ring_SP_features_CSL.svmlight"
    dataset_write_filename_k_disk = f"{k}_disk_SP_features_CSL.svmlight"
    dataset_write_filename_vertex = f"Vertex_SP_features_CSL.svmlight"

    # sp_gen = R_S_Ring_SP_Feature_Generator(dataset = dataset_csl, r = r, s = s, node_pred = False, samples = None, dataset_write_path = csl_path, dataset_write_filename = dataset_write_filename_r_s_ring, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = csl_properties_path, write_properties_root_path = csl_path, write_properties_filename = filename)
    # sp_gen = K_Disk_SP_Feature_Generator(dataset = dataset_csl, k = k, node_pred = False, samples = None, dataset_write_path = csl_path, dataset_write_filename = dataset_write_filename_k_disk, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = csl_properties_path, write_properties_root_path = csl_path, write_properties_filename = filename)
    sp_gen = Vertex_SP_Feature_Generator(dataset = dataset_csl, node_pred = False, samples = None, dataset_write_path = csl_path, dataset_write_filename = dataset_write_filename_vertex, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = csl_properties_path, write_properties_root_path = csl_path, write_properties_filename = filename)

    # True only for the Vertex SP Features
    graph_mode = True

    # NOTE: since the dataset only holds 150 graphs, the chunksize has to be sufficiently small

    print('Multi process performance: ')
    ts_multi = time.time_ns()
    sp_gen.generate_features(num_processes = 1, chunksize = 32, vector_buffer_size = 16_384, comment = None, log_times = False, dump_times = False, time_summary_path = csl_path, graph_mode = graph_mode)
    time_multi = (time.time_ns() - ts_multi) / 1_000_000
    print('Multi threaded time: ' + str(time_multi))


# Needs to be executed to download the proximity datasets. This is done here since multiple proximity datasets are bundled.
def download_proximity(root: str):

    # Downloads and unzips the dataset
    datasets_url = 'https://zenodo.org/records/6557736/files/Proximity.zip?download=1'
    filename = 'Proximity.zip'
    download_url(datasets_url, root)

    with zipfile.ZipFile(osp.join(root, filename), 'r') as zip_ref:
        zip_ref.extractall(root)

    # Downloads the data splits
    prox_split_urls = {1: 'https://github.com/radoslav11/SP-MPNN/raw/refs/heads/main/data_splits/Prox/1-Prox_splits.json',
                       3: 'https://github.com/radoslav11/SP-MPNN/raw/refs/heads/main/data_splits/Prox/3-Prox_splits.json',
                       5: 'https://github.com/radoslav11/SP-MPNN/raw/refs/heads/main/data_splits/Prox/5-Prox_splits.json',
                       8: 'https://github.com/radoslav11/SP-MPNN/raw/refs/heads/main/data_splits/Prox/8-Prox_splits.json',
                       10: 'https://github.com/radoslav11/SP-MPNN/raw/refs/heads/main/data_splits/Prox/10-Prox_splits.json'
    }

    for h, url in prox_split_urls.items():
        download_url(url = url, folder = osp.join(root, f"{h}-Prox"))


def run_proximity():

    h = 1

    path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir, 'data', 'Proximity')
    # download_proximity(root = path)
    prox_path = osp.join(path, f"{h}-Prox")

    result_mmap_path = 'results.np'
    editmask_mmap_path = 'editmask.np'
    dataset_prox = ProximityDataset(root = prox_path, h = h)

    filename = "properties.json"
    prox_properties_path = None # osp.join(prox_path, filename)

    # split_idx = dataset_csl.get_idx_split()
    
    k = 3
    r = 3
    s = 6
    dataset_write_filename_r_s_ring = f"{r}_{s}_ring_SP_features_{h}-Prox.svmlight"
    dataset_write_filename_k_disk = f"{k}_disk_SP_features_{h}-Prox.svmlight"
    dataset_write_filename_vertex = f"Vertex_SP_features_{h}-Prox.svmlight"

    rng = np.random.default_rng(seed = constants.SEED)
    ran_samples = rng.integers(low = 0, high = 9015, size = 50)

    # sp_gen = R_S_Ring_SP_Feature_Generator(dataset = dataset_prox, r = r, s = s, node_pred = False, samples = None, dataset_write_path = prox_path, dataset_write_filename = dataset_write_filename_r_s_ring, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = prox_properties_path, write_properties_root_path = prox_path, write_properties_filename = filename)
    # sp_gen = K_Disk_SP_Feature_Generator(dataset = dataset_prox, k = k, node_pred = False, samples = None, dataset_write_path = prox_path, dataset_write_filename = dataset_write_filename_k_disk, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = prox_properties_path, write_properties_root_path = prox_path, write_properties_filename = filename)
    sp_gen = Vertex_SP_Feature_Generator(dataset = dataset_prox, node_pred = False, samples = None, dataset_write_path = prox_path, dataset_write_filename = dataset_write_filename_vertex, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = prox_properties_path, write_properties_root_path = prox_path, write_properties_filename = filename)

    # True only for the Vertex SP Features
    graph_mode = True

    log_times = False

    # NOTE: The chunksize needs to be reduced for graph mode, normal: 512(/1024), graph: 64-128. Higher values might cause memory usage of the parent process to spike, low values might cause memroy usage of the parent process to continually rise.
    # The chunksize can be estimated based on CPU usage for each process: if the parent process is at max usage, the chunksize should probably be higher since to reduce administrative overhead
    # Additionally, this might lead to constantly increasing memory usage of the parent process since it has to buffer results from the child processes (especially in graph mode). This parameter has a huge impact on performance.
    # This computation should be done on an SSD, a hard drive might not have sufficient data speeds to support the computation without a significant slow down.

    print('Multi process performance: ')
    ts_multi = time.time_ns()
    sp_gen.generate_features(num_processes = 8, chunksize = 64, vector_buffer_size = 16_384, comment = None, log_times = log_times, dump_times = log_times, time_summary_path = prox_path, graph_mode = graph_mode)
    time_multi = (time.time_ns() - ts_multi) / 1_000_000
    print('Multi threaded time: ' + str(time_multi))


if __name__ == '__main__':
    #path = osp.join(osp.abspath(osp.dirname(__file__)), "data", "SP_features")
    # path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'TU')
    # mutag_path = osp.join(path, "MUTAG")
    # dd_path = osp.join(path, "DD")
    # result_mmap_path = 'results.np'
    # editmask_mmap_path = 'editmask.np'
    # dataset_mutag = TUDataset(root=path, name="MUTAG", use_node_attr=True)
    # dataset_dd = TUDataset(root=path, name="DD", use_node_attr=True)
    # filename = "k_disk_sp_properties.json"
    # sp_gen = K_Disk_SP_Feature_Generator(dataset = dataset_mutag, k = 3, dataset_write_path = mutag_path, dataset_write_filename = "k_disk_SP_features_MUTAG.svmlight", result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = osp.join(mutag_path, filename), write_properties_root_path = mutag_path, write_properties_filename = filename)
    
    # Required for pytorch version >= 2.6.0 since torch.load weights_only default value was changed from 'False' to 'True'
    torch.serialization.add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])

    # Testing the OGB data loader compatibility
    run_proximity()

    # path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'OGB')
    # proteins_path = osp.join(path, "PROTEINS")
    # result_mmap_path = 'results.np'
    # editmask_mmap_path = 'editmask.np'
    # dataset_proteins = PygNodePropPredDataset(name = 'ogbn-proteins', root = proteins_path)
    # filename = "k_disk_sp_properties.json"
    # protein_properties_path = None #osp.join(proteins_path, filename)

    # #Necessary to ensure proper function since x is set to None by default
    # dataset_proteins._data.x = torch.zeros((dataset_proteins[0].num_nodes, 1))

    # print(dataset_proteins[0])

    # sp_gen = K_Disk_SP_Feature_Generator(dataset = dataset_proteins, k = 3, dataset_write_path = proteins_path, dataset_write_filename = "k_disk_SP_features_PROTEINS.svmlight", result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = protein_properties_path, write_properties_root_path = proteins_path, write_properties_filename = filename)

    # #sp_gen = K_Disk_SP_Feature_Generator(dataset = dataset_dd, k = 3, dataset_write_path = dd_path, dataset_write_filename = "k_disk_SP_features_DD.svmlight", result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = osp.join(dd_path, filename), write_properties_root_path = dd_path, write_properties_filename = filename)
    # #
    # #  print('Single process performance: ')
    # # ts_single = time.time_ns()
    # # sp_gen.generate_k_disk_SP_features(num_processes = 1, comment = "testcomment1")
    # # time_single = (time.time_ns() - ts_single) / 1_000
    # # print('Single threaded time: ' + str(time_single))
    # print('Multi process performance: ')
    # ts_multi = time.time_ns()
    # sp_gen.generate_features(num_processes = 1, comment = None)
    # time_multi = (time.time_ns() - ts_multi) / 1_000_000
    # print('Multi threaded time: ' + str(time_multi))
    #sp_gen.close_shared_memory()

    #sp_gen = K_Disk_SP_Feature_Generator(dataset = dataset_dd, k=3, write_properties_root_path = dd_path, write_properties_filename = filename)
    #sp_gen = K_Disk_SP_Feature_Generator("", k=3, properties_path = osp.join(path, filename), write_properties_root_path = path, write_properties_filename = filename)

    #path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'TU')
    #dataset_mutag = TUDataset(root=path, name="MUTAG", use_node_attr=True)

    #_data = dataset_mutag.get(0)

    #_vertex_select=4
    #_k=3

    #_r=2
    #_s=_k

    #_data_k_disk = gen_k_disk(_k, _data, _vertex_select)
    #_data_r_s_ring = gen_r_s_ring(r=_r, s=_s, graph_data=_data, vertex=_vertex_select)

    #helpers.drawGraph(_data, vertex_select=[_vertex_select], figure_count=1, draw_labels=True)
    #helpers.drawGraph(_data_k_disk, figure_count=2, draw_labels=True)
    #helpers.drawGraph(_data_r_s_ring, figure_count=3, draw_labels=True)

    #plt.show()

