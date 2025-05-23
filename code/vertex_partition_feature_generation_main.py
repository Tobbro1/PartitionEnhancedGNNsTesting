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
from k_disk_lo_feature_generator import K_Disk_LO_Feature_Generator
from r_s_ring_lo_feature_generator import R_S_Ring_LO_Feature_Generator
from CSL_dataset import CSL_Dataset
from Proximity_dataset import ProximityDataset
import constants
from dataset_property_util import Dataset_Properties_Manager

# OGB
import ogb.nodeproppred as ogb_node
from ogb.nodeproppred import PygNodePropPredDataset
import ogb.graphproppred as ogb_graph
from ogb.graphproppred import PygGraphPropPredDataset

#imports to be removed later
import matplotlib.pyplot as plt


# #test zone

# # Only for testing purposes
# def run_mutag():
#     path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir, 'data', 'TU')
#     mutag_path = osp.join(path, "MUTAG")
#     result_mmap_path = 'results.np'
#     editmask_mmap_path = 'editmask.np'
#     dataset_mutag = TUDataset(root=path, name="MUTAG", use_node_attr=True)
#     filename = "sp_properties.json"
#     mutag_properties_path = osp.join(mutag_path, filename)

#     k = 3
#     r = 3
#     s = 5

#     dataset_write_filename_r_s_ring = f"{r}_{s}_ring_SP_features_MUTAG"
#     dataset_write_filename_k_disk = f"{k}_disk_SP_features_MUTAG"

#     sp_gen = K_Disk_LO_Feature_Generator(dataset = dataset_mutag, k = k, node_pred = False, samples = None, dataset_write_path = mutag_path, dataset_write_filename = dataset_write_filename_k_disk, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = mutag_properties_path, write_properties_root_path = mutag_path, write_properties_filename = filename)
#     #sp_gen = R_S_Ring_SP_Feature_Generator(dataset = dataset_mutag, r = r, s = s, dataset_write_path = mutag_path, dataset_write_filename = dataset_write_filename_r_s_ring, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = mutag_properties_path, write_properties_root_path = mutag_path, write_properties_filename = filename)

#     print('Multi process performance: ')
#     ts_multi = time.time_ns()
#     sp_gen.generate_features(num_processes = 8, comment = None, log_times=True, dump_times = True, time_summary_path = mutag_path)
#     time_multi = (time.time_ns() - ts_multi) / 1_000_000
#     print('Multi threaded time: ' + str(time_multi))

# def run_products():

#     # Testing the OGB data loader compatibility
#     path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir, 'data', 'OGB')
#     products_path = osp.join(path, "PRODUCTS")
#     result_mmap_path = 'results.np'
#     editmask_mmap_path = 'editmask.np'
#     dataset_products = PygNodePropPredDataset(name = 'ogbn-products', root = products_path)
#     filename = "sp_properties.json"
#     products_properties_path = None #osp.join(products_path, filename)

#     # Necessary to ensure proper function since x is set to None by default
#     dataset_products._data.x = torch.zeros((dataset_products[0].num_nodes, 1), dtype = torch.long)

#     print(dataset_products[0])

#     k = 3
#     r = 3
#     s = 4
#     dataset_write_filename_r_s_ring = f"{r}_{s}_ring_SP_features_PRODUCTS"
#     dataset_write_filename_k_disk = f"{k}_disk_SP_features_PRODUCTS"

#     #sp_gen = K_Disk_SP_Feature_Generator(dataset = dataset_products, k = k, dataset_write_path = products_path, dataset_write_filename = dataset_write_filename_k_disk, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = products_properties_path, write_properties_root_path = products_path, write_properties_filename = filename)
#     sp_gen = R_S_Ring_SP_Feature_Generator(dataset = dataset_products, r = r, s = s, dataset_write_path = products_path, dataset_write_filename = dataset_write_filename_r_s_ring, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = products_properties_path, write_properties_root_path = products_path, write_properties_filename = filename)

#     print('Multi process performance: ')
#     ts_multi = time.time_ns()
#     sp_gen.generate_features(num_processes = 1, comment = None, log_times=True, dump_times = False, time_summary_path = products_path)
#     time_multi = (time.time_ns() - ts_multi) / 1_000_000
#     print('Multi threaded time: ' + str(time_multi))

# def run_arxiv():
#     # Testing the OGB data loader compatibility
#     path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir, 'data', 'OGB')
#     arxiv_path = osp.join(path, "ARXIV")
#     result_mmap_path = 'results.np'
#     editmask_mmap_path = 'editmask.np'
#     dataset_arxiv = PygNodePropPredDataset(name = 'ogbn-arxiv', root = arxiv_path)
#     filename = "sp_properties.json"
#     arxiv_properties_path = None #osp.join(products_path, filename)

#     # Necessary to ensure proper function since x is set to None by default
#     dataset_arxiv._data.x = torch.zeros((dataset_arxiv[0].num_nodes, 1), dtype = torch.long)

#     print(dataset_arxiv[0])

#     k = 6
#     r = 3
#     s = 4
#     dataset_write_filename_r_s_ring = f"{r}_{s}_ring_SP_features_ARXIV"
#     dataset_write_filename_k_disk = f"{k}_disk_SP_features_ARXIV"

#     #sp_gen = K_Disk_SP_Feature_Generator(dataset = dataset_arxiv, k = k, dataset_write_path = arxiv_path, dataset_write_filename = dataset_write_filename_k_disk, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = arxiv_properties_path, write_properties_root_path = arxiv_path, write_properties_filename = filename)
#     sp_gen = R_S_Ring_SP_Feature_Generator(dataset = dataset_arxiv, r = r, s = s, dataset_write_path = arxiv_path, dataset_write_filename = dataset_write_filename_r_s_ring, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = arxiv_properties_path, write_properties_root_path = arxiv_path, write_properties_filename = filename)

#     print('Multi process performance: ')
#     ts_multi = time.time_ns()
#     sp_gen.generate_features(num_processes = 1, comment = None, log_times=True, dump_times = False, time_summary_path = arxiv_path)
#     time_multi = (time.time_ns() - ts_multi) / 1_000_000
#     print('Multi threaded time: ' + str(time_multi))

# def run_molhiv():
#     # Testing the OGB data loader compatibility
#     path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir, 'data', 'OGB')
#     molhiv_path = osp.join(path, "MOL_HIV")
#     result_mmap_path = 'results.np'
#     editmask_mmap_path = 'editmask.np'
#     dataset_molhiv = PygGraphPropPredDataset(name = 'ogbg-molhiv', root = molhiv_path)
#     filename = "sp_properties.json"
#     molhiv_properties_path = osp.join(molhiv_path, filename)
#     k = 6

#     split_idx = dataset_molhiv.get_idx_split()

#     #remove vertex features
#     dataset_molhiv._data.x = torch.zeros((dataset_molhiv._data.x.size()[0], 1), dtype = torch.long)
    
#     k = 6
#     r = 3
#     s = 6
#     dataset_write_filename_r_s_ring = f"{r}_{s}_ring_SP_features_MOLHIV"
#     dataset_write_filename_k_disk = f"{k}_disk_SP_features_MOLHIV"
#     dataset_write_filename_vertex = f"Vertex_SP_features_MOLHIV"

#     # split_idx["train"]

#     # sp_gen = R_S_Ring_SP_Feature_Generator(dataset = dataset_molhiv, r = r, s = s, node_pred = False, samples = None, dataset_write_path = molhiv_path, dataset_write_filename = dataset_write_filename_r_s_ring, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = molhiv_properties_path, write_properties_root_path = molhiv_path, write_properties_filename = filename)
#     # sp_gen = K_Disk_SP_Feature_Generator(dataset = dataset_molhiv, k = k, node_pred = False, samples = None, dataset_write_path = molhiv_path, dataset_write_filename = dataset_write_filename_k_disk, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = molhiv_properties_path, write_properties_root_path = molhiv_path, write_properties_filename = filename)
#     sp_gen = Vertex_SP_Feature_Generator(dataset = dataset_molhiv, node_pred = False, samples = None, dataset_write_path = molhiv_path, dataset_write_filename = dataset_write_filename_vertex, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = molhiv_properties_path, write_properties_root_path = molhiv_path, write_properties_filename = filename)

#     # Debugging purposes
#     # graph_id, vertex_id = sp_gen.get_vertex_identifier_from_dataset_idx(83879)
#     # graph = dataset_molhiv.get(graph_id)
#     # graph = dataset_molhiv.get(0)
#     # print(f"Number of vertices in the given graph: {graph.num_nodes}")
#     # helpers.drawGraph(graph = graph)
#     # graph = dataset_molhiv.get(1)
#     # print(f"Number of vertices in the given graph: {graph.num_nodes}")
#     # helpers.drawGraph(graph = graph, figure_count=2)
#     # graph = dataset_molhiv.get(3)
#     # print(f"Number of vertices in the given graph: {graph.num_nodes}")
#     # helpers.drawGraph(graph = graph, figure_count=3)
#     # plt.show()

#     # vector_buffer_size = 16_384
#     # vector_buffer_size = 4096

#     # chunksize = 128

#     graph_mode = True

#     print('Multi process performance: ')
#     ts_multi = time.time_ns()
#     # sp_gen.generate_features(num_processes = 8, chunksize = 512, vector_buffer_size = 16_384, comment = None, log_times = False, dump_times = False, time_summary_path = molhiv_path)
#     sp_gen.generate_features(num_processes = 8, chunksize = 512, vector_buffer_size = 16_384, comment = None, log_times = False, dump_times = False, time_summary_path = molhiv_path, graph_mode = graph_mode)
#     time_multi = (time.time_ns() - ts_multi) / 1_000_000
#     print('Multi threaded time: ' + str(time_multi))

class UnlabeledPreTranform():
    def __init__(self):
        pass

    def __call__(self, data: Data) -> Data:
        data.x = torch.zeros(size = (data.num_nodes,1), dtype = torch.long)
        return data


def run_exp():
    raise NotImplementedError

def run_tu_dataset(dataset_str: str, sp_k_vals: Optional[List[int]] = None, sp_r_vals: Optional[List[int]] = None, sp_s_vals: Optional[List[int]] = None, lo_k_vals: Optional[List[int]] = None, lo_r_vals: Optional[List[int]] = None, lo_s_vals: Optional[List[int]] = None, lo_graph_sizes_range: Optional[Tuple[int,int]] = None, lo_num_samples: Optional[Tuple[int,int]] = None, gen_vertex_sp_features: bool = False, root_path: Optional[str] = None, use_editmask: bool = False, re_gen_properties: bool = False) -> None:
    
    assert dataset_str in ["NCI1", "ENZYMES", "PROTEINS", "DD", "COLLAB"]
    
    if root_path is None:
        root_path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir)

    absolute_path_prefix = root_path

    path = osp.join('data', 'TU', dataset_str)

    # select some constant values
    num_processes = constants.num_processes
    vector_buffer_size = constants.vector_buffer_size

    result_mmap_path = osp.join(root_path, path, 'results.np')
    editmask_mmap_path = osp.join(root_path, path, 'editmask.np')

    # Perhaps distinguish between datasets for use_node_attr
    if dataset_str == 'COLLAB':
        dataset = TUDataset(name = dataset_str, root = osp.join(absolute_path_prefix, path), use_node_attr = False, pre_transform = UnlabeledPreTranform())
    else:
        dataset = TUDataset(name = dataset_str, root = osp.join(absolute_path_prefix, path), use_node_attr = False)

        # # Add labels
        # total_num_vertices = 0
        # for idx in range(dataset.len()):
        #     graph = dataset.get(idx)
        #     num_vertices = graph.size(0)
        #     total_num_vertices += num_vertices
        #     dataset.get(idx).x = torch.zeros(size = (num_vertices,), dtype = torch.long)
        # dataset._x = torch.zeros(size = (total_num_vertices,), dtype = torch.long)
        # dataset.x = torch.zeros(size = (total_num_vertices,), dtype = torch.long)
        # dataset._data_list = None

    output_path = osp.join(path, 'results')
    dataset_prop_filename = "properties.json"
    lookup_filename = "idx_lookup.pkl"
    dataset_properties_path = osp.join(output_path, dataset_prop_filename)
    lookup_path = osp.join(output_path, lookup_filename)

    if not osp.exists(dataset_properties_path) or not osp.exists(lookup_path):
        re_gen_properties = True

    if re_gen_properties:
        # Generate dataset properties and lookup files, they are later read and not generated again
        prop_manager = Dataset_Properties_Manager(properties_path = None, dataset = dataset, node_pred = False, absolute_path_prefix = absolute_path_prefix, write_properties_root_path = output_path, write_properties_filename = dataset_prop_filename)
        prop_manager.initialize_idx_lookups(lookup_path = None, samples = None, write_lookup_root_path = output_path, write_lookup_filename = lookup_filename)

    dataset_desc = f"TU-{dataset_str}"

    if gen_vertex_sp_features:
        # Generate Vertex SP features

        vertex_sp_path = osp.join(output_path, f'vertex_SP_features')
        dataset_write_filename = f"{dataset_str}_vertex_SP_features"

        metadata_filename = f'{dataset_write_filename}_metadata.json'

        chunksize = constants.graph_chunksize

        gen = Vertex_SP_Feature_Generator(dataset = dataset, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = vertex_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
        print(f"---   Generating {dataset_str} Vertex SP features   ---")
        gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = vertex_sp_path, metadata_filename = metadata_filename, graph_mode = True)
        print(f"---   Finished generating {dataset_str} Vertex SP features   ---")

    if sp_k_vals is not None and len(sp_k_vals) > 0:
        for k in sp_k_vals:
            # Generate k-disk SP feature vectors

            k_disk_sp_path = osp.join(output_path, f'{k}-disk_SP_features')
            dataset_write_filename = f"{dataset_str}_{k}-disk_SP_features"

            metadata_filename = f'{dataset_write_filename}_metadata.json'

            chunksize = constants.vertex_chunksize

            gen = K_Disk_SP_Feature_Generator(dataset = dataset, k = k, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = k_disk_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
            print(f"---   Generating {dataset_str} {k}-Disk SP features   ---")
            gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = k_disk_sp_path, metadata_filename = metadata_filename, graph_mode = False)
            print(f"---   Finished generating {dataset_str} {k}-Disk SP features   ---")

    if sp_r_vals is not None and sp_s_vals is not None and len(sp_r_vals) > 0 and len(sp_s_vals) == len(sp_r_vals):
        for idx in range(len(sp_r_vals)):
            # Generate r-s-ring SP feature vectors

            r = sp_r_vals[idx]
            s = sp_s_vals[idx]

            if s < r:
                continue

            r_s_ring_sp_path = osp.join(output_path, f'{r}-{s}-ring_SP_features')
            dataset_write_filename = f"{dataset_str}_{r}-{s}-ring_SP_features"

            metadata_filename = f'{dataset_write_filename}_metadata.json'

            chunksize = constants.vertex_chunksize

            gen = R_S_Ring_SP_Feature_Generator(dataset = dataset, r = r, s = s, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = r_s_ring_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
            print(f"---   Generating {dataset_str} {r}-{s}-Ring SP features   ---")
            gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = r_s_ring_sp_path, metadata_filename = metadata_filename, graph_mode = False)
            print(f"---   Finished generating {dataset_str} {r}-{s}-Ring SP features   ---")

    if lo_k_vals is not None and len(lo_k_vals) > 0:
        assert lo_graph_sizes_range is not None and lo_graph_sizes_range[0] > 0 and lo_graph_sizes_range[1] > 0 and lo_num_samples is not None

        for k in lo_k_vals:
            # Generate k-disk SP feature vectors
            for id in range(constants.num_lo_gens):

                k_disk_sp_path = osp.join(output_path, f'{k}-disk_lo_features')

                if constants.num_lo_gens > 1:
                    dataset_write_filename = f"{dataset_str}_{k}-disk_lo_features_{id}"
                else:
                    dataset_write_filename = f"{dataset_str}_{k}-disk_lo_features"

                metadata_filename = f'{dataset_write_filename}_metadata.json'

                chunksize = constants.vertex_chunksize

                gen = K_Disk_LO_Feature_Generator(dataset = dataset, k = k, lo_graph_sizes_range = lo_graph_sizes_range, lo_s = lo_num_samples, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = k_disk_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
                print(f"---   Generating {dataset_str} {k}-Disk Lovasz features   ---")
                gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = k_disk_sp_path, metadata_filename = metadata_filename, graph_mode = False)
                print(f"---   Finished generating {dataset_str} {k}-Disk Lovasz features   ---")
            

    if lo_r_vals is not None and lo_s_vals is not None and len(lo_r_vals) > 0 and len(lo_s_vals) == len(lo_r_vals):
        assert lo_graph_sizes_range is not None and lo_graph_sizes_range[0] > 0 and lo_graph_sizes_range[1] > 0 and lo_num_samples is not None

        for idx in range(len(lo_r_vals)):
            # Generate r-s-ring SP feature vectors

            r = lo_r_vals[idx]
            s = lo_s_vals[idx]

            if s < r:
                continue

            for id in range(constants.num_lo_gens):

                r_s_ring_sp_path = osp.join(output_path, f'{r}-{s}-ring_lo_features')

                if constants.num_lo_gens > 1:
                    dataset_write_filename = f"{dataset_str}_{r}-{s}-ring_lo_features_{id}"
                else:
                    dataset_write_filename = f"{dataset_str}_{r}-{s}-ring_lo_features"

                metadata_filename = f'{dataset_write_filename}_metadata.json'

                chunksize = constants.vertex_chunksize

                gen = R_S_Ring_LO_Feature_Generator(dataset = dataset, r = r, s = s, lo_graph_sizes_range = lo_graph_sizes_range, lo_s = lo_num_samples, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = r_s_ring_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
                print(f"---   Generating {dataset_str} {r}-{s}-Ring Lovasz features   ---")
                gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = r_s_ring_sp_path, metadata_filename = metadata_filename, graph_mode = False)
                print(f"---   Finished generating {dataset_str} {r}-{s}-Ring Lovasz features   ---")

def run_ppa(sp_k_vals: Optional[List[int]] = None, sp_r_vals: Optional[List[int]] = None, sp_s_vals: Optional[List[int]] = None, lo_k_vals: Optional[List[int]] = None, lo_r_vals: Optional[List[int]] = None, lo_s_vals: Optional[List[int]] = None, lo_graph_sizes_range: Optional[Tuple[int,int]] = None, lo_num_samples: Optional[Tuple[int,int]] = None, gen_vertex_sp_features: bool = False, root_path: Optional[str] = None, use_editmask: bool = False, re_gen_properties: bool = False) -> None:
    
    if root_path is None:
        root_path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir)

    absolute_path_prefix = root_path

    path = osp.join('data', 'OGB')

    path = osp.join(path, 'PPA')

    # select some constant values
    num_processes = constants.num_processes
    vector_buffer_size = constants.vector_buffer_size

    result_mmap_path = osp.join(root_path, path, 'results.np')
    editmask_mmap_path = osp.join(root_path, path, 'editmask.np')

    dataset_ppa = PygGraphPropPredDataset(name = 'ogbg-ppa', root = osp.join(absolute_path_prefix, path), pre_transform = UnlabeledPreTranform())

    # # Remove features
    # dataset_ppa._data.x = torch.zeros((dataset_ppa._data.x.size()[0], 1), dtype = torch.long)
    # if dataset_ppa._data_list is not None:
    #     for data in dataset_ppa._data_list:
    #         data.x = torch.zeros((data.num_nodes, 1), dtype = torch.long)

    output_path = osp.join(path, 'results')
    dataset_prop_filename = "properties.json"
    lookup_filename = "idx_lookup.pkl"
    dataset_properties_path = osp.join(output_path, dataset_prop_filename)
    lookup_path = osp.join(output_path, lookup_filename)

    if not osp.exists(dataset_properties_path) or not osp.exists(lookup_path):
        re_gen_properties = True

    if re_gen_properties:
        # Generate dataset properties and lookup files, they are later read and not generated again
        prop_manager = Dataset_Properties_Manager(properties_path = None, dataset = dataset_ppa, node_pred = False, absolute_path_prefix = absolute_path_prefix, write_properties_root_path = output_path, write_properties_filename = dataset_prop_filename)
        prop_manager.initialize_idx_lookups(lookup_path = None, samples = None, write_lookup_root_path = output_path, write_lookup_filename = lookup_filename)

    dataset_desc = "OGBG-PPA"


    if gen_vertex_sp_features:
        # Generate Vertex SP features

        vertex_sp_path = osp.join(output_path, f'vertex_SP_features')
        dataset_write_filename = f"PPA_vertex_SP_features"

        metadata_filename = f'{dataset_write_filename}_metadata.json'

        chunksize = constants.graph_chunksize

        gen = Vertex_SP_Feature_Generator(dataset = dataset_ppa, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = vertex_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
        print(f"---   Generating PPA Vertex SP features   ---")
        gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = vertex_sp_path, metadata_filename = metadata_filename, graph_mode = True)
        print(f"---   Finished generating PPA Vertex SP features   ---")

    if sp_k_vals is not None and len(sp_k_vals) > 0:
        for k in sp_k_vals:
            # Generate k-disk SP feature vectors

            k_disk_sp_path = osp.join(output_path, f'{k}-disk_SP_features')
            dataset_write_filename = f"PPA_{k}-disk_SP_features"

            metadata_filename = f'{dataset_write_filename}_metadata.json'

            chunksize = constants.vertex_chunksize

            gen = K_Disk_SP_Feature_Generator(dataset = dataset_ppa, k = k, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = k_disk_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
            print(f"---   Generating PPA {k}-Disk SP features   ---")
            gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = k_disk_sp_path, metadata_filename = metadata_filename, graph_mode = False)
            print(f"---   Finished generating PPA {k}-Disk SP features   ---")

    if sp_r_vals is not None and sp_s_vals is not None and len(sp_r_vals) > 0 and len(sp_s_vals) == len(sp_r_vals):
        for idx in range(len(sp_r_vals)):
            # Generate r-s-ring SP feature vectors

            r = sp_r_vals[idx]
            s = sp_s_vals[idx]

            if s < r:
                continue

            r_s_ring_sp_path = osp.join(output_path, f'{r}-{s}-ring_SP_features')
            dataset_write_filename = f"PPA_{r}-{s}-ring_SP_features"

            metadata_filename = f'{dataset_write_filename}_metadata.json'

            chunksize = constants.vertex_chunksize

            gen = R_S_Ring_SP_Feature_Generator(dataset = dataset_ppa, r = r, s = s, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = r_s_ring_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
            print(f"---   Generating PPA {r}-{s}-Ring SP features   ---")
            gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = r_s_ring_sp_path, metadata_filename = metadata_filename, graph_mode = False)
            print(f"---   Finished generating PPA {r}-{s}-Ring SP features   ---")

    if lo_k_vals is not None and len(lo_k_vals) > 0:
        assert lo_graph_sizes_range is not None and lo_graph_sizes_range[0] > 0 and lo_graph_sizes_range[1] > 0 and lo_num_samples is not None

        for k in lo_k_vals:
            # Generate k-disk SP feature vectors
            for id in range(constants.num_lo_gens):

                k_disk_sp_path = osp.join(output_path, f'{k}-disk_lo_features')

                if constants.num_lo_gens > 1:
                    dataset_write_filename = f"PPA_{k}-disk_lo_features_{id}"
                else:
                    dataset_write_filename = f"PPA_{k}-disk_lo_features"

                metadata_filename = f'{dataset_write_filename}_metadata.json'

                chunksize = constants.vertex_chunksize

                gen = K_Disk_LO_Feature_Generator(dataset = dataset_ppa, k = k, lo_graph_sizes_range = lo_graph_sizes_range, lo_s = lo_num_samples, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = k_disk_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
                print(f"---   Generating PPA {k}-Disk Lovasz features   ---")
                gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = k_disk_sp_path, metadata_filename = metadata_filename, graph_mode = False)
                print(f"---   Finished generating PPA {k}-Disk Lovasz features   ---")
            

    if lo_r_vals is not None and lo_s_vals is not None and len(lo_r_vals) > 0 and len(lo_s_vals) == len(lo_r_vals):
        assert lo_graph_sizes_range is not None and lo_graph_sizes_range[0] > 0 and lo_graph_sizes_range[1] > 0 and lo_num_samples is not None

        for idx in range(len(lo_r_vals)):
            # Generate r-s-ring SP feature vectors

            r = lo_r_vals[idx]
            s = lo_s_vals[idx]

            if s < r:
                continue

            for id in range(constants.num_lo_gens):

                r_s_ring_sp_path = osp.join(output_path, f'{r}-{s}-ring_lo_features')

                if constants.num_lo_gens > 1:
                    dataset_write_filename = f"PPA_{r}-{s}-ring_lo_features_{id}"
                else:
                    dataset_write_filename = f"PPA_{r}-{s}-ring_lo_features"

                metadata_filename = f'{dataset_write_filename}_metadata.json'

                chunksize = constants.vertex_chunksize

                gen = R_S_Ring_LO_Feature_Generator(dataset = dataset_ppa, r = r, s = s, lo_graph_sizes_range = lo_graph_sizes_range, lo_s = lo_num_samples, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = r_s_ring_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
                print(f"---   Generating PPA {r}-{s}-Ring Lovasz features   ---")
                gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = r_s_ring_sp_path, metadata_filename = metadata_filename, graph_mode = False)
                print(f"---   Finished generating PPA {r}-{s}-Ring Lovasz features   ---")

def run_molhiv(sp_k_vals: Optional[List[int]] = None, sp_r_vals: Optional[List[int]] = None, sp_s_vals: Optional[List[int]] = None, lo_k_vals: Optional[List[int]] = None, lo_r_vals: Optional[List[int]] = None, lo_s_vals: Optional[List[int]] = None, lo_graph_sizes_range: Optional[Tuple[int,int]] = None, lo_num_samples: Optional[Tuple[int,int]] = None, gen_vertex_sp_features: bool = False, root_path: Optional[str] = None, use_editmask: bool = False, re_gen_properties: bool = False) -> None:
    
    if root_path is None:
        root_path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir)

    absolute_path_prefix = root_path

    path = osp.join('data', 'OGB')

    # select some constant values
    num_processes = constants.num_processes
    vector_buffer_size = constants.vector_buffer_size

    result_mmap_path = osp.join(root_path, path, 'results.np')
    editmask_mmap_path = osp.join(root_path, path, 'editmask.np')

    path = osp.join(path, 'MOL_HIV')

    dataset_molhiv = PygGraphPropPredDataset(name = 'ogbg-molhiv', root = osp.join(absolute_path_prefix, path))

    # Remove features
    dataset_molhiv._data.x = torch.zeros((dataset_molhiv._data.x.size()[0], 1), dtype = torch.long)
    if dataset_molhiv._data_list is not None:
        for data in dataset_molhiv._data_list:
            data.x = torch.zeros((data.num_nodes, 1), dtype = torch.long)

    output_path = osp.join(path, 'results')
    dataset_prop_filename = "properties.json"
    lookup_filename = "idx_lookup.pkl"
    dataset_properties_path = osp.join(output_path, dataset_prop_filename)
    lookup_path = osp.join(output_path, lookup_filename)

    if not osp.exists(dataset_properties_path) or not osp.exists(lookup_path):
        re_gen_properties = True

    if re_gen_properties:
        # Generate dataset properties and lookup files, they are later read and not generated again
        # We skip diameter calculation
        prop_manager = Dataset_Properties_Manager(properties_path = None, dataset = dataset_molhiv, node_pred = False, absolute_path_prefix = absolute_path_prefix, write_properties_root_path = output_path, write_properties_filename = dataset_prop_filename)
        prop_manager.initialize_idx_lookups(lookup_path = None, samples = None, write_lookup_root_path = output_path, write_lookup_filename = lookup_filename)

    dataset_desc = "OGBG-MOL_HIV"


    if gen_vertex_sp_features:
        # Generate Vertex SP features

        vertex_sp_path = osp.join(output_path, f'vertex_SP_features')
        dataset_write_filename = f"MOLHIV_vertex_SP_features"

        metadata_filename = f'{dataset_write_filename}_metadata.json'

        chunksize = constants.graph_chunksize

        gen = Vertex_SP_Feature_Generator(dataset = dataset_molhiv, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = vertex_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
        print(f"---   Generating MOLHIV Vertex SP features   ---")
        gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = vertex_sp_path, metadata_filename = metadata_filename, graph_mode = True)
        print(f"---   Finished generating MOLHIV Vertex SP features   ---")

    if sp_k_vals is not None and len(sp_k_vals) > 0:
        for k in sp_k_vals:
            # Generate k-disk SP feature vectors

            k_disk_path = osp.join(output_path, f'{k}-disk_SP_features')
            dataset_write_filename = f"MOLHIV_{k}-disk_SP_features"

            metadata_filename = f'{dataset_write_filename}_metadata.json'

            chunksize = constants.vertex_chunksize

            gen = K_Disk_SP_Feature_Generator(dataset = dataset_molhiv, k = k, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = k_disk_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
            print(f"---   Generating MOLHIV {k}-Disk SP features   ---")
            gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = k_disk_path, metadata_filename = metadata_filename, graph_mode = False)
            print(f"---   Finished generating MOLHIV {k}-Disk SP features   ---")

    if sp_r_vals is not None and sp_s_vals is not None and len(sp_r_vals) > 0 and len(sp_s_vals) == len(sp_r_vals):
        for idx in range(len(sp_r_vals)):
            # Generate r-s-ring SP feature vectors

            r = sp_r_vals[idx]
            s = sp_s_vals[idx]

            if s < r:
                continue

            r_s_ring_path = osp.join(output_path, f'{r}-{s}-ring_SP_features')
            dataset_write_filename = f"MOLHIV_{r}-{s}-ring_SP_features"

            metadata_filename = f'{dataset_write_filename}_metadata.json'

            chunksize = constants.vertex_chunksize

            gen = R_S_Ring_SP_Feature_Generator(dataset = dataset_molhiv, r = r, s = s, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = r_s_ring_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
            print(f"---   Generating MOLHIV {r}-{s}-Ring SP features   ---")
            gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = r_s_ring_path, metadata_filename = metadata_filename, graph_mode = False)
            print(f"---   Finished generating MOLHIV {r}-{s}-Ring SP features   ---")

    if lo_k_vals is not None and len(lo_k_vals) > 0:
        assert lo_graph_sizes_range is not None and lo_graph_sizes_range[0] > 0 and lo_graph_sizes_range[1] > 0 and lo_num_samples is not None

        for k in lo_k_vals:
            # Generate k-disk SP feature vectors

            for id in range(constants.num_lo_gens):
                k_disk_path = osp.join(output_path, f'{k}-disk_lo_features')

                if constants.num_lo_gens > 1:
                    dataset_write_filename = f"MOLHIV_{k}-disk_lo_features_{id}"
                else:
                    dataset_write_filename = f"MOLHIV_{k}-disk_lo_features"

                metadata_filename = f'{dataset_write_filename}_metadata.json'

                chunksize = constants.vertex_chunksize

                gen = K_Disk_LO_Feature_Generator(dataset = dataset_molhiv, k = k, lo_graph_sizes_range = lo_graph_sizes_range, lo_s = lo_num_samples, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = k_disk_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
                print(f"---   Generating MOLHIV {k}-Disk Lovasz features   ---")
                gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = k_disk_path, metadata_filename = metadata_filename, graph_mode = False)
                print(f"---   Finished generating MOLHIV {k}-Disk Lovasz features   ---")

    if lo_r_vals is not None and lo_s_vals is not None and len(lo_r_vals) > 0 and len(lo_s_vals) == len(lo_r_vals):
        assert lo_graph_sizes_range is not None and lo_graph_sizes_range[0] > 0 and lo_graph_sizes_range[1] > 0 and lo_num_samples is not None

        for idx in range(len(lo_r_vals)):
            # Generate r-s-ring SP feature vectors

            r = lo_r_vals[idx]
            s = lo_s_vals[idx]

            if s < r:
                continue

            for id in range(constants.num_lo_gens):
                r_s_ring_path = osp.join(output_path, f'{r}-{s}-ring_lo_features')

                if constants.num_lo_gens > 1:
                    dataset_write_filename = f"MOLHIV_{r}-{s}-ring_lo_features_{id}"
                else:
                    dataset_write_filename = f"MOLHIV_{r}-{s}-ring_lo_features"

                metadata_filename = f'{dataset_write_filename}_metadata.json'

                chunksize = constants.vertex_chunksize

                gen = R_S_Ring_LO_Feature_Generator(dataset = dataset_molhiv, r = r, s = s, lo_graph_sizes_range = lo_graph_sizes_range, lo_s = lo_num_samples, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = r_s_ring_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
                print(f"---   Generating MOLHIV {r}-{s}-Ring Lovasz features   ---")
                gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = r_s_ring_path, metadata_filename = metadata_filename, graph_mode = False)
                print(f"---   Finished generating MOLHIV {r}-{s}-Ring Lovasz features   ---")


# If None is passed as root_path, the parent directory of this file is chosen as the root directory
def run_csl(sp_k_vals: Optional[List[int]] = None, sp_r_vals: Optional[List[int]] = None, sp_s_vals: Optional[List[int]] = None, lo_k_vals: Optional[List[int]] = None, lo_r_vals: Optional[List[int]] = None, lo_s_vals: Optional[List[int]] = None, lo_graph_sizes_range: Optional[Tuple[int,int]] = None, lo_num_samples: Optional[Tuple[int,int]] = None, gen_vertex_sp_features: bool = False, root_path: Optional[str] = None, use_editmask: bool = False, re_gen_properties: bool = False) -> None:

    if root_path is None:
        root_path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir)

    absolute_path_prefix = root_path

    path = osp.join('data', 'CSL')

    # select some constant values
    num_processes = constants.num_processes
    vector_buffer_size = constants.vector_buffer_size

    # Generate features for every h in h_vals
    result_mmap_path = osp.join(root_path, path, 'results.np')
    editmask_mmap_path = osp.join(root_path, path, 'editmask.np')

    path = osp.join(path, f'CSL_dataset')
    dataset_csl = CSL_Dataset(root = osp.join(absolute_path_prefix, path))

    output_path = osp.join(path, 'results')
    dataset_prop_filename = "properties.json"
    lookup_filename = "idx_lookup.pkl"
    dataset_properties_path = osp.join(output_path, dataset_prop_filename)
    lookup_path = osp.join(output_path, lookup_filename)

    if not osp.exists(dataset_properties_path) or not osp.exists(lookup_path):
        re_gen_properties = True
    
    if re_gen_properties:
        # Generate dataset properties and lookup files, they are later read and not generated again
        prop_manager = Dataset_Properties_Manager(properties_path = None, dataset = dataset_csl, node_pred = False, absolute_path_prefix = absolute_path_prefix, write_properties_root_path = output_path, write_properties_filename = dataset_prop_filename)
        prop_manager.initialize_idx_lookups(lookup_path = None, samples = None, write_lookup_root_path = output_path, write_lookup_filename = lookup_filename)

    dataset_desc = f"CSL"

    if gen_vertex_sp_features:
        # Generate Vertex SP features

        vertex_sp_path = osp.join(output_path, f'vertex_SP_features')
        dataset_write_filename = f"CSL_vertex_SP_features"

        metadata_filename = f'{dataset_write_filename}_metadata.json'

        chunksize = constants.graph_chunksize

        gen = Vertex_SP_Feature_Generator(dataset = dataset_csl, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = vertex_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
        print(f"---   Generating CSL Vertex SP features   ---")
        gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = vertex_sp_path, metadata_filename = metadata_filename, graph_mode = True)
        print(f"---   Finished generating CSL Vertex SP features   ---")

    if sp_k_vals is not None and len(sp_k_vals) > 0:
        for k in sp_k_vals:
            # Generate k-disk SP feature vectors

            k_disk_sp_path = osp.join(output_path, f'{k}-disk_SP_features')
            dataset_write_filename = f"CSL_{k}-disk_SP_features"

            metadata_filename = f'{dataset_write_filename}_metadata.json'

            chunksize = constants.vertex_chunksize

            gen = K_Disk_SP_Feature_Generator(dataset = dataset_csl, k = k, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = k_disk_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
            print(f"---   Generating CSL {k}-Disk SP features   ---")
            gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = k_disk_sp_path, metadata_filename = metadata_filename, graph_mode = False)
            print(f"---   Finished generating CSL {k}-Disk SP features   ---")

    if sp_r_vals is not None and sp_s_vals is not None and len(sp_r_vals) > 0 and len(sp_s_vals) == len(sp_r_vals):
        for idx in range(len(sp_r_vals)):
            # Generate r-s-ring SP feature vectors

            r = sp_r_vals[idx]
            s = sp_s_vals[idx]

            if s < r:
                continue

            r_s_ring_sp_path = osp.join(output_path, f'{r}-{s}-ring_SP_features')
            dataset_write_filename = f"CSL_{r}-{s}-ring_SP_features"

            metadata_filename = f'{dataset_write_filename}_metadata.json'

            chunksize = constants.vertex_chunksize

            gen = R_S_Ring_SP_Feature_Generator(dataset = dataset_csl, r = r, s = s, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = r_s_ring_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
            print(f"---   Generating CSL {r}-{s}-Ring SP features   ---")
            gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = r_s_ring_sp_path, metadata_filename = metadata_filename, graph_mode = False)
            print(f"---   Finished generating CSL {r}-{s}-Ring SP features   ---")

    if lo_k_vals is not None and len(lo_k_vals) > 0:
        assert lo_graph_sizes_range is not None and lo_graph_sizes_range[0] > 0 and lo_graph_sizes_range[1] > 0 and lo_num_samples is not None

        for k in lo_k_vals:
            # Generate k-disk SP feature vectors

            for id in range(constants.num_lo_gens):

                k_disk_sp_path = osp.join(output_path, f'{k}-disk_lo_features')

                if constants.num_lo_gens > 1:
                    dataset_write_filename = f"CSL_{k}-disk_lo_features_{id}"
                else:
                    dataset_write_filename = f"CSL_{k}-disk_lo_features"

                metadata_filename = f'{dataset_write_filename}_metadata.json'

                chunksize = constants.vertex_chunksize

                gen = K_Disk_LO_Feature_Generator(dataset = dataset_csl, k = k, lo_graph_sizes_range = lo_graph_sizes_range, lo_s = lo_num_samples, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = k_disk_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
                print(f"---   Generating CSL {k}-Disk Lovasz features   ---")
                gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = k_disk_sp_path, metadata_filename = metadata_filename, graph_mode = False)
                print(f"---   Finished generating CSL {k}-Disk Lovasz features   ---")

    if lo_r_vals is not None and lo_s_vals is not None and len(lo_r_vals) > 0 and len(lo_s_vals) == len(lo_r_vals):
        assert lo_graph_sizes_range is not None and lo_graph_sizes_range[0] > 0 and lo_graph_sizes_range[1] > 0 and lo_num_samples is not None

        for idx in range(len(lo_r_vals)):
            # Generate r-s-ring SP feature vectors

            r = lo_r_vals[idx]
            s = lo_s_vals[idx]

            if s < r:
                continue

            for id in range(constants.num_lo_gens):
                r_s_ring_sp_path = osp.join(output_path, f'{r}-{s}-ring_lo_features')

                if constants.num_lo_gens > 1:
                    dataset_write_filename = f"CSL_{r}-{s}-ring_lo_features_{id}"
                else:
                    dataset_write_filename = f"CSL_{r}-{s}-ring_lo_features"

                metadata_filename = f'{dataset_write_filename}_metadata.json'

                chunksize = constants.vertex_chunksize

                gen = R_S_Ring_LO_Feature_Generator(dataset = dataset_csl, r = r, s = s, lo_graph_sizes_range = lo_graph_sizes_range, lo_s = lo_num_samples, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = r_s_ring_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
                print(f"---   Generating CSL {r}-{s}-Ring Lovasz features   ---")
                gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = r_s_ring_sp_path, metadata_filename = metadata_filename, graph_mode = False)
                print(f"---   Finished generating CSL {r}-{s}-Ring Lovasz features   ---")


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


# If None is passed as root_path, the parent directory of this file is chosen as the root directory
def run_proximity(h_vals: List[int], sp_k_vals: Optional[List[int]] = None, sp_r_vals: Optional[List[int]] = None, sp_s_vals: Optional[List[int]] = None, lo_k_vals: Optional[List[int]] = None, lo_r_vals: Optional[List[int]] = None, lo_s_vals: Optional[List[int]] = None, lo_graph_sizes_range: Optional[Tuple[int,int]] = None, lo_num_samples: Optional[Tuple[int,int]] = None, gen_vertex_sp_features: bool = False, root_path: Optional[str] = None, use_editmask: bool = False, re_gen_properties: bool = False) -> None:

    if root_path is None:
        root_path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir)

    absolute_path_prefix = root_path

    path = osp.join('data', 'Proximity')

    possible_h = [1,3,5,8,10]

    # sanity checks
    assert [h in possible_h for h in h_vals]

    need_download = False
    for h in h_vals:
        if not osp.exists(osp.join(path, f'{h}-Prox')):
            need_download = True

    if need_download:
        download_proximity(root = path)

    # select some constant values
    num_processes = constants.num_processes
    vector_buffer_size = constants.vector_buffer_size

    # Generate features for every h in h_vals
    result_mmap_path = osp.join(root_path, path, 'results.np')
    editmask_mmap_path = osp.join(root_path, path, 'editmask.np')

    for h in h_vals:
        path = osp.join(path, f'{h}-Prox')
        dataset_prox = ProximityDataset(root = osp.join(absolute_path_prefix, path), h = h)

        output_path = osp.join(path, 'results')
        dataset_prop_filename = "properties.json"
        lookup_filename = "idx_lookup.pkl"
        dataset_properties_path = osp.join(output_path, dataset_prop_filename)
        lookup_path = osp.join(output_path, lookup_filename)

        if not osp.exists(dataset_properties_path) or not osp.exists(lookup_path):
            re_gen_properties = True
        
        if re_gen_properties:
            # Generate dataset properties and lookup files, they are later read and not generated again
            prop_manager = Dataset_Properties_Manager(properties_path = None, dataset = dataset_prox, node_pred = False, absolute_path_prefix = absolute_path_prefix, write_properties_root_path = output_path, write_properties_filename = dataset_prop_filename)
            prop_manager.initialize_idx_lookups(lookup_path = None, samples = None, write_lookup_root_path = output_path, write_lookup_filename = lookup_filename)

        dataset_desc = f"{h}-Prox"


        if gen_vertex_sp_features:
            # Generate Vertex SP features

            vertex_sp_path = osp.join(output_path, f'vertex_SP_features')
            dataset_write_filename = f"{h}-Prox_vertex_SP_features"

            metadata_filename = f'{dataset_write_filename}_metadata.json'

            chunksize = constants.graph_chunksize

            gen = Vertex_SP_Feature_Generator(dataset = dataset_prox, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = vertex_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
            print(f"---   Generating {h}-Prox Vertex SP features   ---")
            gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = vertex_sp_path, metadata_filename = metadata_filename, graph_mode = True)
            print(f"---   Finished generating {h}-Prox Vertex SP features   ---")

        if sp_k_vals is not None and len(sp_k_vals) > 0:
            for k in sp_k_vals:
                # Generate k-disk SP feature vectors

                k_disk_sp_path = osp.join(output_path, f'{k}-disk_SP_features')
                dataset_write_filename = f"{h}-Prox_{k}-disk_SP_features"

                metadata_filename = f'{dataset_write_filename}_metadata.json'

                chunksize = constants.vertex_chunksize

                gen = K_Disk_SP_Feature_Generator(dataset = dataset_prox, k = k, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = k_disk_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
                print(f"---   Generating {h}-Prox {k}-Disk SP features   ---")
                gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = k_disk_sp_path, metadata_filename = metadata_filename, graph_mode = False)
                print(f"---   Finished generating {h}-Prox {k}-Disk SP features   ---")

        if sp_r_vals is not None and sp_s_vals is not None and len(sp_r_vals) > 0 and len(sp_s_vals) == len(sp_r_vals):
            for idx in range(len(sp_r_vals)):
                s = sp_s_vals[idx]
                r = sp_r_vals[idx]
                if s < r:
                    continue

                # Generate r-s-ring SP feature vectors
                r_s_ring_sp_path = osp.join(output_path, f'{r}-{s}-ring_SP_features')
                dataset_write_filename = f"{h}-Prox_{r}-{s}-ring_SP_features"

                metadata_filename = f'{dataset_write_filename}_metadata.json'

                chunksize = constants.vertex_chunksize

                gen = R_S_Ring_SP_Feature_Generator(dataset = dataset_prox, r = r, s = s, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = r_s_ring_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
                print(f"---   Generating {h}-Prox {r}-{s}-Ring SP features   ---")
                gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = r_s_ring_sp_path, metadata_filename = metadata_filename, graph_mode = False)
                print(f"---   Finished generating {h}-Prox {r}-{s}-Ring SP features   ---")

        if lo_k_vals is not None and len(lo_k_vals) > 0:
            assert lo_graph_sizes_range is not None and lo_graph_sizes_range[0] > 0 and lo_graph_sizes_range[1] > 0 and lo_num_samples is not None

            for k in lo_k_vals:
                # Generate k-disk SP feature vectors

                for id in range(constants.num_lo_gens):
                    k_disk_sp_path = osp.join(output_path, f'{k}-disk_lo_features')

                    if constants.num_lo_gens > 1:
                        dataset_write_filename = f"{h}-Prox_{k}-disk_lo_features_{id}"
                    else:
                        dataset_write_filename = f"{h}-Prox_{k}-disk_lo_features"

                    metadata_filename = f'{dataset_write_filename}_metadata.json'

                    chunksize = constants.vertex_chunksize

                    gen = K_Disk_LO_Feature_Generator(dataset = dataset_prox, k = k, lo_graph_sizes_range = lo_graph_sizes_range, lo_s = lo_num_samples, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = k_disk_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
                    print(f"---   Generating {h}-Prox {k}-Disk Lovasz features   ---")
                    gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = k_disk_sp_path, metadata_filename = metadata_filename, graph_mode = False)
                    print(f"---   Finished generating {h}-Prox {k}-Disk Lovasz features   ---")

        if lo_r_vals is not None and lo_s_vals is not None and len(lo_r_vals) > 0 and len(lo_s_vals) == len(lo_r_vals):
            assert lo_graph_sizes_range is not None and lo_graph_sizes_range[0] > 0 and lo_graph_sizes_range[1] > 0 and lo_num_samples is not None

            for idx in range(len(lo_r_vals)):
                s = lo_s_vals[idx]
                r = lo_r_vals[idx]
                if s < r:
                    continue

                for id in range(constants.num_lo_gens):
                    # Generate r-s-ring SP feature vectors
                    r_s_ring_sp_path = osp.join(output_path, f'{r}-{s}-ring_lo_features')

                    if constants.num_lo_gens > 1:
                        dataset_write_filename = f"{h}-Prox_{r}-{s}-ring_lo_features_{id}"
                    else:
                        dataset_write_filename = f"{h}-Prox_{r}-{s}-ring_lo_features"

                    metadata_filename = f'{dataset_write_filename}_metadata.json'

                    chunksize = constants.vertex_chunksize

                    gen = R_S_Ring_LO_Feature_Generator(dataset = dataset_prox, r = r, s = s, lo_graph_sizes_range = lo_graph_sizes_range, lo_s = lo_num_samples, node_pred = False, samples = None, absolute_path_prefix = absolute_path_prefix, dataset_write_path = r_s_ring_sp_path, dataset_desc = dataset_desc, use_editmask = use_editmask, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = dataset_properties_path, idx_lookup_path = lookup_path)
                    print(f"---   Generating {h}-Prox {r}-{s}-Ring Lovasz features   ---")
                    gen.generate_features(write_filename = dataset_write_filename, chunksize = chunksize, vector_buffer_size = vector_buffer_size, num_processes = num_processes, log_times = False, metadata_path = r_s_ring_sp_path, metadata_filename = metadata_filename, graph_mode = False)
                    print(f"---   Finished generating {h}-Prox {r}-{s}-Ring Lovasz features   ---")


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

    root_path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir)
    # run_proximity(h_vals = [3], gen_vertex_sp_features = True, root_path = root_path, use_editmask = False, re_gen_properties = False)
    run_csl(k_vals = [3], gen_vertex_sp_features = True, root_path = root_path, use_editmask = False, re_gen_properties = True)

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

