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

#parallelization
import multiprocessing
from multiprocessing.shared_memory import SharedMemory

#pytorch, pytorch geometric
import torch
from torch import Tensor

from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
import torch_geometric.utils

#numpy
import numpy as np

#scikit-learn
from sklearn import datasets

#import for the r_s_ring_subgraph implementation
from torch_geometric.utils.num_nodes import maybe_num_nodes

#import own functionality
import developmentHelpers as helpers
import SP_features as spf

# OGB
import ogb.nodeproppred as ogb_node
from ogb.nodeproppred import PygNodePropPredDataset
import ogb.graphproppred as ogb_graph
from ogb.graphproppred import PygGraphPropPredDataset

#imports to be removed later
import matplotlib.pyplot as plt

# Adaptation of the torch_geometric.utils.k_hop_subgraph method for r-s-Rings
# NOTE: this implementation only works for undirected graphs unlike the initial implementation of k_hop_subgraph
def r_s_ring_subgraph(
    node_idx: Union[int, List[int], Tensor],
    r: int,
    s: int,
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    #returns num_nodes if defined or the number of nodes based on the given edge_index tensor

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    #row, col are the two tensors representing edges, meaning {row[i],col[i]} are the edges defined by the graph

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
    #row.size(0) returns the size of the given dimension of the tensor (here: dim=0). This tensor should be 1-dimensional thus, this should return the size of the tensor (meaning number of (directed) edges or number of undirected edges x2)
    #Tensor.new_empty returns a new tensor of (by default) same type and device as the original tensor. The first argument is the size of the tensor. The data values are uninitialized.

    if isinstance(node_idx, int):
        node_idx = torch.tensor([node_idx], device=row.device)
    elif isinstance(node_idx, (list, tuple)):
        node_idx = torch.tensor(node_idx, device=row.device)
    else:
        node_idx = node_idx.to(row.device)
    #torch.tensor is a constructor of a tensor that copies data (the first argument). Thus this creates a 1-dimensional tensor with node_idx as data (either a single integer or a list/tuple of indices)

    subsets = [node_idx]

    #generate the s-disk of the given vertices
    for _ in range(s):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        #subsets[-1] returns the last element of subsets
        torch.index_select(node_mask, 0, row, out=edge_mask)
        #returns a tensor (copied into edge_mask) with the same number of dimensions as node_mask where the 0-th dimension has the same size as row
        subsets.append(col[edge_mask])

    #remove the elements corresponding to the r-disk of the given vertices
    r_disk_subset = torch.cat(subsets[0:r]).unique()
    
    #valid_node_mask evaluates True for every entry that is not contained in the r-disk around the given vertex
    valid_node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    valid_node_mask.fill_(True)
    valid_node_mask[r_disk_subset]=False

    for i in range(r,s+1):
        #for every subset of vertices in a given range, remove the vertices already contained in the r-disk.
        #torch.index_select(valid_node_mask, 0, subsets[i]) returns a 1-dim tensor that evaluates True if valid_node_mask[subsets[i][j]] evaluates True
        #subsets[i][torch.index_select(valid_node_mask, 0, subsets[i])] returns subsets[i] constricted to the indices where torch.index_select(valid_node_mask, 0, subsets[i]) evaluates to True.
        subsets[i]=subsets[i][torch.index_select(valid_node_mask, 0, subsets[i])]

    #discard the first r-1 subsets (the vertices with distance smaller than r)
    subsets=subsets[r:]

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True

    #select the edges included in the subgraph induced by node_mask
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        mapping = row.new_full((num_nodes, ), -1)
        mapping[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = mapping[edge_index]

    return subset, edge_index, inv, edge_mask


#generate k-disk around a vertex v for a given graph G
def gen_k_disk(k: int, graph_data: Data, vertex: int) -> Data:

    # graph should be a pytorch_geometric Data object
    # vertex is the index of the considered vertex

    # NOTE: It is necessary to specify num_nodes in case of graphs containing isolated vertices (as the correct number of vertices cannot be inferenced from edge_index in this case)
    _subset, _edge_index, _, _ = torch_geometric.utils.k_hop_subgraph(node_idx=vertex, num_hops=k, edge_index=graph_data.edge_index, relabel_nodes=True, num_nodes = graph_data.num_nodes)
    return Data(x=graph_data.x[_subset], edge_index=_edge_index)

#generate r-s-ring around a vertex v for a given graph G
def gen_r_s_ring(r: int, s: int, graph_data: Data, vertex: int) -> Data:

    # graph should be a pytorch_geometric Data object
    # vertex is the index of the considered vertex

    # NOTE: It is necessary to specify num_nodes in case of graphs containing isolated vertices (as the correct number of vertices cannot be inferenced from edge_index in this case)
    _subset, _edge_index, _, _ = r_s_ring_subgraph(node_idx=vertex, r=r, s=s, edge_index=graph_data.edge_index, relabel_nodes=True, num_nodes = graph_data.num_nodes)
    return Data(x=graph_data.x[_subset], edge_index=_edge_index)

# An enum used to encode events for time logging if desired
class TimeLoggingEvent(Enum):
    load_graph = 0
    gen_k_disk = 1
    gen_r_s_ring = 2
    floyd_warshall = 3
    SP_vector_computation = 4
    get_database_idx = 5


# Superclass implementing some basic logging and property extraction functionality
# NOTE: This class should be treated as an abstract class and never be directly initialized
class Feature_Generator():

    # NOTE: dataset._data.x must NOT be set to None to ensure proper function and num_features must be at least 1. In the unlabeled case, x should simply be set to the constant 0 vector.
    # samples defines the indices of the vertices in the dataset that should be considered.
    def __init__(self, dataset: Dataset, node_pred: bool, samples: Optional[List[int]] | Optional[Tensor], dataset_write_path: str, dataset_write_filename: str, result_mmap_dest: str, editmask_mmap_dest: str, properties_path: Optional[str] = None, write_properties_root_path: Optional[str] = None, write_properties_filename: Optional[str] = None):
        super().__init__()

        self.dataset = dataset
        self.dataset_path = dataset_write_path
        self.dataset_filename = dataset_write_filename

        self.properties = {}

        if properties_path is None:
            #compute general dataset properties => number of vertices, label alphabet, graph sizes; There may be further properties that need defining in the subclasses (such as a distance alphabet)
            self.properties["num_vertices"] = 0
            if dataset.num_features > 1:
                self.properties["label_alphabet"] = list(range(dataset.num_features))
            else:
                self.properties["label_alphabet"] = []

            self.properties["graph_sizes"] = []
            
            for i in range(dataset.len()):
                _curgraph_data = dataset.get(i)
                _curgraph_num_vertices = _curgraph_data.x.shape[0]
                self.properties["num_vertices"] += _curgraph_num_vertices
                self.properties["graph_sizes"].append(_curgraph_num_vertices)

                if dataset.num_features == 1:
                    _new_labels_tensor = torch.unique(input = _curgraph_data.x, sorted = False)
                    _new_labels = _new_labels_tensor.flatten().tolist()
                    self.properties["label_alphabet"].extend(_new_labels)
                    self.properties["label_alphabet"] = list(set(self.properties["label_alphabet"]))

            #if write_properties_path is specified, save properties as a json file on disk
            if write_properties_root_path is not None:
                assert write_properties_filename is not None

                if not osp.exists(write_properties_root_path):
                    os.makedirs(write_properties_root_path)

                p = osp.join(write_properties_root_path, write_properties_filename)
                if not osp.exists(p):
                    open(p, 'w').close()  

                with open(p, "w") as file:
                    file.write(json.dumps(self.properties, indent=4))
        else:
            #read properties file
            if not osp.exists(properties_path):
                raise FileNotFoundError
            else:
                with open(properties_path, "r") as file:
                    self.properties = json.loads(file.read())

        # Ensure important properties are set
        assert "num_vertices" in self.properties and "label_alphabet" in self.properties and "graph_sizes" in self.properties
        assert len(self.properties["graph_sizes"]) > 0 and len(self.properties["label_alphabet"]) > 0 and self.properties["num_vertices"] > 0

        # convert self.properties["graph_sizes"] to a numpy array for easier indexing later
        self.properties["graph_sizes"] = np.array(self.properties["graph_sizes"])

        # remove duplicates if necessary
        self.properties["label_alphabet"] = list(set(self.properties["label_alphabet"]))
        self.properties["label_alphabet"].sort()

        # Variables that need to be initialized by the subclasses
        # The computed dataset
        self.shared_dataset_result_dtype = None
        self.shared_dataset_result_shape = None
        self.shared_dataset_result_mmap_dest = None

        # Variables that need to be initialized by the subclasses
        # An editmask for the computed dataset (utilised in cropping)
        self.shared_editmask_dtype = None
        self.shared_editmask_shape = None
        self.shared_editmask_mmap_dest = None

        # Variables that need to be initialized by the subclasses
        # String describing the computed features
        self.title_str = None


        # Used in dictionary that will be filled with times if time logging is enabled
        self.num_events = len(TimeLoggingEvent)

        # Nature of the prediction task
        self.node_pred = node_pred

        # A dictionary utilised to speed up index computation methods. 
        # In the node prediction setting this corresponds to a map dataset_idx -> database_idx with entries for every dataset_idx contained in samples.
        # In the graph prediction setting this corresponds to a map dataset_idx -> graph_id, vertex_id with entries for every dataset_idx contained in the graphs in samples
        self.dataset_idx_lookup = {}

        # A dictionary containing key-value pairs of the form {graph_id: database_start_idx} giving the starting idx of vertices of the given graph in the database. 
        # This does not correspond to the idx in the dataset. This is used to efficiently compute a database_vertex_idx from a graph_id.
        # Only utilised in the graph prediction setting
        self.database_graph_start_idx = None

        if samples is None:
            # We compute the feature vectors for the whole dataset
            self.samples = []

            if node_pred:
                # One graph
                for dataset_idx in range(self.properties["num_vertices"]):
                    self.dataset_idx_lookup[dataset_idx] = dataset_idx
                    self.samples.append(dataset_idx)
            else:
                # Multiple graphs
                self.database_graph_start_idx = {}

                graph_id = 0
                cur_start_idx = 0
                self.database_graph_start_idx[0] = 0
                for dataset_idx in range(self.properties["num_vertices"]):
                    if (dataset_idx - cur_start_idx) == int(self.properties["graph_sizes"][graph_id]):
                        graph_id += 1
                        cur_start_idx = dataset_idx
                        self.database_graph_start_idx[graph_id] = dataset_idx

                    self.dataset_idx_lookup[dataset_idx] = tuple([graph_id, dataset_idx - cur_start_idx])
                    self.samples.append(dataset_idx)

        else:
            if torch.is_tensor(samples):
                samples = samples.tolist()

            assert len(samples) > 0

            if node_pred:
                # One graph
                self.samples = []

                # Pre-compute the dataset_idx_lookup dictionary, a map representing dataset_idx -> database_idx
                for dataset_idx in samples:
                    self.dataset_idx_lookup[dataset_idx] = len(self.samples)
                    self.samples.append(dataset_idx)

            else:
                # Multiple graphs
                assert samples is not None and len(samples) > 0
                self.samples = []

                self.database_graph_start_idx = {}

                # We compute a list of the vertices included in the graphs (defined by the graph_ids in samples). Additionaly we store the start_idx property to efficiently compute the inverse of this operation via look-up.
                for graph_id in samples:
                    start_idx = int(sum(self.properties["graph_sizes"][:graph_id]))

                    # Store database_start_idx for a given graph. This is given by the length of the sample list before adding the vertices of the given graph.
                    assert graph_id not in self.database_graph_start_idx
                    self.database_graph_start_idx[graph_id] = len(self.samples)

                    # Refine samples
                    num_nodes = int(self.properties["graph_sizes"][graph_id])
                    self.samples.extend(list(range(start_idx, start_idx + num_nodes)))

                    # Pre-compute the dataset_idx_lookup dictionary
                    for dataset_idx in range(start_idx, start_idx + num_nodes):
                        self.dataset_idx_lookup[dataset_idx] = tuple([graph_id, dataset_idx - start_idx])

        
    # NOTE: currently unused
    # #helpers for indexing and editing of the database array. NOTE: The database idx is the idx used to store results, not in the dataset itself (which would be the dataset idx).
    # def get_vertex_identifier_from_database_idx(self, database_idx: int) -> Tuple[int, int]:
    #     cur_graph_id = 0
    #     idx = database_idx
    #     for size in self.properties["graph_sizes"][self.sampled_graphs]:
    #         size = int(size)
    #         if idx >= size:
    #             idx -= size
    #             cur_graph_id += 1
    #         else:
    #             return tuple([cur_graph_id, idx])
    
    # Helper to get the graph_id and vertex_id from a dataset idx. NOTE: The dataset idx is the idx used in the dataset, not in the tensor used to store results (which would be the database idx).
    def get_vertex_identifier_from_dataset_idx(self, dataset_idx: int) -> Tuple[int, int]:
        
        if self.node_pred:
            return tuple([0, dataset_idx])
        else:
            # We use a pre-computed dictionary representing dataset_idx -> graph_id, vertex_id to speed up the computation
            return self.dataset_idx_lookup[dataset_idx]
        
        # cur_graph_id = 0
        # idx = dataset_idx
        # for size in self.properties["graph_sizes"]:
        #     size = int(size)
        #     if idx >= size:
        #         idx -= size
        #         cur_graph_id += 1
        #     else:
        #         return tuple([cur_graph_id, idx])

    # NOTE: In the node prediction scenario this method currently does not support computing a sample of the vertices of the dataset.
    def get_database_idx_from_vertex_identifier(self, vertex_identifier: Tuple[int,int]) -> int:   
    
        if self.node_pred:
            # Only one graph
            # vertex_identifier[1] is the dataset_idx of the given vertex. We use dataset_idx_lookup; a dictionary representing a map dataset_idx -> database_idx directly
            return self.dataset_idx_lookup[vertex_identifier[1]]
        else:
            # Multiple graphs, vertex_identifier[0] represents the graph_id and vertex_identifier[1] represents the idx of the vertex within the graph. Utilise graph_start_idx as a lookup method
            assert self.database_graph_start_idx is not None and vertex_identifier[0] in self.database_graph_start_idx

            return self.database_graph_start_idx[vertex_identifier[0]] + vertex_identifier[1]

        # # In the graph prediction setting, this function is sped up using a lookup of pre-computed vertex ranges
        # start = time.time()
        # graphs = [x for x in range(vertex_identifier[0]) if x in self.sampled_graphs]
        # result = int(sum(self.properties["graph_sizes"][graphs]) + vertex_identifier[1])
        # print(time.time() - start)
        # return result
        # #return int(sum(self.properties["graph_sizes"][:vertex_identifier[0]]) + vertex_identifier[1])
    
    #NOTE: this method requires a fully edited vector, meaning it has to already include the graph_id and vertex_id of the specific feature vector
    def edit_database_result_by_index(self, database_idx: int, vector: np.array):
        #assert vector.shape[0] == self.shared_dataset_result_shape[1]
        #assert 0 <= database_idx and database_idx < self.shared_dataset_result_shape[0]

        shared_dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'r+', shape = self.shared_dataset_result_shape)
        shared_dataset_result[database_idx,:] = vector
        shared_dataset_result.flush()

    # NOTE: this method requires a fully edited vector, meaning it has to already include the graph_id and vertex_id of the specific feature vector
    # Works with an array of indices and an array of vectors
    def edit_database_result_by_indices(self, database_idx: np.array, vectors: np.array, editmask: np.array, count: Optional[int] = None):

        if count is None:
            # A complete batch should be stored
            shared_dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'r+', shape = self.shared_dataset_result_shape)
            shared_dataset_result[database_idx,:] = vectors[:]
            shared_dataset_result.flush()

            # Edit the editmask in the same step
            editmask_mmap = np.memmap(self.shared_editmask_mmap_dest, dtype = self.shared_editmask_dtype, mode = 'r+', shape = self.shared_editmask_shape)
            editmask_mmap[:] = np.logical_or(editmask_mmap, editmask)
            editmask_mmap.flush()
        else:
            # Only the first count elements should be stored
            shared_dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'r+', shape = self.shared_dataset_result_shape)
            shared_dataset_result[database_idx[:count],:] = vectors[:count,:]
            shared_dataset_result.flush()

            # Edit the editmask in the same step
            editmask_mmap = np.memmap(self.shared_editmask_mmap_dest, dtype = self.shared_editmask_dtype, mode = 'r+', shape = self.shared_editmask_shape)
            editmask_mmap[:] = np.logical_or(editmask_mmap, editmask)
            editmask_mmap.flush()


    # NOTE: This method has to be overwritten by any subclass
    def compute_feature(self, vertex_idx: int):
        raise NotImplementedError
    
    # Utilised for analysis of runtimes, might include significant memory and calculation time overhead
    def compute_feature_log_times(self, vertex_idx: int):
        raise NotImplementedError
    
    # generate features of all vertices and store them on disk
    # NOTE: specifying a chunksize greater than 1 might increase the lag (esp of the progress bar) but often times yields a much faster computation speed and is highly advised
    def generate_features(self, chunksize: int = 1, vector_buffer_size: int = 256, num_processes: int=1, comment: Optional[str]=None, log_times: bool = False, dump_times: bool = False, time_summary_path: str = "", time_summary_filename: Optional[str] = None):

        assert num_processes > 0
        assert self.samples is not None and len(self.samples) > 0

        start_time = time.time()

        vector_buffer = np.zeros(shape = (vector_buffer_size, self.shared_dataset_result_shape[1]), dtype = self.shared_dataset_result_dtype)
        vector_buffer_count = 0
        index_buffer = np.zeros(shape = (vector_buffer_size), dtype = int)
        editmask_buffer = np.full(shape = self.shared_editmask_shape, dtype = self.shared_editmask_dtype, fill_value = False)

        pool = multiprocessing.Pool(processes = num_processes)
        if log_times:
            self.times = np.full(shape = (len(self.samples), self.num_events), dtype = np.float32, fill_value = -1)

            completed = 0
            num_samples = len(self.samples)

            # Store the time results
            for vector_res in tqdm.tqdm(pool.imap_unordered(self.compute_feature_log_times, self.samples, chunksize = chunksize), total = len(self.samples)):
                # Since the writing of intermediary results is by far the most time intensive task, we try to write batches of 256 vectors in the main process
                # vector_res is in the shape of database_idx, vector
                index_buffer[vector_buffer_count] = vector_res[0]
                vector_buffer[vector_buffer_count,:] = vector_res[1]
                vector_buffer_count += 1
                completed += 1
                editmask_buffer[:] = np.logical_or(editmask_buffer, vector_res[2])

                if vector_buffer_count == vector_buffer_size:
                    self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer)
                    vector_buffer_count = 0
                elif completed == num_samples:
                    # Remaining tasks completed successfully but are not yet stored
                    self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer, count = vector_buffer_count)

                self.times[vector_res[0],:] = vector_res[3][:]
        else:
            completed = 0
            num_samples = len(self.samples)

            for vector_res in tqdm.tqdm(pool.imap_unordered(self.compute_feature, self.samples, chunksize = chunksize), total = num_samples):
                # Since the writing of intermediary results is by far the most time intensive task, we try to write batches of vector_buffer_size vectors in the main process
                # vector_res is in the shape of database_idx, vector
                index_buffer[vector_buffer_count] = vector_res[0]
                vector_buffer[vector_buffer_count,:] = vector_res[1]
                vector_buffer_count += 1
                completed += 1
                editmask_buffer[:] = np.logical_or(editmask_buffer, vector_res[2])

                if vector_buffer_count == vector_buffer_size:
                    self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer)
                    vector_buffer_count = 0
                elif completed == num_samples:
                    # Remaining tasks completed successfully but are not yet stored
                    self.edit_database_result_by_indices(database_idx = index_buffer, vectors = vector_buffer, editmask = editmask_buffer, count = vector_buffer_count)

        pool.close()
        pool.join()

        computation_time = time.time() - start_time

        #should only be called after the processes are joined again
        if comment is not None:
            comment = self.title_str + "\n" + comment + f"\nComputation time: {computation_time}\n"
        else:
            comment = self.title_str + f"\nComputation time: {computation_time}\n"
        
        self.save_features(comment)

        if log_times:
            if time_summary_path != "":
                self.calculate_time_summary(time_summary_path = time_summary_path, time_summary_filename = time_summary_filename)
                if dump_times:
                    self.dump_times(time_dump_path = time_summary_path)

    #NOTE: This function must only be called after joining parallelized processes to ensure correct function
    # Implements a simple cropping of the feature vectors 
    def save_features(self, comment: Optional[str]):

        if not osp.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        p = osp.join(self.dataset_path, self.dataset_filename)
        if not osp.exists(p):
            open(p, 'w').close()

        dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'r+', shape = self.shared_dataset_result_shape)
        editmask = np.memmap(self.shared_editmask_mmap_dest, dtype = self.shared_editmask_dtype, mode = 'r+', shape = self.shared_editmask_shape)

        if comment is None:
            datasets.dump_svmlight_file(X = dataset_result[:,editmask], y = np.zeros(len(self.samples)), f = p, comment = "")
        else:
            datasets.dump_svmlight_file(X = dataset_result[:,editmask], y = np.zeros(len(self.samples)), f = p, comment = comment)

    # Helper to evaluate time complexity of individual operations
    def log_time(self, event: TimeLoggingEvent, value: float):
        
        self.times[event.value] = value

    def calculate_time_summary(self, time_summary_path: str, time_summary_filename: Optional[str] = None):
        assert len(self.times) > 0
        if time_summary_filename is None:
            time_summary_filename = "time_summary.txt"

        summary_str_list = []

        summary_str_list.append(f"Time summary for {self.title_str}:\n\n")

        # calculate average times for each step (/event)
        # The keys of summary are the events and the values are tuples (implemented as list with 2 elements) of the number of times a time for the specified event has been stored and the sum of the stored times.
        # The number of times a time for an event has been stored is useful to detect potentially erronious computation
        summary = np.zeros(shape = (self.num_events), dtype = np.float32)
        count = np.zeros(shape = (self.num_events), dtype = np.int32)

        for t in self.times:
            mask = (t >= 0)
            count[mask] += 1
            summary[mask] += t[mask]

        for event in TimeLoggingEvent:
            if count[event.value] == 0:
                continue

            if count[event.value] != len(self.samples):
                summary_str_list.append(f"WARNING: a time for {event.name} has not been stored for every iteration\n")

            avg = summary[event.value] / count[event.value]
            summary_str_list.append(f"{event.name} has been computed in {avg}s on average over {count[event.value]} iterations\n")

        summary_str = "".join(summary_str_list)
        
        # Write the summary into a file
        if not osp.exists(time_summary_path):
            os.makedirs(time_summary_path)

        p = osp.join(time_summary_path, time_summary_filename)
        if not osp.exists(p):
            open(p, 'w').close()  

        with open(p, "w") as file:
            file.write(summary_str)

    # Can be used to dump a json file with all the stored times
    def dump_times(self, time_dump_path: str, time_dump_filename: Optional[str] = None):
        if time_dump_filename is None:
            time_dump_filename = "times_dump.json"

        dump = {}
        for idx in self.samples:
            dump[idx] = {}
            for event in TimeLoggingEvent:
                if self.times[idx, event.value] >= 0:
                    dump[idx][event.name] = float(self.times[idx, event.value])

        if not osp.exists(time_dump_path):
            os.makedirs(time_dump_path)

        p = osp.join(time_dump_path, time_dump_filename)
        if not osp.exists(p):
            open(p, 'w').close()  

        with open(p, "w") as file:
            file.write(json.dumps(dump, indent=4))


#classes that includes generator functionality for features that can be clustered and implement parallelization functionality

class K_Disk_SP_Feature_Generator(Feature_Generator):

    #dataset: The dataset of which the features should be generated, k: the k-value for the k-disk generation, 
    #properties_path: if set specifies path of a json file including the dataset properties such that they do not need to be computed again, 
    #write_properties_path: if set specifies the path to which computed properties are saved in a json file; this path is recommended to coincide with the path, the dataset is saved to
    #write_properties_filename: if specified set the filename for the properties json file
    def __init__(self, dataset: Dataset, k: int, node_pred: bool, samples: Optional[List[int]], dataset_write_path: str, dataset_write_filename: str, result_mmap_dest: str, editmask_mmap_dest: str, properties_path: Optional[str] = None, write_properties_root_path: Optional[str] = None, write_properties_filename: Optional[str] = None):
        super().__init__(dataset = dataset, node_pred = node_pred, samples = samples, dataset_write_path = dataset_write_path, dataset_write_filename = dataset_write_filename, result_mmap_dest = result_mmap_dest, editmask_mmap_dest = editmask_mmap_dest, properties_path = properties_path, write_properties_root_path = write_properties_root_path, write_properties_filename = write_properties_filename)

        #sanity checks
        assert k > 0

        self.k = k

        # The property distances_alphabet is required for the SP features
        self.properties["distances_alphabet"] = list(range(2*self.k + 2))
        # To denote disconnected label pairs
        self.properties["distances_alphabet"].append(-1)
        
        self.title_str = f"{self.k}-Disk SP features"

        assert "distances_alphabet" in self.properties
        assert len(self.properties["distances_alphabet"]) > 0

        # remove duplicates if necessary
        self.properties["distances_alphabet"] = list(set(self.properties["distances_alphabet"]))
        self.properties["distances_alphabet"].sort()

        # create a dataset np array as shared memory to store the computed feature vectors
        self.shared_dataset_result_dtype = 'float64'
        self.shared_dataset_result_shape = tuple([len(self.samples), ((len(self.properties["label_alphabet"])**2) * len(self.properties["distances_alphabet"])) + 2])
        self.shared_dataset_result_mmap_dest = result_mmap_dest

        dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'w+', shape = self.shared_dataset_result_shape)
        dataset_result.fill(0)
        dataset_result.flush()

        # implement the editmask as a memmap
        self.shared_editmask_mmap_dest = editmask_mmap_dest
        self.shared_editmask_shape = tuple([((len(self.properties["label_alphabet"])**2) * len(self.properties["distances_alphabet"])) + 2,])
        self.shared_editmask_dtype = np.bool

        editmask = np.memmap(self.shared_editmask_mmap_dest, dtype=self.shared_editmask_dtype, mode = 'w+', shape=self.shared_editmask_shape)
        editmask.fill(False)
        editmask[0] = True
        editmask[1] = True
        editmask.flush()
        
        #create graph features object for computation purposes
        self.sp_features = spf.SP_graph_features(label_alphabet = self.properties["label_alphabet"], distances_alphabet = self.properties["distances_alphabet"], graph_sizes = self.properties["graph_sizes"])

    # Start method of a new process
    # Compute the k-disk SP feature vector for a single data point.
    # vertex_idx is a dataset_idx, NOT a database_idx
    def compute_feature(self, vertex_idx: int):
        #get the graph associated with vertex_idx
        vertex_identifier = self.get_vertex_identifier_from_dataset_idx(vertex_idx)
        graph_id = vertex_identifier[0]
        vertex_id = vertex_identifier[1]
        cur_graph = self.dataset.get(graph_id)

        #generate k disk
        k_disk = gen_k_disk(k = self.k, graph_data = cur_graph, vertex = vertex_id)

        #run floyd warshall on the k disk to compute a distances matrix
        floyd_warshall_distances = self.sp_features.floyd_warshall(graph = k_disk)

        #compute a dictionary representing the sp distances in the k disk
        sp_map = self.sp_features.sp_feature_map(distances = floyd_warshall_distances, x = k_disk.x)
        result, editmask_res = self.sp_features.sp_feature_vector_from_feature_map(dict = sp_map, vertex_identifier = vertex_identifier)

        # editmask = np.memmap(self.shared_editmask_mmap_dest, dtype = self.shared_editmask_dtype, mode = 'r+', shape = self.shared_editmask_shape)
        # editmask[:] = np.logical_or(editmask, editmask_res)
        database_idx = self.get_database_idx_from_vertex_identifier(vertex_identifier = vertex_identifier)
        #self.edit_database_result_by_index(database_idx = database_idx, vector = result)

        return database_idx, result, editmask_res

    # Start method of a new process
    # Compute the k-disk SP feature vector for a single data point with logging the times using the log_time() method
    # Returns a tuple of the computed idx (necessary since the order in which this function is called is arbitrary) and the computed times.
    # NOTE: logging the computation times is significantly slower than running the computation without, thus this method should only be utilised on a limited amount of vertices
    def compute_feature_log_times(self, vertex_idx: int):
        self.times = np.full(shape = (self.num_events), dtype = np.float32, fill_value = -1)

        #get the graph associated with vertex_idx
        get_graph_start = time.time()
        #graph_id, vertex_id = self.get_vertex_identifier_from_dataset_idx(vertex_idx)
        vertex_identifier = self.get_vertex_identifier_from_dataset_idx(vertex_idx)
        graph_id = vertex_identifier[0]
        vertex_id = vertex_identifier[1]
        cur_graph = self.dataset.get(graph_id)
        get_graph_time = time.time() - get_graph_start
        self.log_time(event = TimeLoggingEvent.load_graph, value = get_graph_time)

        #generate k disk
        gen_k_disk_start = time.time()
        k_disk = gen_k_disk(k = self.k, graph_data = cur_graph, vertex = vertex_id)
        gen_k_disk_time = time.time() - gen_k_disk_start
        self.log_time(event = TimeLoggingEvent.gen_k_disk, value = gen_k_disk_time)

        #run floyd warshall on the k disk to compute a distances matrix
        floyd_warshall_start = time.time()
        floyd_warshall_distances = self.sp_features.floyd_warshall(graph = k_disk)
        floyd_warshall_time = time.time() - floyd_warshall_start
        self.log_time(event = TimeLoggingEvent.floyd_warshall, value = floyd_warshall_time)

        #compute a dictionary representing the sp distances in the k disk
        sp_map_start = time.time()
        sp_map = self.sp_features.sp_feature_map(distances = floyd_warshall_distances, x = k_disk.x)
        result, editmask_res = self.sp_features.sp_feature_vector_from_feature_map(dict = sp_map, vertex_identifier = vertex_identifier)
        sp_map_time = time.time() - sp_map_start
        self.log_time(event = TimeLoggingEvent.SP_vector_computation, value = sp_map_time)

        save_res_start = time.time()
        # editmask_start = time.time()
        # editmask = np.memmap(self.shared_editmask_mmap_dest, dtype = self.shared_editmask_dtype, mode = 'r+', shape = self.shared_editmask_shape)
        # editmask[:] = np.logical_or(editmask, editmask_res)
        # editmask_time = time.time() - editmask_start
        # index_calc_start = time.time()
        database_idx = self.get_database_idx_from_vertex_identifier(vertex_identifier = vertex_identifier)
        # index_calc_time = time.time() - index_calc_start
        # databasewrite_start = time.time()
        #self.edit_database_result_by_index(database_idx = database_idx, vector = result)
        # databasewrite_time = time.time()-databasewrite_start
        save_res_time = time.time() - save_res_start
        self.log_time(event = TimeLoggingEvent.get_database_idx, value = save_res_time)

        return database_idx, result, editmask_res, self.times

class R_S_Ring_SP_Feature_Generator(Feature_Generator):

    #dataset: The dataset of which the features should be generated, k: the k-value for the k-disk generation, 
    #properties_path: if set specifies path of a json file including the dataset properties such that they do not need to be computed again, 
    #write_properties_path: if set specifies the path to which computed properties are saved in a json file; this path is recommended to coincide with the path, the dataset is saved to
    #write_properties_filename: if specified set the filename for the properties json file
    def __init__(self, dataset: Dataset, r: int, s: int, node_pred: bool, samples: Optional[List[int]], dataset_write_path: str, dataset_write_filename: str, result_mmap_dest: str, editmask_mmap_dest: str, properties_path: Optional[str] = None, write_properties_root_path: Optional[str] = None, write_properties_filename: Optional[str] = None):
        super().__init__(dataset = dataset, node_pred = node_pred, samples = samples, dataset_write_path = dataset_write_path, dataset_write_filename = dataset_write_filename, result_mmap_dest = result_mmap_dest, editmask_mmap_dest = editmask_mmap_dest, properties_path = properties_path, write_properties_root_path = write_properties_root_path, write_properties_filename = write_properties_filename)

        # sanity checks
        assert r > 0
        assert s > 0
        assert s > r

        self.r = r
        self.s = s

        # The property distances_alphabet is required for the SP features
        self.properties["distances_alphabet"] = list(range(2*self.s + 2))
        # We add -1 to the possible distances to denote a pair of labels which are disconnected.
        self.properties["distances_alphabet"].append(-1)
        
        self.title_str = f"{self.r}-{self.s}-Ring SP features"

        assert "distances_alphabet" in self.properties
        assert len(self.properties["distances_alphabet"]) > 0

        # remove duplicates if necessary
        self.properties["distances_alphabet"] = list(set(self.properties["distances_alphabet"]))
        self.properties["distances_alphabet"].sort()

        # create a dataset np array as shared memory to store the computed feature vectors
        self.shared_dataset_result_dtype = 'float64'
        self.shared_dataset_result_shape = tuple([len(self.samples), ((len(self.properties["label_alphabet"])**2) * len(self.properties["distances_alphabet"])) + 2])
        self.shared_dataset_result_mmap_dest = result_mmap_dest

        dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'w+', shape = self.shared_dataset_result_shape)
        dataset_result.fill(0)
        dataset_result.flush()

        # implement the editmask as a memmap
        self.shared_editmask_mmap_dest = editmask_mmap_dest
        self.shared_editmask_shape = tuple([((len(self.properties["label_alphabet"])**2) * len(self.properties["distances_alphabet"])) + 2,])
        self.shared_editmask_dtype = np.bool

        editmask = np.memmap(self.shared_editmask_mmap_dest, dtype=self.shared_editmask_dtype, mode = 'w+', shape=self.shared_editmask_shape)
        editmask.fill(False)
        editmask[0] = True
        editmask[1] = True
        editmask.flush()
        
        #create graph features object for computation purposes
        self.sp_features = spf.SP_graph_features(label_alphabet = self.properties["label_alphabet"], distances_alphabet = self.properties["distances_alphabet"], graph_sizes = self.properties["graph_sizes"])

    # Start method of a new process
    # Compute the r-s-ring SP feature vector for a single data point
    def compute_feature(self, vertex_idx: int):
        #get the graph associated with vertex_idx
        vertex_identifier = self.get_vertex_identifier_from_dataset_idx(vertex_idx)
        graph_id = vertex_identifier[0]
        vertex_id = vertex_identifier[1]
        cur_graph = self.dataset.get(graph_id)

        #generate r-s-ring
        r_s_ring = gen_r_s_ring(r = self.r, s = self.s, graph_data = cur_graph, vertex = vertex_id)

        #run floyd warshall on the k disk to compute a distances matrix
        floyd_warshall_distances = self.sp_features.floyd_warshall(graph = r_s_ring)

        #compute a dictionary representing the sp distances in the k disk
        sp_map = self.sp_features.sp_feature_map(distances = floyd_warshall_distances, x = r_s_ring.x)
        result, editmask_res = self.sp_features.sp_feature_vector_from_feature_map(dict = sp_map, vertex_identifier = vertex_identifier)

        # editmask = np.memmap(self.shared_editmask_mmap_dest, dtype = self.shared_editmask_dtype, mode = 'r+', shape = self.shared_editmask_shape)
        # editmask[:] = np.logical_or(editmask, editmask_res)
        database_idx = self.get_database_idx_from_vertex_identifier(vertex_identifier = vertex_identifier)
        #self.edit_database_result_by_index(database_idx = database_idx, vector = result)

        return database_idx, result, editmask_res

    # Start method of a new process
    # Compute the r-s-ring SP feature vector for a single data point with logging the times using the log_time() method
    # NOTE: logging the computation times is significantly slower than running the computation without, thus this method should only be utilised on a limited amount of vertices
    def compute_feature_log_times(self, vertex_idx: int):
        self.times = np.full(shape = (self.num_events), dtype = np.float32, fill_value = -1)

        #get the graph associated with vertex_idx
        get_graph_start = time.time()
        vertex_identifier = self.get_vertex_identifier_from_dataset_idx(vertex_idx)
        graph_id = vertex_identifier[0]
        vertex_id = vertex_identifier[1]
        cur_graph = self.dataset.get(graph_id)
        get_graph_time = time.time() - get_graph_start
        self.log_time(event = TimeLoggingEvent.load_graph, value = get_graph_time)

        #generate k disk
        gen_r_s_ring_start = time.time()
        r_s_ring = gen_r_s_ring(r = self.r, s = self.s, graph_data = cur_graph, vertex = vertex_id)
        gen_r_s_ring_time = time.time() - gen_r_s_ring_start
        self.log_time(event = TimeLoggingEvent.gen_r_s_ring, value = gen_r_s_ring_time)

        #run floyd warshall on the k disk to compute a distances matrix
        floyd_warshall_start = time.time()
        floyd_warshall_distances = self.sp_features.floyd_warshall(graph = r_s_ring)
        floyd_warshall_time = time.time() - floyd_warshall_start
        self.log_time(event= TimeLoggingEvent.floyd_warshall, value = floyd_warshall_time)

        #compute a dictionary representing the sp distances in the k disk
        sp_map_start = time.time()
        sp_map = self.sp_features.sp_feature_map(distances = floyd_warshall_distances, x = r_s_ring.x)
        result, editmask_res = self.sp_features.sp_feature_vector_from_feature_map(dict = sp_map, vertex_identifier = tuple([graph_id, vertex_id]))
        sp_map_time = time.time() - sp_map_start
        self.log_time(event = TimeLoggingEvent.SP_vector_computation, value = sp_map_time)

        save_res_start = time.time()
        # editmask = np.memmap(self.shared_editmask_mmap_dest, dtype = self.shared_editmask_dtype, mode = 'r+', shape = self.shared_editmask_shape)
        # editmask[:] = np.logical_or(editmask, editmask_res)
        database_idx = self.get_database_idx_from_vertex_identifier(vertex_identifier = vertex_identifier)
        #self.edit_database_result_by_index(database_idx = database_idx, vector = result)
        save_res_time = time.time() - save_res_start
        self.log_time(event= TimeLoggingEvent.get_database_idx, value = save_res_time)

        return database_idx, result, editmask_res, self.times

#test zone

# Only for testing purposes
def run_mutag():
    path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'TU')
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
    path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'OGB')
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
    path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'OGB')
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
    path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'OGB')
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

    # split_idx["train"]

    sp_gen = R_S_Ring_SP_Feature_Generator(dataset = dataset_molhiv, r = r, s = s, node_pred = False, samples = None, dataset_write_path = molhiv_path, dataset_write_filename = dataset_write_filename_r_s_ring, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = molhiv_properties_path, write_properties_root_path = molhiv_path, write_properties_filename = filename)
    #sp_gen = K_Disk_SP_Feature_Generator(dataset = dataset_molhiv, k = k, node_pred = False, samples = split_idx["train"], dataset_write_path = molhiv_path, dataset_write_filename = dataset_write_filename_k_disk, result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = molhiv_properties_path, write_properties_root_path = molhiv_path, write_properties_filename = filename)


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

    print('Multi process performance: ')
    ts_multi = time.time_ns()
    sp_gen.generate_features(num_processes = 8, chunksize = 128, vector_buffer_size = 16_384, comment = None, log_times = False, dump_times = False, time_summary_path = molhiv_path)
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
    run_molhiv()

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

