#general imports
import multiprocessing.shared_memory
from typing import List, Optional, Tuple, Union
import os
import os.path as osp
import json #to implement saving and loading the properties of datasets without having to run the computation multiple times

#parallelization
import multiprocessing
from multiprocessing.shared_memory import SharedMemory

#pytorch, pytorch geometric
import torch
from torch import Tensor

from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Dataset
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

#imports to be removed later
import matplotlib.pyplot as plt
import time

#adaptation of the torch_geometric.utils.k_hop_subgraph method for r-s-Rings
#NOTE: this implementation only works for undirected graphs unlike the initial implementation of k_hop_subgraph
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

    #graph should be a pytorch_geometric Data object
    #vertex is the index of the considered vertex

    _subset, _edge_index, _, _ = torch_geometric.utils.k_hop_subgraph(node_idx=vertex, num_hops=k, edge_index=graph_data.__getitem__('edge_index'), relabel_nodes=True)
    return Data(x=graph_data.__getitem__('x')[_subset], edge_index=_edge_index)

#generate r-s-ring around a vertex v for a given graph G
def gen_r_s_ring(r: int, s: int, graph_data: Data, vertex: int) -> Data:

    #graph should be a pytorch_geometric Data object
    #vertex is the index of the considered vertex

    _subset, _edge_index, _, _ = r_s_ring_subgraph(node_idx=vertex, r=r, s=s, edge_index=graph_data.__getitem__('edge_index'), relabel_nodes=True)
    return Data(x=graph_data.__getitem__('x')[_subset], edge_index=_edge_index)

# Superclass implementing some basic logging and property extraction functionality
# NOTE: This class should be treated as an abstract class and never be directly initialized
class Feature_Generator():

    def __init__(self, dataset: Dataset, dataset_write_path: str, dataset_write_filename: str, result_mmap_dest: str, editmask_mmap_dest: str, properties_path: Optional[str] = None, write_properties_root_path: Optional[str] = None, write_properties_filename: Optional[str] = None):
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
                    _new_labels_tensor, _, _ = torch.unique(input = _curgraph_data.x, sorted = False)
                    _new_labels = _new_labels_tensor.flatten().tolist()
                    self.properties["label_alphabet"].append(_new_labels)
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

        # remove duplicates if necessary
        self.properties["label_alphabet"] = list(set(self.properties["label_alphabet"]))
        self.properties["label_alphabet"].sort()

        # Variables that need to be initialized by the subclasses
        # The computed dataset
        self.shared_dataset_result_dtype = None
        self.shared_dataset_result_shape = None
        self.shared_dataset_result_mmap_dest = None

        # An editmask for the computed dataset (utilised in cropping)
        self.shared_editmask_dtype = None
        self.shared_editmask_shape = None
        self.shared_editmask_mmap_dest = None

        # String describing the computed features
        self.title_str = None


    #helpers for indexing and editing of the database array
    def get_vertex_identifier_from_dataset_idx(self, dataset_idx: int) -> Tuple[int, int]:
        cur_graph_id = 0
        idx = dataset_idx
        for size in self.properties["graph_sizes"]:
            if idx >= size:
                idx -= size
                cur_graph_id += 1
            else:
                return tuple([cur_graph_id, idx])

    def get_dataset_idx_from_vertex_identifier(self, vertex_identifier: Tuple[int,int]) -> int:      
        return sum(self.properties["graph_sizes"][:vertex_identifier[0]]) + vertex_identifier[1]
    
    #NOTE: this method requires a fully edited vector, meaning it has to already include the graph_id and vertex_id of the specific feature vector
    def edit_dataset_result_by_index(self, dataset_idx: int, vector: np.array):
        #assert vector.shape == self.shared_dataset_result[0,:].shape
        assert vector.shape[0] == self.shared_dataset_result_shape[1]
        #assert 0 <= dataset_idx and dataset_idx < self.shared_dataset_result[:,0].shape[0]
        assert 0 <= dataset_idx and dataset_idx < self.shared_dataset_result_shape[0]

        shared_dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'r+', shape = self.shared_dataset_result_shape)
        shared_dataset_result[dataset_idx,:] = vector
        shared_dataset_result.flush()

    # NOTE: This method has to be overwritten by any subclass
    def compute_feature(self, vertex_idx: int):
        raise NotImplementedError
    
    #generate SP features from k-Disks of all vertices and store them on disk
    def generate_features(self, num_processes: int=1, comment: Optional[str]=None):

        assert num_processes > 0

        start_time = time.time()

        pool = multiprocessing.Pool(processes = num_processes)
        pool.imap_unordered(self.compute_feature, range(self.properties["num_vertices"]))
        pool.close()
        pool.join()

        computation_time = time.time() - start_time

        #should only be called after the processes are joined again
        if comment is not None:
            comment = self.title_str + "\n" + comment + f"\nComputation time: {computation_time}\n"
        else:
            comment = self.title_str + f"\nComputation time: {computation_time}\n"
        
        self.save_k_disk_SP_features(comment)

    #NOTE: This function must only be called after joining parallelized processes to ensure correct function
    # Implements a simple cropping of the feature vectors 
    def save_k_disk_SP_features(self, comment: Optional[str]):

        if not osp.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        p = osp.join(self.dataset_path, self.dataset_filename)
        if not osp.exists(p):
            open(p, 'w').close()

        #editmask = np.ndarray(shape = self.editmask_shape, dtype=self.editmask_dtype, buffer = self.editmask.buf)
        dataset_result = np.memmap(self.shared_dataset_result_mmap_dest, dtype = self.shared_dataset_result_dtype, mode = 'r+', shape = self.shared_dataset_result_shape)
        editmask = np.memmap(self.shared_editmask_mmap_dest, dtype = self.shared_editmask_dtype, mode = 'r+', shape = self.shared_editmask_shape)

        if comment is None:
            datasets.dump_svmlight_file(X = dataset_result[:,editmask], y = np.zeros(self.properties["num_vertices"]), f = p, comment = "")
        else:
            datasets.dump_svmlight_file(X = dataset_result[:,editmask], y = np.zeros(self.properties["num_vertices"]), f = p, comment = comment)

#classes that includes generator functionality for features that can be clustered and implement parallelization functionality

class K_Disk_SP_Feature_Generator(Feature_Generator):

    #dataset: The dataset of which the features should be generated, k: the k-value for the k-disk generation, 
    #properties_path: if set specifies path of a json file including the dataset properties such that they do not need to be computed again, 
    #write_properties_path: if set specifies the path to which computed properties are saved in a json file; this path is recommended to coincide with the path, the dataset is saved to
    #write_properties_filename: if specified set the filename for the properties json file
    def __init__(self, dataset: Dataset, k: int, dataset_write_path: str, dataset_write_filename: str, result_mmap_dest: str, editmask_mmap_dest: str, properties_path: Optional[str] = None, write_properties_root_path: Optional[str] = None, write_properties_filename: Optional[str] = None):
        super().__init__(dataset = dataset, dataset_write_path = dataset_write_path, dataset_write_filename = dataset_write_filename, result_mmap_dest = result_mmap_dest, editmask_mmap_dest = editmask_mmap_dest, properties_path = properties_path, write_properties_root_path = write_properties_root_path, write_properties_filename = write_properties_filename)

        #sanity checks
        assert k > 0

        self.k = k

        self.properties["distances_alphabet"] = list(range(2*self.k + 1))
        
        self.title_str = f"{self.k}-Disk SP features"

        assert "distances_alphabet" in self.properties
        assert len(self.properties["distances_alphabet"]) > 0

        # remove duplicates if necessary
        self.properties["distances_alphabet"] = list(set(self.properties["distances_alphabet"]))
        self.properties["distances_alphabet"].sort()

        # create a dataset np array as shared memory to store the computed feature vectors
        self.shared_dataset_result_dtype = 'float64'
        self.shared_dataset_result_shape = tuple([self.properties["num_vertices"], ((len(self.properties["label_alphabet"])**2) * len(self.properties["distances_alphabet"])) + 2])
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

    #start method of a new process
    #compute the k_disk SP feature vector for a single data point
    def compute_feature(self, vertex_idx: int):
        graph_id, vertex_id = self.get_vertex_identifier_from_dataset_idx(vertex_idx)
        cur_graph = self.dataset.get(graph_id)

        #generate k disk
        k_disk = gen_k_disk(k = self.k, graph_data = cur_graph, vertex = vertex_id)

        #run floyd warshall on the k disk to compute a distances matrix
        floyd_warshall_distances = self.sp_features.floyd_warshall(graph = k_disk)

        #compute a dictionary representing the sp distances in the k disk
        sp_map = self.sp_features.sp_feature_map(distances = floyd_warshall_distances, x = k_disk.x)
        result, editmask_res = self.sp_features.sp_feature_vector_from_feature_map(dict = sp_map, vertex_identifier = tuple([graph_id, vertex_id]))

        editmask = np.memmap(self.shared_editmask_mmap_dest, dtype = self.shared_editmask_dtype, mode = 'r+', shape = self.shared_editmask_shape)
        editmask[:] = np.logical_or(editmask, editmask_res)
        self.edit_dataset_result_by_index(dataset_idx = vertex_idx, vector = result)


#test zone
if __name__ == '__main__':
    #path = osp.join(osp.abspath(osp.dirname(__file__)), "data", "SP_features")
    path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'TU')
    mutag_path = osp.join(path, "MUTAG")
    dd_path = osp.join(path, "DD")
    result_mmap_path = 'results.np'
    editmask_mmap_path = 'editmask.np'
    dataset_mutag = TUDataset(root=path, name="MUTAG", use_node_attr=True)
    dataset_dd = TUDataset(root=path, name="DD", use_node_attr=True)
    filename = "k_disk_sp_properties.json"
    sp_gen = K_Disk_SP_Feature_Generator(dataset = dataset_mutag, k = 3, dataset_write_path = mutag_path, dataset_write_filename = "k_disk_SP_features_MUTAG.svmlight", result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = osp.join(mutag_path, filename), write_properties_root_path = mutag_path, write_properties_filename = filename)
    #sp_gen = K_Disk_SP_Feature_Generator(dataset = dataset_dd, k = 3, dataset_write_path = dd_path, dataset_write_filename = "k_disk_SP_features_DD.svmlight", result_mmap_dest = result_mmap_path, editmask_mmap_dest = editmask_mmap_path, properties_path = osp.join(dd_path, filename), write_properties_root_path = dd_path, write_properties_filename = filename)
    #
    #  print('Single process performance: ')
    # ts_single = time.time_ns()
    # sp_gen.generate_k_disk_SP_features(num_processes = 1, comment = "testcomment1")
    # time_single = (time.time_ns() - ts_single) / 1_000
    # print('Single threaded time: ' + str(time_single))
    print('Multi process performance: ')
    ts_multi = time.time_ns()
    sp_gen.generate_features(num_processes = 8, comment = None)
    time_multi = (time.time_ns() - ts_multi) / 1_000_000
    print('Multi threaded time: ' + str(time_multi))
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

