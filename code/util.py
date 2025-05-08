#general imports
from typing import List, Optional, Tuple, Union
import json
import os.path as osp
import os
import pickle
import random

import numpy as np

#pytorch, pytorch geometric
import torch
from torch import Tensor

from torch_geometric.data import Data, Dataset
import torch_geometric.utils

#import for the r_s_ring_subgraph implementation
from torch_geometric.utils.num_nodes import maybe_num_nodes

import constants

# Helper to initialize random seeds for reproducability
def initialize_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    constants.random_generator = np.random.default_rng(seed)

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

def one_hot_to_labels(x: Tensor) -> Tensor:
    #if the features are not labels they are assumed to be a one-hot encoding
    if x.dim() > 1 and x.size()[1] > 1:
        assert (x.sum(dim=-1) == 1).sum() == x.size(0)
        x = x.argmax(dim=-1).to(dtype = torch.long)  # one-hot -> integer.
    assert x.dtype == torch.long

    return x

def read_metadata_file(path: str):
    if not osp.exists(path):
        raise FileNotFoundError
    else:
        with open(path, "r") as file:
            metadata = json.loads(file.read())

    return metadata

def write_metadata_file(path: str, filename: str, data) -> None:
    if not osp.exists(path):
        os.makedirs(path)

    path = osp.join(path, filename)
    if not osp.exists(path):
        open(path, 'w').close()

    with open(path, "w") as file:
        file.write(json.dumps(data, indent=4))

def write_numpy_txt(path: str, filename: str, data: np.array, comment: Optional[str]) -> None:
    if comment is None:
        comment = ""

    if not osp.exists(path):
        os.makedirs(path)

    path = osp.join(path, filename)
    if not osp.exists(path):
        open(path, 'w').close()

    with open(path, "w") as file:
        np.savetxt(fname = file, X = data, comments = '#', header = comment)

def read_numpy_txt(path: str) -> np.array:

    if not osp.exists(path):
        raise FileNotFoundError
    
    return np.loadtxt(fname = path, dtype = np.float64, comments = '#')

def write_pickle(path: str, filename: str, obj) -> None:
    if not osp.exists(path):
        os.makedirs(path)

    path = osp.join(path, filename)
    if not osp.exists(path):
        open(path, 'wb').close()

    with open(path, 'wb') as file:
        pickle.dump(obj = obj, file = file)

def read_pickle(path: str):
    if not osp.exists(path):
        raise FileNotFoundError
    
    with open(path, 'rb') as f:
        res = pickle.load(file = f)

    return res

def generate_tu_splits(root_path: str, dataset_path: str, dataset: Dataset, split_path: Optional[str] = None):
    splits = {}

    if split_path is not None:
        splits = read_metadata_file(path = split_path)
    else:
        # check whether a splits file exists at the default path
        def_path = osp.join(osp.join(root_path, dataset_path), "splits")
        def_filename = "splits.json"
        if osp.exists(osp.join(def_path, def_filename)):
            splits = read_metadata_file(osp.join(def_path, def_filename))
            return splits

        # generate test data, we sample test data according to the specified settings uniformly at random before generating the folds
        k = constants.num_k_fold

        for idx in range(k):
            splits[idx] = {}
            splits[idx]["test"] = [] # Note that the test set is the same for each fold, we store it multiple times to be consistent with the Prox Dataset splits
            splits[idx]["train"] = []
            splits[idx]["val"] = []

        num_graphs = dataset.len()

        total_indices = np.array(range(num_graphs), dtype = np.int64)
        num_test_graphs = int(constants.k_fold_test_ratio * num_graphs)
        possible_indices = total_indices.copy()
        test_indices = np.random.choice(possible_indices, size = num_test_graphs, replace = False)
        test_indices_list = test_indices.tolist()

        # generate k folds
        fold_elements = np.delete(total_indices, test_indices)
        num_remaining_graphs = fold_elements.shape[0]
        num_select = int(num_remaining_graphs / k)

        # remaining_elements include all elements that have not yet been included in a validation set
        remaining_elements = np.copy(fold_elements)

        for idx in range(k-1):
            splits[idx]["test"].extend(test_indices_list)

            val_elements = np.random.choice(remaining_elements, size = num_select, replace = False)
            splits[idx]["train"].extend(np.delete(fold_elements, np.isin(fold_elements, val_elements)).tolist())
            splits[idx]["val"].extend(val_elements.tolist())
            remaining_elements = np.delete(remaining_elements, np.isin(remaining_elements, val_elements))

        # the last split is generated by using all remaining elements that have not been included in a validation set yet
        splits[k-1]["test"].extend(test_indices_list)
        val_elements = remaining_elements
        splits[k-1]["train"].extend(np.delete(fold_elements, np.isin(fold_elements, val_elements)).tolist())
        splits[k-1]["val"].extend(val_elements.tolist())

        # Just as a precaution
        remaining_elements = np.delete(remaining_elements, np.isin(remaining_elements, val_elements))
        assert len(remaining_elements.tolist()) == 0

        # sort
        for idx in range(k):
            splits[idx]["train"].sort()
            splits[idx]["val"].sort()
            splits[idx]["test"].sort()

        # Write generated splits to default location
        write_metadata_file(path = def_path, filename = def_filename, data = splits)

    return splits