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

from torch_geometric.data import Data
import torch_geometric.utils

#import for the r_s_ring_subgraph implementation
from torch_geometric.utils.num_nodes import maybe_num_nodes

# Helper to initialize random seeds for reproducability
def initialize_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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