import torch
from torch import Tensor

from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data
import torch_geometric.utils
import os.path as osp
import networkx as nx
import matplotlib.pyplot as plt

import developmentHelpers as helpers

#import for r_s_ring_subgraph
from typing import List, Optional, Tuple, Union
from torch_geometric.utils.num_nodes import maybe_num_nodes

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

#test zone
path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'TU')
dataset_mutag = TUDataset(root=path, name="MUTAG", use_node_attr=True)

_data = dataset_mutag.get(0)

_vertex_select=4
_k=3

_r=2
_s=_k

_data_k_disk = gen_k_disk(_k, _data, _vertex_select)
_data_r_s_ring = gen_r_s_ring(r=_r, s=_s, graph_data=_data, vertex=_vertex_select)

helpers.drawGraph(_data, vertex_select=[_vertex_select], figure_count=1, draw_labels=True)
helpers.drawGraph(_data_k_disk, figure_count=2, draw_labels=True)
helpers.drawGraph(_data_r_s_ring, figure_count=3, draw_labels=True)

plt.show()

