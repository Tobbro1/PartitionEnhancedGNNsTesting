#general imports
from typing import List, Optional, Tuple, Union
import os
import os.path as osp
import json #to implement saving and loading the properties of datasets without having to run the computation multiple times

#pytorch, pytorch geometric
import torch
from torch import Tensor

from torch_geometric.datasets import TUDataset
from torch_geometric.data import Data, Dataset
import torch_geometric.utils

#import for the r_s_ring_subgraph implementation
from torch_geometric.utils.num_nodes import maybe_num_nodes

#import own functionality
import developmentHelpers as helpers
#import SP_features as spf

#imports to be removed later
import matplotlib.pyplot as plt

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



#classes that includes generator functionality for features that can be clustered

class K_Disk_SP_Feature_Generator():

    #dataset: The dataset of which the features should be generated, k: the k-value for the k-disk generation, 
    #properties_path: if set specifies path of a json file including the dataset properties such that they do not need to be computed again, 
    #write_properties_path: if set specifies the path to which computed properties are saved in a json file; this path is recommended to coincide with the path, the dataset is saved to
    #write_properties_filename: if specified set the filename for the properties json file
    def __init__(self, dataset: Dataset, k: int, properties_path: Optional[str] = None, write_properties_root_path: Optional[str] = None, write_properties_filename: Optional[str] = None):
        super().__init__()

        #sanity checks
        assert k > 0

        properties = {}

        if properties_path is None:
            assert write_properties_filename is not None

            #compute dataset properties => number of vertices, label alphabet, distances alphabet (maximum diameter of a k-disk given by 2*k + 1), graph sizes


            #test values
            properties["num_vertices"] = 200
            properties["label_alphabet"] = [0,2,4,5,6,7,8,9,10,11]
            properties["distances_alphabet"] = list(range(2*k + 1))
            properties["graph_sizes"] = [10,20,14,55,3,4,1]

            #if write_properties_path is specified, save properties as a json file on disk
            if write_properties_root_path is not None:
                if not osp.exists(write_properties_root_path):
                    os.makedirs(write_properties_root_path)

                p = osp.join(write_properties_root_path, write_properties_filename)
                if not osp.exists(p):
                    open(p, 'w').close()  

                with open(p, "w") as file:
                    file.write(json.dumps(properties, indent=4))
        else:
            #read properties file
            if not osp.exists(properties_path):
                raise FileNotFoundError
            else:
                with open(properties_path, "r") as file:
                    properties = json.loads(file.read())

        assert "num_vertices" in properties and "label_alphabet" in properties and "distances_alphabet" in properties and "graph_sizes" in properties
        #create graph features object
        #self.sp_features = spf.SP_graph_features()

    #generate SP features from k-Disks of all vertices and store them on disk
    #TODO: Implement multi-threading
    def generate_k_disk_SP_features(self):

        raise NotImplementedError
    

    def save_k_disk_SP_features():

        raise NotImplementedError


#test zone
path = osp.join(osp.abspath(osp.dirname(__file__)), "data", "SP_features")
filename = "k_disk_sp_properties.json"
sp_gen = K_Disk_SP_Feature_Generator("", k=3, write_properties_root_path = path, write_properties_filename = filename)
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

