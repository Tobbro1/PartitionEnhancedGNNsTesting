import torch
from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.utils import (
    is_sparse,
    scatter,
    sort_edge_index,
    to_edge_index,
)
from itertools import permutations
from torch_geometric.data import Data

from torch_geometric.datasets import TUDataset
import os.path as osp
import vertex_partition_feature_embedding.developmentHelpers as helpers
import matplotlib.pyplot as plt

#Adaptation of the WLConv in the standard torch_geometric convolutions to implement the 2-WL algorithm
class TwoWLConv(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.hashmap = {}

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.hashmap = {}

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        r"""Runs the forward pass of the module."""

        out = torch.full((x.size(0), x.size(0)), -1, dtype=torch.long, device=x.device)

        for _v1 in range(x.size(0)):
            for _v2 in range(x.size(0)):
                #generate the multisets _M1 and _M2 which are the multisets of the labels of N_1((_v1,_v2)) and N_2((_v1,_v2))

                _M1 = x[_v1,:].sort()[0].tolist()
                _M2 = x[:,_v2].sort()[0].tolist()

                #generate a new label for the tuple (_v1, _v2)
                idx = hash(tuple([x[_v1][_v2].item()] + _M1 + _M2))
                if idx not in self.hashmap:
                    self.hashmap[idx] = len(self.hashmap)
                out[_v1][_v2] = self.hashmap[idx]

        return out

    #batch is a vector assigning each vertex to the identifier of its batch. If not None, multiple graphs are summarized in one disconnected graph and the batch vector identifies the initial graph origin of a vertex
    def histogram(self, x: Tensor,
                  norm: bool = False) -> Tensor:
        r"""Given a node coloring :obj:`x`, computes the color histograms of
        the respective graphs (separated by :obj:`batch`).
        """
        
        num_colors = len(self.hashmap)

        #convert the labels to a one-dimensional tensor
        index = x.flatten()

        out = scatter(torch.ones_like(index), index, dim=0,
                      dim_size=num_colors, reduce='sum')

        if norm:
            out = out.to(torch.float)
            out /= out.norm(dim=-1, keepdim=True)

        return out
    
#NOTE: This implementation is only designed for undirected graphs
class TwoWL(torch.nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList([TwoWLConv() for _ in range(num_layers)])
        self.atomic_hashmap = {}

    def atomic_histogram(self, x: Tensor,
                  norm: bool = False) -> Tensor:
        r"""Given a node coloring :obj:`x`, computes the color histograms of
        the respective graphs (separated by :obj:`batch`).
        """
        
        num_colors = len(self.atomic_hashmap)

        #convert the labels to a one-dimensional tensor
        index = x.flatten()

        out = scatter(torch.ones_like(index), index, dim=0,
                      dim_size=num_colors, reduce='sum')

        if norm:
            out = out.to(torch.float)
            out /= out.norm(dim=-1, keepdim=True)

        return out

    def forward(self, x: Tensor, edge_index: Adj, norm: bool = False) -> Tensor:

        #determines whether the feature dimension is larger than 1 => it is assumed to be a one-hot encoding in this case
        if x.dim() > 1:
            assert (x.sum(dim=-1) == 1).sum() == x.size(0)
            x = x.argmax(dim=-1)  # one-hot -> integer.
        assert x.dtype == torch.long

        #handles the different possible types of edge_index
        if is_sparse(edge_index):
            col_and_row, _ = to_edge_index(edge_index)
            col = col_and_row[0]
            row = col_and_row[1]
        else:
            edge_index = sort_edge_index(edge_index, num_nodes=x.size(0),
                                         sort_by_row=False)
            row, col = edge_index[0], edge_index[1]

        #generate tuples
        _tuples = list(permutations(range(x.size(0)), 2))
        for i in range(x.size(0)):
            _tuples.append((i,i))

        #generate initial labels for 2-tuples based on atomic type, i.e. two tuples (v,w), (v',w') get assigned the same atomic type if v->v', w->w' is an isomorphism
        #that means the tuples get assigned the same atomic type iff x[v] = x[v'], x[w] = x[w'] and {v,w} in E(G) iff {v',w'} in E(G)
        _labels = torch.full((x.size(0),x.size(0)), -1, dtype = torch.long, device = x.device)

        for _t in _tuples:
            _v1 = _t[0]
            _v2 = _t[1]

            #check whether _v1 and _v2 are connected by an edge
            _connected = False
            _indices = ((row == _v1).nonzero(as_tuple=True)[0])
            for i in _indices.tolist():
                if col[i]==_v2:
                    _connected = True

            #utilise a hash function to generate labels based on the atomic type
            _idx = hash(tuple([x[_v1].item()] + [x[_v2].item()] + [_connected]))
            if _idx not in self.atomic_hashmap:
                self.atomic_hashmap[_idx] = len(self.atomic_hashmap)
            _labels[_v1][_v2] = self.atomic_hashmap[_idx]


        #generate the histograms 
        histograms = []

        #add the histogram based on the initial labeling
        histograms.append(self.atomic_histogram(_labels, norm=norm))

        for i,conv in enumerate(self.convs):
            _labels = conv(x = _labels)
            histograms.append(conv.histogram(_labels, norm=norm))
        
        return histograms
    

#Testing the implementation
#_edge_index = torch.tensor([[0,1,1,2,3,3,4,4,1,1], [1,0,2,1,4,1,3,1,3,4]], dtype=torch.long)
#_x = torch.tensor([0,0,0,0,0], dtype=torch.long)

#_data = Data(x=_x, edge_index = _edge_index)
#path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'TU')
#dataset_mutag = TUDataset(root=path, name="MUTAG", use_node_attr=True)

#_data = dataset_mutag.get(13)

#_num_layers = 2#_data.num_nodes - 1

#_wl = TwoWL(num_layers = _num_layers)
#_histograms = _wl(_data.x, _data.edge_index, norm=False)

_edge_index = torch.tensor([[0,1], [1,0]], dtype=torch.long)
_x = torch.tensor([0,0], dtype=torch.long)
_data = Data(x=_x, edge_index = _edge_index)
_num_layers = 1
_wl = TwoWL(num_layers = _num_layers)
_histograms = _wl(_data.x, _data.edge_index, norm=False)

_edge_index2 = torch.tensor([[], []], dtype=torch.long)
_x2 = torch.tensor([0,0], dtype=torch.long)
_data2 = Data(x=_x2, edge_index = _edge_index2)
_num_layers2 = 1
_wl2 = TwoWL(num_layers = _num_layers2)
_histograms2 = _wl2(_data2.x, _data2.edge_index, norm=False)

helpers.drawGraph(_data, labels=_data.x)
helpers.drawGraph(_data2, labels=_data2.x, figure_count=2)
print('Graph1: ' + str(_histograms))
print('Graph2: ' + str(_histograms2))
plt.show()

#                print('(' + str(_v1) + ', ' + str(_v2) + '): ' + str(self.hashmap[idx]))
#                print('X[_v1][_v2]: ' + str(tuple([x[_v1][_v2].item()] + _M1 + _M2)))
