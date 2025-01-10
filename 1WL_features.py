from typing import Optional

import torch
from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.utils import (
    degree,
    is_sparse,
    scatter,
    sort_edge_index,
    to_edge_index,
)

from torch_geometric.datasets import TUDataset
import os.path as osp
import developmentHelpers as helpers
import matplotlib.pyplot as plt

"""@torch.no_grad()
    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        rRuns the forward pass of the module.
        #Adj is either a Tensor or a sparse Tensor
        
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



        # `col` is sorted, so we can use it to `split` neighbors to groups:
        #returns a list with deg[v] = degree(v) for a vertex of index v
        deg = degree(col, x.size(0), dtype=torch.long).tolist()

        out = []
        #zip(x.tolist(), x[row].split(deg)) returns tuples in the shape of v, tensor[h(w_1),...,h(w_k)] where v is a vertex, w_1,...,w_k are the neighbors of v and h(u) is the label of the last iteration of u
        for node, neighbors in zip(x.tolist(), x[row].split(deg)):
            #neighbors.sort()[0].tolist() sorts the multiset of the neighbors labels and converts it to a list (neighbors is a tensor of shape ([x.size(0)]))
            idx = hash(tuple([node] + neighbors.sort()[0].tolist()))
            if idx not in self.hashmap:
                self.hashmap[idx] = len(self.hashmap)
            out.append(self.hashmap[idx])

        return torch.tensor(out, device=x.device)"""

class WLConv(torch.nn.Module):
    r"""The Weisfeiler Lehman (WL) operator from the `"A Reduction of a Graph
    to a Canonical Form and an Algebra Arising During this Reduction"
    <https://www.iti.zcu.cz/wl2018/pdf/wl_paper_translation.pdf>`_ paper.

    :class:`WLConv` iteratively refines node colorings according to:

    .. math::
        \mathbf{x}^{\prime}_i = \textrm{hash} \left( \mathbf{x}_i, \{
        \mathbf{x}_j \colon j \in \mathcal{N}(i) \} \right)

    Shapes:
        - **input:**
          node coloring :math:`(|\mathcal{V}|, F_{in})` *(one-hot encodings)*
          or :math:`(|\mathcal{V}|)` *(integer-based)*,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node coloring :math:`(|\mathcal{V}|)` *(integer-based)*
    """
    def __init__(self):
        super().__init__()
        self.hashmap = {}

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.hashmap = {}

    @torch.no_grad()
    def forward(self, x: Tensor, edge_index: Adj) -> Tensor:
        r"""Runs the forward pass of the module."""

        if x.dim() > 1:
            assert (x.sum(dim=-1) == 1).sum() == x.size(0)
            x = x.argmax(dim=-1)  # one-hot -> integer.
        assert x.dtype == torch.long

        if is_sparse(edge_index):
            col_and_row, _ = to_edge_index(edge_index)
            col = col_and_row[0]
            row = col_and_row[1]
        else:
            edge_index = sort_edge_index(edge_index, num_nodes=x.size(0),
                                         sort_by_row=False)
            row, col = edge_index[0], edge_index[1]

        # `col` is sorted, so we can use it to `split` neighbors to groups:
        deg = degree(col, x.size(0), dtype=torch.long).tolist()

        out = []
        for node, neighbors in zip(x.tolist(), x[row].split(deg)):
            idx = hash(tuple([node] + neighbors.sort()[0].tolist()))
            if idx not in self.hashmap:
                self.hashmap[idx] = len(self.hashmap)
            out.append(self.hashmap[idx])

        return torch.tensor(out, device=x.device)

    def histogram(self, x: Tensor, batch: Optional[Tensor] = None,
                  norm: bool = False) -> Tensor:
        r"""Given a node coloring :obj:`x`, computes the color histograms of
        the respective graphs (separated by :obj:`batch`).
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        num_colors = len(self.hashmap)
        batch_size = int(batch.max()) + 1

        index = batch * num_colors + x
        #torch.ones_like(index) returns a tensor of the same properties as index filled with the value 1
        out = scatter(torch.ones_like(index), index, dim=0,
                      dim_size=num_colors * batch_size, reduce='sum')
        out = out.view(batch_size, num_colors)

        if norm:
            out = out.to(torch.float)
            out /= out.norm(dim=-1, keepdim=True)

        return out
    
#for num_layers select the diameter of input graph to ensure that every 1-WL distinguishable graph is distinguished
class WL(torch.nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList([WLConv() for _ in range(num_layers)])

    def forward(self, x, edge_index, batch=None):
        histograms = []
        for conv in self.convs:
            x = conv(x, edge_index)
            print('Histograms')
            histograms.append(conv.histogram(x, batch, norm=False))
            print(histograms)
        
        return histograms


#testing the function
path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'TU')
dataset_mutag = TUDataset(root=path, name="MUTAG", use_node_attr=True)

_data = dataset_mutag.get(0)
_num_layers = _data.num_nodes - 1

print('X: ')
print(_data.x)
print('X.dim: ')
print(_data.x.dim())

_wl = WL(num_layers = _num_layers)
_histograms = _wl(_data.x, _data.edge_index, _data.batch)

helpers.drawGraph(_data)
print(_histograms)
plt.show()