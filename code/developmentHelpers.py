from torch_geometric.data import Data
from typing import Optional, List

import torch_geometric.utils
import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch import Tensor

#Utilises the networkx draw function to draw a given torch_geometric data object with matplotlib
#NOTE: if labels are set and not 1-dimensional, it is assumed to be a one-hot-encoding.
#NOTE: utilising the networkx draw functionality is generally disencouraged
def drawGraph(graph: Data, vertex_select: Optional[List[int]] = None, figure_count: int=1, draw_labels: bool=True, labels: Optional[Tensor] = None):

    _graph_vis = torch_geometric.utils.to_networkx(graph, to_undirected=True)
    _graph_colors=['yellow' for _ in _graph_vis.nodes()]

    if vertex_select is not None:
        for j in vertex_select:
            _graph_colors[j]='green'

    plt.figure(figure_count)

    if labels is None:
        nx.draw(_graph_vis, node_color=_graph_colors, with_labels=draw_labels)
    else:
        #determines whether the feature dimension is larger than 1 => it is assumed to be a one-hot encoding in this case
        if labels.dim() > 1:
            assert (labels.sum(dim=-1) == 1).sum() == labels.size(0)
            labels = labels.argmax(dim=-1)  # one-hot -> integer.
        assert labels.dtype == torch.long

        _labeldict = {}
        for i,v in enumerate(_graph_vis):
            _labeldict[v] = labels[i].item()
        
        nx.draw(_graph_vis, labels=_labeldict, node_color=_graph_colors, with_labels=draw_labels)