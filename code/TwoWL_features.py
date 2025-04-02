# general
from typing import Optional, Tuple, Dict
import itertools

# Numpy
import numpy as np

# pytorch
import torch

# pytorch geometric
from torch_geometric.data import Data
from torch_geometric.utils import (
    is_sparse,
    sort_edge_index,
    to_edge_index,
)

class TwoWL():

    # One big challenge is to make the calculation of the vectors consistent across multiple processes. Concretely, we need to utilise a single hashmap storing the colors among all processes.
    # Implementing this might lead to race conditions. Thus it is probably advisable to return the coloring dictionary as a result. 
    # The parent process that spawned the sub-processes should join the processes and then calculate the concrete vectors or alternatively the calculation of the vectors from the histograms can be spread among multiple processes again
    # 

    # num_layers and num_labels are independent on the concrete graph and should only depend on the dataset and experiment setup
    def __init__(self, num_layers: int, num_labels: int):

        assert num_layers > 0
        self.num_layers = num_layers

        self.num_labels = num_labels
    
    # NOTE: This function is intended to be called internally
    # Input: graph is the graph for which the atomic types of 2-tuples should be computed
    # Output: an |V|x|V| array of which the entry (v1, v2) corresponds to the atomic type of the tupe (v1, v2)
    def compute_atomic_types(self, graph: Data, num_vertices: Optional[int] = None) -> np.array:
        x, edge_index = graph.x, graph.edge_index

        if num_vertices is None:
            num_vertices = x.size(0)

        assert num_vertices > 0

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

        # two tuples (x,y) and (x', y') get assigned the same atomic type iff x->x' and y->y' is an isomorphism
        # That means, the tuples get assinged the same atomic type iff l(x) = l(x'), l(y) = l(y') and {x,y} in E iff {x', y'} in E
        # That means if there are L possible labels for vertices, there are L*L*2 possible atomic types (label v1, lavel v2, connected?)
        # We utilise this information to generate atomic labelings that are consistent among different processes and graphs.
        # An atomic type (l1, l2, con?) is mapped to the number 2*L*l1 + 2*l2 + con?
        
        # Atomic types is an array of shape |V|x|V| holding the atomic types for each tuple (v1, v2)
        atomic_types = np.full(shape = (num_vertices, num_vertices), fill_value = -1, dtype = np.int64)

        for v1, v2 in itertools.product(range(num_vertices), repeat = 2):
            # we consider the tuple (v1, v2)

            # check whether v1 and v2 are connected, 0 corresponds to disconnected, 1 corresponds to connected
            connected = 0
            indices = ((row == v1).nonzero(as_tuple=True)[0])
            for i in indices.tolist():
                if col[i] == v2:
                    connected = 1

            # The value for the tuple (v1, v2) is given by 2*L*l1 + 2*l2 + con?
            atomic_types[v1, v2] = 2 * self.num_labels * x[v1].item() + 2 * x[v2].item() + connected

        return atomic_types

    # Input: The graph for which the two WL coloring dictionary should be computed
    # Output: coloring: an array of shape |V|x|V| which represents a coloring of all the tuples of the given graph.
    def compute_two_wl(self, graph: Data, num_vertices: Optional[int] = None) -> np.array:

        if num_vertices is None:
            num_vertices = graph.x.size(0)

        # The initial tuple coloring are the atomic types
        coloring = self.compute_atomic_types(graph = graph)

        tuples = itertools.product(range(num_vertices), repeat = 2)

        # Update the coloring num_layers times
        for _ in range(self.num_layers):

            new_colors = np.full(shape = coloring.shape, fill_value = -1, dtype = coloring.dtype)

            for v1, v2 in tuples:
                # We generate the Multisets of the colors in the 1 neighborhood and 2 neighborhood of a tuple
                _M1 = coloring[v1,:].tolist()
                _M2 = coloring[:,v2].tolist()
                _M1.sort()
                _M2.sort()

                color = hash(tuple(coloring[v1, v2] + _M1 + _M2))

                # NOTE: Contrary to popular implementations of e.g. the 1-WL (torch geometric implementation) we use the result of the hashfunction as the color directly.
                #       This is motivated by ensuring consistency across processes and multiple graphs. 
                new_colors[v1, v2] = color
            
            coloring = new_colors

        return coloring
    
    # Input: array detailing the final coloring, expected to be the result of compute_two_wl
    # Output: Dictionary mapping color (as hash) -> frequency
    def compute_color_frequencies(self, coloring: np.array) -> Dict[int, int]:
        coloring = np.ravel(coloring)
        values = set(coloring)
        result = {}

        for color in values:
            result[color] = coloring[coloring == color].size

        return result

    # NOTE: This function is mostly  for testing purposes and not intended to be used in the final computation as it does NOT ensure consistency across graphs and processes
    def compute_single_graph_vector(self, color_frequencies: Dict[int, int]) -> np.array:

        result = np.full(shape = (len(color_frequencies),), dtype = int, fill_value = -1)

        for color_idx, (_, freq) in enumerate(color_frequencies.items()):
            result[color_idx] = freq

        return result


#Test zone
if __name__ == '__main__':

    _num_layers = 1
    wl = TwoWL(num_layers = _num_layers, num_labels = 1)

    _edge_index = torch.tensor([[0,1], [1,0]], dtype=torch.long)
    _x = torch.tensor([0,0], dtype=torch.long)
    graph1 = Data(x=_x, edge_index = _edge_index)
    
    coloring1 = wl.compute_two_wl(graph = graph1)
    col_freq1 = wl.compute_color_frequencies(coloring = coloring1)
    histogram1 = wl.compute_single_graph_vector(color_frequencies = col_freq1)

    _edge_index2 = torch.tensor([[], []], dtype=torch.long)
    _x2 = torch.tensor([0,0], dtype=torch.long)
    graph2 = Data(x=_x2, edge_index = _edge_index2)

    coloring2 = wl.compute_two_wl(graph = graph2)
    col_freq2 = wl.compute_color_frequencies(coloring = coloring2)
    histogram2 = wl.compute_single_graph_vector(color_frequencies = col_freq2)

    print('Graph1: ' + str(histogram1))
    print('Graph2: ' + str(histogram2))
