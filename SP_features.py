from torch_geometric.data import Data
import torch_geometric.utils
import networkx as nx
import numpy as np
from torch import Tensor
import torch
from typing import Dict
from typing import Tuple
from sklearn import datasets
import os
import os.path as osp
from typing import Optional, List


from torch_geometric.datasets import TUDataset
import os.path as osp
import developmentHelpers as helpers
import matplotlib.pyplot as plt


class SP_graph_features():

    def __init__(self, num_samples: int, label_alphabet: List[int], distances_alphabet: List[int], graph_sizes: Optional[List[int]] = None):
        super().__init__()
        self.label_alphabet = label_alphabet
        self.distances_alphabet = distances_alphabet
        self.graph_sizes = graph_sizes
        #editmask represents a mask of which entries have been edited by vectors in order to efficiently crop the final vectors. It is important that entries of this array are only ever set to true in order to prevent race conditions when parallelizing
        self.editmask = np.full(((len(label_alphabet)**2) * len(distances_alphabet)) + 2, False)
        #graph_id and vertex_id
        self.editmask[0] = True
        self.editmask[1] = True
        self.dataset = np.full((num_samples, ((len(label_alphabet)**2) * len(distances_alphabet)) + 2), 0)

    #helpers for indexing and editing of the database array
    def get_vertex_identifier_from_dataset_idx(self, dataset_idx: int) -> Tuple[int, int]:

        if self.graph_sizes is None:
            return tuple([0, dataset_idx])
        else:
            cur_graph_id = 0
            idx = dataset_idx
            for size in self.graph_sizes:
                if idx >= size:
                    idx -= size
                    cur_graph_id += 1
                else:
                    return tuple([cur_graph_id, idx])
                
    def get_dataset_idx_from_vertex_identifier(self, vertex_identifier: Tuple[int,int]) -> int:
        if self.graph_sizes is None:
            return vertex_identifier[1]
        else:            
            return sum(self.graph_sizes[:vertex_identifier[0]]) + vertex_identifier[1]

    #NOTE: this method requires a fully edited vector, meaning it has to already include the graph_id and vertex_id of the specific feature vector
    def edit_dataset_by_index(self, dataset_idx: int, vector: np.array):

        assert vector.shape == self.dataset[0,:].shape
        assert 0 <= dataset_idx and dataset_idx < self.dataset[:,0].shape[0]

        self.dataset[dataset_idx,:] = vector

    #helper for computing the path of a file from the local directory
    def get_file_from_local_by_path(self, root_path, filename):
        p = osp.join(osp.abspath(osp.dirname(__file__)), root_path)
        if not osp.exists(p):
            os.makedirs(p)

        p = osp.join(p, filename)
        
        if not osp.exists(p):
            open(p, 'w').close()

        return p

    #directly utilise the networkx implementation of floyd_warshall to construct a distance matrix between the vertices
    def floyd_warshall(self, graph: Data) -> np.array:
        return nx.floyd_warshall_numpy(torch_geometric.utils.to_networkx(graph))

    #generates a vector whose entries are tuples of form (l_1, l_2, d) where l_1, l_2 are labels and d is the distance. The value of each entry corresponds to the number of vertex pairs with matching labesl of shortest distance d
    #NOTE: this implementation is intended for undirected graphs and will not work with directed graphs. Thus it is assumed that the distances matrix is symmetric. 
    def sp_feature_map(self, distances: np.array, x: Tensor) -> Dict[Tuple[Tuple[int, int], int], int]:

        #if the features are not labels they are assumed to be a one-hot encoding
        if x.dim() > 1:
            assert (x.sum(dim=-1) == 1).sum() == x.size(0)
            x = x.argmax(dim=-1)  # one-hot -> integer.
        assert x.dtype == torch.long

        assert distances.shape == (x.size(0), x.size(0))

        _tupledict = {}

        #iterate over every entry of the distance matrix
        for i in range(x.size(0)):
            for j in range(i):
                _llist = [x[i].item(), x[j].item()]
                _llist.sort()
                _t = tuple([tuple(_llist), int(distances[i][j])])
                if _t not in _tupledict:
                    _tupledict[_t] = 1
                else:
                    _tupledict[_t] += 1

        return _tupledict

    #Takes a dictionary representing a vector to generate a feature vector (that is consistent between all feature vectors of the same alphabets). We utilise np arrays since further processing is done with clustering algorithms
    def sp_feature_vector_from_feature_map(self, dict: Dict[Tuple[Tuple[int, int], int], int], vertex_identifier: Tuple[int,int]) -> np.array:

        result = np.zeros((len(self.label_alphabet)**2) * len(self.distances_alphabet) + 2)
        #write the graph_id and vertex_id into the dataset array
        result[0] = vertex_identifier[0]
        result[1] = vertex_identifier[1]
        dataset_idx = self.get_dataset_idx_from_vertex_identifier(vertex_identifier = vertex_identifier)

        for i_x, x in enumerate(self.label_alphabet):
            for i_y, y in enumerate(self.label_alphabet):
                for i_d, d in enumerate(self.distances_alphabet):
                    id = i_x * len(self.label_alphabet) * len(self.distances_alphabet) + i_y * len(self.distances_alphabet) + i_d
                    key = tuple([tuple([x,y]),d])
                    if key in dict:
                        result[id + 2] = dict[key]
                        self.editmask[id] = True

        self.edit_dataset_by_index(dataset_idx = dataset_idx, vector = result)

        return result
    
    #NOTE: This function must only be called after joining parallelized threads to ensure correct function
    #crops feature vectors so that any column not modified in any feature vector is removed
    def crop_feature_vectors(self):
        self.dataset = self.dataset[:, self.editmask]
    
    def get_dataset(self) -> np.array:
        return self.dataset
    
    def write_dataset(self, path: str, comment: str):
        datasets.dump_svmlight_file(X = self.dataset, y = np.zeros(self.dataset[:,0].shape), f = path, comment = comment)
    


#TODO: Implement cropping etc
#computes the vertex sp feature map, that is result_(l,d) with label l and distance d is the number of vertices with label l and distance d to the vertex v
#NOTE: this implementation is intended for undirected graphs and will not work with directed graphs. Thus it is assumed that the distances matrix is symmetric. 
def vertex_sp_feature_map(distances: np.array, x: Tensor, vertex_id: int) -> Dict[Tuple[int, int], int]:

    #if the features are not labels they are assumed to be a one-hot encoding
    if x.dim() > 1:
        assert (x.sum(dim=-1) == 1).sum() == x.size(0)
        x = x.argmax(dim=-1)  # one-hot -> integer.
    assert x.dtype == torch.long
    assert distances.shape == (x.size(0), x.size(0))
    assert vertex_id in range(x.size(0))

    _dict = {}

    for i in range(x.size(0)):
        if i==vertex_id:
            continue

        _t = tuple([x[i].item(), int(distances[vertex_id, i])])
        if _t not in _dict:
            _dict[_t] = 1
        else:
            _dict[_t] += 1

    return _dict



#Test area
#_edge_index = torch.tensor([[0,1,1,2,3,3,4,4,1,1], [1,0,2,1,4,1,3,1,3,4]], dtype=torch.long)
#_x = torch.tensor([1,0,0,0,0], dtype=torch.long)

#_data = Data(x=_x, edge_index = _edge_index)

path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'TU')
dataset_mutag = TUDataset(root=path, name="MUTAG", use_node_attr=True)

_data = dataset_mutag.get(13)
_data2 = dataset_mutag.get(15)

_graph_sizes = [_data.x.shape[0], _data2.x.shape[0]]
print('Graph sizes: ' + str(_graph_sizes))

sp_feature_generator = SP_graph_features(num_samples = sum(_graph_sizes) , label_alphabet=range(3), distances_alphabet=range(1,12), graph_sizes = _graph_sizes)
#testing the indexing functions
a = sp_feature_generator.get_dataset_idx_from_vertex_identifier(tuple([1,4]))
print(a)
print(sp_feature_generator.get_vertex_identifier_from_dataset_idx(a))

_floyd_warshall_res = sp_feature_generator.floyd_warshall(_data)
_floyd_warshall_res2 = sp_feature_generator.floyd_warshall(_data2)

_sp_map = sp_feature_generator.sp_feature_map(distances=_floyd_warshall_res, x=_data.x)
_sp_map2 = sp_feature_generator.sp_feature_map(distances=_floyd_warshall_res2, x=_data2.x)

vector = sp_feature_generator.sp_feature_vector_from_feature_map(dict = _sp_map, vertex_identifier=tuple([0,1]))
print('Feature vector 1: ' + str(vector))
vector2 = sp_feature_generator.sp_feature_vector_from_feature_map(dict = _sp_map2, vertex_identifier=tuple([1,0]))
print('Feature vector 2: ' + str(vector2))

root_path = osp.join('data', 'SP_features')

print('Writing the uncropped dataset..')
sp_feature_generator.write_dataset(path = sp_feature_generator.get_file_from_local_by_path(root_path = root_path, filename = 'SP_features.svmlight'), comment='Uncropped SP features')
print('Writing the cropped dataset..')
sp_feature_generator.crop_feature_vectors()
print('Cropped Feature vector 1: ' + str(sp_feature_generator.get_dataset()[sp_feature_generator.get_dataset_idx_from_vertex_identifier(tuple([0,1])),:]))
print('Cropped Feature vector 2: ' + str(sp_feature_generator.get_dataset()[sp_feature_generator.get_dataset_idx_from_vertex_identifier(tuple([1,0])),:]))
sp_feature_generator.write_dataset(path=sp_feature_generator.get_file_from_local_by_path(root_path = root_path, filename = 'SP_features_cropped.svmlight'), comment='Cropped SP features')

#_vertex_sp_map = vertex_sp_feature_map(distances=_floyd_warshall_res, x=_data.x, vertex_id = 4)
#print(_vertex_sp_map)

helpers.drawGraph(_data, labels=_data.x)
helpers.drawGraph(_data2, labels=_data2.x, figure_count=2)
plt.show()