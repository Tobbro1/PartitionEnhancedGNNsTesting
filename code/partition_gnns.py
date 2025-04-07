# General
from typing import Optional, Tuple, Callable, Union

# Numpy
import numpy as np

# Pytorch
import torch
from torch import Tensor
import torch.nn.functional as F

# Pytorch geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import MLP, GINConv, GCNConv, global_add_pool
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.logging import log
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor, OptTensor

# System
import os.path as osp
import os
import time
import random

# Import constants
import constants
from CSL_dataset import CSL_Dataset

# The pytorch geometric GIN conv edited to serve our purposes (concretely, we pass an bool mask determining which indices we need to compute in a given convolusion)
# We note that this method still computes the message passing for every vertex in this step due to the computation being done in matrix format. However, we only compute the new feature vector for selected indices (including the application of the MLP)
class Partition_enhanced_GIN_conv(GINConv):
     
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = False, **kwargs):
        super().__init__(nn, eps, train_eps, **kwargs)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        mask: Tensor,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        x_r = x[1][mask,:]
        if x_r is not None:
            out[mask,:] = out[mask,:] + (1 + self.eps) * x_r

        return self.nn(out[mask,:])

class Partition_enhanced_GIN(torch.nn.Module):

    # In the GIN model the trainable parameters are represented by the MLPs applied after each aggregation, since the aggregation is a simple sum

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_clusters):
        super().__init__()

        assert hidden_channels >= in_channels

        self.num_clusters = num_clusters
        self.num_layers = num_layers

        self.hidden_channels = hidden_channels
        self.hidden_in_channel_diff = hidden_channels - in_channels

        self.device = constants.device

        self.convs = torch.nn.ModuleList()

        # properties of the parameters per layer and in total
        self.parameter_props = {}
        self.parameter_props["total"] = {}
        self.parameter_props["total"]["num"] = -1
        self.parameter_props["total"]["dtype"] = ""
        self.parameter_props["total"]["size"] = -1
        self.parameter_props["layer"] = {}

        sum_p = 0
        sum_p_size = 0

        for l in range(self.num_layers):
            convs_p = []
            for _ in range(self.num_clusters):
                # 2-layer MLP
                mlp = MLP([in_channels, hidden_channels, hidden_channels], act = 'relu')
                conv = Partition_enhanced_GIN_conv(nn = mlp, train_eps = False)
                convs_p.append(conv)
                self.convs.append(conv)

            # Saving parameter properties
            sum_p_layer = 0
            sum_p_size_layer = 0
            dtype = None
            for conv in convs_p:
                for p in conv.parameters():
                    sum_p_layer += p.numel()
                    sum_p_size_layer += p.nbytes
                    dtype = str(p.dtype)

            sum_p += sum_p_layer
            sum_p_size += sum_p_size_layer

            self.parameter_props["layer"][l] = {}
            self.parameter_props["layer"][l]["num"] = sum_p_layer
            self.parameter_props["layer"][l]["dtype"] = dtype
            self.parameter_props["layer"][l]["size"] = sum_p_size_layer

            self.parameter_props["total"]["dtype"] = dtype

            in_channels = hidden_channels

        self.parameter_props["total"]["num"] = sum_p
        self.parameter_props["total"]["size"] = sum_p_size

        # graph level pooling
        self.pool_mlp = MLP([hidden_channels * self.num_layers, hidden_channels, out_channels], act = 'relu')

        sum_p_layer = 0
        sum_p_size_layer = 0
        for p in self.pool_mlp.parameters():
            sum_p_layer += p.numel()
            sum_p_size_layer += p.nbytes
            dtype = str(p.dtype)
        self.parameter_props["layer"]["pool"] = {}
        self.parameter_props["layer"]["pool"]["num"] = sum_p_layer
        self.parameter_props["layer"]["pool"]["dtype"] = dtype
        self.parameter_props["layer"]["pool"]["size"] = sum_p_size_layer

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    # vertex_idx_start represents the first idx of clustering_labels which corresponds to a vertex of the graph data
    def forward(self, data: Data):
        x = torch.cat((data.x[:,1:], torch.zeros((data.x.size()[0], self.hidden_in_channel_diff)).to(self.device)), dim = 1)
        clustering_labels = data.x[:,0]
        edge_index = data.edge_index
        batch = data.batch

        num_graphs_in_batch = torch.unique(batch).size()[0]

        # We need to store the sum of all features after each layer in order to compute the graph level readout
        layer_global_add_pool_res = torch.empty((self.num_layers, num_graphs_in_batch, self.hidden_channels), device = self.device)

        # We edit only the subset
        for t in range(self.num_layers):
            for c in range(self.num_clusters):
                conv_idx = t * self.num_clusters + c

                # To implement: x[id,:] = convs[conv_idx](x[id,:]) iff clustering_labels[id] == j
                mask = (clustering_labels == c).view(-1)

                # This distinction is necessary. We cannot simply override x after a single convolution since we apply different convolutions within the same layers (Including the initial layer where the dimensionality is increased to the value of hidden_channels).
                # That means, we need to store intermediary results, meaning we artificially enlarge the feature dimension of x in the first step, and we have to account for the artificially large dimension in the first step layer.
                if t == 0:
                    x[mask,:] = self.convs[conv_idx](x[:,:(x.size()[1] - self.hidden_in_channel_diff)], edge_index, mask)
                else:
                    x[mask,:] = self.convs[conv_idx](x, edge_index, mask)
                # x = self.convs[conv_idx](torch.masked_select(x, mask), edge_index)

            # store pooling res
            # global_add_pool result shape is (num_unique_graphs_in_batch, feature_dim)
            layer_global_add_pool_res[t,:,:] = global_add_pool(x, batch)


        # graph level readout

        # concatenate the global_add_pools
        x = layer_global_add_pool_res[0,:,:]
        for t in range(1, self.num_layers):
            x = torch.cat((x, layer_global_add_pool_res[t,:,:]), dim = 1)

        return self.pool_mlp(x)
    
    

# classic GIN
class GIN_Classic(torch.nn.Module):
    # In the GIN model the trainable parameters are represented by the MLPs applied after each aggregation, since the aggregation is a simple sum

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        assert hidden_channels >= in_channels

        self.num_layers = num_layers

        self.hidden_channels = hidden_channels

        self.device = constants.device

        self.convs = torch.nn.ModuleList()

        # properties of the parameters per layer and in total
        self.parameter_props = {}
        self.parameter_props["total"] = {}
        self.parameter_props["total"]["num"] = -1
        self.parameter_props["total"]["dtype"] = ""
        self.parameter_props["total"]["size"] = -1
        self.parameter_props["layer"] = {}

        sum_p = 0
        sum_p_size = 0

        for l in range(self.num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels], act = 'relu')
            conv = GINConv(nn = mlp, train_eps = False)
            self.convs.append(conv)

            # Saving parameter properties
            sum_p_layer = 0
            sum_p_size_layer = 0
            dtype = None
            for p in conv.parameters():
                sum_p_layer += p.numel()
                sum_p_size_layer += p.nbytes
                dtype = str(p.dtype)
            sum_p += sum_p_layer
            sum_p_size += sum_p_size_layer

            self.parameter_props["layer"][l] = {}
            self.parameter_props["layer"][l]["num"] = sum_p_layer
            self.parameter_props["layer"][l]["dtype"] = dtype
            self.parameter_props["layer"][l]["size"] = sum_p_size_layer

            self.parameter_props["total"]["dtype"] = dtype

            in_channels = hidden_channels

        self.parameter_props["total"]["num"] = sum_p
        self.parameter_props["total"]["size"] = sum_p_size

        # graph level pooling
        self.pool_mlp = MLP([hidden_channels * self.num_layers, hidden_channels, out_channels], act = 'relu')

        sum_p_layer = 0
        sum_p_size_layer = 0
        for p in self.pool_mlp.parameters():
            sum_p_layer += p.numel()
            sum_p_size_layer += p.nbytes
            dtype = str(p.dtype)
        self.parameter_props["layer"]["pool"] = {}
        self.parameter_props["layer"]["pool"]["num"] = sum_p_layer
        self.parameter_props["layer"]["pool"]["dtype"] = dtype
        self.parameter_props["layer"]["pool"]["size"] = sum_p_size_layer

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    # vertex_idx_start represents the first idx of clustering_labels which corresponds to a vertex of the graph data
    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        num_graphs_in_batch = torch.unique(batch).size()[0]

        # We need to store the sum of all features after each layer in order to compute the graph level readout
        layer_global_add_pool_res = torch.empty((self.num_layers, num_graphs_in_batch, self.hidden_channels), device = self.device)

        # We edit only the subset
        for conv in self.convs:
            
            x = conv(x, edge_index)

            # store pooling res
            # global_add_pool result shape is (num_unique_graphs_in_batch, feature_dim)
            layer_global_add_pool_res[t,:,:] = global_add_pool(x, batch)


        # graph level readout

        # concatenate the global_add_pools
        x = layer_global_add_pool_res[0,:,:]
        for t in range(1, self.num_layers):
            x = torch.cat((x, layer_global_add_pool_res[t,:,:]), dim = 1)

        return self.pool_mlp(x)

class Partition_enhanced_GCN_conv(GCNConv):
    def __init__(self, in_channels, out_channels, improved, cached, normalize, bias, **kwargs):
        super().__init__(in_channels = in_channels, out_channels = out_channels, cached = cached, improved = improved, normalize = normalize, bias = bias, **kwargs)


    def forward(self, x: Tensor, edge_index: Adj, mask,
            edge_weight: OptTensor = None) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                            f"of node features as input while this layer "
                            f"does not support bipartite message passing. "
                            f"Please try other layers such as 'SAGEConv' or "
                            f"'GraphConv' instead")

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        x = self.lin(x)

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight)

        if self.bias is not None:
            out = out + self.bias

        return out[mask,:]

class Partition_enhanced_GCN(torch.nn.Module):

    # In the GCN model the trainable parameters are represented by the parameter matrices within the convolutions

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_clusters, normalize: bool = True, bias: bool = True):
        super().__init__()

        assert hidden_channels >= in_channels

        self.num_clusters = num_clusters
        self.num_layers = num_layers

        self.hidden_channels = hidden_channels
        self.hidden_in_channel_diff = hidden_channels - in_channels

        self.device = constants.device

        self.convs = torch.nn.ModuleList()

        # properties of the parameters per layer and in total
        self.parameter_props = {}
        self.parameter_props["total"] = {}
        self.parameter_props["total"]["num"] = -1
        self.parameter_props["total"]["dtype"] = ""
        self.parameter_props["total"]["size"] = -1
        self.parameter_props["layer"] = {}

        sum_p = 0
        sum_p_size = 0

        for l in range(self.num_layers):
            convs_p = []
            for _ in range(self.num_clusters):
                conv = Partition_enhanced_GCN_conv(in_channels = in_channels, out_channels = hidden_channels, improved = False, cached = False, normalize = normalize, bias = bias)
                convs_p.append(conv)
                self.convs.append(conv)

            # Saving parameter properties
            sum_p_layer = 0
            sum_p_size_layer = 0
            dtype = None
            for conv in convs_p:
                for p in conv.parameters():
                    sum_p_layer += p.numel()
                    sum_p_size_layer += p.nbytes
                    dtype = str(p.dtype)

            sum_p += sum_p_layer
            sum_p_size += sum_p_size_layer

            self.parameter_props["layer"][l] = {}
            self.parameter_props["layer"][l]["num"] = sum_p_layer
            self.parameter_props["layer"][l]["dtype"] = dtype
            self.parameter_props["layer"][l]["size"] = sum_p_size_layer

            self.parameter_props["total"]["dtype"] = dtype

            in_channels = hidden_channels

        self.parameter_props["total"]["num"] = sum_p
        self.parameter_props["total"]["size"] = sum_p_size

        # graph level pooling
        self.pool_mlp = MLP([hidden_channels * self.num_layers, hidden_channels, out_channels], act = 'relu')

        sum_p_layer = 0
        sum_p_size_layer = 0
        for p in self.pool_mlp.parameters():
            sum_p_layer += p.numel()
            sum_p_size_layer += p.nbytes
            dtype = str(p.dtype)
        self.parameter_props["layer"]["pool"] = {}
        self.parameter_props["layer"]["pool"]["num"] = sum_p_layer
        self.parameter_props["layer"]["pool"]["dtype"] = dtype
        self.parameter_props["layer"]["pool"]["size"] = sum_p_size_layer

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    # vertex_idx_start represents the first idx of clustering_labels which corresponds to a vertex of the graph data
    def forward(self, data: Data):
        x = torch.cat((data.x[:,1:], torch.zeros((list(data.x.size())[0], self.hidden_in_channel_diff)).to(self.device)), dim = 1)
        clustering_labels = data.x[:,0]
        edge_index = data.edge_index
        batch = data.batch

        num_graphs_in_batch = torch.unique(batch).size()[0]

        # We need to store the sum of all features after each layer in order to compute the graph level readout
        layer_global_add_pool_res = torch.empty((self.num_layers, num_graphs_in_batch, self.hidden_channels), device = self.device)

        # We edit only the subset
        for t in range(self.num_layers):
            # remove in place operations
            x2 = torch.zeros(size = x.size(), device = self.device)

            for c in range(self.num_clusters):
                conv_idx = t * self.num_clusters + c

                # To implement: x[id,:] = convs[conv_idx](x[id,:]) iff clustering_labels[id] == j
                mask = (clustering_labels == c).view(-1)

                # This distinction is necessary. We cannot simply override x after a single convolution since we apply different convolutions within the same layers (Including the initial layer where the dimensionality is increased to the value of hidden_channels).
                # That means, we need to store intermediary results, meaning we artificially enlarge the feature dimension of x in the first step, and we have to account for the artificially large dimension in the first step layer.
                # We note that the current implementation computes the whole GCN layer for each datapoint irrespective of whether it is later utilised. The computation is mostly a linear transform and needs to be computed before aggregation, meaning it is likely inefficient to try to speed this step up.
                if t == 0:
                    x2[mask,:] = self.convs[conv_idx](x[:,:(list(x.size())[1] - self.hidden_in_channel_diff)], edge_index, mask)
                else:
                    x2[mask,:] = self.convs[conv_idx](x, edge_index, mask)
                
            x = x2

            # store pooling res
            # global_add_pool result shape is (num_unique_graphs_in_batch, feature_dim)
            layer_global_add_pool_res[t,:,:] = global_add_pool(x, batch)

        # concatenate the global_add_pools
        x = layer_global_add_pool_res[0,:,:]
        for t in range(1, self.num_layers):
            x = torch.cat((x, layer_global_add_pool_res[t,:,:]), dim = 1)

        return self.pool_mlp(x)

# classic GCN
class GCN_Classic(torch.nn.Module):
    # In the GIN model the trainable parameters are represented by the MLPs applied after each aggregation, since the aggregation is a simple sum

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, normalize: bool = True, bias: bool = True):
        super().__init__()

        assert hidden_channels >= in_channels

        self.num_layers = num_layers

        self.hidden_channels = hidden_channels

        self.device = constants.device

        self.convs = torch.nn.ModuleList()

        # properties of the parameters per layer and in total
        self.parameter_props = {}
        self.parameter_props["total"] = {}
        self.parameter_props["total"]["num"] = -1
        self.parameter_props["total"]["dtype"] = ""
        self.parameter_props["total"]["size"] = -1
        self.parameter_props["layer"] = {}

        sum_p = 0
        sum_p_size = 0

        for l in range(self.num_layers):
            conv = GCNConv(in_channels = in_channels, out_channels = hidden_channels, improved = False, cached = False, normalize = normalize, bias = bias)
            self.convs.append(conv)

            # Saving parameter properties
            sum_p_layer = 0
            sum_p_size_layer = 0
            dtype = None
            for p in conv.parameters():
                sum_p_layer += p.numel()
                sum_p_size_layer += p.nbytes
                dtype = str(p.dtype)
            sum_p += sum_p_layer
            sum_p_size += sum_p_size_layer

            self.parameter_props["layer"][l] = {}
            self.parameter_props["layer"][l]["num"] = sum_p_layer
            self.parameter_props["layer"][l]["dtype"] = dtype
            self.parameter_props["layer"][l]["size"] = sum_p_size_layer

            self.parameter_props["total"]["dtype"] = dtype

            in_channels = hidden_channels

        self.parameter_props["total"]["num"] = sum_p
        self.parameter_props["total"]["size"] = sum_p_size

        # graph level pooling
        self.pool_mlp = MLP([hidden_channels * self.num_layers, hidden_channels, out_channels], act = 'relu')

        sum_p_layer = 0
        sum_p_size_layer = 0
        for p in self.pool_mlp.parameters():
            sum_p_layer += p.numel()
            sum_p_size_layer += p.nbytes
            dtype = str(p.dtype)
        self.parameter_props["layer"]["pool"] = {}
        self.parameter_props["layer"]["pool"]["num"] = sum_p_layer
        self.parameter_props["layer"]["pool"]["dtype"] = dtype
        self.parameter_props["layer"]["pool"]["size"] = sum_p_size_layer

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    
    # vertex_idx_start represents the first idx of clustering_labels which corresponds to a vertex of the graph data
    def forward(self, data: Data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        num_graphs_in_batch = torch.unique(batch).size()[0]

        # We need to store the sum of all features after each layer in order to compute the graph level readout
        layer_global_add_pool_res = torch.empty((self.num_layers, num_graphs_in_batch, self.hidden_channels), device = self.device)

        # We edit only the subset
        for conv in self.convs:
            
            x = conv(x, edge_index)

            # store pooling res
            # global_add_pool result shape is (num_unique_graphs_in_batch, feature_dim)
            layer_global_add_pool_res[t,:,:] = global_add_pool(x, batch)


        # graph level readout

        # concatenate the global_add_pools
        x = layer_global_add_pool_res[0,:,:]
        for t in range(1, self.num_layers):
            x = torch.cat((x, layer_global_add_pool_res[t,:,:]), dim = 1)

        return self.pool_mlp(x)

# A manager class for GNNs and partition enhanced GNNs
class GNN_Manager():

    def __init__(self):
        super().__init__()

        self.clustering_labels = None
        self.dataset = None
        self.num_features = -1
        self.num_classes = -1

        self.device = constants.device

        self.model = None
        self.optimizer = None

        # Store information about the managed model
        # device
        # optimizer: desc, lr
        # gnn: model desc, model_id (0 -> ENH_GIN, 1 -> CLASSIC_GIN, 2 -> ENH_GCN, 3 -> CLASSIC_GCN)
        #      config: in_channels, hidden_channels, out_channels, num_layers
        #      parameters: num_parameters, parameters dtype, parameters size
        #                  per layer num_parameters, parameters dtype, parameters size
        self.metadata = {}
        self.metadata["device"] = self.device.type
        self.metadata["optimizer"] = {}
        self.metadata["optimizer"]["desc"] = ""
        self.metadata["optimizer"]["lr"] = -1.0
        self.metadata["gnn"] = {}
        self.metadata["gnn"]["desc"] = ""
        self.metadata["gnn"]["model_id"] = -1
        self.metadata["gnn"]["config"] = {}
        self.metadata["gnn"]["config"]["num_layers"] = -1
        self.metadata["gnn"]["config"]["in_channels"] = -1
        self.metadata["gnn"]["config"]["hidden_channels"] = -1
        self.metadata["gnn"]["config"]["out_channels"] = -1
        self.metadata["gnn"]["parameters"] = {}
        self.metadata["gnn"]["parameters"]["num_total"] = -1
        self.metadata["gnn"]["parameters"]["dtype"] = ""
        self.metadata["gnn"]["parameters"]["size"] = -1

    def reset_parameters(self):
        self.device = constants.device
        self.metadata["device"] = self.device.type
        self.model = None
        self.optimizer = None
        self.metadata["optimizer"] = {}
        self.metadata["optimizer"]["desc"] = ""
        self.metadata["optimizer"]["lr"] = -1.0
        self.metadata["gnn"] = {}
        self.metadata["gnn"]["desc"] = ""
        self.metadata["gnn"]["model_id"] = -1
        self.metadata["gnn"]["config"] = {}
        self.metadata["gnn"]["config"]["num_layers"] = -1
        self.metadata["gnn"]["config"]["in_channels"] = -1
        self.metadata["gnn"]["config"]["hidden_channels"] = -1
        self.metadata["gnn"]["config"]["out_channels"] = -1
        self.metadata["gnn"]["parameters"] = {}
        self.metadata["gnn"]["parameters"]["num_total"] = -1
        self.metadata["gnn"]["parameters"]["dtype"] = ""
        self.metadata["gnn"]["parameters"]["size"] = -1

        
    # # Read clustering results from disk
    # def read_clustering_labels(self, path: str):
    #     self.clustering_labels = torch.from_numpy(np.loadtxt(fname = path, dtype = np.int32, comments = '#')).view(-1,1)

    # We need to modify the dataset such that each entry includes the original vertex_identifier to uniquely identify it later
    def load_dataset(self, dataset: Dataset, num_clusters: int):

        self.dataset = dataset
        self.num_features = self.dataset.num_features
        # If we include the cluster_id, num_features is increased by one
        if num_clusters > 0:
            self.num_features -= 1
        self.num_clusters = num_clusters
        self.num_classes = self.dataset.num_classes

    def generate_partition_enhanced_GIN_model(self, hidden_channels: int, num_layers: int, lr: float) -> None:
        assert self.dataset is not None

        self.model = Partition_enhanced_GIN(
            in_channels = self.num_features,
            hidden_channels = hidden_channels,
            out_channels =  self.num_classes,
            num_layers = num_layers,
            num_clusters = self.num_clusters
        ).to(self.device)

        self.metadata["gnn"]["desc"] = "enhanced_gin"
        self.metadata["gnn"]["model_id"] = 0
        self.metadata["gnn"]["config"]["num_layers"] = num_layers
        self.metadata["gnn"]["config"]["in_channels"] = self.num_features
        self.metadata["gnn"]["config"]["hidden_channels"] = hidden_channels
        self.metadata["gnn"]["config"]["out_channels"] = self.num_classes
        self.metadata["gnn"]["parameters"] = self.model.parameter_props

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.metadata["optimizer"]["desc"] = "adam"
        self.metadata["optimizer"]["lr"] = lr

    def generate_classic_GIN_model(self, hidden_channels: int, num_layers: int, lr: float) -> None:
        assert self.dataset is not None

        self.model = GIN_Classic(
            in_channels = self.num_features,
            hidden_channels = hidden_channels,
            out_channels =  self.num_classes,
            num_layers = num_layers,
        ).to(self.device)

        self.metadata["gnn"]["desc"] = "classic_gin"
        self.metadata["gnn"]["model_id"] = 1
        self.metadata["gnn"]["config"]["num_layers"] = num_layers
        self.metadata["gnn"]["config"]["in_channels"] = self.num_features
        self.metadata["gnn"]["config"]["hidden_channels"] = hidden_channels
        self.metadata["gnn"]["config"]["out_channels"] = self.num_classes
        self.metadata["gnn"]["parameters"] = self.model.parameter_props

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.metadata["optimizer"]["desc"] = "adam"
        self.metadata["optimizer"]["lr"] = lr

    def generate_partition_enhanced_GCN_model(self, hidden_channels: int, num_layers: int, lr: float) -> None:
        assert self.dataset is not None

        num_clusters = torch.unique(self.clustering_labels).size(dim = 0)

        self.model = Partition_enhanced_GCN(
            in_channels = self.num_features,
            hidden_channels = hidden_channels,
            out_channels =  self.num_classes,
            num_layers = num_layers,
            num_clusters = num_clusters
        ).to(self.device)

        self.metadata["gnn"]["desc"] = "enhanced_gcn"
        self.metadata["gnn"]["model_id"] = 2
        self.metadata["gnn"]["config"]["num_layers"] = num_layers
        self.metadata["gnn"]["config"]["in_channels"] = self.num_features
        self.metadata["gnn"]["config"]["hidden_channels"] = hidden_channels
        self.metadata["gnn"]["config"]["out_channels"] = self.num_classes
        self.metadata["gnn"]["config"]["normalize"] = True
        self.metadata["gnn"]["config"]["use_bias"] = True
        self.metadata["gnn"]["parameters"] = self.model.parameter_props

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.metadata["optimizer"]["desc"] = "adam"
        self.metadata["optimizer"]["lr"] = lr

    def generate_classic_GCN_model(self, hidden_channels: int, num_layers: int, lr: float) -> None:
        assert self.dataset is not None

        self.model = GCN_Classic(
            in_channels = self.num_features,
            hidden_channels = hidden_channels,
            out_channels =  self.num_classes,
            num_layers = num_layers,
        ).to(self.device)

        self.metadata["gnn"]["desc"] = "classic_gcn"
        self.metadata["gnn"]["model_id"] = 3
        self.metadata["gnn"]["config"]["num_layers"] = num_layers
        self.metadata["gnn"]["config"]["in_channels"] = self.num_features
        self.metadata["gnn"]["config"]["hidden_channels"] = hidden_channels
        self.metadata["gnn"]["config"]["out_channels"] = self.num_classes
        self.metadata["gnn"]["config"]["normalize"] = True
        self.metadata["gnn"]["config"]["use_bias"] = True
        self.metadata["gnn"]["parameters"] = self.model.parameter_props

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.metadata["optimizer"]["desc"] = "adam"
        self.metadata["optimizer"]["lr"] = lr

    def get_metadata(self):
        return self.metadata

def get_data_loaders(gnn: GNN_Manager, batch_size: int, shuffle: bool, training_ratio: float, test_ratio: float) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    assert gnn.dataset is not None

    if training_ratio + test_ratio < 1:
        train_loader = DataLoader(dataset = gnn.dataset[:training_ratio], batch_size = batch_size, shuffle = shuffle)
        test_loader = DataLoader(dataset = gnn.dataset[training_ratio:training_ratio+test_ratio], batch_size = batch_size)
        valid_loader = DataLoader(dataset = gnn.dataset[training_ratio+test_ratio:], batch_size = batch_size)

        return train_loader, test_loader, valid_loader
    else:
        train_loader = DataLoader(dataset = gnn.dataset[:training_ratio], batch_size = batch_size, shuffle = shuffle)
        test_loader = DataLoader(dataset = gnn.dataset[training_ratio:], batch_size = batch_size)

        return train_loader, test_loader, None

def train_partition_enhanced_GNN(gnn: GNN_Manager, train_loader: DataLoader):
    device = constants.device
    
    gnn.model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        gnn.optimizer.zero_grad()
        out = gnn.model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        gnn.optimizer.step()
        total_loss += float(loss) * data.num_graphs

    return total_loss/len(train_loader.dataset)

@torch.no_grad()
def test_partition_enhanced_GNN(gnn: GNN_Manager, test_loader: DataLoader):
    device = constants.device

    gnn.model.eval()

    total_correct = 0
    for data in test_loader:
        data = data.to(device)
        out = gnn.model(data)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())

    return total_correct/len(test_loader.dataset)

#Test zone
if __name__ == '__main__':

    # reproducability
    constants.initialize_random_seeds()

    torch.autograd.set_detect_anomaly(True)

    gnn_generator = GNN_Manager()

    root_path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir)
    csl_path = osp.join('data', 'CSL', 'CSL_dataset')
    dataset_csl = CSL_Dataset(root = osp.join(root_path, csl_path))

    gnn_generator.load_dataset(dataset_csl, 0)

    gnn_generator.generate_classic_GIN_model(hidden_channels = 32, num_layers = 3, lr = 0.1)

    params = gnn_generator.model.parameters()

    for param in params:
        print(type(param), param.size())

    # path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'TU')
    # mutag_path = osp.join(path, "MUTAG")
    # dd_path = osp.join(path, "DD")
    # mutag_dataset_filename = "k_disk_SP_features_MUTAG.svmlight"
    # dd_dataset_filename = "k_disk_SP_features_DD.svmlight"
    # dataset_path = osp.join(path, mutag_path, mutag_dataset_filename)
    # # clusterlabels_path = osp.join(path, mutag_path, 'cluster_labels.txt')
    # clusterlabels_path = osp.join(path, mutag_path, 'cluster_labels_zero.txt')

    # gnn_generator.read_clustering_labels(path = clusterlabels_path)
    # # NOTE: force_reload needs to be set to True in order to update the clustering information in the dataset. When re-running the program with identical settings it may be left False. Otherwise read_clustering_labels does not apply the update leading to potentially incorrect, thus unpredictable behaviour
    # gnn_generator.load_dataset(root_path=osp.join(path, 'enhanced_gnn'), dataset_name = 'MUTAG', force_reload = True)

    # batch_size = 128
    # hidden_channels = 32
    # num_layers = 3
    # lr = 0.01
    # epochs = 100

    # # Test the GIN
    # print("Testing GIN: ")

    # gnn_generator.generate_partition_enhanced_GIN_model(hidden_channels = hidden_channels, num_layers = num_layers, lr = lr)

    # train_loader, test_loader, valid_loader = get_data_loaders(gnn_generator, batch_size = batch_size, shuffle = True, training_ratio = 0.9, test_ratio = 0.1)

    # times = []
    # for epoch in range(1, epochs + 1):
    #     start = time.time()
    #     loss = train_partition_enhanced_GNN(gnn = gnn_generator, train_loader = train_loader)
    #     train_acc = test_partition_enhanced_GNN(gnn = gnn_generator, test_loader = train_loader)
    #     test_acc = test_partition_enhanced_GNN(gnn = gnn_generator, test_loader = test_loader)
    #     log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
    #     times.append(time.time() - start)
    # print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')

    # # Test the GCN
    # print("Testing GCN: ")

    # gnn_generator.reset_parameters()
    # gnn_generator.generate_partition_enhanced_GCN_model(hidden_channels = hidden_channels, num_layers = num_layers, lr = lr)

    # times = []
    # for epoch in range(1, epochs + 1):
    #     start = time.time()
    #     loss = train_partition_enhanced_GNN(gnn = gnn_generator, train_loader = train_loader)
    #     train_acc = test_partition_enhanced_GNN(gnn = gnn_generator, test_loader = train_loader)
    #     test_acc = test_partition_enhanced_GNN(gnn = gnn_generator, test_loader = test_loader)
    #     log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
    #     times.append(time.time() - start)
    # print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')