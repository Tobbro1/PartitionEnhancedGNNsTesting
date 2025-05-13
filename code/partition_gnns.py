# General
from typing import Optional, Tuple, Callable, Union
from copy import deepcopy

# Numpy
import numpy as np

# Pytorch
import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn.norm import BatchNorm

# Pytorch geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import MLP, GINConv, GCNConv, global_add_pool
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.logging import log
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor, OptTensor
from torch_geometric.utils import spmm, to_dense_adj, to_torch_csr_tensor
from torch_geometric.nn.dense.linear import Linear

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

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        out = self.nn(out)

        return out[mask,:]

class Partition_enhanced_GIN(torch.nn.Module):

    # In the GIN model the trainable parameters are represented by the MLPs applied after each aggregation, since the aggregation is a simple sum

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_clusters, use_batch_norm: bool):
        super().__init__()

        assert hidden_channels >= in_channels

        self.num_clusters = num_clusters
        self.num_layers = num_layers

        self.hidden_channels = hidden_channels
        self.in_channels = in_channels

        self.use_batch_norm = use_batch_norm

        self.device = constants.device

        self.convs = torch.nn.ModuleList()
        self.norm_convs = torch.nn.ModuleList()

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

                if self.use_batch_norm:
                    batch_norm = BatchNorm(in_channels = hidden_channels, allow_single_element = True)
                    self.norm_convs.append(batch_norm)
                    convs_p.append(batch_norm)

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
        
        for norm in self.norm_convs:
            norm.reset_parameters()
    
    # vertex_idx_start represents the first idx of clustering_labels which corresponds to a vertex of the graph data
    def forward(self, data: Data):
        x = torch.cat((data.x[:,1:], torch.zeros((data.x.size()[0], (self.hidden_channels - self.in_channels))).to(self.device)), dim = 1).to(dtype = torch.float32)
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
                    x[mask,:] = self.convs[conv_idx](x[:,:(x.size()[1] - (self.hidden_channels - self.in_channels))], edge_index, mask)
                else:
                    x[mask,:] = self.convs[conv_idx](x, edge_index, mask)
                # x = self.convs[conv_idx](torch.masked_select(x, mask), edge_index)

                if self.use_batch_norm:
                    x[mask,:] = self.norm_convs[conv_idx](x[mask,:])

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

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, use_batch_norm: bool):
        super().__init__()

        assert hidden_channels >= in_channels

        self.num_layers = num_layers

        self.hidden_channels = hidden_channels

        self.use_batch_norm = use_batch_norm

        self.device = constants.device

        self.convs = torch.nn.ModuleList()
        self.norm_convs = torch.nn.ModuleList()

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
            conv_p = []

            mlp = MLP([in_channels, hidden_channels, hidden_channels], act = 'relu')
            conv = GINConv(nn = mlp, train_eps = False)
            conv.node_dim = 0
            self.convs.append(conv)
            conv_p.append(conv)

            if self.use_batch_norm:
                batch_norm = BatchNorm(in_channels = hidden_channels)
                self.norm_convs.append(batch_norm)
                conv_p.append(batch_norm)

            # Saving parameter properties
            sum_p_layer = 0
            sum_p_size_layer = 0
            dtype = None
            for conv in conv_p:
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

        for conv in self.norm_convs:
            conv.reset_parameters()
    
    # vertex_idx_start represents the first idx of clustering_labels which corresponds to a vertex of the graph data
    def forward(self, data: Data):
        x = data.x.to(dtype = torch.float32, copy = True)
        edge_index = data.edge_index
        batch = data.batch

        num_graphs_in_batch = torch.unique(batch).size()[0]

        # We need to store the sum of all features after each layer in order to compute the graph level readout
        layer_global_add_pool_res = torch.empty((self.num_layers, num_graphs_in_batch, self.hidden_channels), device = self.device)

        # We edit only the subset
        for t, conv in enumerate(self.convs):
            
            x = conv(x, edge_index)

            if self.use_batch_norm:
                x = self.norm_convs[t](x)

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

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_clusters, use_batch_norm: bool, normalize: bool = True, bias: bool = False):
        super().__init__()

        assert hidden_channels >= in_channels

        self.num_clusters = num_clusters
        self.num_layers = num_layers

        self.hidden_channels = hidden_channels
        self.hidden_in_channel_diff = hidden_channels - in_channels

        self.use_batch_norm = use_batch_norm

        self.device = constants.device

        self.convs = torch.nn.ModuleList()
        self.norm_convs = torch.nn.ModuleList()

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

                if self.use_batch_norm:
                    batch_norm = BatchNorm(in_channels = hidden_channels, allow_single_element = True)
                    self.norm_convs.append(batch_norm)
                    convs_p.append(batch_norm)

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

        for conv in self.norm_convs:
            conv.reset_parameters()
    
    # vertex_idx_start represents the first idx of clustering_labels which corresponds to a vertex of the graph data
    def forward(self, data: Data):
        x = torch.cat((data.x[:,1:], torch.zeros((list(data.x.size())[0], self.hidden_in_channel_diff)).to(self.device)), dim = 1).to(dtype = torch.float32)
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
                
                # Apply ReLU
                x2[mask,:] = F.relu(x2[mask,:])

                if self.use_batch_norm:
                    x2[mask,:] = self.norm_convs[conv_idx](x2[mask,:])

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

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, use_batch_norm: bool, normalize: bool = True, bias: bool = True):
        super().__init__()

        assert hidden_channels >= in_channels

        self.num_layers = num_layers

        self.hidden_channels = hidden_channels
        self.use_batch_norm = use_batch_norm

        if not normalize:
            add_self_loops = True
        else:
            add_self_loops = False

        self.device = constants.device

        self.convs = torch.nn.ModuleList()
        self.norm_convs = torch.nn.ModuleList()

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

            conv = GCNConv(in_channels = in_channels, out_channels = hidden_channels, add_self_loops = add_self_loops, improved = False, cached = False, normalize = normalize, bias = bias)
            self.convs.append(conv)
            convs_p.append(conv)

            if self.use_batch_norm:
                batch_norm = BatchNorm(in_channels = hidden_channels)
                self.norm_convs.append(batch_norm)
                convs_p.append(batch_norm)

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

        for conv in self.norm_convs:
            conv.reset_parameters()
    
    # vertex_idx_start represents the first idx of clustering_labels which corresponds to a vertex of the graph data
    def forward(self, data: Data):
        x = data.x.to(dtype = torch.float32, copy = True)
        edge_index = data.edge_index
        batch = data.batch

        num_graphs_in_batch = torch.unique(batch).size()[0]

        # We need to store the sum of all features after each layer in order to compute the graph level readout
        layer_global_add_pool_res = torch.empty((self.num_layers, num_graphs_in_batch, self.hidden_channels), device = self.device)

        # We edit only the subset
        for t, conv in enumerate(self.convs):
            
            x = conv(x, edge_index)

            # Apply ReLU
            x = F.relu(x)

            if self.use_batch_norm:
                x = self.norm_convs[t](x)

            # store pooling res
            # global_add_pool result shape is (num_unique_graphs_in_batch, feature_dim)
            layer_global_add_pool_res[t,:,:] = global_add_pool(x, batch)


        # graph level readout

        # concatenate the global_add_pools
        x = layer_global_add_pool_res[0,:,:]
        for t in range(1, self.num_layers):
            x = torch.cat((x, layer_global_add_pool_res[t,:,:]), dim = 1)

        return self.pool_mlp(x)

class GPNNEdgeConv(GINConv):
    def __init__(self, nn: Callable, eps: float = 0., train_eps: bool = True, **kwargs):
        super().__init__(nn, eps, train_eps, **kwargs)

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        size: Size = None,
    ) -> Tensor:

        # x corresponds to alpha 
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor)

        out = self.propagate(edge_index, alpha = x, size=size, num_vertices = edge_index.size()[1])

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)
    
    # # Overwrite the message
    # def message(self, x_i: Tensor, x_j: Tensor, alpha: Tensor, v: int, w: int) -> Tensor:
    #     r"""Constructs messages from node :math:`j` to node :math:`i`
    #     in analogy to :math:`\phi_{\mathbf{\Theta}}` for each edge in
    #     :obj:`edge_index`.
    #     This function can take any argument as input which was initially
    #     passed to :meth:`propagate`.
    #     Furthermore, tensors passed to :meth:`propagate` can be mapped to the
    #     respective nodes :math:`i` and :math:`j` by appending :obj:`_i` or
    #     :obj:`_j` to the variable name, *.e.g.* :obj:`x_i` and :obj:`x_j`.
    #     """

    #     #idx_j is a hack to get j by passing a tensor (0,...,n) as idx
    #     i = x_i.item()
    #     j = x_j.item()

    #     # Needs to return alpha(u, x_j,) + alpha(v, x_j)
    #     return x_j
    
    # def aggregate(
    #     self,
    #     inputs: Tensor,
    #     index: Tensor,
    #     ptr: Optional[Tensor] = None,
    #     dim_size: Optional[int] = None,
    # ) -> Tensor:
    #     r"""Aggregates messages from neighbors as
    #     :math:`\bigoplus_{j \in \mathcal{N}(i)}`.

    #     Takes in the output of message computation as first argument and any
    #     argument which was initially passed to :meth:`propagate`.

    #     By default, this function will delegate its call to the underlying
    #     :class:`~torch_geometric.nn.aggr.Aggregation` module to reduce messages
    #     as specified in :meth:`__init__` by the :obj:`aggr` argument.
    #     """
    #     return self.aggr_module(inputs, index, ptr=ptr, dim_size=dim_size,
    #                             dim=self.node_dim)

    def message_and_aggregate(self, adj_t: Adj, alpha: OptPairTensor, num_vertices: int) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)

        local_alpha = alpha[0]
        alpha_copy = alpha[1]
        f = local_alpha.size()[1]

        for v in range(num_vertices):
            for w in range(num_vertices):
                # alpha_vw corresponds to alpha_{v*n+w}
                A = torch.empty(size = (num_vertices, f), dtype = local_alpha.dtype)
                for j in range(num_vertices):
                    A[j,:] = alpha_copy[v*num_vertices+j,:] + alpha_copy[w*num_vertices+j,:]

                local_alpha[v*num_vertices + w] = spmm(adj_t, A, reduce=self.aggr)

        return local_alpha

class GPNN(torch.nn.Module):

    def __init__(self, num_gpnn_layers: int, gpnn_channels: int, num_classes: int, base_gnn: str, base_gnn_layers: int, base_gnn_in_channels: int,
                 base_gnn_hidden_channels: int, out_channels: int, use_batch_norm: bool) -> None:
        super().__init__()

        # Stores the base GNN (i.e. the classic GIN/GCN)
        assert base_gnn == 'gin' or base_gnn == 'gcn'

        self.num_gpnn_layers = num_gpnn_layers
        self.num_classes = num_classes
        self.gpnn_channels = gpnn_channels

        self.base_gnn_layers = base_gnn_layers
        self.base_gnn_in_channels = base_gnn_in_channels
        self.base_gnn_hidden_channels = base_gnn_hidden_channels
        self.use_batch_norm = use_batch_norm
        self.base_gnn = base_gnn

        self.device = constants.device

        self.vertex_convs = torch.nn.ModuleList()
        self.edge_convs = torch.nn.ModuleList()

        self.gnn_convs = torch.nn.ModuleList()
        self.norm_convs = torch.nn.ModuleList()

        self.W = torch.nn.ParameterList()
        self.omega = torch.nn.ParameterList()
        for c in range(self.num_classes):
            self.omega.append(torch.nn.Parameter(torch.randn(1,1)))
            self.W.append(Linear(2 * gpnn_channels + num_classes, gpnn_channels, bias = False, weight_initializer = 'glorot'))

        # properties of the parameters per layer and in total
        self.parameter_props = {}
        self.parameter_props["total"] = {}
        self.parameter_props["total"]["num"] = -1
        self.parameter_props["total"]["dtype"] = ""
        self.parameter_props["total"]["size"] = -1
        self.parameter_props["gpnn"] = {}
        self.parameter_props["gpnn"]["layer"] = {}
        self.parameter_props["base_gnn"] = {}
        self.parameter_props["base_gnn"]["layer"] = {}

        sum_p = 0
        sum_p_size = 0

        gpnn_in_channels = 1

        # generate GPNN
        for l in range(self.num_gpnn_layers):
            convs_p = []

            vertex_mlp = MLP([gpnn_in_channels, self.gpnn_channels, self.gpnn_channels], act = 'relu')
            vertex_conv = GINConv(nn = vertex_mlp, train_eps = True)
            self.vertex_convs.append(vertex_conv)

            edge_mlp = MLP([gpnn_in_channels, self.gpnn_channels, self.gpnn_channels], act = 'relu')
            edge_conv = GPNNEdgeConv(nn = edge_mlp, train_eps = True)
            self.edge_convs.append(edge_conv)

            convs_p.append(vertex_conv)
            convs_p.append(edge_conv)

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

            self.parameter_props["gpnn"]["layer"][l] = {}
            self.parameter_props["gpnn"]["layer"][l]["num"] = sum_p_layer
            self.parameter_props["gpnn"]["layer"][l]["dtype"] = dtype
            self.parameter_props["gpnn"]["layer"][l]["size"] = sum_p_size_layer

            self.parameter_props["total"]["dtype"] = dtype

            gpnn_in_channels = self.gpnn_channels

        for p in self.omega.parameters():
            sum_p += p.numel()
            sum_p_size += p.nbytes

        for w in self.W.parameters():
            sum_p += w.numel()
            sum_p_size += w.nbytes

        # generate base GNN
        for l in range(base_gnn_layers):
            convs_p = []

            if self.base_gnn == 'gcn':
                conv = GCNConv(in_channels = base_gnn_in_channels, out_channels = self.base_gnn_hidden_channels, add_self_loops = False, improved = False, cached = False, normalize = True, bias = True)
                self.gnn_convs.append(conv)
                convs_p.append(conv)
            elif self.base_gnn == 'gin':
                mlp = MLP([base_gnn_in_channels, self.base_gnn_hidden_channels, self.base_gnn_hidden_channels], act = 'relu')
                conv = GINConv(nn = mlp, train_eps = False)
                conv.node_dim = 0
                self.gnn_convs.append(conv)
                convs_p.append(conv)

            if self.use_batch_norm:
                batch_norm = BatchNorm(in_channels = self.base_gnn_hidden_channels)
                self.norm_convs.append(batch_norm)
                convs_p.append(batch_norm)

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

            self.parameter_props["base_gnn"]["layer"][l] = {}
            self.parameter_props["base_gnn"]["layer"][l]["num"] = sum_p_layer
            self.parameter_props["base_gnn"]["layer"][l]["dtype"] = dtype
            self.parameter_props["base_gnn"]["layer"][l]["size"] = sum_p_size_layer

            base_gnn_in_channels = self.base_gnn_hidden_channels

        self.parameter_props["total"]["num"] = sum_p
        self.parameter_props["total"]["size"] = sum_p_size

        # graph level pooling
        self.pool_mlp = MLP([(self.base_gnn_hidden_channels * self.base_gnn_layers) + (self.num_gpnn_layers * self.gpnn_channels), self.gpnn_channels + self.base_gnn_hidden_channels, out_channels], act = 'relu')

        sum_p_layer = 0
        sum_p_size_layer = 0
        for p in self.pool_mlp.parameters():
            sum_p_layer += p.numel()
            sum_p_size_layer += p.nbytes
            dtype = str(p.dtype)
        self.parameter_props["pool"] = {}
        self.parameter_props["pool"]["num"] = sum_p_layer
        self.parameter_props["pool"]["dtype"] = dtype
        self.parameter_props["pool"]["size"] = sum_p_size_layer

    def reset_parameters(self):
        for conv in self.vertex_convs:
            conv.reset_parameters()

        for conv in self.edge_convs:
            conv.reset_parameters()

        for conv in self.gnn_convs:
            conv.reset_parameters()

        for conv in self.norm_convs:
            conv.reset_parameters()

        for p in self.omega:
            p = torch.nn.Parameter(torch.randn(1,1))

        for w in self.W:
            w.reset_parameters()
    
    # vertex_idx_start represents the first idx of clustering_labels which corresponds to a vertex of the graph data
    def forward(self, data: Data):
        batch = data.batch
        num_graphs_in_batch = torch.unique(batch).size()[0]

        edge_index = torch.clone(data.edge_index)
        # gamma_0(v) := clustering_labels[v]
        gamma = torch.clone(data.x[:,0])
        
        # generate alpha => alpha_0(v,w) = edge_label(v,w)
        # generate the initial edge colors
        num_vertices = data.x.size()[0]

        # generate c adjecency matrices to later quickly evalute the corresponding neighborhoods
        class_adjacencies = []
        # we explicitely convert the edge_index to a sparse tensor
        for c in range(self.num_classes):
            vertices = torch.arange(0, num_vertices, dtype = torch.long).to(self.device)
            c_vertices = vertices[(gamma == c).nonzero().view(-1)]
            edge_index_adj = torch.empty(size = (2,0), dtype = torch.long).to(self.device)
            for v in c_vertices:
                edge_index_adj = torch.cat((edge_index_adj, edge_index[:, (edge_index[1] == v).nonzero().view(-1)]), dim = 1)
            class_adjacencies.append(to_torch_csr_tensor(edge_index_adj))

        # Generate the initial alpha
        alpha = torch.empty(size = (num_vertices ** 2,), dtype = torch.float32)
        for v in range(num_vertices):
            for w in range(num_vertices):
                class_v = gamma[v].item()
                class_w = gamma[w].item()

                # check whether there exists an edge between v and w
                # we assume the graphs to be undirected
                possible_edges = edge_index[0][edge_index[1] == v]
                edge_exists = w in possible_edges
                
                if class_v == class_w and edge_exists:
                    # case 1
                    alpha[v*num_vertices + w] = hash(tuple([class_v, 0, class_w]))
                elif class_v != class_w and edge_exists:
                    # case 2
                    alpha[v*num_vertices + w] = hash(tuple([class_v, 1, class_w]))
                else:
                    # case 3, edge does not exist
                    alpha[v*num_vertices + w] = hash(tuple([class_v, 2, class_w]))

        # We need to store the sum of all features after each layer in order to compute the graph level readout
        gpnn_layer_global_add_pool_res = torch.empty((self.num_gpnn_layers, num_graphs_in_batch, self.gpnn_channels), device = self.device)

        # compute GPNN
        for l in range(self.num_gpnn_layers):
            beta = self.vertex_convs[l](gamma, edge_index)

            # update alpha
            alpha = self.edge_convs[l](alpha, edge_index)

            # update gamma
            gamma = torch.zeros((num_vertices,self.gpnn_channels), dtype = alpha.dtype).to(self.device)
            for c in range(self.num_classes):
                u_c = torch.zeros(size = (self.num_classes)).to(self.device)
                u_c[c] = 1
                vector = torch.empty((num_vertices, self.gpnn_channels)).to(self.device)
                for v in range(num_vertices):
                    vector[v,:] = torch.concat((beta, alpha[v * num_vertices:(v+1) * num_vertices,:], u_c), dim = 1)
                    vector = self.W[c](vector)
                    gamma_v = spmm(class_adjacencies[c], vector)
                    gamma[v,:] += self.omega[c] * gamma_v

            gpnn_layer_global_add_pool_res[t,:,:] = global_add_pool(x, batch)

        # compute GNN
        x = torch.cat((torch.clone(data.x[:,1:]), torch.zeros((list(data.x.size())[0], self.base_gnn_hidden_channels - self.base_gnn_in_channels)).to(self.device)), dim = 1).to(dtype = torch.float32)
        edge_index = torch.clone(data.edge_index)

        # We need to store the sum of all features after each layer in order to compute the graph level readout
        gnn_layer_global_add_pool_res = torch.empty((self.base_gnn_layers, num_graphs_in_batch, self.base_gnn_hidden_channels), device = self.device)

        for t in range(self.base_gnn_layers):
            x = self.gnn_convs[t](x, edge_index)

            if self.base_gnn == 'gcn':
                # Apply ReLU
                x = F.relu(x)

            if self.use_batch_norm:
                x = self.norm_convs[t](x)

            # store pooling res
            # global_add_pool result shape is (num_unique_graphs_in_batch, feature_dim)
            # concatenate with the GPNN result vector
            gnn_layer_global_add_pool_res[t,:,:] = global_add_pool(x, batch)

        gamma = gpnn_layer_global_add_pool_res[0,:,:]
        for t in range(1, self.num_gpnn_layers):
            gamma = torch.cat((gamma, gpnn_layer_global_add_pool_res[t,:,:]), dim = 1)

        # concatenate the global_add_pools to get resulting GNN vector
        x = gnn_layer_global_add_pool_res[0,:,:]
        for t in range(1, self.base_gnn_layers):
            x = torch.cat((x, gnn_layer_global_add_pool_res[t,:,:]), dim = 1)
            
        return self.pool_mlp(torch.cat((x, gamma), dim = 1))

# A manager class for GNNs and partition enhanced GNNs
class GNN_Manager():

    def __init__(self):
        super().__init__()

        self.clustering_labels = None
        self.num_features = -1
        self.out_channels = -1

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
        self.metadata["gpnn"] = {}
        self.metadata["gpnn"]["desc"] = "unused"
        self.metadata["gnn"] = {}
        self.metadata["gnn"]["desc"] = ""
        self.metadata["gnn"]["model_id"] = -1
        self.metadata["gnn"]["config"] = {}
        self.metadata["gnn"]["config"]["num_layers"] = -1
        self.metadata["gnn"]["config"]["in_channels"] = -1
        self.metadata["gnn"]["config"]["hidden_channels"] = -1
        self.metadata["gnn"]["config"]["out_channels"] = -1
        self.metadata["gnn"]["config"]["use_batch_norm"] = False
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
        self.metadata["gpnn"] = {}
        self.metadata["gpnn"]["desc"] = "unused"
        self.metadata["gnn"] = {}
        self.metadata["gnn"]["desc"] = ""
        self.metadata["gnn"]["model_id"] = -1
        self.metadata["gnn"]["config"] = {}
        self.metadata["gnn"]["config"]["num_layers"] = -1
        self.metadata["gnn"]["config"]["in_channels"] = -1
        self.metadata["gnn"]["config"]["hidden_channels"] = -1
        self.metadata["gnn"]["config"]["out_channels"] = -1
        self.metadata["gnn"]["config"]["use_batch_norm"] = False
        self.metadata["gnn"]["parameters"] = {}
        self.metadata["gnn"]["parameters"]["num_total"] = -1
        self.metadata["gnn"]["parameters"]["dtype"] = ""
        self.metadata["gnn"]["parameters"]["size"] = -1

        
    # # Read clustering results from disk
    # def read_clustering_labels(self, path: str):
    #     self.clustering_labels = torch.from_numpy(np.loadtxt(fname = path, dtype = np.int32, comments = '#')).view(-1,1)

    # We need to modify the dataset such that each entry includes the original vertex_identifier to uniquely identify it later
    def set_dataset_parameters(self, num_features: int, num_classes: int, num_clusters: int):

        self.num_features = num_features
        self.num_clusters = num_clusters
        self.out_channels = num_classes

    def generate_GPNN_model(self, num_gpnn_layers: int, gpnn_channels: int, base_gnn_str: str, num_gnn_layers: int, gnn_hidden_channels: int, lr: float) -> None:
        assert base_gnn_str == 'gcn' or base_gnn_str == 'gin'
        assert self.num_features > 0 and self.out_channels > 0 and self.num_clusters > 0

        self.model = GPNN(num_gpnn_layers = num_gpnn_layers,
                          gpnn_channels = gpnn_channels,
                          num_classes = self.num_clusters,
                          base_gnn = base_gnn_str,
                          base_gnn_layers = num_gnn_layers,
                          base_gnn_in_channels = self.num_features,
                          base_gnn_hidden_channels = gnn_hidden_channels,
                          out_channels = self.out_channels,
                          use_batch_norm = constants.use_batch_norm
                          ).to(self.device)
        
        self.metadata["gnn"]["desc"] = base_gnn_str
        if base_gnn_str == 'gin':
            self.metadata["gpnn"]["desc"] = "gin-GPNN"
            self.metadata["gnn"]["model_id"] = 1
        elif base_gnn_str == 'gcn':
            self.metadata["gpnn"]["desc"] = "gcn-GPNN"
            self.metadata["gnn"]["model_id"] = 3
        self.metadata["gnn"]["config"]["num_layers"] = num_gnn_layers
        self.metadata["gnn"]["config"]["in_channels"] = self.num_features
        self.metadata["gnn"]["config"]["hidden_channels"] = gnn_hidden_channels
        self.metadata["gnn"]["config"]["use_batch_norm"] = constants.use_batch_norm
        
        self.metadata["gpnn"]["config"] = {}
        self.metadata["gpnn"]["config"]["num_layers"] = num_gpnn_layers
        self.metadata["gpnn"]["config"]["gpnn_channels"] = gpnn_channels
        self.metadata["gpnn"]["config"]["out_channels"] = self.out_channels
        self.metadata["gpnn"]["parameters"] = self.model.parameter_props

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.metadata["optimizer"]["desc"] = "adam"
        self.metadata["optimizer"]["lr"] = lr

    def generate_partition_enhanced_GIN_model(self, hidden_channels: int, num_layers: int, lr: float) -> None:
        assert self.num_features > 0 and self.out_channels > 0 and self.num_clusters > 0

        self.model = Partition_enhanced_GIN(
            in_channels = self.num_features,
            hidden_channels = hidden_channels,
            out_channels =  self.out_channels,
            num_layers = num_layers,
            num_clusters = self.num_clusters,
            use_batch_norm = constants.use_batch_norm
        ).to(self.device)

        self.metadata["gnn"]["desc"] = "enhanced_gin"
        self.metadata["gnn"]["model_id"] = 0
        self.metadata["gnn"]["config"]["num_layers"] = num_layers
        self.metadata["gnn"]["config"]["in_channels"] = self.num_features
        self.metadata["gnn"]["config"]["hidden_channels"] = hidden_channels
        self.metadata["gnn"]["config"]["out_channels"] = self.out_channels
        self.metadata["gnn"]["config"]["use_batch_norm"] = constants.use_batch_norm
        self.metadata["gnn"]["parameters"] = self.model.parameter_props

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.metadata["optimizer"]["desc"] = "adam"
        self.metadata["optimizer"]["lr"] = lr

    def generate_classic_GIN_model(self, hidden_channels: int, num_layers: int, lr: float) -> None:
        assert self.num_features > 0 and self.out_channels > 0

        self.model = GIN_Classic(
            in_channels = self.num_features,
            hidden_channels = hidden_channels,
            out_channels =  self.out_channels,
            num_layers = num_layers,
            use_batch_norm = constants.use_batch_norm
        ).to(self.device)

        self.metadata["gnn"]["desc"] = "classic_gin"
        self.metadata["gnn"]["model_id"] = 1
        self.metadata["gnn"]["config"]["num_layers"] = num_layers
        self.metadata["gnn"]["config"]["in_channels"] = self.num_features
        self.metadata["gnn"]["config"]["hidden_channels"] = hidden_channels
        self.metadata["gnn"]["config"]["out_channels"] = self.out_channels
        self.metadata["gnn"]["config"]["use_batch_norm"] = constants.use_batch_norm
        self.metadata["gnn"]["parameters"] = self.model.parameter_props

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.metadata["optimizer"]["desc"] = "adam"
        self.metadata["optimizer"]["lr"] = lr

    def generate_partition_enhanced_GCN_model(self, hidden_channels: int, num_layers: int, lr: float) -> None:
        assert self.num_features > 0 and self.out_channels > 0 and self.num_clusters > 0

        self.model = Partition_enhanced_GCN(
            in_channels = self.num_features,
            hidden_channels = hidden_channels,
            out_channels =  self.out_channels,
            num_layers = num_layers,
            num_clusters = self.num_clusters,
            use_batch_norm = constants.use_batch_norm
        ).to(self.device)

        self.metadata["gnn"]["desc"] = "enhanced_gcn"
        self.metadata["gnn"]["model_id"] = 2
        self.metadata["gnn"]["config"]["num_layers"] = num_layers
        self.metadata["gnn"]["config"]["in_channels"] = self.num_features
        self.metadata["gnn"]["config"]["hidden_channels"] = hidden_channels
        self.metadata["gnn"]["config"]["out_channels"] = self.out_channels
        self.metadata["gnn"]["config"]["normalize"] = True
        self.metadata["gnn"]["config"]["use_bias"] = True
        self.metadata["gnn"]["config"]["use_batch_norm"] = constants.use_batch_norm
        self.metadata["gnn"]["parameters"] = self.model.parameter_props

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.metadata["optimizer"]["desc"] = "adam"
        self.metadata["optimizer"]["lr"] = lr

    def generate_classic_GCN_model(self, hidden_channels: int, num_layers: int, lr: float) -> None:
        assert self.num_features > 0 and self.out_channels > 0

        self.model = GCN_Classic(
            in_channels = self.num_features,
            hidden_channels = hidden_channels,
            out_channels =  self.out_channels,
            num_layers = num_layers,
            use_batch_norm = constants.use_batch_norm
        ).to(self.device)

        self.metadata["gnn"]["desc"] = "classic_gcn"
        self.metadata["gnn"]["model_id"] = 3
        self.metadata["gnn"]["config"]["num_layers"] = num_layers
        self.metadata["gnn"]["config"]["in_channels"] = self.num_features
        self.metadata["gnn"]["config"]["hidden_channels"] = hidden_channels
        self.metadata["gnn"]["config"]["out_channels"] = self.out_channels
        self.metadata["gnn"]["config"]["normalize"] = True
        self.metadata["gnn"]["config"]["use_bias"] = True
        self.metadata["gnn"]["config"]["use_batch_norm"] = constants.use_batch_norm
        self.metadata["gnn"]["parameters"] = self.model.parameter_props

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        self.metadata["optimizer"]["desc"] = "adam"
        self.metadata["optimizer"]["lr"] = lr

    def get_metadata(self):
        return deepcopy(self.metadata)

# def get_data_loaders(gnn: GNN_Manager, batch_size: int, shuffle: bool, training_ratio: float, test_ratio: float) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
#     assert gnn.dataset is not None

#     if training_ratio + test_ratio < 1:
#         train_loader = DataLoader(dataset = gnn.dataset[:training_ratio], batch_size = batch_size, shuffle = shuffle)
#         test_loader = DataLoader(dataset = gnn.dataset[training_ratio:training_ratio+test_ratio], batch_size = batch_size)
#         valid_loader = DataLoader(dataset = gnn.dataset[training_ratio+test_ratio:], batch_size = batch_size)

#         return train_loader, test_loader, valid_loader
#     else:
#         train_loader = DataLoader(dataset = gnn.dataset[:training_ratio], batch_size = batch_size, shuffle = shuffle)
#         test_loader = DataLoader(dataset = gnn.dataset[training_ratio:], batch_size = batch_size)

#         return train_loader, test_loader, None

# def train_partition_enhanced_GNN(gnn: GNN_Manager, train_loader: DataLoader):
#     device = constants.device
    
#     gnn.model.train()

#     total_loss = 0
#     for data in train_loader:
#         data = data.to(device)
#         gnn.optimizer.zero_grad()
#         out = gnn.model(data)
#         loss = F.cross_entropy(out, data.y)
#         loss.backward()
#         gnn.optimizer.step()
#         total_loss += float(loss) * data.num_graphs

#     return total_loss/len(train_loader.dataset)

# @torch.no_grad()
# def test_partition_enhanced_GNN(gnn: GNN_Manager, test_loader: DataLoader):
#     device = constants.device

#     gnn.model.eval()

#     total_correct = 0
#     for data in test_loader:
#         data = data.to(device)
#         out = gnn.model(data)
#         pred = out.argmax(dim=-1)
#         total_correct += int((pred == data.y).sum())

#     return total_correct/len(test_loader.dataset)

#Test zone
# if __name__ == '__main__':

#     # reproducability
#     constants.initialize_random_seeds()

#     torch.autograd.set_detect_anomaly(True)

#     gnn_generator = GNN_Manager()

#     root_path = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir)
#     csl_path = osp.join('data', 'CSL', 'CSL_dataset')
#     dataset_csl = CSL_Dataset(root = osp.join(root_path, csl_path))

#     gnn_generator.load_dataset(dataset_csl, 0)

#     gnn_generator.generate_classic_GIN_model(hidden_channels = 32, num_layers = 3, lr = 0.1)

#     params = gnn_generator.model.parameters()

#     for param in params:
#         print(type(param), param.size())

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

#     # # Test the GIN
#     # print("Testing GIN: ")

#     # gnn_generator.generate_partition_enhanced_GIN_model(hidden_channels = hidden_channels, num_layers = num_layers, lr = lr)

#     # train_loader, test_loader, valid_loader = get_data_loaders(gnn_generator, batch_size = batch_size, shuffle = True, training_ratio = 0.9, test_ratio = 0.1)

#     # times = []
#     # for epoch in range(1, epochs + 1):
#     #     start = time.time()
#     #     loss = train_partition_enhanced_GNN(gnn = gnn_generator, train_loader = train_loader)
#     #     train_acc = test_partition_enhanced_GNN(gnn = gnn_generator, test_loader = train_loader)
#     #     test_acc = test_partition_enhanced_GNN(gnn = gnn_generator, test_loader = test_loader)
#     #     log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
#     #     times.append(time.time() - start)
#     # print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')

#     # # Test the GCN
#     # print("Testing GCN: ")

#     # gnn_generator.reset_parameters()
#     # gnn_generator.generate_partition_enhanced_GCN_model(hidden_channels = hidden_channels, num_layers = num_layers, lr = lr)

#     # times = []
#     # for epoch in range(1, epochs + 1):
#     #     start = time.time()
#     #     loss = train_partition_enhanced_GNN(gnn = gnn_generator, train_loader = train_loader)
#     #     train_acc = test_partition_enhanced_GNN(gnn = gnn_generator, test_loader = train_loader)
#     #     test_acc = test_partition_enhanced_GNN(gnn = gnn_generator, test_loader = test_loader)
#     #     log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
#     #     times.append(time.time() - start)
#     # print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')