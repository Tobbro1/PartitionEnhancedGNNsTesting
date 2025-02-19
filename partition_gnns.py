
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
from torch_geometric.nn import MLP, GINConv, global_add_pool
from torch_geometric.logging import log
from torch_geometric.typing import Adj, OptPairTensor, Size

# System
import os.path as osp
import time

# Import constants
import constants

# Defines a pre-transform required to add the clustering ID to the dataset features of any loaded dataset. This is necessary in case the dataset gets shuffled when training the GNNs
class ClusteringIDPreTransform():
    def __init__(self, clustering_labels):
        self.start_index = 0
        self.clustering_labels = clustering_labels

    def __call__(self, data: Data):
        num_vertices = data.num_nodes
        data.x = torch.cat((self.clustering_labels[self.start_index:self.start_index + num_vertices,:], data.x), dim = 1)
        self.start_index += num_vertices

        return data

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

        self.hidden_in_channel_diff = hidden_channels - in_channels

        self.device = constants.device

        self.convs = torch.nn.ModuleList()

        for _ in range(self.num_layers):
            for _ in range(self.num_clusters):
                # 2-layer MLP
                mlp = MLP([in_channels, hidden_channels, hidden_channels], act = 'relu')
                self.convs.append(Partition_enhanced_GIN_conv(nn = mlp, train_eps = False))

            in_channels = hidden_channels

        # graph level pooling
        self.pool_mlp = MLP([hidden_channels, hidden_channels, out_channels], act = 'relu')

    
    # vertex_idx_start represents the first idx of clustering_labels which corresponds to a vertex of the graph data
    def forward(self, data: Data):
        x = torch.cat((data.x[:,1:], torch.zeros((list(data.x.size())[0], self.hidden_in_channel_diff)).to(self.device)), dim = 1)
        clustering_labels = data.x[:,0]
        edge_index = data.edge_index
        batch = data.batch

        # We edit only the subset
        for i in range(self.num_layers):
            for j in range(self.num_clusters):
                conv_idx = i * self.num_clusters + j

                # To implement: x[id,:] = convs[conv_idx](x[id,:]) iff clustering_labels[id] == j
                mask = (clustering_labels == j).view(-1)

                # This distinction is necessary. We cannot simply override x after a single convolution since we apply different convolutions in different layers.
                # That means, we need to store intermediary results, meaning we artificially enlarge the feature dimension of x in the first step, and we have to account for the artificially large dimension in the first step layer.
                if i == 0:
                    x[mask,:] = self.convs[conv_idx](x[:,:(list(x.size())[1] - self.hidden_in_channel_diff)], edge_index, mask)
                else:
                    x[mask,:] = self.convs[conv_idx](x, edge_index, mask)
                #x = self.convs[conv_idx](torch.masked_select(x, mask), edge_index)

        # graph level readout
        x = global_add_pool(x, batch)
        return self.pool_mlp(x)
        

class Partition_enhanced_GNN():

    def __init__(self):
        super().__init__()

        self.clustering_labels = None
        self.dataset = None
        self.num_features = -1
        self.num_classes = -1

        self.device = constants.device

        self.model = None
        self.optimizer = None
        
    # Read clustering results from disk
    def read_clustering_labels(self, path: str):
        self.clustering_labels = torch.from_numpy(np.loadtxt(fname = path, dtype = np.int32, comments = '#')).view(-1,1)

    # We need to modify the dataset such that each entry includes the original vertex_identifier to uniquely identify it later
    def load_dataset(self, dataset_path: Optional[str] = None, root_path: Optional[str] = None, dataset_name: Optional[str] = None, force_reload: bool = False):

        assert dataset_path is not None or (dataset_name is not None and root_path is not None)
        assert self.clustering_labels is not None

        if dataset_name is not None and root_path is not None:
            if dataset_name == 'MUTAG':
                self.dataset = TUDataset(root = root_path, name = 'MUTAG', use_node_attr = True, force_reload = force_reload, pre_transform = ClusteringIDPreTransform(clustering_labels = self.clustering_labels))
                # Since we increase the feature dimension artificially by including the cluster_id
                self.num_features = self.dataset.num_features - 1
                self.num_classes = self.dataset.num_classes


    def generate_partition_enhanced_GIN_model(self, hidden_channels: int, num_layers: int, lr: float):
        assert self.dataset is not None
        assert self.clustering_labels is not None

        num_clusters = torch.unique(self.clustering_labels).size(dim = 0)

        self.model = Partition_enhanced_GIN(
            in_channels = self.num_features,
            hidden_channels = hidden_channels,
            out_channels =  self.num_classes,
            num_layers = num_layers,
            num_clusters = num_clusters
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)

def get_data_loaders(gnn: Partition_enhanced_GNN, batch_size: int, shuffle: bool, training_ratio: float, test_ratio: float) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
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

def train_partition_enhanced_GIN(gin: Partition_enhanced_GNN, train_loader: DataLoader):
    device = constants.device
    
    gin.model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        gin.optimizer.zero_grad()
        out = gin.model(data)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        gin.optimizer.step()
        total_loss += float(loss) * data.num_graphs

    return total_loss/len(train_loader.dataset)

@torch.no_grad()
def test_partition_enhanced_GIN(gin: Partition_enhanced_GNN, test_loader: DataLoader):
    device = constants.device

    gin.model.eval()

    total_correct = 0
    for data in test_loader:
        data = data.to(device)
        out = gin.model(data)
        pred = out.argmax(dim=-1)
        total_correct += int((pred == data.y).sum())

    return total_correct/len(test_loader.dataset)

#Test zone
if __name__ == '__main__':

    #reproducability
    np.random.seed(constants.SEED)
    torch.manual_seed(constants.SEED)

    gnn_generator = Partition_enhanced_GNN()

    path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'TU')
    mutag_path = osp.join(path, "MUTAG")
    dd_path = osp.join(path, "DD")
    mutag_dataset_filename = "k_disk_SP_features_MUTAG.svmlight"
    dd_dataset_filename = "k_disk_SP_features_DD.svmlight"
    dataset_path = osp.join(path, mutag_path, mutag_dataset_filename)
    clusterlabels_path = osp.join(path, mutag_path, 'cluster_labels.txt')

    gnn_generator.read_clustering_labels(path = clusterlabels_path)
    gnn_generator.load_dataset(root_path=osp.join(path, 'enhanced_gnn'), dataset_name = 'MUTAG')

    batch_size = 128
    hidden_channels = 32
    num_layers = 3
    lr = 0.01
    epochs = 100

    gnn_generator.generate_partition_enhanced_GIN_model(hidden_channels = hidden_channels, num_layers = num_layers, lr = lr)

    train_loader, test_loader, valid_loader = get_data_loaders(gnn_generator, batch_size = batch_size, shuffle = True, training_ratio = 0.9, test_ratio = 0.1)

    times = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        loss = train_partition_enhanced_GIN(gin = gnn_generator, train_loader = train_loader)
        train_acc = test_partition_enhanced_GIN(gin = gnn_generator, test_loader = train_loader)
        test_acc = test_partition_enhanced_GIN(gin = gnn_generator, test_loader = test_loader)
        log(Epoch=epoch, Loss=loss, Train=train_acc, Test=test_acc)
        times.append(time.time() - start)
    print(f'Median time per epoch: {torch.tensor(times).median():.4f}s')