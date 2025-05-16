import numpy as np
from sklearn.datasets import load_svmlight_file
from CSL_dataset import CSL_Dataset
import os.path as osp
import os
import developmentHelpers
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx

if __name__ == '__main__':

    featuretype = "10-disk"

    path = f"E:\\workspace\\PartitionEnhancedGNNsTesting\\data\\CSL\\CSL_dataset\\results\\{featuretype}_SP_features\\CSL_{featuretype}_SP_features.svmlight"

    data, _ = load_svmlight_file(f = path, dtype = np.float64, zero_based = True)
    data = data[:,2:].toarray()

    dataset = CSL_Dataset(root = osp.join(osp.abspath(osp.dirname(__file__)), os.pardir, 'data', 'CSL', 'CSL_dataset'))

    

    examples = []

    for c in range(dataset.num_classes):
        for idx in range(dataset.len()):
            if dataset.get(idx).y.item() == c:
                examples.append(tuple([c, idx]))
                break

    print(examples)

    r_examples = []
    for r in range(2, 17):
        for _, graph_id in examples:
            graph = to_networkx(dataset.get(idx), to_undirected= True)
            if graph.has_edge(0, r):
                r_examples.append(tuple([r, graph_id]))
    
    print(r_examples)


    for g in range(dataset.num_classes):
        for h in range(g, dataset.num_classes):
            if g == h:
                continue
            if (data[41 * examples[g][1],:] == data[41 * examples[h][1],:]).all():
                print(f"Graph 1: {examples[g][1]}, class {examples[g][0]}; Graph 2: {examples[h][1]}, class {examples[h][0]}")
                print(f"Feature Graph 1: {data[41 * examples[g][1],:]}")
                print(f"Feature Graph 2: {data[41 * examples[h][1],:]}")

    print(f"Graph 60: {data[41 * 60,:]}")
    print(f"Graph 135: {data[41 * 135,:]}")
    print(f"Graph 75: {data[41 * 75,:]}")
    print(f"Graph 105: {data[41 * 105,:]}")

    # vsp: r = 5, 13?, class 4, 9
    # 3-disk:  class 5, 7