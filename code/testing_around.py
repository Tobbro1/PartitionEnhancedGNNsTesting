import numpy as np
from sklearn.datasets import load_svmlight_file
from CSL_dataset import CSL_Dataset
import os.path as osp
import os
import developmentHelpers as helpers
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
import networkx as nx

if __name__ == '__main__':

    featuretypes = ["10-disk", "vertex"]
    fig_count = 1

    for featuretype in featuretypes:
        print(f'Feature type {featuretype}:')
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

        print(f'Class examples: {examples}')

        # r_examples = []
        # for r in range(2, 17):
        #     for _, graph_id in examples:
        #         graph = to_networkx(dataset.get(idx), to_undirected = True)
        #         if graph.has_edge(0, r):
        #             r_examples.append(tuple([r, graph_id]))
        
        # print(r_examples)


        r_examples = []

        # Get example graphs for each class
        example_graphs = []
        for c, graph_id in examples:
            ex_graph = to_networkx(dataset.get(graph_id), to_undirected = True)
            example_graphs.append(tuple([c, graph_id, ex_graph]))

        # Find r for each class by creating corresponding graphs and testing for isomorphism
        for r in [2,3,4,5,6,9,11,12,13,16]:
            # Create graph
            r_graph = nx.Graph()
            r_graph.add_nodes_from(list(range(41)))
            # Create circle by adding corresponding edges
            for i in range(41):
                r_graph.add_edge(i, (i+1) % 41)
            # Create skip link edges
            for i in range(41):
                r_graph.add_edge(i, (i+r) % 41)
            # Ensure the graph is undirected
            r_graph = r_graph.to_undirected()

            if featuretype == 'vertex':
                if r in [6,16]:
                    # Compute the distances between vertices to compute a vertex coloring depending on distances from vertex 0
                    distances = nx.floyd_warshall(G = r_graph)
                    res = [dict(b) for a,b in distances.items() if a == 0]
                    # print(f'FW results: {res}')
                    
                    # Generate colors
                    num_colors = 

                    # Draw the r_graph
                    circle_positions = nx.circular_layout(G = r_graph)
                    helpers.drawGraph(graph = r_graph, figure_count = fig_count, pos = circle_positions)
                    fig_count += 1
            
            # Find isomorphic example graph
            for c, ex_graph_id, ex_graph in example_graphs:
                if nx.is_isomorphic(r_graph, ex_graph):
                    r_examples.append(tuple([c, r, ex_graph_id]))

        print(f'r examples: {r_examples}')

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

    plt.show()
    # vsp: r = 5, 13?, class 4, 9
    # 3-disk:  class 5, 7