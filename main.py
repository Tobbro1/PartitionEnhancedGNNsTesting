import os.path as osp
import numpy as np

#load dataset
#select graph
#foreach vertex
#generate subgraph by k-disk



if __name__ == '__main__':
    # Generate an empty cluster feature vector for testing purposes

    comment = 'Zero-vector to simulate MUTAG clustering into the same clustering class (simulate a "normal" GIN)'
    num_vertices = 3371

    path = osp.join(osp.abspath(osp.dirname(__file__)), 'data', 'TU')
    mutag_path = osp.join(path, "MUTAG")
    clusterlabels_path = osp.join(path, mutag_path, 'cluster_labels_zero.txt')

    np.savetxt(fname = clusterlabels_path, X = np.zeros((num_vertices,1)), comments = '#', fmt = '%d', header = comment)