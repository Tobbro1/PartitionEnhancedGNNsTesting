import torch

# If not specified, the seed below is used
SEED = 37

# Used for feature generation
num_processes = 8
graph_chunksize = 64    # Note that the chunksizes should be tested on hardware. If the parent process is limiting, the chunksize should be increased.
vertex_chunksize = 512 
vector_buffer_size = 16_384

# Used in the experiments
num_reruns = 3 # number of re-initializations of GNNs in order to account for random parameter initialization
num_k_fold = 5 # used for CSL experiments
k_fold_test_ratio = 0.3 # used for CSL experiments
# clustering
mbk_batch_size = 1024
mbk_n_init = 10
mbk_max_no_improvement = 20
mbk_max_iter = 2000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
