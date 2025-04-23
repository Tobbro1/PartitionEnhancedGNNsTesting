import torch

# If not specified, the seed below is used
SEED = 37

random_generator = None # Random generator if necessary

# Used for feature generation
num_processes = 8
graph_chunksize = 64    # Note that the chunksizes (graph + vertex) should be tested on hardware. If the parent process is the CPU bottleneck, the chunksize should be increased.
vertex_chunksize = 512 
vector_buffer_size = 16_384

# Used for lovasz features
num_lo_gens = 2 # Number of times the Lovasz features are generated in order to 
sdp_max_iters = 20 # maximum number of iterations done in the sdp solver used in computing the lo-theta value of a graph
lo_small_val = float("1e-13") # Used to make the zero matrix pos definite if necessary
welzl_tolerance = float("1e-10")

# Used in the experiments
num_workers = 0 # The number of worker processes used in the dataloaders for gnn training, 0 means not utilising parallelization in this step
num_reruns = 3 # number of re-initializations of GNNs in order to account for random parameter initialization
num_k_fold = 5 # used for CSL experiments
k_fold_test_ratio = 0.3 # used for CSL experiments
max_patience = 100 # patience value used in gnn training. If the validation performance does not improve for max_patience training steps, the training is prematurely halted.
use_batch_norm = False # Specifiy whether batch normalization is utilised in gnn training.
# clustering
mbk_batch_size = 1024
mbk_n_init = 3
mbk_max_no_improvement = 10
mbk_max_iter = 100

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
