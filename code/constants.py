import random
import torch
import numpy as np

SEED = 37

num_processes = 8
graph_chunksize = 64
vertex_chunksize = 512
vector_buffer_size = 16_384

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper to initialize random seeds for reproducability
def initialize_random_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)