import random
import torch
import numpy as np

SEED = 37

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper to initialize random seeds for reproducability
def initialize_random_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)