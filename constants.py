import torch

SEED = 37

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')