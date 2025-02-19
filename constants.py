import torch

SEED = 37

if torch.cuda.is_available():
    device = torch.device('cuda')  
else:
    device = torch.device('cpu')
