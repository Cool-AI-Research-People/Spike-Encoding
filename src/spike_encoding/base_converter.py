import torch

class BaseConverter(torch.nn.Module):
    def __init__(self):
        super(BaseConverter, self).__init__()

    def encode(self, tensor):
        raise NotImplementedError
    
    def decode(self, spikes: torch.Tensor):
        raise NotImplementedError
    
    def optimize(self, data: torch.Tensor):
        raise NotImplementedError