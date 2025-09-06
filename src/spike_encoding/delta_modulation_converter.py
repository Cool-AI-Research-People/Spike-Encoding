import torch
from .base_converter import BaseConverter



class DeltaModulationConverter(BaseConverter):
    """Encodes a tensor of input values by the difference bewteen two subsequent timesteps, and simulates the spikes that
    occur when meets threshold.

    Parameters:
        delta (float): Input with a change greater than the thresold across one timestep will generate a spike, defaults to ``0.1``
        padding (bool): If ``True``, the first time step will be compared with itself resulting in ``0``'s in spikes. If ``False``, it will be padded with ``0``'s, defaults to ``False``
        off_spike: If ``True``, spikes for negative changes less than ``-threshold``, defaults to ``False``
    """

    def __init__(
        self,
        delta: float = 0.1,
        normalized: bool = True,
        padding: bool = False,
        off_spike: bool = False,
    ):
        super(DeltaModulationConverter, self).__init__()
        self.delta = delta
        self.normalized = normalized
        self.padding = padding
        self.off_spike = off_spike

    def forward(self, tensor):
        if tensor.ndimension() < 2:
            tensor = tensor.unsqueeze(0)
        cols = torch.split(tensor, 1, 1)

        if not self.normalized:
            for i in range(tensor.shape[0]):
                tensor[i] = (tensor[i] - torch.min(tensor[i])) / (torch.max(tensor[i]) - torch.min(tensor[i]))

        if self.padding:
            data_offset = torch.cat((cols[0], tensor), dim=1)[:, :-1]  # duplicate first time step, remove final step
        else:
            data_offset = torch.cat((torch.zeros_like(cols[0]), tensor), dim=1)[:,
                          :-1]  # add 0's to first step, remove final step

        if not self.off_spike:
            return torch.stack((torch.ones_like(tensor) * ((tensor - data_offset) >= self.delta), torch.zeros_like(tensor)))
        else:
            on_spk = torch.ones_like(tensor) * ((tensor - data_offset) >= self.delta)
            off_spk = -torch.ones_like(tensor) * ((tensor - data_offset) <= -self.delta)
            return torch.stack((on_spk, off_spk))
        
    def decode(self, spikes: torch.Tensor):
        raise NotImplementedError