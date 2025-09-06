from typing import Iterable, Union
import numpy as np
import torch
from .base_converter import BaseConverter


class RateStepForwardConverter(BaseConverter):

    def __init__(
        self,
        n_feat: int,
        threshold: Union[torch.Tensor, np.array] = torch.ones(1),
        padding: bool = True,
    ):
        """Encodes a tensor of input values by the difference between a timestep and an adaptive threshold.
        Generates spike probabilities proportoinal to the threshold plus a moving baseline based on previous values.

        Args:
            n_feat (int):
                Number of features to be passed to the encoder
            threshold (Union[torch.Tensor, np.array], optional):
                Threshold to compare the diffence against for each timestep.
                Defaults to torch.ones(1).
            padding (bool, optional):
                If ``True``, the first time step will be compared with itself resulting in ``0``'s in spikes.
                If ``False``, it will be padded with ``0``'s.
                Defaults to False.
        """
        super(RateStepForwardConverter, self).__init__()
        self.n_feat = n_feat
        if threshold.shape[0] != n_feat:
            threshold = torch.ones(n_feat) * threshold[0]
        self.threshold = threshold
        self.padding = padding

    @staticmethod
    def _encode_step_(
        current: Union[torch.Tensor, np.array],
        base: Union[torch.Tensor, np.array],
        threshold: Union[torch.Tensor, np.array],
    ) -> Iterable[Union[Union[torch.Tensor, np.array], Union[torch.Tensor, np.array]]]:
        """Performs an encoding step, by comparing the current value to the residual value + threshold for up spike and to residual value - threshold for down spikes

        Args:
            current (Union[torch.Tensor, np.array]): Value of the input array at the current time step.
            base (Union[torch.Tensor, np.array]): Residual values from previous timesteps.
            threshold (Union[torch.Tensor, np.array]): Threshold to compare against.

        Returns:
            base (Union[torch.Tensor, np.array]): Updated residual value.
            p_spike (Union[torch.Tensor, np.array]): Spike proability.
        """

        p_spike = (current - base) / threshold
        base = current

        return base, p_spike

    def encode(
        self,
        signal: Union[torch.Tensor, np.array],
        threshold: Union[torch.Tensor, np.array] = None,
    ) -> Union[torch.Tensor, np.array]:
        """Converts the original signal into spike probabilities.

        Args:
            signal (Union[torch.Tensor, np.array]):
                Tensor containing signal to be converted.
                Must have shape [batch, features, time_steps].
            threshold (Union[torch.Tensor, np.array], optional):
                Inputs greater than moving baseline + threshold generate a positive spike probability.
                Inputs smaller than moving baseline - threshold generate a negative spike probability.
                Must have shape [n_features]
                Defaults to self.threshold if None.
                Defaults to None.

        Returns:
            p_spike (Union[torch.Tensor, np.array]):
                Converted spike probabilities.
        """

        threshold = self.threshold if threshold == None else threshold
        assert signal.ndimension() == 3
        assert threshold.ndimension() == 1

        # Signal has shape: [batch, features, time_steps]
        batch, features, time_steps = signal.shape

        threshold = (signal[:, :, 0] + threshold - signal[:, :, 0]).reshape(-1)
        signal = signal.reshape(threshold.shape[0], time_steps)

        # The signal now has shape = [features * batch, time_steps] and threshold has shape = [features * batch]
        p_spikes = torch.zeros_like(signal)

        if type(signal) == torch.Tensor:
            device = signal.device
            signal = signal.detach().clone()
            threshold = torch.Tensor(threshold).to(device)
        else:
            threshold = torch.Tensor(threshold).detach().cpu().numpy()
            p_spikes = p_spikes.detach().cpu().numpy()

        if self.padding:
            base = signal[:, 0]
        else:
            base = signal[:, 0] * 0

        for t in range(1, time_steps):
            current = signal[:, t]
            base, p = self._encode_step_(current, base, threshold)
            p_spikes[:, t] = p

        p_spikes = p_spikes.reshape(batch, features, time_steps)

        return p_spikes

    def decode(
        self,
        spikes: Union[torch.Tensor, np.array],
        initial_value: Union[torch.Tensor, np.array] = None,
        threshold: Union[torch.Tensor, np.array] = None,
    ) -> Union[torch.Tensor, np.array]:
        """Attempts to reconstruct the original signal using the given spikes or spike probabilities.

        Args:
            spikes (Union[torch.Tensor, np.array]):
                Contains spike information derived from the original signal.
                Should either contain up and down spikes (dtype=int) with shape [batch, features, time_steps, num_steps]
                Or spike probabilities (dtype=float) with shape [batch, features, time_steps]
            initial_value (Union[torch.Tensor, np.array], optional):
                Initial value to compare against.
                Shape [batch, features]
                If None, will default to zero.
                Defaults to None.
            threshold (Union[torch.Tensor, np.array], optional):
                Inputs greater than moving baseline + threshold generate a positive spike probability.
                Inputs smaller than moving baseline - threshold generate a negative spike probability.
                Must have shape [n_features]
                Defaults to self.threshold if None.
                Defaults to None.

        Returns:
            Union[torch.Tensor, np.array]: Decoded signal with 3 dims in the form [batch, features, time_steps].
        """

        assert spikes.ndim() == 4 | 3

        if spikes.ndimension() == 4:
            # Assume shape [batch, features, time_steps, num_steps]
            spikes = spikes[0].add(spikes[1])
        spikes = torch.atleast_2d(spikes)

        threshold = self.threshold if threshold == None else threshold

        weighted_spikes = threshold[:, None] * spikes.squeeze()

        # Shape of weighted_spikes is now [features, time_steps]
        time_steps = weighted_spikes.shape[1]
        start = initial_value
        reconstructed = torch.zeros_like(weighted_spikes)
        reconstructed[:, 0] = start
        for t in range(1, time_steps):
            value = weighted_spikes[:, t]
            reconstructed[:, t] = reconstructed[:, t - 1] + value
        return reconstructed

    def optimize(
        self, data: Union[torch.Tensor, np.array]
    ) -> Union[torch.Tensor, np.array]:
        """Attempts find optimal threshold values for the given data per feature.

        Args:
            data (Union[torch.Tensor, np.array]):
                Contains data for which to optimize the threshold.
                Must have shape [batch, features, time_steps]

        Returns:
            threshold (Union[torch.Tensor, np.array]): The optimized threshold with shape [features].
        """
        assert data.ndim == 3
        batch, features, time_steps = data.shape

        diffs = data[:, :, 1:] - data[:, :, :-1]
        diffs = diffs.abs().swapaxes(0, 1).reshape(features, -1)
        diffs = diffs[:, ~diffs[0].isnan()]
        diffs = diffs[:, ~diffs[0].isinf()]
        threshold = diffs.max(axis=1).values
        self.threshold = threshold

        return self.threshold
