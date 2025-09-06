from typing import List, Literal, Union
import numpy as np
from numpy.typing import NDArray as ndarray
import numpy.random as rndm

from spike_encoding.base_converter import BaseConverter
from spike_encoding.gymnasium_bounds_finder import ScalerFactory
from spike_encoding.encoder_common import poisson, rate


def gaussian_response(x, mu, sigma=0.3):
    return np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sigma, 2.0)))


def transform_firing_rates(firing_rates, n_bins, sigma=0.3):
    # Assuming firing rates are scaled to [0, 1]
    bin_centers = np.linspace(0, 1, n_bins)
    transformed_rates = []

    for rate in firing_rates:
        responses = [gaussian_response(rate, mu, sigma=sigma) for mu in bin_centers]
        # Normalize the responses to sum to 1
        normalized_responses = responses / np.sum(responses)
        transformed_rates += normalized_responses.tolist()

    return np.array(transformed_rates)


class BinEncoder(BaseConverter):
    """
    This encoder creates spike trains from gymnasium observations. To use it, first install gymnasium https://gymnasium.farama.org/

    Args:
        seq_length: The number of timesteps in the spike train. E.g. [[1], [0], [1]] has a seq_length of 3
        scaler: This is important for scaling inputs internally. For example, if your input goes up to 10, the scaler needs to scale instances with value 10 to a firing rate of 1.0
        spike_train_conversion_method: determines how a firing rate is converted into a spike train. In poisson encoding, a firing rate of 0.1 means there is a 10% chance for any given timestep to be a spike. By chance there could be more or fewer spikes. If instead "deterministic" is chosen, you are guraranteed that 10% of timesteps are spikes
        max_firing_rate: multiplier for maximum firing rate. typically the maximum is 1.0 (i.e. spikes at every step). You can set this to a lower value like 0.5 so on average you will only get a spike at every other step. This may be important for some R-STDP scenarios, where very high firing rates can impact synaptic tags


    Returns:
        spike train: an array of arrays of spikes.
    """

    spike_train_conversion_method: Literal["poisson", "deterministic"] = "poisson"

    def __init__(
        self,
        seq_length: int,
        min_values: Union[List[float], ndarray],
        max_values: Union[List[float], ndarray],
        spike_train_conversion_method: Literal["poisson", "deterministic"] = "poisson",
        n_bins=10,
        max_firing_rate=1.0,
        sigma=0.1,
    ):
        self.sigma = sigma
        self.seq_length = seq_length if seq_length <= 1 else seq_length
        scaler_factory = ScalerFactory()
        self.scaler = scaler_factory.from_known_values(min_values, max_values)
        self.n_bins = n_bins
        self.spike_train_conversion_method = spike_train_conversion_method
        self.max_firing_rate = max_firing_rate

        self.seed = 42
        rndm.seed(self.seed)

    def encode(self, state: ndarray) -> ndarray:
        # NOTE this uses batches for scaling and coding, but not for binning
        p_spikes = self.scaler.transform(np.atleast_2d(state))[0]
        p_bins = np.atleast_2d(
            [transform_firing_rates(p_spikes, self.n_bins, self.sigma)]
        )
        p_bins *= self.max_firing_rate

        if self.spike_train_conversion_method == "poisson":
            output = poisson(p_bins, self.seq_length)  # type: ignore
        else:
            output = rate(p_bins, self.seq_length)  # type: ignore

        return output
