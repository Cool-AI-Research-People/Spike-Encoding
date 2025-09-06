from typing import Literal, Optional, Tuple, Union
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from numpy.typing import NDArray as ndarray
import numpy.random as rndm

from spike_encoding.base_converter import BaseConverter

from .rate_step_forward_converter import RateStepForwardConverter as Converter
from spike_encoding.encoder_common import poisson, rate


class GymnasiumEncoder(BaseConverter):
    """
    This encoder creates spike trains from gymnasium observations. To use it, first install gymnasium https://gymnasium.farama.org/

    Args:
        n_features: The number of features in the input data. E.g. 4 in cartpole
        seq_length: The number of timesteps in the spike train. E.g. [[1], [0], [1]] has a seq_length of 3
        scaler: This is important for scaling inputs internally. For example, if your input goes up to 10, the scaler needs to scale instances with value 10 to a firing rate of 1.0
        split_exc_inh: Flag for splitting feature into 2 features at middle value. For example, if your angle can be between -3 and 3, it creates 2 spike trains. The first contains a proportional number of spikes for when the value is > 0 and the second for when it is < 0.
        spike_train_conversion_method: determines how a firing rate is converted into a spike train. In poisson encoding, a firing rate of 0.1 means there is a 10% chance for any given timestep to be a spike. By chance there could be more or fewer spikes. If instead "deterministic" is chosen, you are guraranteed that 10% of timesteps are spikes
        add_inverted_inputs: Flag for adding another set of inputs with inverse firing rate. E.g. if input 0 has firing rate 0.8, its inverse has 0.2
        max_firing_rate: multiplier for maximum firing rate. typically the maximum is 1.0 (i.e. spikes at every step). You can set this to a lower value like 0.5 so on average you will only get a spike at every other step. This may be important for some R-STDP scenarios, where very high firing rates can impact synaptic tags


    Returns:
        spike train: an array of arrays of spikes.
    """

    spike_train_conversion_method: Literal["poisson", "deterministic"] = "deterministic"

    def __init__(
        self,
        n_features: int,
        batch_size: int,
        seq_length: int,
        scaler: MinMaxScaler,
        converter_th: Union[ndarray, None] = None,
        converter: Union[Converter, None] = None,
        # Encodes the absolute value (e.g. position first normalized -1 and 1 -> 0.7 -> spike train with spiking probability of 0.7 for given number of steps)
        rate_coder: bool = True,
        step_coder: bool = False,
        split_exc_inh: bool = True,
        spike_train_conversion_method: Literal[
            "poisson", "deterministic"
        ] = "deterministic",
        add_inverted_inputs=False,
        max_firing_rate=1.0,
        seed=None,
    ):
        self.split_exc_inh = split_exc_inh
        self.batch_size = batch_size
        self.seq_length = seq_length if seq_length <= 1 else seq_length
        self.features = n_features
        self.converter_th = converter_th
        self.step_coder = step_coder
        self.rate_coder = rate_coder
        self.scaler = scaler
        self.converter = converter
        self.spike_train_conversion_method = spike_train_conversion_method
        self.add_inverted_inputs = add_inverted_inputs
        self.max_firing_rate = max_firing_rate
        self.seed = seed

        if not self.seed is None:
            rndm.seed(self.seed)

        if self.step_coder and converter is None:
            print(
                "Error - using step_coder without providing a converter. Please pass a converter in the parameters"
            )

    def _scale(self, state: ndarray) -> ndarray:
        return self.scaler.transform(np.atleast_2d(state))

    def reset(self, state: Optional[ndarray] = None) -> None:
        if not self.seed is None:
            rndm.seed(self.seed)
        # state.shape == (batch, features)
        if state is None:
            self.prev_state = np.zeros((self.batch_size, self.features))
        else:
            self.prev_state = self._scale(state)

        self.base = self.prev_state

    def encode(
        self, state: ndarray, return_norm=False
    ) -> Union[ndarray, Tuple[ndarray, Optional[ndarray]]]:
        """
        converts a list of lists of floats such as [1.7, 2.1] to a spike train like [[0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 1]]

        Parameters
        ----------
        state : ndarray
            The state to encode. I.e., if there are x-coordinates and angles, a state might be (2, 0.5) representing a coordinate (2) and an angle of (0.5)

        return_norm : Optional[bool], default=False
            If true, returns the normalized state as well. The normalized state is scaled between 0 and 1 using the known upper/lower bounds for a feature
        """
        state = self._scale(state)

        if self.step_coder:
            assert type(self.converter) == Converter
            self.base, p_spikes = self.converter._encode_step_(
                state, self.base, self.converter_th
            )
            p_spikes = (p_spikes / 2) + 0.5
            self.prev_state = state

        else:
            p_spikes = np.array([[]])

        if self.rate_coder:
            if self.step_coder:
                p_spikes = np.append(p_spikes, state, axis=1)
            else:
                p_spikes = state

        up = p_spikes * 0
        down = p_spikes * 0
        up[p_spikes >= 0.5] = (p_spikes[p_spikes >= 0.5] - 0.5) * 2  # type:ignore
        down[p_spikes < 0.5] = (0.5 - p_spikes[p_spikes < 0.5]) * 2  # type:ignore
        p_spikes_split = np.append(up, down, axis=1)

        if self.add_inverted_inputs and self.split_exc_inh:
            inverse_p_spikes = self.max_firing_rate - p_spikes_split
            p_spikes = np.append(p_spikes_split, inverse_p_spikes, axis=1)
        elif self.add_inverted_inputs:
            inverse_p_spikes = self.max_firing_rate - p_spikes
            p_spikes = np.append(p_spikes, inverse_p_spikes, axis=1)
        elif self.split_exc_inh:
            p_spikes = p_spikes_split

        # Rescale probs between [0.0474, 0.9526]
        p_spikes *= self.max_firing_rate

        if self.spike_train_conversion_method == "poisson":
            output = poisson(p_spikes, self.seq_length)  # type: ignore
        else:
            output = rate(p_spikes, self.seq_length)  # type: ignore

        # Burst Spikes
        # output[int(3*output.shape[0]/4):, :, :] = 0
        if return_norm:
            return output, p_spikes  # type:ignore
        else:
            return output
