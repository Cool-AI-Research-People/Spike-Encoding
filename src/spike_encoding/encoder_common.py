from numpy.typing import NDArray as ndarray
import numpy as np
import numpy.random as rndm


def poisson(spike_probabilities: ndarray, max_t: int) -> ndarray:
    spikes = (
        np.asarray(
            [
                spike_probabilities > rndm.random(spike_probabilities.shape)
                for i in range(int(max_t))
            ]
        )
        + 0
    )

    return spikes


def rate(spike_probabilities: ndarray, max_t: int) -> ndarray:
    spikes = np.zeros((max_t, *spike_probabilities.shape), dtype=int)

    for index in np.ndindex(spike_probabilities.shape):
        i = spike_probabilities[index]
        idx = np.array(np.arange(max_t * i, 0, -1) / i, dtype=int) - 1
        idx = np.clip(idx, 0, max_t - 1)  # Ensure indices are within bounds

        # Use tuple unpacking to set the value at the specified indices
        spikes[(idx,) + index] = 1

    return spikes
