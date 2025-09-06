import unittest
import numpy as np

from encoding.bin_encoder import BinEncoder


class TestBinEncoder(unittest.TestCase):
    def test_encodes_correct_number_of_bins(self):
        encoder = BinEncoder(
            10,
            min_values=np.array([-2, -5]),
            max_values=np.array([2, 5]),
            n_bins=3,
        )
        spike_train = encoder.encode(np.array([1.8, 0]))
        self.assertEqual(spike_train.shape, (10, 1, 6))

    def test_respects_max_firing_rate(self):
        seq_length = 100
        max_firing_rate = 0.6
        encoder = BinEncoder(
            100,
            min_values=np.array([0]),
            max_values=np.array([1]),
            n_bins=2,
            max_firing_rate=0.6,
            spike_train_conversion_method="deterministic",
        )
        # NOTE second bin is at 0.75, thus encoding 0.75 gives the max_firing_rate in that bin
        # however, firing rates sum to 1, so the actual firing rate in that bin is a little lower
        spike_train = encoder.encode(np.array([0.75]))
        n_spikes = np.sum(spike_train[:], axis=0)
        n_spikes_in_second_bin = n_spikes[0][1]
        self.assertGreaterEqual(
            n_spikes_in_second_bin, (seq_length * max_firing_rate) * 0.9
        )
