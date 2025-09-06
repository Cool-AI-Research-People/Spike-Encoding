import unittest
import numpy as np

from encoding.gymnasium_bounds_finder import ScalerFactory
from encoding.gymnasium_encoder import GymnasiumEncoder


class TestGymnasiumEncoder(unittest.TestCase):
    def create_encoder(
        self,
        max_values: np.ndarray,
        seq_length=10,
        poisson=False,
        split_exc_inh=True,
        max_firing_rate=1.0,
        add_inverted_inputs=False,
    ):
        scaler_factory = ScalerFactory()
        scaler = scaler_factory.from_known_values(-max_values, max_values)
        encoder = GymnasiumEncoder(
            max_values.shape[0],
            batch_size=1,
            seq_length=seq_length,
            scaler=scaler,
            rate_coder=True,
            step_coder=False,
            split_exc_inh=split_exc_inh,
            spike_train_conversion_method="poisson" if poisson else "deterministic",
            max_firing_rate=max_firing_rate,
            add_inverted_inputs=add_inverted_inputs,
        )
        return encoder

    def test_encodes_max_value_as_max_positive_spikes(self):
        encoder = self.create_encoder(max_values=np.array([2.4]), seq_length=10)
        spike_train = encoder.encode(np.array([[2.4]]))
        n_spikes = np.sum(spike_train[:], axis=0)
        n_positive_spikes = n_spikes[0][0]
        self.assertEqual(n_positive_spikes, 10)

    def test_has_more_outputs_when_split_exc_inh_is_true(self):
        split_encoder = self.create_encoder(max_values=np.array([2.4]), seq_length=10)
        split_spike_train = split_encoder.encode(np.array([[2.4]]))
        self.assertEqual(split_spike_train.shape, (10, 1, 2))
        regular_encoder = self.create_encoder(
            max_values=np.array([2.4]), seq_length=10, split_exc_inh=False
        )
        regular_spike_train = regular_encoder.encode(np.array([[-2.4]]))
        self.assertEqual(regular_spike_train.shape, (10, 1, 1))

    def test_encodes_max_value_as_zero_negative_spikes(self):
        encoder = self.create_encoder(max_values=np.array([2.4]), seq_length=10)
        spike_train = encoder.encode(np.array([[2.4]]))
        n_spikes = np.sum(spike_train[:], axis=0)
        n_negative_spikes = n_spikes[0][1]
        self.assertEqual(n_negative_spikes, 0)

    def test_encodes_min_value_as_zero_positive_spikes(self):
        encoder = self.create_encoder(max_values=np.array([2.4]), seq_length=10)
        spike_train = encoder.encode(np.array([[-2.4]]))
        n_spikes = np.sum(spike_train[:], axis=0)
        n_positive_spikes = n_spikes[0][0]
        self.assertEqual(n_positive_spikes, 0)

    def test_encodes_min_value_as_max_negative_spikes(self):
        encoder = self.create_encoder(max_values=np.array([2.4]), seq_length=10)
        spike_train = encoder.encode(np.array([[-2.4]]))
        n_spikes = np.sum(spike_train[:], axis=0)
        n_negative_spikes = n_spikes[0][1]
        self.assertEqual(n_negative_spikes, 10)

    def test_encodes_low_positive_value_as_no_negative_spikes(self):
        encoder = self.create_encoder(max_values=np.array([2.4]), seq_length=10)
        spike_train = encoder.encode(np.array([[0.1]]))
        n_spikes = np.sum(spike_train[:], axis=0)
        n_negative_spikes = n_spikes[0][1]
        self.assertEqual(n_negative_spikes, 0)

    def test_poisson_encoding_is_exact_at_min_and_max(self):
        encoder = self.create_encoder(
            max_values=np.array([2.4]), seq_length=10, split_exc_inh=False, poisson=True
        )
        spike_train_0 = encoder.encode(np.array([[-2.4]]))
        n_spikes_0 = np.sum(spike_train_0[:], axis=0)[0][0]
        self.assertEqual(n_spikes_0, 0)
        spike_train_max = encoder.encode(np.array([[2.4]]))
        n_spikes_max = np.sum(spike_train_max[:], axis=0)[0][0]
        self.assertEqual(n_spikes_max, 10)

    def test_poisson_encoding_is_approximately_correct(self):
        np.random.seed(0)
        encoder = self.create_encoder(
            max_values=np.array([2.4]),
            seq_length=1000,
            split_exc_inh=True,
            poisson=True,
        )
        spike_train = encoder.encode(np.array([[1.2]]))
        n_spikes = np.sum(spike_train[:], axis=0)[0][0]
        difference_in_spikes = abs(500 - n_spikes)
        self.assertLess(difference_in_spikes, 50)
        self.assertGreater(difference_in_spikes, 0)

    def test_respects_max_firing_rate(self):
        encoder = self.create_encoder(
            max_values=np.array([2.4]), seq_length=100, max_firing_rate=0.5
        )
        spike_train = encoder.encode(np.array([[2.4]]))
        n_spikes = np.sum(spike_train[:], axis=0)
        n_positive_spikes = n_spikes[0][0]
        self.assertGreater(n_positive_spikes, 35)
        self.assertLess(n_positive_spikes, 51)

    def test_add_inverse_doubles_feature_count(self):
        inverse_added_encoder = self.create_encoder(
            max_values=np.array([2.4]),
            seq_length=10,
            split_exc_inh=False,
            add_inverted_inputs=True,
        )
        split_spike_train = inverse_added_encoder.encode(np.array([[2.4]]))
        self.assertEqual(split_spike_train.shape, (10, 1, 2))

    def test_add_inverse_and_split_exc_inh_quadruples_feature_count(self):
        encoder = self.create_encoder(
            max_values=np.array([2.4]),
            seq_length=10,
            split_exc_inh=True,
            add_inverted_inputs=True,
        )
        spike_train = encoder.encode(np.array([[2.4]]))
        self.assertEqual(spike_train.shape, (10, 1, 4))
        n_spikes = np.sum(spike_train[:], axis=0)
        n_positive_spikes = n_spikes[0][0]
        self.assertEqual(n_positive_spikes, 10)
        n_negative_spikes = n_spikes[0][1]
        self.assertEqual(n_negative_spikes, 0)
        n_positive_spikes_inverted = n_spikes[0][2]
        self.assertEqual(n_positive_spikes_inverted, 0)
        n_negative_spikes_inverted = n_spikes[0][3]
        self.assertEqual(n_negative_spikes_inverted, 10)

    def test_inverse_spike_trains_have_inverse_firing_rate(self):
        encoder = self.create_encoder(
            max_values=np.array([2.4]),
            seq_length=10,
            split_exc_inh=False,
            add_inverted_inputs=True,
            poisson=False,
        )
        spike_train = encoder.encode(np.array([[2.4]]))
        n_spikes = np.sum(spike_train[:], axis=0)
        n_original_spikes = n_spikes[0][0]
        n_inverted_spikes = n_spikes[0][1]
        self.assertEqual(n_original_spikes, 10)
        self.assertEqual(n_inverted_spikes, 0)


if __name__ == "__main__":
    unittest.main()
