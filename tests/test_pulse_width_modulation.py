import unittest
import torch
import numpy as np
from encoding.pulse_width_modulation import PulseWidthModulation


class TestPulseWidthModulation(unittest.TestCase):
    def create_encoder(
        self,
        frequency=1.0,
        init_val=0.0,
        min_value=0.0,
        max_value=1.0,
        scale_factor=1.0,
        down_spike=True,
    ):
        return PulseWidthModulation(
            frequency=torch.tensor([frequency]),
            init_val=torch.tensor([init_val]),
            min_value=torch.tensor([min_value]),
            max_value=torch.tensor([max_value]),
            scale_factor=torch.tensor([scale_factor]),
            down_spike=down_spike,
        )

    def test_normalize_tensor(self):
        encoder = self.create_encoder()
        signal = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        normalized = encoder.normalize_tensor(signal)
        self.assertTrue(torch.all(normalized >= 0) and torch.all(normalized <= 1))

    def test_sawtooth(self):
        encoder = self.create_encoder()
        sawtooth_signal = encoder.sawtooth(freq=1.0, lenght=1000)
        self.assertEqual(len(sawtooth_signal), 1000)
        self.assertTrue(np.all(sawtooth_signal >= 0) and np.all(sawtooth_signal <= 1))

    def test_moving_average(self):
        encoder = self.create_encoder()
        signal = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        filtered_signal = encoder.moving_average(signal, window_size=3)
        expected_output = torch.tensor([1 / 3, 2 / 3, 1 / 3])
        self.assertTrue(torch.allclose(filtered_signal, expected_output, atol=1e-6))

    def test_encode_decode_consistency(self):
        encoder = self.create_encoder()
        signal = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]])
        encoded = encoder.encode(signal)
        decoded = encoder.decode(encoded)
        encoded_again = encoder.encode(decoded)
        self.assertTrue(torch.allclose(encoded, encoded_again, atol=0.1))


if __name__ == "__main__":
    unittest.main()
