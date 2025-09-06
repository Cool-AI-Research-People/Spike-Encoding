import unittest
import torch
from encoding.lif_based_encoding import LIFBasedEncoding


class TestLIFBasedEncoding(unittest.TestCase):
    def create_encoder(
        self, threshold=0.5, membrane_constant=0.9, down_spike=True, padding=False
    ):
        return LIFBasedEncoding(
            threshold=torch.tensor([threshold]),
            membrane_constant=torch.tensor([membrane_constant]),
            down_spike=down_spike,
            padding=padding,
        )

    def test_down_spike_effect(self):
        up_spike_encoder = self.create_encoder(down_spike=False)
        down_spike_encoder = self.create_encoder(down_spike=True)
        signal = torch.tensor([[0.1, 0.9, 0.1]])
        up_spike_encoded = up_spike_encoder.encode(signal)
        down_spike_encoded = down_spike_encoder.encode(signal)
        self.assertFalse(torch.allclose(up_spike_encoded, down_spike_encoded))

    def test_normalize_tensor(self):
        encoder = self.create_encoder()
        signal = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        normalized = encoder.normalize_tensor(signal)
        self.assertTrue(torch.all(normalized >= -1) and torch.all(normalized <= 1))

    def test_adjust_threshold_shape(self):
        encoder = self.create_encoder()
        threshold = torch.tensor([0.5, 0.6])
        signal = torch.tensor([[0.1, 0.5, 0.9], [0.2, 0.6, 1.0], [0.3, 0.7, 1.1]])
        adjusted_threshold = encoder._adjust_threshold_shape(threshold, signal)
        self.assertEqual(adjusted_threshold.shape[0], signal.shape[0])

    def test_threshold_as_tensor(self):
        encoder = self.create_encoder()
        threshold_float = 0.5
        threshold_tensor = encoder._threshold_as_tensor(threshold_float)
        self.assertIsInstance(threshold_tensor, torch.Tensor)
        self.assertEqual(threshold_tensor.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
