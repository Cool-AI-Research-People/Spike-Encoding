import unittest
import torch
from encoding.bens_spiker_algorithm import BensSpikerAlgorithm


class TestBensSpikerAlgorithm(unittest.TestCase):
    def create_encoder(
        self, threshold=0.5, down_spike=True, filter_order=10, filter_cutoff=0.2
    ):
        return BensSpikerAlgorithm(
            threshold=threshold,
            down_spike=down_spike,
            filter_order=filter_order,
            filter_cutoff=filter_cutoff,
        )

    def test_encode_multi_channel_signal(self):
        encoder = self.create_encoder(
            threshold=[0.5, 0.6], filter_order=[10, 15], filter_cutoff=[0.2, 0.3]
        )
        signal = torch.tensor([[0.1, 0.5, 0.9], [0.2, 0.6, 1.0]])
        encoded = encoder.encode(signal)
        self.assertEqual(encoded.shape, (2, 3))

    def test_decode_multi_channel_signal(self):
        encoder = self.create_encoder(
            threshold=[0.5, 0.6], filter_order=[10, 15], filter_cutoff=[0.2, 0.3]
        )
        signal = torch.tensor([[0.1, 0.5, 0.9], [0.2, 0.6, 1.0]])
        encoded = encoder.encode(signal)
        decoded = encoder.decode(encoded)
        self.assertEqual(len(decoded), 2)
        self.assertEqual(len(decoded[0]), 3)
        self.assertEqual(len(decoded[1]), 3)

    def test_normalize_tensor(self):
        encoder = self.create_encoder()
        signal = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        normalized = encoder.normalize_tensor(signal)
        self.assertTrue(torch.all(normalized >= 0) and torch.all(normalized <= 1))

    def test_fir_filter(self):
        encoder = self.create_encoder(filter_order=[10, 15], filter_cutoff=[0.2, 0.3])
        fir_coeffs = encoder.fir_filter()
        self.assertEqual(len(fir_coeffs), 2)
        self.assertEqual(len(fir_coeffs[0]), 11)  # filter_order + 1
        self.assertEqual(len(fir_coeffs[1]), 16)  # filter_order + 1


if __name__ == "__main__":
    unittest.main()
