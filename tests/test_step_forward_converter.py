import unittest
from torch import Tensor
from encoding import StepForwardConverter
import numpy as np
import pandas as pd


def generate_stepwise_data(seed: int, steps: int, step_length: int):
    """Generate stepwise data for testing.

    Args:
        seed: Random seed for reproducibility
        steps: Number of steps in the data
        step_length: Length of each step

    Returns:
        DataFrame containing stepwise data with columns ['x', 'x_norm']
    """
    np.random.seed(seed)

    # Generate step values
    step_values = np.random.uniform(-1, 1, steps)
    # Repeat each value step_length times
    x = np.repeat(step_values, step_length)

    df = pd.DataFrame({"x": x})
    df["x_norm"] = (df["x"] - df["x"].mean()) / df["x"].std()
    return df


class TestStepForwardConverter(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = 10
        self.steps = 10
        self.step_length = 100
        self.threshold = 0.1
        self.data = generate_stepwise_data(
            seed=self.seed, steps=self.steps, step_length=self.step_length
        )
        # Note: tests in this class expect self.data to have at least 2 dimensions
        self.signal = Tensor(self.data.values).swapaxes(0, 1)
        self.signal_1d = Tensor(self.data[self.data.columns[0]].values)

    def test_encode(self):
        converter = StepForwardConverter(threshold=self.threshold, down_spike=True)
        # Check encoded data shape for 1D data
        events = converter.encode(self.signal_1d)
        self.assertTupleEqual(events.shape, (2, 1, self.steps * self.step_length))
        # Ensure that threshold has not been altered during execution
        self.assertAlmostEqual(converter.threshold.item(), self.threshold)
        # Check if down spikes are dropped correctly
        events_nodown = converter.encode(self.signal_1d, down_spike=False)
        self.assertTrue(events_nodown[1].count_nonzero() == 0)
        ## Up spikes should still be the same
        self.assertSequenceEqual(events[0].tolist(), events_nodown[0].tolist())
        # Check encoding of multi-dimensional data
        event_cols_multi = converter.encode(self.signal)
        self.assertTupleEqual(
            event_cols_multi.shape,
            (2, len(self.data.columns), self.steps * self.step_length),
        )
        # Threshold dimension larger than data dimension should trigger a warning
        converter.threshold = Tensor([self.threshold, self.threshold])
        self.assertWarns(SyntaxWarning, converter.encode, self.signal_1d)
        # This warning should be gone if we explicitly state the threshold to use
        converter.encode(self.signal_1d, threshold=converter.threshold[0])

    def test_decode(self):
        # Ensure shape of reconstructed signal matches the original one
        converter = StepForwardConverter(threshold=self.threshold, down_spike=True)
        # 1D
        events = converter.encode(self.signal_1d)
        reconstruction = converter.decode(events)
        self.assertTupleEqual(reconstruction.shape, self.signal_1d.unsqueeze(0).shape)
        self.assertNotEqual(reconstruction.count_nonzero(), 0)
        # 2D
        events = converter.encode(self.signal)
        reconstruction = converter.decode(events)
        self.assertTupleEqual(reconstruction.shape, self.signal.shape)
        self.assertNotEqual(reconstruction.count_nonzero(), 0)
        # Threshold dimension mismatch
        converter.threshold = Tensor([self.threshold, self.threshold])
        with self.assertWarns(SyntaxWarning):
            events = converter.encode(self.signal_1d)
        with self.assertWarns(SyntaxWarning):
            reconstruction = converter.decode(events)
        self.assertTupleEqual(reconstruction.shape, self.signal_1d.unsqueeze(0).shape)
        self.assertNotEqual(reconstruction.count_nonzero(), 0)
