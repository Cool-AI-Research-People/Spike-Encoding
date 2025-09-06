import torch
import torch.nn.functional as F
from torchmetrics import MeanSquaredError
import numpy as np
import scipy as scipy
from .base_converter import BaseConverter
import optuna


class PulseWidthModulation(BaseConverter):
    def __init__(
        self,
        frequency: torch.Tensor = torch.tensor(
            [1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float
        ),
        init_val: torch.Tensor = torch.tensor(
            [0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float
        ),
        min_value: float = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float),
        max_value: float = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float),
        scale_factor: float = torch.tensor(
            [1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float
        ),
        down_spike: bool = True,
    ):
        super(PulseWidthModulation, self).__init__()

        if isinstance(frequency, (float, int)):
            self.frequency = torch.tensor([frequency], dtype=torch.float)
        elif isinstance(frequency, (list, tuple)):
            self.frequency = torch.tensor(frequency, dtype=torch.float)
        elif isinstance(frequency, torch.Tensor):
            self.frequency = frequency.float()
        else:
            raise ValueError(
                "frequency must be a float, int, list, tuple, or torch.Tensor"
            )
        if isinstance(init_val, (float, int)):
            self.init_val = torch.tensor([init_val], dtype=torch.float)
        elif isinstance(init_val, (list, tuple)):
            self.init_val = torch.tensor(init_val, dtype=torch.float)
        elif isinstance(init_val, torch.Tensor):
            self.init_val = init_val.float()
        else:
            raise ValueError(
                "init_val must be a float, int, list, tuple, or torch.Tensor"
            )
        self.min_value = min_value
        self.max_value = max_value
        self.scale_factor = scale_factor
        self.down_spike = down_spike

    def encode(
        self,
        signal: torch.Tensor,
        frequency: torch.Tensor = torch.tensor([], dtype=torch.float),
        down_spike: bool = None,
        isNormed: bool = False,
    ):
        """
        Encode a signal into spikes

        Args:
        - signal: The signal to be encoded as a tensor
        - frequency: The frequency of the carrier signal in Hertz
        - down_spike: Boolean that defines whether negative spikes are generated

        Returns:
        - spikes: The encoded spikes as a tensor
        """
        signal.squeeze_()
        signal = torch.atleast_2d(signal)
        if signal.ndimension() > 2:
            raise ValueError(
                f"{type(self).__name__}.{self.encode.__name__}() only supports input tensors with dimension <=2, but got dimension {signal.ndim}."
            )
        if isNormed:
            signal_norm = signal
        else:
            signal_norm = self.normalize_tensor(signal)
        self.init_val = signal[:][0]
        frequency = self.frequency if frequency.numel() == 0 else frequency
        self.frequency = frequency
        down_spike = self.down_spike if down_spike == None else down_spike

        while len(frequency) < signal_norm.shape[0]:
            frequency = torch.cat(
                (frequency, frequency[: signal_norm.shape[0] - len(frequency)])
            )

        frequency = frequency[: signal_norm.shape[0]]

        up_spikes = torch.zeros_like(signal_norm)
        down_spikes = torch.zeros_like(signal_norm)
        pwm = torch.zeros_like(signal_norm)

        carrier_signal = np.linspace(0, 1, 1000)

        for r in range(signal_norm.shape[0]):
            count = 0
            for t in range(1, signal_norm.shape[1]):
                if signal_norm[r, t] < (
                    np.fmod(frequency[r] * carrier_signal[count], 1.0)
                ):
                    pwm[r, t] = 1
                else:
                    pwm[r, t] = 0
                if (
                    signal_norm[r, t]
                    > (1 - (np.fmod(frequency[r] * carrier_signal[count], 1.0)))
                ) & down_spike:
                    pwm[r, t] = -1

                count += 1
                if count == 1000:
                    count = 0
                elif np.fmod(frequency[r] * carrier_signal[count], 1.0) < np.fmod(
                    frequency[r] * carrier_signal[count - 1], 1.0
                ):
                    count = 0

        for t in range(1, signal_norm.shape[1]):
            up_spikes[(pwm[:, t] == 1) & (pwm[:, t - 1] != 1), t] = 1
            down_spikes[(pwm[:, t] == -1) & (pwm[:, t - 1] != -1), t] = -1

        if not down_spike:
            return torch.stack((up_spikes, torch.zeros_like(signal_norm[0])))
        else:
            return torch.stack((up_spikes, down_spikes))

    def decode(
        self,
        spikes: torch.Tensor,
        init_val: torch.Tensor = torch.tensor([], dtype=torch.float),
    ):
        """
        decode the spike train to the original signal

        Args:
        - spikes: the encoded signal as Tensor
        - frequency: the given frequency of the carrier signal in Hz

        Returns:
        - reconstructed_signal: the reconstructed signal.
        """
        if spikes.ndimension() == 3:
            # Assume shape [polarity, features, spikes]
            spikes = spikes[0].add(spikes[1])
        spikes = torch.atleast_2d(spikes)

        reconstructed_signal = torch.zeros_like(spikes)
        self.init_val = self.init_val if init_val.numel() == 0 else init_val.clone()
        if len(self.init_val) < spikes.shape[0]:
            print("init_val is not the same length as the number of signals")
            for i in range(spikes.shape[0]):
                self.init_val[i] = self.init_val[0]
        self.init_val = self.init_val[: len(spikes)]

        for i in range(len(self.init_val)):
            self.init_val[i] = self.scale_factor[i] * (
                self.init_val[i] - self.min_value[i]
            )

        frequency = self.frequency
        carrier_signal = np.linspace(0, 1, 1000)

        reconstructed_signal[:, 0] = self.init_val
        # for t in range(1, spikes.shape[1]) :
        #   reconstructed_signal[:, t] = reconstructed_signal[: ,t-1]
        #  reconstructed_signal[spikes[:, t] != spikes[:, t-1], t] = carrier_signal[t]

        for r in range(spikes.shape[0]):
            last_spike_time = 0
            count = 0
            for t in range(1, spikes.shape[1]):
                if spikes[r, t] == 1:
                    lins = np.linspace(
                        reconstructed_signal[r, last_spike_time],
                        np.fmod(frequency[r] * carrier_signal[count], 1.0),
                        (1 + t - last_spike_time),
                    )
                    for t_ in range(last_spike_time, t + 1):
                        reconstructed_signal[r, t_] = lins[t_ - last_spike_time]
                    last_spike_time = t
                elif spikes[r, t] == -1:
                    lins = np.linspace(
                        reconstructed_signal[r, last_spike_time],
                        (1.0 - np.fmod(frequency[r] * carrier_signal[count], 1.0)),
                        (1 + t - last_spike_time),
                    )
                    for t_ in range(last_spike_time, t + 1):
                        reconstructed_signal[r, t_] = lins[t_ - last_spike_time]
                    last_spike_time = t
                elif t == spikes.shape[1] - 1:
                    # Last time step is treated as if there was a spike
                    lins = np.linspace(
                        reconstructed_signal[r, last_spike_time],
                        (1.0 - np.fmod(frequency[r] * carrier_signal[count], 1.0)),
                        (1 + t - last_spike_time),
                    )
                    for t_ in range(last_spike_time, t + 1):
                        reconstructed_signal[r, t_] = lins[t_ - last_spike_time]

                count += 1
                if count == 1000:
                    count = 0
                elif np.fmod(frequency[r] * carrier_signal[count], 1.0) < np.fmod(
                    frequency[r] * carrier_signal[count - 1], 1.0
                ):
                    count = 0

        for i in range(reconstructed_signal.shape[0]):
            for j in range(reconstructed_signal.shape[1]):
                reconstructed_signal[i][j] = (
                    reconstructed_signal[i][j].item() / self.scale_factor[i]
                    + self.min_value[i]
                )

        return reconstructed_signal

    def optimize(
        self,
        data: torch.Tensor,
        error_function=MeanSquaredError(),
        trials=100,
        down_spike=True,
        plot_history=False,
    ):

        self.down_spike = down_spike
        """
        Optimize the frequency of the carrier signal to an optimum MeanSquaredError

        Args:
        - data: The given Signal

        Returns:
        - best_freqencey: best frequency from the optimize function.
        """

        if data.ndim > 2:
            raise ValueError(
                f"{type(self).__name__}.{self.optimize.__name__}() only supports input tensors with dimension <=2, but got dimension {data.ndim}."
            )
        data = torch.atleast_2d(data)
        # synth_sig = self.normalize_tensor(data)
        synth_sig = torch.zeros_like(data)

        bounds = [1.0, 750.0]
        _freq = torch.zeros(len(data))
        _ma_win = torch.zeros(len(data))

        for i in range(len(data)):

            def loss_function(trial):
                frequency = trial.suggest_float("frequency", bounds[0], bounds[1])
                MA_window = trial.suggest_int("MA_window", 1, 1)
                MA_window = 2 * MA_window - 1
                synth_sig[
                    i, ((MA_window - 1) // 2) : (len(data[i]) - (MA_window - 1) // 2)
                ] = self.moving_average(data[i], MA_window)
                frequency = torch.tensor([frequency])
                self.frequency = frequency
                encoded_signal = self.encode(
                    synth_sig[i], frequency, isNormed=False, down_spike=down_spike
                )
                decoded_signal = self.decode(encoded_signal)
                decoded_signal.squeeze_()
                decoded_signal = decoded_signal[: len(data[i])]
                loss = error_function(decoded_signal, data[i])
                return loss

            optuna.logging.set_verbosity(optuna.logging.WARNING)  # hide logging
            study = optuna.create_study(direction="minimize")
            study.optimize(loss_function, n_trials=trials, show_progress_bar=True)
            _freq[i] = study.best_params["frequency"]
            _ma_win[i] = study.best_params["MA_window"]

            if plot_history:
                # Plot the optimization values
                optuna.visualization.plot_optimization_history(study).show()
                optuna.visualization.plot_slice(
                    study, params=["frequency", "MA_window"]
                ).show()

        self.frequency = _freq.float()
        print(f"best_frequency={self.frequency}")
        print(f"best_MA_window={_ma_win.float()}")

        return self.frequency

    def sawtooth(self, freq: float, lenght: int):
        """Generate the carrier signal in the form of a sawtooth waveform"""
        carrier_signal = np.linspace(0, 1, 1000)
        count = 0
        sawtooth_signal = np.zeros(lenght)

        for t in range(1, lenght):
            sawtooth_signal[t] = np.fmod(freq * carrier_signal[count], 1.0)

            count += 1
            if count == 1000:
                count = 0
            elif np.fmod(freq * carrier_signal[count], 1.0) < np.fmod(
                freq * carrier_signal[count - 1], 1.0
            ):
                count = 0

        return sawtooth_signal

    def normalize_tensor(self, tensor):
        """Normalize a tensor to the range between 0 and 1."""
        self.min_value = torch.min(tensor, dim=1)[0]
        self.max_value = torch.max(tensor, dim=1)[0]
        self.scale_factor = 1.0 / (self.max_value - self.min_value)

        normalized_tensor = torch.zeros(tensor.shape)
        for i in range(len(tensor)):
            normalized_tensor[i] = self.scale_factor[i] * (
                tensor[i] - self.min_value[i]
            )
        return normalized_tensor

    def moving_average(self, signal, window_size):
        # Ensure signal is a 2D tensor with shape (batch_size, channels, length)
        filtered_signal = signal.unsqueeze(0).unsqueeze(0)

        # Create a window (kernel) of the given size
        window = torch.ones(1, 1, window_size) / window_size

        # Apply the convolution between the signal and the window
        filtered_signal = F.conv1d(filtered_signal, window, padding=0)

        # Remove the extra dimensions added
        filtered_signal = filtered_signal.squeeze(0).squeeze(0)

        return filtered_signal
