import torch
from torch import Tensor
from torchmetrics import MeanSquaredError
from scipy.signal import firwin
import optuna
from optuna.samplers import TPESampler

from encoding.base_converter import BaseConverter


class BensSpikerAlgorithm(BaseConverter):
    def __init__(
        self,
        threshold: torch.Tensor = torch.tensor(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float
        ),
        down_spike: bool = True,
        min_value=None,
        max_value=None,
        scale_factor=None,
        filter_order: torch.Tensor = torch.tensor(
            [10, 10, 10, 10, 10, 10, 10], dtype=torch.int
        ),
        filter_cutoff: torch.Tensor = torch.tensor(
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=torch.float
        ),
    ):
        super().__init__()
        if isinstance(threshold, (float, int)):
            self.threshold = torch.tensor([threshold], dtype=torch.float)
        elif isinstance(threshold, (list, tuple)):
            self.threshold = torch.tensor(threshold, dtype=torch.float)
        elif isinstance(threshold, torch.Tensor):
            self.threshold = threshold.float()
        else:
            raise ValueError(
                "threshold must be a float, int, list, tuple, or torch.Tensor"
            )
        self.down_spike = down_spike
        self.min_value = min_value
        self.max_value = max_value
        self.scale_factor = scale_factor
        if isinstance(filter_order, int):
            self.filter_order = torch.tensor([filter_order], dtype=torch.int)
        elif isinstance(filter_order, (list, tuple)):
            self.filter_order = torch.tensor(filter_order, dtype=torch.int)
        elif isinstance(filter_order, torch.Tensor):
            self.filter_order = filter_order.int()
        else:
            raise ValueError("filter_order must be an int, list, tuple or torch.Tensor")

        if isinstance(filter_cutoff, (float, int)):
            self.filter_cutoff = torch.tensor([filter_cutoff], dtype=torch.float)
        elif isinstance(filter_cutoff, (list, tuple)):
            self.filter_cutoff = torch.tensor(filter_cutoff, dtype=torch.float)
        elif isinstance(filter_cutoff, torch.Tensor):
            self.filter_cutoff = filter_cutoff.float()
        else:
            raise ValueError(
                "filter_cutoff must be a float, int, list, tuple, or torch.Tensor"
            )

    def encode(
        self,
        signal: torch.Tensor,
        filter_order: torch.Tensor = torch.tensor([], dtype=torch.int),
        filter_cutoff: torch.Tensor = torch.tensor([], dtype=torch.float),
        threshold: torch.Tensor = torch.tensor([], dtype=torch.float),
        isNormed: bool = False,
    ):
        """
        Encodes a signal into spikes using an FIR filter and a threshold.

        Args:
            signal (torch.Tensor): The signal to be encoded as a tensor.
            filter_order (torch.Tensor): possible to give a order, DEFAULT: 100
            cutoff (torch.Tensor): the cutoff percentage for the FIR-filter. DEFAULT: 0.2
            threshold (torch.Tensor): The threshold to generate spikes. DEFAULT: 0.5

        Returns:
            spikes (torch.Tensor): The encoded spikes as a tensor.
        """
        # === Standardize signal shape to [B, C, T] ===
        original_shape = signal.shape
        signal = signal.unsqueeze(0) if signal.ndim == 1 else signal  # [T] → [1, T]
        signal = signal.unsqueeze(0) if signal.ndim == 2 else signal  # [C, T] → [1, C, T]
        B, C, T = signal.shape
        print(f"[BSA] Input shape adjusted from {original_shape} to {signal.shape} (B={B}, C={C}, T={T})")

        B, C, T = signal.shape
        signal = signal.reshape(B * C, T)  # Flatten to [B*C, T] for batch-wise processing

        if isNormed:
            signal_norm = signal
        else:
            signal_norm = self.normalize_tensor(signal)

        self.filter_order = (
            self.filter_order if filter_order.numel() == 0 else filter_order
        )
        self.filter_cutoff = (
            self.filter_cutoff if filter_cutoff.numel() == 0 else filter_cutoff
        )
        self.threshold = self.threshold if threshold.numel() == 0 else threshold

        if (
            len(self.filter_order) < len(signal)
            or len(self.filter_cutoff) < len(signal)
            or len(self.threshold) < len(signal)
        ):
            raise ValueError(
                "filter_order, filter_cutoff and threshold must be a tensor of the same length as the input signal."
            )
        self.filter_order = self.filter_order[: len(signal)]
        self.filter_cutoff = self.filter_cutoff[: len(signal)]
        self.threshold = self.threshold[: len(signal)]

        FIR = self.fir_filter()
        num_rows, L = signal.shape

        # print(f"Encoding {num_rows} signals with length {L}. filter_order={self.filter_order}, filter_cutoff={self.filter_cutoff}, threshold={self.threshold} \nfir_filter={FIR}")

        spike_trains = torch.zeros(num_rows, L)

        for row in range(num_rows):
            F = len(FIR[row])

            s = signal_norm[row]

            out = torch.zeros(L)
            for t in range(1, L):
                err1 = 0
                err2 = 0

                for k in range(1, F):
                    if t + k - 1 < L:
                        err1 += torch.abs(s[t + k - 1] - FIR[row][k])
                        err2 += torch.abs(s[t + k - 1])

                if err1 <= err2 - self.threshold[row]:
                    out[t] = 1
                    for k in range(1, F):
                        if t + k - 1 < L:
                            s[t + k - 1] -= FIR[row][k]

            spike_trains[row] = out

        spike_trains = spike_trains.reshape(B, C, T)
        squeeze_shape = spike_trains.shape
        spike_trains = spike_trains.squeeze()
        print(f"[BSA] Output shape reshaped to {squeeze_shape}, squeezed to {spike_trains.shape} "
              f"(B={'kept' if squeeze_shape[0] > 1 else 'squeezed'}, "
              f"C={'kept' if squeeze_shape[1] > 1 else 'squeezed'}, "
              f"T={'kept' if squeeze_shape[2] > 1 else 'squeezed'})")

        return spike_trains

    def decode(self, spikes: torch.Tensor):

        spikes.squeeze_()
        spikes = torch.atleast_2d(spikes)

        FIR = self.fir_filter()

        feature_size = spikes.shape[0]
        L = spikes.shape[1]

        reconstructed_signals = torch.zeros((feature_size, L))

        for row in range(feature_size):
            spike_train = spikes[row].clone().detach()

            out = torch.zeros(L)
            F = len(FIR[row])
            padding = F // 2
            padded_input = torch.cat(
                (torch.zeros(padding), spike_train, torch.zeros(padding))
            )

            for i in range(padding, L):
                for j in range(F):
                    if i + j < L + padding * 2:
                        out[i] += padded_input[i + j - padding + 1] * FIR[row][j]

            reconstructed_signals[row] = out

        result_vector = []
        for i in range(feature_size):
            scaled_signal = (
                reconstructed_signals[i] / self.scale_factor[i] + self.min_value[i]
            )
            result_vector.append(scaled_signal.tolist())

        if len(result_vector) == 1:
            return torch.tensor(result_vector[0])
        else:
            return torch.tensor(result_vector)

    def optimize(
        self,
        data: torch.Tensor,
        trials: int = 100,
        error_function=MeanSquaredError(),
        plot_history=False,
    ):
        """
        Optimize the threshold and the cutoff persentage of the FIR-Filter of the encoding

        Args:
        - data (torch.Tensor): The given Signal

        Returns:
        - self.threshold: best threshold for the Singnal.
        - self.filter_cutoff: best cutoff for the FIR-filter
        """

        if data.ndim > 2:
            raise ValueError(
                f"{type(self).__name__}.{self.optimize.__name__}() only supports input tensors with dimension <=2, but got dimension {data.ndim}."
            )
        data = torch.atleast_2d(data)
        synth_sig = data  # self.normalize_tensor(data)
        f_order = torch.zeros(len(synth_sig))
        f_cutoff = torch.zeros(len(synth_sig))
        _threshold = torch.zeros(len(synth_sig))

        for i in range(len(synth_sig)):

            def loss_function(trial):
                filter_order = trial.suggest_int("filter_order", 35, 50)
                filter_cutoff = trial.suggest_float(
                    "filter_cutoff", 0.001, 0.25
                )  # 0 < cutoff < fs/2
                threshold = trial.suggest_float("threshold", 0.4, 1.1)
                filter_order_T = torch.tensor([filter_order], dtype=torch.int)
                filter_cutoff_T = torch.tensor([filter_cutoff], dtype=torch.float)
                threshold_T = torch.tensor([threshold], dtype=torch.float)
                encoded_signal = self.encode(
                    synth_sig[i],
                    filter_order=filter_order_T,
                    filter_cutoff=filter_cutoff_T,
                    threshold=threshold_T,
                    isNormed=False,
                )
                decoded_signal = self.decode(encoded_signal)
                decoded_tensor = Tensor(decoded_signal)
                if decoded_tensor.shape != data[i].shape:
                    decoded_tensor = decoded_tensor.squeeze()
                loss = error_function(decoded_tensor, data[i])
                return loss

            print(f"Optimizing for signal {i+1}/{len(synth_sig)}:")

            optuna.logging.set_verbosity(optuna.logging.WARNING)  # hide logging
            study = optuna.create_study(direction="minimize", sampler=TPESampler())
            study.optimize(loss_function, n_trials=trials, show_progress_bar=True)
            f_order[i] = study.best_params["filter_order"]
            f_cutoff[i] = study.best_params["filter_cutoff"]
            _threshold[i] = study.best_params["threshold"]

            print(
                f"best_order={f_order[i]}, best_cutoff={f_cutoff[i]}, best_threshold={_threshold[i]}"
            )

            if plot_history:
                # Plot the optimization values
                optuna.visualization.plot_optimization_history(study).show()
                optuna.visualization.plot_slice(
                    study, params=["filter_order", "filter_cutoff", "threshold"]
                ).show()

        self.filter_order = f_order.int()
        self.filter_cutoff = f_cutoff.float()
        self.threshold = _threshold.float()

        return self.filter_order, self.filter_cutoff, self.threshold

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

    def fir_filter(self):
        """set up the FIR-filter coefficents"""
        filter_coeffs = []
        for i in range(len(self.filter_order)):
            fir = firwin(
                self.filter_order[i].item() + 1, self.filter_cutoff[i].item(), fs=1.0
            )  # fs=1.0 for normalized frequencies
            filter_coeffs.append(torch.tensor(fir, dtype=torch.float32))

        return filter_coeffs
