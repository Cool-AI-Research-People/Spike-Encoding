from typing import Union
import torch
from torch import Tensor
from torchmetrics import MeanSquaredError
import optuna
from warnings import warn
from colorama import init as colorama_init
from colorama import Fore, Style

from .base_converter import BaseConverter


class LIFBasedEncoding(BaseConverter):
    """Encodes a tensor of input values by the difference between a timestep and an adaptive threshold.
    Generates spikes when an input value is greater than the threshold plus a moving baseline based on previous values.

    Parameters:
        threshold (float): Inputs greater than threshold + moving baseline generate a spike, defaults to ``1.0``
        padding (bool): If ``True``, the first time step will be compared with itself resulting in ``0``'s in spikes. If ``False``, it will be padded with ``0``'s, defaults to ``False``
        down_spike: If ``True``, spikes for negative changes less than ``-threshold``. If ``False``, down spikes are dropped. Defaults to ``True``
    """

    def __init__(
        self,
        threshold: torch.Tensor = torch.tensor(
            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=torch.float
        ),
        membrane_constant: torch.Tensor = torch.tensor(
            [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=torch.float
        ),
        padding: bool = False,
        down_spike: bool = True,
        min_value=None,
        max_value=None,
        scale_factor=None,
        initial_value=None,
    ):
        super(LIFBasedEncoding, self).__init__()
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

        if isinstance(membrane_constant, (float, int)):
            self.membrane_constant = torch.tensor(
                [membrane_constant], dtype=torch.float
            )
        elif isinstance(membrane_constant, (list, tuple)):
            self.membrane_constant = torch.tensor(membrane_constant, dtype=torch.float)
        elif isinstance(membrane_constant, torch.Tensor):
            self.membrane_constant = membrane_constant.float()
        else:
            raise ValueError(
                "membrane_constant must be a float, int, list, tuple, or torch.Tensor"
            )

        self.padding = padding
        self.down_spike = down_spike
        self.min_value = min_value
        self.max_value = max_value
        self.scale_factor = scale_factor
        self.initial_value = initial_value

    def encode(
        self,
        signal: torch.Tensor,
        threshold: Union[int, float, torch.Tensor] = None,
        membrane_constant: Union[int, float, torch.Tensor] = None,
        down_spike: bool = None,
    ):
        """Converts the original signal into spikes.

        Parameters
        ----------
        signal : torch.Tensor
            Tensor containing signal to be converted.
            Must have 1 [time_steps] or 2 [features, time_steps] dimensions.
        threshold : float
            Inputs greater than threshold + moving baseline generate a spike, defaults to self.threshold if None.
        membrane_constant : float
            The membrane constant to be used for the moving baseline, defaults to 0.9. Must be between 0 and 1.
        down_spike : bool
            If ``True``, spikes for negative changes less than ``-threshold``. If ``False``, down spikes are dropped. Defaults to self.down_spike if None.

        Returns
        -------
        torch.Tensor
            A tensor containing spike information derived from the original signal.
        """
        if signal.ndimension() > 2:
            return self.encode_3d(signal, threshold)
        else:
            if signal.ndimension() < 2:
                signal = signal.unsqueeze(0)

            signal = torch.atleast_2d(signal)
            signal = self.normalize_tensor(signal)  # Normalize the signal to [-1, 1]

            self.initial_value = signal[:, 0].detach().clone()

            threshold = (
                self.threshold
                if threshold == None
                else self._threshold_as_tensor(threshold)
            )
            threshold = self._adjust_threshold_shape(threshold, signal)

            membrane_constant = (
                self.membrane_constant
                if membrane_constant == None
                else membrane_constant
            )
            if membrane_constant.shape[0] < signal.shape[0]:
                if membrane_constant.shape[0] > 1:
                    colorama_init()
                    print(
                        f"{type(self).__name__} - {Fore.YELLOW}Warning: Membrane constant shape (={membrane_constant.shape[0]}) smaller than "
                        + f"data shape at first dim (={signal.shape[0]}) but larger than 1. A new tensor based on membrane_constant[0]: "
                        + f"{membrane_constant[0]} is created. All other membrane constant values will be ignored.{Style.RESET_ALL}"
                    )
                membrane_constant = torch.full_like(signal[:, 0], membrane_constant[0])

            if down_spike == None:
                down_spike = self.down_spike

            # The signal now has shape = [features, time_steps] and threshold has shape = [features]
            time_steps = signal.shape[1]
            up_spikes = torch.zeros_like(signal)
            down_spikes = torch.zeros_like(signal)
            # voltage = signal[:, 0].detach().clone()
            # base = voltage#gone
            voltage = torch.zeros_like(signal[:, 0])
            base = voltage  # gone
            for t in range(1, time_steps):
                current = signal[:, t].detach().clone()
                voltage += current
                up_spikes[voltage - base > threshold, t] = 1  # voltage
                down_spikes[voltage - base < -threshold, t] = -1  # voltage
                for r in range(len(signal)):
                    if up_spikes[r, t] != 0:
                        voltage[r] = 0
                        base[r] += threshold[r]  # gone
                        base[r] = 0
                    if down_spikes[r, t] != 0:
                        voltage[r] = 0
                        base[r] -= threshold[r]  # gone
                        base[r] = 0
                # base = base + (current - base * membrane_constant)#gone
                # base = (base + (voltage - current * membrane_constant))#gone
                # voltage = (voltage - base) * membrane_constant + base#without base
                voltage = voltage * membrane_constant  # gone
            if not down_spike:
                return torch.stack((up_spikes, torch.zeros_like(signal)))
            else:
                return torch.stack((up_spikes, down_spikes))

    def decode(
        self,
        spikes: torch.Tensor,
        initial_value: Union[int, float, torch.Tensor] = None,
        threshold=None,
        membrane_constant: Union[int, float, torch.Tensor] = None,
    ):
        """Attempts to reconstruct the original signal using the given spikes.

        Parameters
        ----------
        spikes : torch.Tensor
            Tensor containing spike information derived from the original signal. Should contain up and down spikes.
            Must have 1 [spikes], 2 [features, spikes] or 3 [polarity, features, spikes] dimensions.

        Returns
        -------
        torch.Tensor
            A tensor of dimension 2 in the form [features, signal_values].
        """
        if spikes.ndimension() == 3:
            # Assume shape [polarity, 1, spikes]
            spikes = spikes[0].add(spikes[1])
        # Assume shape [features, spikes]
        spikes = torch.atleast_2d(spikes)

        initial_value = (
            self.initial_value if initial_value == None else Tensor(initial_value)
        )
        if initial_value.shape[0] < spikes.shape[0]:
            if initial_value.shape[0] > 1:
                colorama_init()
                print(
                    f"{type(self).__name__} - {Fore.YELLOW}Warning: Initial value shape (={initial_value.shape[0]}) smaller than "
                    + f"data shape at first dim (={spikes.shape[0]}) but larger than 1. A new tensor based on initial_value[0]: "
                    + f"{initial_value[0]} is created. All other initial value values will be ignored.{Style.RESET_ALL}"
                )
            initial_value = torch.full_like(spikes[:, 0], initial_value[0])

        threshold = (
            self.threshold
            if threshold == None
            else self._threshold_as_tensor(threshold)
        )
        threshold = self._adjust_threshold_shape(threshold, spikes)
        weighted_spikes = threshold[:, None] * spikes.squeeze()

        membrane_constant = (
            self.membrane_constant if membrane_constant == None else membrane_constant
        )
        if membrane_constant.shape[0] < spikes.shape[0]:
            if membrane_constant.shape[0] > 1:
                colorama_init()
                print(
                    f"{type(self).__name__} - {Fore.YELLOW}Warning: Membrane constant shape (={membrane_constant.shape[0]}) smaller than "
                    + f"data shape at first dim (={spikes.shape[0]}) but larger than 1. A new tensor based on membrane_constant[0]: "
                    + f"{membrane_constant[0]} is created. All other membrane constant values will be ignored.{Style.RESET_ALL}"
                )
            membrane_constant = torch.full_like(spikes[:, 0], membrane_constant[0])

        # Shape of weighted_spikes is now [features, time_steps]
        time_steps = spikes.shape[1]
        reconstructed = torch.zeros_like(spikes)
        voltage = torch.zeros_like(spikes[:, 0])
        # base = initial_value#gone
        base = voltage  # gone
        # reconstructed[:, 0] = initial_value
        for r in range(spikes.shape[0]):
            for t in range(1, time_steps):
                # reconstructed[r, t] = reconstructed[r, t - 1] * (membrane_constant[r]) + weighted_spikes[r, t]#without base
                # reconstructed[r, t] = reconstructed[r, t - 1] * (membrane_constant[r]) + weighted_spikes[r, t]#without base

                # reconstructed[r, t] = base[r] + weighted_spikes[r, t]#gone
                # base[r] = reconstructed[r, t - 1]#gone
                # base[r] = (base[r] + (reconstructed[r, t] - reconstructed[r, t - 1]) * 1/2)#gone #whatif membrane

                # voltage[r] = voltage[r] / membrane_constant[r]#gone

                # if spikes[r, t - 1] == 0:
                #    reconstructed[r, t] = - base[r]#gone

                # base + threshold - base[if no spikes at last time step else 0] = current

                # voltage[r] -= weighted_spikes[r, t]#gone

                if spikes[r, t] == 1:
                    base[r] += threshold[r]
                    # base[r] = 0
                if spikes[r, t] == -1:
                    base[r] -= threshold[r]
                    # base[r] = 0

                reconstructed[r, t] = (
                    reconstructed[r, t - 1] * (membrane_constant[r])
                    + weighted_spikes[r, t]
                )  # gone

                # voltage[r] = base[r]

                # voltage[voltage[r] - base[r] > threshold[r]] = 0

                # reconstructed[r, t] += voltage[r]

            # reconstructed[r] = reconstructed[r] + initial_value[r]

        reconstructed = (reconstructed + 1) / 2
        for i in range(reconstructed.shape[0]):
            reconstructed[i, :] = (
                reconstructed[i, :] / self.scale_factor[i] + self.min_value[i]
            )
        return reconstructed

    def optimize(
        self,
        data: torch.Tensor,
        error_function=MeanSquaredError(),
        trials=100,
        plot_history=False,
    ):
        """Attempts find optimal threshold values for the given data per feature.

        Parameters
        ----------
        data : torch.Tensor
            Tensor containing the signal the threshold should be optimized for.
            Must have 1 [time_steps] or 2 [features, time_steps] dimensions.
        error_function
            Function used to calculate the error between original and reconstructed signal.
        tolerance : float
            Tolerance the optimization algorithm uses for termination.

        Returns
        -------
        torch.Tensor
            A tensor of dimension 1 in the form [input_features] containing one threshold per feature in the input data.
        """
        if data.ndim > 2:
            raise ValueError(
                "Optimize function got dimension {data.ndim}. Please use optimize_3d() for 3d data."
            )

        data = torch.atleast_2d(data)
        _threshold = torch.zeros(len(data))
        _membrane_constant = torch.zeros(len(data))

        for i in range(len(data)):
            print(f"Optimizing threshold for feature {i + 1}/{len(data)}")

            def loss_function(trial):
                threshold = trial.suggest_float("threshold", 0.1, 0.6)
                membrane_constant = trial.suggest_float("membrane_constant", 0.1, 0.5)
                thresh = Tensor([threshold])
                membrane_constant = Tensor([membrane_constant])
                spikes = self.encode(
                    data[i], thresh, membrane_constant=membrane_constant
                )
                reconstruction = self.decode(
                    spikes,
                    initial_value=None,
                    threshold=thresh,
                    membrane_constant=membrane_constant,
                )
                error_val = error_function(reconstruction, torch.atleast_2d(data[i]))
                return error_val

            optuna.logging.set_verbosity(optuna.logging.WARNING)  # hide logging
            study = optuna.create_study(direction="minimize")
            study.optimize(loss_function, n_trials=trials, show_progress_bar=True)
            _threshold[i] = study.best_params["threshold"]
            _membrane_constant[i] = study.best_params["membrane_constant"]

            if plot_history:
                optuna.visualization.plot_optimization_history(study).show()
                optuna.visualization.plot_slice(
                    study, params=["threshold", "membrane_constant"]
                ).show()

        self.threshold = _threshold.float()
        self.membrane_constant = _membrane_constant.float()
        return self.threshold, self.membrane_constant

    def _adjust_threshold_shape(self, threshold: Tensor, signal: Tensor) -> Tensor:
        """Adjusts threshold shape to match the shape of the given signal."""
        if threshold.shape[0] < signal.shape[0]:
            if threshold.shape[0] > 1:
                colorama_init()
                print(
                    f"{type(self).__name__} - {Fore.YELLOW}Warning: Threshold shape (={threshold.shape[0]}) smaller than "
                    + f"data shape at first dim (={signal.shape[0]}) but larger than 1. A new tensor based on threshold[0]: "
                    + f"{threshold[0]} is created. All other threshold values will be ignored.{Style.RESET_ALL}"
                )
            return torch.full_like(signal[:, 0], threshold[0])
        elif threshold.shape[0] > signal.shape[0]:
            colorama_init()
            warn(
                f"\n{type(self).__name__} - {Fore.YELLOW}Threshold shape (={threshold.shape[0]}) larger than "
                + f"data shape at first dim (={signal.shape[0]}). Using only the first {signal.shape[0]} value(s) of threshold. "
                + f"Make sure to give the matching data or explicitly give the correct threshold values as parameter!{Style.RESET_ALL}",
                SyntaxWarning,
            )
            return threshold[: signal.shape[0]]
        else:
            return threshold

    @staticmethod
    def _threshold_as_tensor(threshold) -> Tensor:
        """Tries to cast given threshold value to a torch.Tensor."""
        if type(threshold) in (float, int):
            return torch.Tensor([threshold])
        elif type(threshold) == Tensor:
            # This catches cases where the user has given a scalar tensor with 0 dimensions (i.e. tensor(0.1)),
            # which would break all our array based calculations. Might happen if the function is called with a
            # subset of the stored thresholds, e.g. encode(data, threshold=converter.threshold[0]).
            return torch.atleast_1d(threshold)
        else:
            raise TypeError(
                f"Inappropriate type for argument 'threshold'. Must be one of float|int|torch.Tensor but got {type(threshold)}."
            )

    def normalize_tensor(self, tensor):
        """Normalize a tensor to the range between -1 and 1."""
        self.min_value = torch.min(tensor, dim=1)[0]
        self.max_value = torch.max(tensor, dim=1)[0]
        self.scale_factor = 1.0 / (self.max_value - self.min_value)

        normalized_tensor = torch.zeros(tensor.shape)
        for i in range(len(tensor)):
            normalized_tensor[i] = (
                self.scale_factor[i] * (tensor[i] - self.min_value[i])
            ) * 2 - 1
        return normalized_tensor
