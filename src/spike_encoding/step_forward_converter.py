from typing import Union
import torch
from torch import Tensor
from torchmetrics import MeanSquaredError
from scipy.optimize import minimize_scalar
import optuna
from joblib import Parallel, delayed
from warnings import warn
from colorama import init as colorama_init
from colorama import Fore, Style

from .base_converter import BaseConverter


class StepForwardConverter(BaseConverter):
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
    ):
        super(StepForwardConverter, self).__init__()
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

    def encode_3d(self, signal: torch.Tensor, threshold):
        # Signal has shape: [batch, features, time_steps]
        batch, features, time_steps = signal.shape
        # self.threshold has shape [features]
        device = signal.device
        if threshold == None:
            threshold = self.threshold.to(device)
        else:
            threshold = Tensor([threshold]).to(device)

        threshold = (signal[:, :, 0] + threshold - signal[:, :, 0]).reshape(-1)
        signal = signal.reshape(threshold.shape[0], time_steps).detach().clone()
        # The signal now has shape = [features, time_steps] and threshold has shape = [features]
        up_spikes = torch.zeros_like(signal)
        down_spikes = torch.zeros_like(signal)
        base = signal[:, 0].detach().clone()
        for t in range(1, time_steps):
            current = signal[:, t].detach().clone()
            up_spikes[current > base + threshold, t] = 1
            base[current > base + threshold] += threshold[current > base + threshold]
            down_spikes[current < base - threshold, t] = -1
            base[current < base - threshold] -= threshold[current < base - threshold]

        up_spikes = up_spikes.reshape(batch, features, time_steps)
        down_spikes = down_spikes.reshape(batch, features, time_steps)

        if not self.down_spike:
            return torch.stack((up_spikes, torch.zeros_like(signal)))
        else:
            return torch.stack((up_spikes, down_spikes))

    def encode(
        self,
        signal: torch.Tensor,
        threshold: Union[int, float, torch.Tensor] = None,
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
            threshold = (
                self.threshold
                if threshold == None
                else self._threshold_as_tensor(threshold)
            )
            threshold = self._adjust_threshold_shape(threshold, signal)
            if down_spike == None:
                down_spike = self.down_spike

            # The signal now has shape = [features, time_steps] and threshold has shape = [features]
            time_steps = signal.shape[1]
            up_spikes = torch.zeros_like(signal)
            down_spikes = torch.zeros_like(signal)
            base = signal[:, 0].detach().clone()
            for t in range(1, time_steps):
                current = signal[:, t].detach().clone()
                up_spikes[current > base + threshold, t] = 1
                base[current > base + threshold] += threshold[
                    current > base + threshold
                ]
                down_spikes[current < base - threshold, t] = -1
                base[current < base - threshold] -= threshold[
                    current < base - threshold
                ]
            if not down_spike:
                return torch.stack((up_spikes, torch.zeros_like(signal)))
            else:
                return torch.stack((up_spikes, down_spikes))

    def decode_3d(
        self, spikes: torch.Tensor, initial_value: float = 0.0, threshold=None
    ):
        assert spikes.ndimension() == 4
        # Assume shape [polarity, batch, features, time_steps]
        polarity, batch, features, time_steps = spikes.shape
        spikes = spikes[0].add(spikes[1])
        # spikes has shape: [batch, features, time_steps]

        # self.threshold has shape [features]
        device = spikes.device
        if threshold == None:
            threshold = self.threshold.to(device)
        else:
            threshold = Tensor([threshold]).to(device)
        threshold = (spikes[:, :, 0] + threshold - spikes[:, :, 0]).reshape(-1)
        spikes = spikes.reshape(threshold.shape[0], time_steps).detach().clone()

        if type(initial_value) == float or type(initial_value) == int:
            initial_value = Tensor([initial_value]).to(device)
            initial_value = torch.ones((batch, features)).to(device) * initial_value

        # initial_value has shape [batch, features]
        initial_value = initial_value.reshape(threshold.shape[0])

        weighted_spikes = threshold[:, None] * spikes.squeeze()

        # Shape of weighted_spikes is now [batch * features, time_steps]
        time_steps = weighted_spikes.shape[1]
        start = initial_value
        reconstructed = torch.zeros_like(weighted_spikes)
        reconstructed[:, 0] = start
        for t in range(1, time_steps):
            value = weighted_spikes[:, t]
            reconstructed[:, t] = reconstructed[:, t - 1] + value
        return reconstructed.reshape(batch, features, time_steps)

    def decode(self, spikes: torch.Tensor, initial_value: float = 0, threshold=None):
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
        if spikes.ndimension() > 3:
            return self.decode_3d(spikes, initial_value, threshold)
        if spikes.ndimension() == 3:
            # Assume shape [polarity, 1, spikes]
            spikes = spikes[0].add(spikes[1])
        # Assume shape [features, spikes]
        spikes = torch.atleast_2d(spikes)
        threshold = (
            self.threshold
            if threshold == None
            else self._threshold_as_tensor(threshold)
        )
        threshold = self._adjust_threshold_shape(threshold, spikes)
        weighted_spikes = threshold[:, None] * spikes.squeeze()

        # Shape of weighted_spikes is now [features, time_steps]
        time_steps = weighted_spikes.shape[1]
        start = initial_value
        reconstructed = torch.zeros_like(weighted_spikes)
        reconstructed[:, 0] = start
        for t in range(1, time_steps):
            value = weighted_spikes[:, t]
            reconstructed[:, t] = reconstructed[:, t - 1] + value
        return reconstructed

    # Note: optimize_3d is not implemented with optuna
    def optimize_3d(
        self,
        data: torch.Tensor,
        error_function=MeanSquaredError(),
        tolerance: float = 0.01,
    ):
        options = dict(maxiter=50, disp=0, xatol=tolerance)
        device = data.device
        error_function = error_function.to(device)

        def minimize_scalar_parallel(args):
            i, bounds = args

            def loss_function(threshold):
                thresh = Tensor([threshold]).to(device)
                spikes = self.encode(data[:, i], thresh)
                reconstruction = self.decode(
                    spikes, initial_value=data[:, i, 0], threshold=thresh
                )
                error_val = error_function(reconstruction, data[:, i]).cpu()
                return error_val

            result = minimize_scalar(
                loss_function,
                options=options,
                bounds=bounds,
                method="bounded",
            ).x
            if type(result) == torch.tensor:
                result = result.item()

            return result

        bounds = [
            (0.000000001, i.item())
            for i in (data[:, :, :-1] - data[:, :, 1:])
            .abs()
            .max(dim=-1)
            .values.max(dim=0)
            .values
        ]
        new_threshold = Parallel(n_jobs=-1)(
            delayed(minimize_scalar_parallel)(args) for args in enumerate(bounds)
        )
        self.threshold = Tensor(new_threshold)
        return self.threshold

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

        for i in range(len(data)):
            print(f"Optimizing threshold for feature {i}...")

            def loss_function(trial):
                threshold = trial.suggest_float("threshold", 0.001, 0.5)
                thresh = Tensor([threshold])
                spikes = self.encode(data[i], thresh)
                reconstruction = self.decode(
                    spikes, initial_value=data[i, 0], threshold=thresh
                )
                error_val = error_function(reconstruction, torch.atleast_2d(data[i]))
                return error_val

            optuna.logging.set_verbosity(optuna.logging.WARNING)  # hide logging
            study = optuna.create_study(direction="minimize")
            study.optimize(loss_function, n_trials=trials, show_progress_bar=True)
            _threshold[i] = study.best_params["threshold"]

            if plot_history:
                # Plot the optimization values
                optuna.visualization.plot_optimization_history(study).show()
                optuna.visualization.plot_slice(study, params=["threshold"]).show()

        self.threshold = _threshold.float()
        return self.threshold

    def optimizeWithMembraneConstant(
        self, data: torch.Tensor, error_function=MeanSquaredError(), trials=100
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

            def loss_function(trial):
                threshold = trial.suggest_float("threshold", 0.001, 0.02)
                membrane_constant = trial.suggest_float("membrane_constant", 0.01, 0.99)
                thresh = Tensor([threshold])
                membrane_constant = Tensor([membrane_constant])
                spikes = self.encodeWithMembraneConstant(
                    data[i], thresh, membrane_constant=membrane_constant
                )
                reconstruction = self.decodeWithMembraneConstant(
                    spikes,
                    initial_value=data[i, 0],
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

            # Plot the optimization values
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
