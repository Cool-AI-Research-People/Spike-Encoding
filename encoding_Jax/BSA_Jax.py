import jax
import jax.numpy as jnp
import numpy as np
from scipy.signal import firwin


class BensSpikerAlgorithmJAX:
    def __init__(
        self,
        threshold=jnp.array([0.5] * 7, dtype=jnp.float32),
        down_spike=True,
        min_value=None,
        max_value=None,
        scale_factor=None,
        filter_order=jnp.array([10] * 7, dtype=jnp.int32),
        filter_cutoff=jnp.array([0.2] * 7, dtype=jnp.float32),
    ):
        self.threshold = threshold.astype(jnp.float32)
        self.down_spike = down_spike
        self.min_value = min_value
        self.max_value = max_value
        self.scale_factor = scale_factor
        self.filter_order = filter_order.astype(jnp.int32)
        self.filter_cutoff = filter_cutoff.astype(jnp.float32)

    def normalize_tensor(self, tensor):
        self.min_value = jnp.min(tensor, axis=1)
        self.max_value = jnp.max(tensor, axis=1)
        self.scale_factor = 1.0 / (self.max_value - self.min_value)

        normalized_tensor = jnp.zeros_like(tensor)
        for i in range(tensor.shape[0]):
            normalized_tensor = normalized_tensor.at[i].set(
                self.scale_factor[i] * (tensor[i] - self.min_value[i])
            )
        return normalized_tensor

    def fir_filter(self):
        coeffs = []
        for i in range(len(self.filter_order)):
            fir = firwin(
                self.filter_order[i].item() + 1,
                self.filter_cutoff[i].item(),
                fs=1.0
            )
            coeffs.append(jnp.array(fir, dtype=jnp.float32))
        return coeffs

    def encode(self, signal, filter_order=None, filter_cutoff=None, threshold=None, isNormed=False):
        signal = jnp.atleast_2d(jnp.squeeze(signal))
        if not isNormed:
            signal = self.normalize_tensor(signal)

        if filter_order is not None:
            self.filter_order = filter_order
        if filter_cutoff is not None:
            self.filter_cutoff = filter_cutoff
        if threshold is not None:
            self.threshold = threshold

        FIR = self.fir_filter()
        num_rows, L = signal.shape
        spike_trains = jnp.zeros((num_rows, L))

        for row in range(num_rows):
            F = len(FIR[row])
            s = signal[row].copy()  # 拷贝防止原信号被改
            out = jnp.zeros(L)

            for t in range(1, L):
                if t + F - 1 >= L:
                    continue  # 避免索引越界

                window = s[t : t + F - 1]
                if window.shape[0] != F - 1:
                    continue  # 形状不一致跳过

                err1 = jnp.sum(jnp.abs(window - FIR[row][1:F]))
                err2 = jnp.sum(jnp.abs(window))

                if err1 <= err2 - self.threshold[row]:
                    out = out.at[t].set(1.0)
                    # 更新 s[t : t + F - 1]
                    s = s.at[t : t + F - 1].add(-FIR[row][1:F])

            spike_trains = spike_trains.at[row].set(out)

        return spike_trains

    def decode(self, spikes):
        spikes = jnp.atleast_2d(jnp.squeeze(spikes))
        FIR = self.fir_filter()
        feature_size, L = spikes.shape
        reconstructed_signals = jnp.zeros((feature_size, L))

        for row in range(feature_size):
            spike_train = spikes[row]
            F = len(FIR[row])
            padding = F // 2
            padded_input = jnp.concatenate((jnp.zeros(padding), spike_train, jnp.zeros(padding)))
            out = jnp.zeros(L)

            for i in range(padding, L):
                for j in range(F):
                    idx = i + j - padding + 1
                    if idx < len(padded_input):
                        out = out.at[i].add(padded_input[idx] * FIR[row][j])

            reconstructed_signals = reconstructed_signals.at[row].set(out)

        result_vector = []
        for i in range(feature_size):
            scaled = reconstructed_signals[i] / self.scale_factor[i] + self.min_value[i]
            result_vector.append(np.array(scaled))

        return result_vector

    def optimize(self, data, trials=50, plot=False):
        import optuna
        from optuna.samplers import TPESampler

        def mse_jax(a, b):
            return jnp.mean((a - b) ** 2)

        data = jnp.atleast_2d(data)
        f_order = jnp.zeros(data.shape[0])
        f_cutoff = jnp.zeros(data.shape[0])
        _threshold = jnp.zeros(data.shape[0])

        for i in range(data.shape[0]):
            def objective(trial):
                fo = trial.suggest_int("filter_order", 10, 50)
                fc = trial.suggest_float("filter_cutoff", 0.01, 0.25)
                th = trial.suggest_float("threshold", 0.3, 1.0)

                encoded = self.encode(
                    data[i],
                    filter_order=jnp.array([fo]),
                    filter_cutoff=jnp.array([fc]),
                    threshold=jnp.array([th]),
                    isNormed=False
                )
                decoded = self.decode(encoded)[0]
                return float(mse_jax(data[i], decoded))

            study = optuna.create_study(direction="minimize", sampler=TPESampler())
            study.optimize(objective, n_trials=trials)
            f_order = f_order.at[i].set(study.best_params["filter_order"])
            f_cutoff = f_cutoff.at[i].set(study.best_params["filter_cutoff"])
            _threshold = _threshold.at[i].set(study.best_params["threshold"])

        self.filter_order = f_order.astype(jnp.int32)
        self.filter_cutoff = f_cutoff.astype(jnp.float32)
        self.threshold = _threshold.astype(jnp.float32)
        return self.filter_order, self.filter_cutoff, self.threshold