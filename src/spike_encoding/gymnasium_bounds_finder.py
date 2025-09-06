from typing import List, Union
import gymnasium as gym
import tqdm
from joblib import Parallel, delayed
import joblib
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
import numpy as np
import os
from numpy.typing import NDArray as ndarray

from .rate_step_forward_converter import RateStepForwardConverter as Converter


def len_2(arr: ndarray) -> int:
    return arr.shape[1]


def get_runs(env_name: str, num_steps: int, workers: int, print_updates=False) -> list:
    if print_updates:
        print("Generating " + str(num_steps) + " random runs...")

    def get_runs() -> list:
        env = gym.make(env_name)
        runs = []
        for i in tqdm.tqdm(range(int(num_steps / workers))):
            # Get the initial state of the environment
            states = []
            state, _ = env.reset()
            done = False
            trunc = False
            while not (done or trunc):
                states.append(state)
                action = env.action_space.sample()
                state, _, done, trunc, _ = env.step(action)

            runs.append(np.array(states).swapaxes(0, 1))

        return runs

    if workers > 1:
        results = Parallel(n_jobs=-1)(delayed(get_runs)() for i in [()] * workers)
        runs = []
        assert results is not None
        for res in results:
            runs.extend(res)
    else:
        runs = get_runs()

    return runs


def save_scaler(path: str, scaler: MinMaxScaler, print_updates=False) -> bool:
    joblib.dump(scaler, path)
    if print_updates:
        print("Saved scaler at " + path)
    return True


def load_presaved_env_data(path: str, print_updates=False) -> MinMaxScaler:
    scaler = joblib.load(path)
    if print_updates:
        print("Loaded scaler at " + path)
    return scaler


def save_treshold(path: str, treshold: Tensor, print_updates=False) -> bool:
    with open(path, "w") as f:
        f.write(str(treshold.tolist()))

    if print_updates:
        print("Saved thresholds at " + path)
    return True


def load_treshold(path: str, print_updates=False) -> Tensor:
    with open(path, "r") as f:
        for line in f:
            th = line
    threshold = [float(nr) for nr in th.split("[")[1].split("]")[0].split(",")]
    if print_updates:
        print("Loaded thresholds at " + path)
    return Tensor(threshold)


def get_path_from_dirs(dirs: list) -> str:
    path = os.getcwd()

    for dir_name in dirs:
        path = os.path.join(path, dir_name)
        if not os.path.exists(path):
            os.mkdir(path)

    return path


class ScalerFactory:
    workers = 64
    num_steps = 1000

    def __init__(
        self,
        print_updates=False,
    ):
        self.print_updates = print_updates

    def has_presaved_env_data(self) -> bool:
        return os.path.exists(self.scaler_path)

    def from_env(self, env: gym.Env):
        assert env.spec is not None
        self.env_path = get_path_from_dirs(["data", "envs", env.spec.id])
        self.scaler_path = os.path.join(self.env_path, "scaler.save")
        if not self.has_presaved_env_data():
            return self.run_env(env)
        return load_presaved_env_data(self.scaler_path)

    def from_known_values(
        self, minima: Union[List[float], ndarray], maxima: Union[List[float], ndarray]
    ):
        scaler = MinMaxScaler()
        data = np.concatenate(([minima], [maxima]), axis=0)
        scaler.fit(data)
        return scaler

    def run_env(self, env: gym.Env):
        assert env.spec is not None
        runs = get_runs(env.spec.id, self.num_steps * 10, workers=self.workers)
        scaler = self._optimize_scaler(runs)
        save_scaler(self.scaler_path, scaler)
        return scaler

    def _optimize_scaler(self, runs: list) -> MinMaxScaler:
        if self.print_updates:
            print("Fitting scaler...")
        # Fit data between 0 and 1
        concat_runs = np.concatenate(runs, axis=1).swapaxes(0, 1)
        # Balance upper and lower bounds
        upper = (
            np.concatenate([abs(concat_runs.min(axis=0)), concat_runs.max(axis=0)])
            .reshape(-1, concat_runs.shape[1])
            .max(axis=0)
        )
        lower = -upper
        concat_runs = np.append(concat_runs, upper.reshape(1, -1), axis=0)
        concat_runs = np.append(concat_runs, lower.reshape(1, -1), axis=0)

        # absolute values generate spikes from the first time steps

        # Multiplyier for modifying the upper and lower bounds
        # Needs to be greater than 0
        # A value between 0 and 1 makes the converter more sensitive, i.e. it will generate more spikes
        # A value greater then 1 makes the converter less sensitive, i.e. it will generate less spikes
        max_modifier: int = 1
        concat_runs = concat_runs * max_modifier
        scaler = MinMaxScaler().fit(concat_runs)

        return scaler


class ConverterFactory:
    workers = 64
    num_steps = 1000

    def __init__(self, env: gym.Env, scaler: MinMaxScaler, print_updates=False):
        self.env = env
        self.print_updates = print_updates
        self.scaler = scaler
        assert env.observation_space.shape is not None
        self.features = env.observation_space.shape[0]
        assert env.spec is not None
        self.env_path = get_path_from_dirs(["data", "envs", env.spec.id])
        self.enc_path = os.path.join(self.env_path, "th.txt")
        self.runs = []

    def initialize(self, runs):
        if len(runs) == 0:
            assert self.env.spec is not None
            runs = get_runs(self.env.spec.id, self.num_steps, workers=self.workers)
        else:
            runs = runs[: self.num_steps]

        if self.print_updates:
            print("Initializing encoder...")
        conv = self._optimize_converter(runs)
        th = conv.threshold
        th_numpy = th.detach().numpy()
        save_treshold(self.enc_path, th_numpy)
        return conv, th_numpy

    def generate(self):
        if not self.initialized():
            return self.initialize(self.runs)
        else:
            th = load_treshold(self.enc_path)
            th_numpy = th.detach().numpy()
            if self.print_updates:
                print("Loaded existing encoder initialization at " + self.enc_path)
            return Converter(self.features, th_numpy), th_numpy

    def initialized(self) -> bool:
        return os.path.exists(self.enc_path)

    def _optimize_converter(self, runs: list) -> Converter:
        # Sort runs by length
        runs.sort(key=len_2)

        if self.print_updates:
            print("Normalizing data")

        runs = [
            self.scaler.transform(arr.swapaxes(0, 1)).swapaxes(0, 1) for arr in runs
        ]

        if self.print_updates:
            print("Reshaping tensors...")

        # Fill all arrays to the same length and stack along run dimension
        max_len = runs[-1].shape[1]
        runs_tensor = Tensor(
            np.stack(
                [
                    np.append(
                        arr,
                        np.ones((self.features, max_len - arr.shape[1])) * np.inf,
                        axis=1,
                    )
                    for arr in runs
                ]
            )
        )

        converter = Converter(self.features, np.ones(self.features) * 0.01)
        if self.print_updates:
            print("Optimizing converter tresholds...")
        converter.optimize(runs_tensor)

        return converter
