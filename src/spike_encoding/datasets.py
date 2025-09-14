from .dataset_utils.download_shd import load_shd
from .dataset_utils.process_shd import process_shd_to_sparse


def load_processed_shd(n_timesteps=100):
    train_path, test_path = load_shd()
    return process_shd_to_sparse(train_path, test_path, n_timesteps)


__all__ = [
    "load_shd",
    "process_shd_to_sparse",
    "load_processed_shd",
]

