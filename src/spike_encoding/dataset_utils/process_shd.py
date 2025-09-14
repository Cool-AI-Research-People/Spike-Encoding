import h5py
import numpy as np
import scipy.sparse as sp


def load_spikes_from_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        times_data = f["spikes"]["times"][...]
        units_data = f["spikes"]["units"][...]
        labels = f["labels"][...]
    return times_data, units_data, labels


def convert_to_sparse_tensor(times_data, units_data, max_time, n_features, n_timesteps):
    dt = max_time / n_timesteps
    sparse_samples = []

    for i in range(len(times_data)):
        times = times_data[i]
        units = units_data[i]

        if len(times) == 0:
            sparse_matrix = sp.coo_matrix((n_timesteps, n_features), dtype=np.float32)
        else:
            time_bins = np.floor(times / dt).astype(int)
            time_bins = np.clip(time_bins, 0, n_timesteps - 1)
            units = np.clip(units, 0, n_features - 1)
            data = np.ones(len(times), dtype=np.float32)

            sparse_matrix = sp.coo_matrix(
                (data, (time_bins, units)),
                shape=(n_timesteps, n_features),
                dtype=np.float32
            )

        sparse_samples.append(sparse_matrix)

    return sparse_samples


def process_shd_to_sparse(train_path, test_path, n_timesteps):
    train_times, train_units, y_train = load_spikes_from_h5(train_path)
    test_times, test_units, y_test = load_spikes_from_h5(test_path)

    max_time = 1.36914
    n_features = 700

    print(f"Converting to sparse tensors...")
    print(f"Parameters: max_time={max_time}, n_features={n_features}, n_timesteps={n_timesteps}")

    X_train_sparse = convert_to_sparse_tensor(train_times, train_units, max_time, n_features, n_timesteps)
    X_test_sparse = convert_to_sparse_tensor(test_times, test_units, max_time, n_features, n_timesteps)

    metadata = {
        'max_time': max_time,
        'n_features': n_features,
        'n_timesteps': n_timesteps,
        'dt': max_time / n_timesteps
    }

    return X_train_sparse, y_train, X_test_sparse, y_test, metadata
