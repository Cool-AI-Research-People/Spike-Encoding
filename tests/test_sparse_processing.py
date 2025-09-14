import pytest
import sys
import os
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spike_encoding.datasets import load_processed_shd


def test_load_processed_shd_sparsity():
    X_train, y_train, X_test, y_test, metadata = load_processed_shd()

    assert len(X_train) > 0
    assert len(X_test) > 0

    sample = X_train[0]
    assert sp.issparse(sample)
    assert isinstance(sample, sp.coo_matrix)

    expected_shape = (metadata['n_timesteps'], metadata['n_features'])
    assert sample.shape == expected_shape

    density = sample.nnz / (sample.shape[0] * sample.shape[1])
    print(f"Sample density: {density:.6f}")
    print(f"Sample nnz: {sample.nnz}")
    print(f"Sample shape: {sample.shape}")
    assert density < 0.5

    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)

    assert 'max_time' in metadata
    assert 'n_features' in metadata
    assert 'n_timesteps' in metadata
    assert 'dt' in metadata

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Metadata: {metadata}")


def test_sparse_matrix_properties():
    X_train, y_train, X_test, y_test, metadata = load_processed_shd()

    sample = X_train[0]

    assert np.all((sample.data == 0) | (sample.data == 1))

    assert hasattr(sample, 'row')
    assert hasattr(sample, 'col')
    assert hasattr(sample, 'data')

    assert np.all(sample.row >= 0)
    assert np.all(sample.col >= 0)
    assert np.all(sample.row < metadata['n_timesteps'])
    assert np.all(sample.col < metadata['n_features'])

    print(f"Sample has {sample.nnz} non-zero entries")
    if sample.nnz > 0:
        print(f"Row range: {sample.row.min()} to {sample.row.max()}")
        print(f"Col range: {sample.col.min()} to {sample.col.max()}")


def test_custom_timesteps():
    X_train, y_train, X_test, y_test, metadata = load_processed_shd(n_timesteps=50)

    sample = X_train[0]
    assert sample.shape[0] == 50
    assert metadata['n_timesteps'] == 50

    print(f"Custom timesteps shape: {sample.shape}")
    print(f"Custom dt: {metadata['dt']}")