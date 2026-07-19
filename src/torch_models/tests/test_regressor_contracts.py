from __future__ import annotations

import numpy as np
import pytest

from torch_models.attention import TorchFractionalAttentionRegressor
from torch_models.base import TorchMLPRegressor
from torch_models.ensemble import TorchMLPEnsembleRegressor


@pytest.mark.parametrize(
    'features',
    [
        np.asarray([1.0, 2.0], dtype=float),
        np.empty((2, 0), dtype=float),
    ],
)
def test_torch_mlp_rejects_non_matrix_or_zero_feature_input(features):
    model = TorchMLPRegressor(max_epochs=1)

    with pytest.raises(ValueError):
        model.fit(features, np.asarray([1.0, 2.0]))


def test_torch_mlp_rejects_length_mismatch_and_single_row_training():
    model = TorchMLPRegressor(max_epochs=1)

    with pytest.raises(ValueError, match='same length'):
        model.fit(np.ones((2, 2)), np.asarray([1.0]))
    with pytest.raises(ValueError, match='at least two training rows'):
        model.fit(np.ones((1, 2)), np.asarray([1.0]))


def test_torch_mlp_predict_requires_fit():
    with pytest.raises(AttributeError, match='must be fitted'):
        TorchMLPRegressor().predict(np.ones((2, 2)))


def test_fractional_attention_rejects_wrong_input_dimension():
    model = TorchFractionalAttentionRegressor(expected_input_dim=118, max_epochs=1)

    with pytest.raises(ValueError, match='input dimension 118'):
        model.fit(np.ones((2, 117)), np.asarray([1.0, 2.0]))


def test_fractional_attention_rejects_invalid_head_divisibility():
    model = TorchFractionalAttentionRegressor(
        embedding_dim=10,
        num_heads=3,
        expected_input_dim=118,
        max_epochs=1,
    )

    with pytest.raises(ValueError, match='embedding_dim must be divisible by num_heads'):
        model.fit(np.ones((2, 118)), np.asarray([1.0, 2.0]))


def test_fractional_attention_auto_device_stays_on_cpu_without_cuda():
    class UnavailableCuda:
        @staticmethod
        def is_available():
            return False

    class AvailableMps:
        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        cuda = UnavailableCuda()
        backends = type('Backends', (), {'mps': AvailableMps()})()

    assert TorchFractionalAttentionRegressor(device='auto')._resolve_device(FakeTorch) == 'cpu'


def test_torch_ensemble_rejects_empty_member_seed_list():
    model = TorchMLPEnsembleRegressor(member_seeds=[])

    with pytest.raises(ValueError, match='at least one integer seed'):
        model.fit(np.ones((2, 2)), np.asarray([1.0, 2.0]))


def test_torch_ensemble_predict_requires_fit():
    with pytest.raises(AttributeError, match='must be fitted'):
        TorchMLPEnsembleRegressor().predict(np.ones((2, 2)))
