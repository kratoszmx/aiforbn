from __future__ import annotations

import numpy as np

from torch_models.base import *
from torch_models.base import _normalize_member_seeds

class TorchMLPEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Seed-ensemble wrapper around TorchMLPRegressor.

    This keeps the same lightweight neural family but reduces the chance that a single lucky
    initialization dominates the BN-facing story. The ensemble mean is used as the model
    prediction, while member-level predictions can be surfaced to the existing uncertainty layer.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        depth: int = 3,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 40,
        patience: int = 6,
        min_delta: float = 1e-4,
        val_fraction: float = 0.15,
        device: str = 'auto',
        random_seed: int = 42,
        member_seeds: tuple[int, ...] | list[int] | None = (42, 43, 44),
    ) -> None:
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.val_fraction = val_fraction
        self.device = device
        self.random_seed = random_seed
        self.member_seeds = member_seeds

    def _member_kwargs(self, seed: int) -> dict:
        return {
            'hidden_dim': self.hidden_dim,
            'depth': self.depth,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'max_epochs': self.max_epochs,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'val_fraction': self.val_fraction,
            'device': self.device,
            'random_seed': int(seed),
        }

    def fit(self, X, y):
        seeds = _normalize_member_seeds(self.member_seeds, default_seed=int(self.random_seed))
        self.member_seeds_ = seeds
        self.members_ = []
        for seed in seeds:
            member = TorchMLPRegressor(**self._member_kwargs(seed))
            member.fit(X, y)
            self.members_.append(member)
        if not self.members_:
            raise RuntimeError('TorchMLPEnsembleRegressor failed to fit any member models')
        self.member_count_ = int(len(self.members_))
        self.member_best_monitor_mae_ = [
            float(getattr(member, 'best_monitor_mae_', np.nan))
            for member in self.members_
        ]
        return self

    def predict_members(self, X) -> np.ndarray:
        if not hasattr(self, 'members_') or not self.members_:
            raise AttributeError('TorchMLPEnsembleRegressor must be fitted before predict_members()')
        return np.asarray([member.predict(X) for member in self.members_], dtype=np.float64)

    def predict(self, X):
        member_predictions = self.predict_members(X)
        return member_predictions.mean(axis=0)

