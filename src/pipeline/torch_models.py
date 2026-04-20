from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


@dataclass(slots=True)
class _TorchTrainingState:
    best_metric: float
    best_state_dict: dict | None
    best_epoch: int


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """A lightweight PyTorch regressor for tabular material features.

    This keeps the repo on a small dependency surface while adding a learned neural baseline
    that can run on composition-only feature tables and, when useful, structure-summary tables.
    The class intentionally exposes a sklearn-like ``fit`` / ``predict`` API so it can plug into
    the existing benchmark and screening pipeline without special-case wrappers.
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

    def _import_torch(self):
        try:
            import torch
            import torch.nn as nn
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised when torch missing
            raise ModuleNotFoundError(
                'torch is required for model_type="torch_mlp". Install torch in the quant '
                'environment before enabling this neural baseline.'
            ) from exc
        return torch, nn

    def _resolve_device(self, torch_module) -> str:
        requested = str(self.device or 'auto').lower()
        if requested != 'auto':
            return requested
        cuda = getattr(torch_module, 'cuda', None)
        if cuda is not None and bool(cuda.is_available()):
            return 'cuda'
        backends = getattr(torch_module, 'backends', None)
        mps_backend = getattr(backends, 'mps', None) if backends is not None else None
        if mps_backend is not None and bool(mps_backend.is_available()):
            return 'mps'
        return 'cpu'

    @staticmethod
    def _to_numpy_2d(X) -> np.ndarray:
        array = np.asarray(X, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError('TorchMLPRegressor expects a 2D feature matrix')
        if array.shape[1] == 0:
            raise ValueError('TorchMLPRegressor requires at least one feature column')
        return array

    def _split_indices(self, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
        if n_rows < 8 or self.val_fraction <= 0:
            return np.arange(n_rows), np.asarray([], dtype=int)
        rng = np.random.default_rng(self.random_seed)
        indices = np.arange(n_rows)
        rng.shuffle(indices)
        val_count = int(round(n_rows * float(self.val_fraction)))
        val_count = max(1, min(val_count, n_rows - 1))
        val_idx = np.sort(indices[:val_count])
        train_idx = np.sort(indices[val_count:])
        return train_idx, val_idx

    def _build_network(self, input_dim: int, nn_module):
        layers = []
        current_dim = input_dim
        hidden_dim = int(self.hidden_dim)
        depth = max(1, int(self.depth))
        dropout = float(self.dropout)
        for _ in range(depth):
            layers.extend([
                nn_module.Linear(current_dim, hidden_dim),
                nn_module.LayerNorm(hidden_dim),
                nn_module.GELU(),
            ])
            if dropout > 0:
                layers.append(nn_module.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn_module.Linear(current_dim, 1))
        return nn_module.Sequential(*layers)

    def fit(self, X, y):
        torch, nn = self._import_torch()
        features = self._to_numpy_2d(X)
        targets = np.asarray(y, dtype=np.float32).reshape(-1)
        if len(features) != len(targets):
            raise ValueError('Feature matrix and target vector must have the same length')
        if len(features) < 2:
            raise ValueError('TorchMLPRegressor requires at least two training rows')

        torch.manual_seed(int(self.random_seed))

        train_idx, val_idx = self._split_indices(len(features))
        train_x = features[train_idx]
        train_y = targets[train_idx]
        val_x = features[val_idx] if len(val_idx) else np.empty((0, features.shape[1]), dtype=np.float32)
        val_y = targets[val_idx] if len(val_idx) else np.empty((0,), dtype=np.float32)

        self.x_mean_ = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.x_std_ = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
        self.x_std_[self.x_std_ < 1e-8] = 1.0
        self.y_mean_ = float(train_y.mean(dtype=np.float64))
        y_std = float(train_y.std(dtype=np.float64))
        self.y_std_ = y_std if y_std >= 1e-8 else 1.0

        train_x = ((train_x - self.x_mean_) / self.x_std_).astype(np.float32)
        val_x = ((val_x - self.x_mean_) / self.x_std_).astype(np.float32)
        train_y = ((train_y - self.y_mean_) / self.y_std_).astype(np.float32)
        val_y = ((val_y - self.y_mean_) / self.y_std_).astype(np.float32)

        self.input_dim_ = int(features.shape[1])
        self.device_ = self._resolve_device(torch)
        self.model_ = self._build_network(self.input_dim_, nn).to(self.device_)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        criterion = nn.SmoothL1Loss(beta=0.5)

        train_x_tensor = torch.as_tensor(train_x, dtype=torch.float32, device=self.device_)
        train_y_tensor = torch.as_tensor(train_y, dtype=torch.float32, device=self.device_).unsqueeze(-1)
        val_x_tensor = torch.as_tensor(val_x, dtype=torch.float32, device=self.device_)
        val_y_tensor = torch.as_tensor(val_y, dtype=torch.float32, device=self.device_).unsqueeze(-1)

        state = _TorchTrainingState(best_metric=np.inf, best_state_dict=None, best_epoch=-1)
        epochs_without_improvement = 0

        for epoch in range(int(self.max_epochs)):
            self.model_.train()
            optimizer.zero_grad(set_to_none=True)
            pred = self.model_(train_x_tensor)
            loss = criterion(pred, train_y_tensor)
            loss.backward()
            optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                if len(val_idx):
                    monitor_pred = self.model_(val_x_tensor)
                    monitor_true = val_y_tensor
                else:
                    monitor_pred = self.model_(train_x_tensor)
                    monitor_true = train_y_tensor
                monitor_metric = float(torch.mean(torch.abs(monitor_pred - monitor_true)).item())

            if monitor_metric + float(self.min_delta) < state.best_metric:
                state.best_metric = monitor_metric
                state.best_epoch = epoch
                state.best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model_.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(self.patience):
                    break

        if state.best_state_dict is None:
            raise RuntimeError('TorchMLPRegressor training failed to produce a checkpoint')

        self.model_.load_state_dict(state.best_state_dict)
        self.model_.eval()
        self.best_epoch_ = int(state.best_epoch)
        self.best_monitor_mae_ = float(state.best_metric)
        return self

    def predict(self, X):
        torch, _ = self._import_torch()
        if not hasattr(self, 'model_'):
            raise AttributeError('TorchMLPRegressor must be fitted before predict()')

        features = self._to_numpy_2d(X)
        normalized = ((features - self.x_mean_) / self.x_std_).astype(np.float32)
        feature_tensor = torch.as_tensor(normalized, dtype=torch.float32, device=self.device_)
        self.model_.eval()
        with torch.no_grad():
            prediction = self.model_(feature_tensor).detach().cpu().numpy().reshape(-1)
        return prediction.astype(np.float64) * float(self.y_std_) + float(self.y_mean_)
