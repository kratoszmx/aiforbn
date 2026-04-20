from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


def _normalize_member_seeds(member_seeds, default_seed: int) -> tuple[int, ...]:
    if member_seeds is None:
        return (int(default_seed),)
    normalized = tuple(int(seed) for seed in member_seeds)
    if not normalized:
        raise ValueError('member_seeds must contain at least one integer seed')
    return normalized


def _build_sparse_fractional_tokens(fractions, *, padding_index: int, torch_module):
    sorted_fractions, sorted_indices = torch_module.sort(
        fractions,
        dim=1,
        descending=True,
    )
    token_mask = sorted_fractions > 0.0
    max_tokens = int(token_mask.sum(dim=1).max().item()) if fractions.shape[0] else 0
    max_tokens = max(1, max_tokens)

    token_fractions = sorted_fractions[:, :max_tokens]
    token_indices = sorted_indices[:, :max_tokens]
    token_mask = token_mask[:, :max_tokens]

    empty_rows = ~token_mask.any(dim=1)
    if bool(empty_rows.any()):
        token_fractions = token_fractions.clone()
        token_indices = token_indices.clone()
        token_mask = token_mask.clone()
        token_fractions[empty_rows, 0] = 1.0
        token_indices[empty_rows, 0] = 0
        token_mask[empty_rows, 0] = True

    token_indices = token_indices.masked_fill(~token_mask, padding_index)
    return token_indices, token_fractions, token_mask


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
        batch_size: int = 256,
        eval_batch_size: int | None = None,
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
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size

    def _import_torch(self):
        try:
            import torch
            import torch.nn as nn
        except ModuleNotFoundError as exc:  # pragma: no cover - exercised when torch missing
            raise ModuleNotFoundError(
                'torch is required for the torch-based neural baselines. Install torch in the '
                'quant environment before enabling these models.'
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

    def _prepare_fit_features(
        self,
        train_x: np.ndarray,
        val_x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.x_mean_ = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.x_std_ = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
        self.x_std_[self.x_std_ < 1e-8] = 1.0
        normalized_train_x = ((train_x - self.x_mean_) / self.x_std_).astype(np.float32)
        normalized_val_x = ((val_x - self.x_mean_) / self.x_std_).astype(np.float32)
        return normalized_train_x, normalized_val_x

    def _prepare_predict_features(self, features: np.ndarray) -> np.ndarray:
        return ((features - self.x_mean_) / self.x_std_).astype(np.float32)

    def _effective_batch_size(self, n_rows: int, *, eval_mode: bool = False) -> int:
        configured_batch_size = self.eval_batch_size if eval_mode else self.batch_size
        if configured_batch_size is None:
            configured_batch_size = self.batch_size
        batch_size = int(configured_batch_size or n_rows or 1)
        if batch_size <= 0:
            batch_size = n_rows or 1
        return max(1, min(batch_size, max(1, int(n_rows))))

    def _batched_mae(
        self,
        model,
        features: np.ndarray,
        targets: np.ndarray,
        *,
        torch_module,
    ) -> float:
        if len(features) == 0:
            return np.inf
        batch_size = self._effective_batch_size(len(features), eval_mode=True)
        total_absolute_error = 0.0
        total_row_count = 0
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            feature_tensor = torch_module.as_tensor(
                features[start:stop],
                dtype=torch_module.float32,
                device=self.device_,
            )
            target_tensor = torch_module.as_tensor(
                targets[start:stop],
                dtype=torch_module.float32,
                device=self.device_,
            ).unsqueeze(-1)
            prediction = model(feature_tensor)
            total_absolute_error += float(
                torch_module.sum(torch_module.abs(prediction - target_tensor)).item()
            )
            total_row_count += int(stop - start)
        if total_row_count <= 0:
            return np.inf
        return total_absolute_error / float(total_row_count)

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

        self.y_mean_ = float(train_y.mean(dtype=np.float64))
        y_std = float(train_y.std(dtype=np.float64))
        self.y_std_ = y_std if y_std >= 1e-8 else 1.0

        train_x, val_x = self._prepare_fit_features(train_x, val_x)
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

        state = _TorchTrainingState(best_metric=np.inf, best_state_dict=None, best_epoch=-1)
        epochs_without_improvement = 0
        train_batch_size = self._effective_batch_size(len(train_x), eval_mode=False)
        rng = np.random.default_rng(int(self.random_seed))

        for epoch in range(int(self.max_epochs)):
            self.model_.train()
            epoch_indices = rng.permutation(len(train_x))
            for start in range(0, len(epoch_indices), train_batch_size):
                batch_indices = epoch_indices[start:start + train_batch_size]
                batch_x_tensor = torch.as_tensor(
                    train_x[batch_indices],
                    dtype=torch.float32,
                    device=self.device_,
                )
                batch_y_tensor = torch.as_tensor(
                    train_y[batch_indices],
                    dtype=torch.float32,
                    device=self.device_,
                ).unsqueeze(-1)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model_(batch_x_tensor)
                loss = criterion(pred, batch_y_tensor)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                if len(val_idx):
                    monitor_metric = self._batched_mae(
                        self.model_,
                        val_x,
                        val_y,
                        torch_module=torch,
                    )
                else:
                    monitor_metric = self._batched_mae(
                        self.model_,
                        train_x,
                        train_y,
                        torch_module=torch,
                    )

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
        prepared_features = self._prepare_predict_features(features)
        batch_size = self._effective_batch_size(len(prepared_features), eval_mode=True)
        self.model_.eval()
        prediction_batches = []
        with torch.no_grad():
            for start in range(0, len(prepared_features), batch_size):
                stop = min(start + batch_size, len(prepared_features))
                feature_tensor = torch.as_tensor(
                    prepared_features[start:stop],
                    dtype=torch.float32,
                    device=self.device_,
                )
                prediction_batches.append(
                    self.model_(feature_tensor).detach().cpu().numpy().reshape(-1)
                )
        prediction = (
            np.concatenate(prediction_batches, axis=0)
            if prediction_batches
            else np.empty((0,), dtype=np.float32)
        )
        return prediction.astype(np.float64) * float(self.y_std_) + float(self.y_mean_)

    def predict_members(self, X) -> np.ndarray:
        return np.asarray([self.predict(X)], dtype=np.float64)


class TorchFractionalAttentionRegressor(TorchMLPRegressor):
    """A lightweight composition-attention regressor over fractional-composition vectors.

    This model is more materials-shaped than a plain MLP: it treats the 118-dimensional
    fractional composition vector as a small set of element tokens, combines a learned element
    embedding with stoichiometric signals, runs a compact Transformer encoder, and pools the
    encoded tokens with fraction weights. It remains candidate-compatible because it uses only
    formula-derived fractional composition inputs.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        head_hidden_dim: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 40,
        patience: int = 6,
        min_delta: float = 1e-4,
        val_fraction: float = 0.15,
        device: str = 'auto',
        random_seed: int = 42,
        expected_input_dim: int = 118,
        batch_size: int = 256,
        eval_batch_size: int | None = None,
    ) -> None:
        super().__init__(
            hidden_dim=head_hidden_dim,
            depth=max(1, int(num_layers)),
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            val_fraction=val_fraction,
            device=device,
            random_seed=random_seed,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
        )
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.head_hidden_dim = head_hidden_dim
        self.expected_input_dim = expected_input_dim

    def _resolve_device(self, torch_module) -> str:
        requested = str(self.device or 'auto').lower()
        if requested != 'auto':
            return requested
        cuda = getattr(torch_module, 'cuda', None)
        if cuda is not None and bool(cuda.is_available()):
            return 'cuda'
        # Transformer encoder padding-mask ops still hit unsupported MPS paths on this stack,
        # so the attention baseline should prefer CPU under auto-selection rather than fail at runtime.
        return 'cpu'

    def _to_numpy_2d(self, X) -> np.ndarray:
        array = super()._to_numpy_2d(X)
        expected_dim = int(self.expected_input_dim)
        model_name = self.__class__.__name__
        if expected_dim > 0 and array.shape[1] != expected_dim:
            raise ValueError(
                f'{model_name} expects the fractional composition vector '
                f'input dimension {expected_dim}, but received {array.shape[1]}. '
                'Use feature_set="fractional_composition_vector" for this model.'
            )
        return array

    @staticmethod
    def _normalize_fraction_rows(array: np.ndarray) -> np.ndarray:
        normalized = np.clip(np.asarray(array, dtype=np.float32), a_min=0.0, a_max=None)
        row_sums = normalized.sum(axis=1, keepdims=True)
        nonzero_mask = row_sums.reshape(-1) > 0
        if bool(np.any(nonzero_mask)):
            normalized[nonzero_mask] = normalized[nonzero_mask] / row_sums[nonzero_mask]
        return normalized.astype(np.float32)

    def _prepare_fit_features(
        self,
        train_x: np.ndarray,
        val_x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        self.x_mean_ = np.zeros(train_x.shape[1], dtype=np.float32)
        self.x_std_ = np.ones(train_x.shape[1], dtype=np.float32)
        return (
            self._normalize_fraction_rows(train_x),
            self._normalize_fraction_rows(val_x),
        )

    def _prepare_predict_features(self, features: np.ndarray) -> np.ndarray:
        return self._normalize_fraction_rows(features)

    def _build_network(self, input_dim: int, nn_module):
        if int(self.embedding_dim) % int(self.num_heads) != 0:
            raise ValueError('embedding_dim must be divisible by num_heads')
        embedding_dim = int(self.embedding_dim)
        num_heads = int(self.num_heads)
        num_layers = max(1, int(self.num_layers))
        head_hidden_dim = int(self.head_hidden_dim)
        dropout = float(self.dropout)
        torch_module, _ = self._import_torch()

        class FractionalCompositionAttentionNetwork(nn_module.Module):
            def __init__(self):
                super().__init__()
                self.element_embedding = nn_module.Embedding(input_dim, embedding_dim)
                self.fraction_encoder = nn_module.Sequential(
                    nn_module.Linear(2, embedding_dim),
                    nn_module.GELU(),
                    nn_module.Linear(embedding_dim, embedding_dim),
                )
                encoder_layer = nn_module.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=max(embedding_dim * 4, head_hidden_dim),
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=False,
                )
                self.encoder = nn_module.TransformerEncoder(
                    encoder_layer,
                    num_layers=num_layers,
                )
                self.token_norm = nn_module.LayerNorm(embedding_dim)
                self.head = nn_module.Sequential(
                    nn_module.LayerNorm(embedding_dim),
                    nn_module.Linear(embedding_dim, head_hidden_dim),
                    nn_module.GELU(),
                    nn_module.Dropout(dropout),
                    nn_module.Linear(head_hidden_dim, 1),
                )

            def forward(self, x):
                fractions = x.clamp(min=0.0)
                present_mask = fractions > 0.0
                if present_mask.ndim != 2:
                    raise ValueError('Expected a 2D fractional composition tensor')
                empty_rows = ~present_mask.any(dim=1)
                if bool(empty_rows.any()):
                    fractions = fractions.clone()
                    present_mask = present_mask.clone()
                    fractions[empty_rows, 0] = 1.0
                    present_mask[empty_rows, 0] = True

                batch_size, token_count = fractions.shape
                element_index = (
                    torch_module.arange(token_count, device=x.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                fraction_features = torch_module.stack(
                    [fractions, torch_module.log1p(fractions)],
                    dim=-1,
                )
                token_embedding = self.element_embedding(element_index)
                token_embedding = token_embedding + self.fraction_encoder(fraction_features)
                token_embedding = token_embedding * (1.0 + fractions.unsqueeze(-1))
                token_embedding = self.token_norm(token_embedding)
                encoded = self.encoder(
                    token_embedding,
                    src_key_padding_mask=~present_mask,
                )
                weight_denominator = fractions.sum(dim=1, keepdim=True).clamp(min=1e-8)
                normalized_weights = fractions / weight_denominator
                pooled = (encoded * normalized_weights.unsqueeze(-1)).sum(dim=1)
                return self.head(pooled)

        return FractionalCompositionAttentionNetwork()


class TorchSparseFractionalAttentionRegressor(TorchFractionalAttentionRegressor):
    """Experimental sparse-token attention over only the present elements in each formula."""

    def _build_network(self, input_dim: int, nn_module):
        if int(self.embedding_dim) % int(self.num_heads) != 0:
            raise ValueError('embedding_dim must be divisible by num_heads')
        embedding_dim = int(self.embedding_dim)
        num_heads = int(self.num_heads)
        num_layers = max(1, int(self.num_layers))
        head_hidden_dim = int(self.head_hidden_dim)
        dropout = float(self.dropout)
        padding_index = int(input_dim)
        torch_module, _ = self._import_torch()

        class SparseFractionalCompositionAttentionNetwork(nn_module.Module):
            def __init__(self):
                super().__init__()
                self.padding_index = padding_index
                self.element_embedding = nn_module.Embedding(
                    input_dim + 1,
                    embedding_dim,
                    padding_idx=padding_index,
                )
                self.rank_embedding = nn_module.Embedding(input_dim, embedding_dim)
                self.fraction_encoder = nn_module.Sequential(
                    nn_module.Linear(2, embedding_dim),
                    nn_module.GELU(),
                    nn_module.Linear(embedding_dim, embedding_dim),
                )
                encoder_layer = nn_module.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=num_heads,
                    dim_feedforward=max(embedding_dim * 4, head_hidden_dim),
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=False,
                )
                self.encoder = nn_module.TransformerEncoder(
                    encoder_layer,
                    num_layers=num_layers,
                )
                self.token_norm = nn_module.LayerNorm(embedding_dim)
                self.head = nn_module.Sequential(
                    nn_module.LayerNorm(embedding_dim),
                    nn_module.Linear(embedding_dim, head_hidden_dim),
                    nn_module.GELU(),
                    nn_module.Dropout(dropout),
                    nn_module.Linear(head_hidden_dim, 1),
                )

            def forward(self, x):
                fractions = x.clamp(min=0.0)
                if fractions.ndim != 2:
                    raise ValueError('Expected a 2D fractional composition tensor')

                token_indices, token_fractions, token_mask = _build_sparse_fractional_tokens(
                    fractions,
                    padding_index=self.padding_index,
                    torch_module=torch_module,
                )
                batch_size, max_tokens = token_indices.shape
                rank_index = (
                    torch_module.arange(max_tokens, device=x.device)
                    .unsqueeze(0)
                    .expand(batch_size, -1)
                )
                fraction_features = torch_module.stack(
                    [token_fractions, torch_module.log1p(token_fractions)],
                    dim=-1,
                )
                token_embedding = self.element_embedding(token_indices)
                token_embedding = token_embedding + self.rank_embedding(rank_index)
                token_embedding = token_embedding + self.fraction_encoder(fraction_features)
                token_embedding = token_embedding * (1.0 + token_fractions.unsqueeze(-1))
                token_embedding = token_embedding * token_mask.unsqueeze(-1).to(token_embedding.dtype)
                token_embedding = self.token_norm(token_embedding)
                encoded = self.encoder(
                    token_embedding,
                    src_key_padding_mask=~token_mask,
                )
                weight_denominator = token_fractions.sum(dim=1, keepdim=True).clamp(min=1e-8)
                normalized_weights = token_fractions / weight_denominator
                pooled = (encoded * normalized_weights.unsqueeze(-1)).sum(dim=1)
                return self.head(pooled)

        return SparseFractionalCompositionAttentionNetwork()


class TorchRoostLikeRegressor(TorchFractionalAttentionRegressor):
    """Experimental present-element stoichiometry network inspired by Roost."""

    def __init__(
        self,
        embedding_dim: int = 96,
        num_message_layers: int = 2,
        message_hidden_dim: int = 128,
        head_hidden_dim: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        max_epochs: int = 40,
        patience: int = 6,
        min_delta: float = 1e-4,
        val_fraction: float = 0.15,
        device: str = 'auto',
        random_seed: int = 42,
        expected_input_dim: int = 118,
        batch_size: int = 256,
        eval_batch_size: int | None = None,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            num_heads=1,
            num_layers=max(1, int(num_message_layers)),
            head_hidden_dim=head_hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            patience=patience,
            min_delta=min_delta,
            val_fraction=val_fraction,
            device=device,
            random_seed=random_seed,
            expected_input_dim=expected_input_dim,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
        )
        self.num_message_layers = num_message_layers
        self.message_hidden_dim = message_hidden_dim

    def _build_network(self, input_dim: int, nn_module):
        embedding_dim = int(self.embedding_dim)
        num_message_layers = max(1, int(self.num_message_layers))
        message_hidden_dim = int(self.message_hidden_dim)
        head_hidden_dim = int(self.head_hidden_dim)
        dropout = float(self.dropout)
        padding_index = int(input_dim)
        torch_module, _ = self._import_torch()

        class RoostLikeMessagePassingBlock(nn_module.Module):
            def __init__(self):
                super().__init__()
                self.query = nn_module.Linear(embedding_dim, embedding_dim, bias=False)
                self.key = nn_module.Linear(embedding_dim, embedding_dim, bias=False)
                self.value = nn_module.Linear(embedding_dim, embedding_dim, bias=False)
                self.edge_bias = nn_module.Sequential(
                    nn_module.Linear(4, message_hidden_dim),
                    nn_module.GELU(),
                    nn_module.Linear(message_hidden_dim, 1),
                )
                self.message_gate = nn_module.Sequential(
                    nn_module.Linear((embedding_dim * 2) + 3, message_hidden_dim),
                    nn_module.GELU(),
                    nn_module.Linear(message_hidden_dim, embedding_dim),
                    nn_module.Sigmoid(),
                )
                self.message_proj = nn_module.Sequential(
                    nn_module.Linear(embedding_dim, embedding_dim),
                    nn_module.GELU(),
                    nn_module.Dropout(dropout),
                )
                self.message_norm = nn_module.LayerNorm(embedding_dim)
                self.feedforward = nn_module.Sequential(
                    nn_module.Linear(embedding_dim, max(message_hidden_dim, embedding_dim)),
                    nn_module.GELU(),
                    nn_module.Dropout(dropout),
                    nn_module.Linear(max(message_hidden_dim, embedding_dim), embedding_dim),
                    nn_module.Dropout(dropout),
                )
                self.feedforward_norm = nn_module.LayerNorm(embedding_dim)

            def forward(self, token_state, token_fractions, token_mask):
                scale = float(embedding_dim) ** -0.5
                query = self.query(token_state)
                key = self.key(token_state)
                value = self.value(token_state)
                attention_logits = torch_module.matmul(query, key.transpose(1, 2)) * scale

                receiver_fraction = token_fractions.unsqueeze(2)
                sender_fraction = token_fractions.unsqueeze(1)
                edge_features = torch_module.stack(
                    [
                        receiver_fraction.expand(-1, -1, token_fractions.shape[1]),
                        sender_fraction.expand(-1, token_fractions.shape[1], -1),
                        torch_module.abs(receiver_fraction - sender_fraction),
                        receiver_fraction * sender_fraction,
                    ],
                    dim=-1,
                )
                attention_logits = attention_logits + self.edge_bias(edge_features).squeeze(-1)

                valid_edges = token_mask.unsqueeze(1) & token_mask.unsqueeze(2)
                attention_logits = attention_logits.masked_fill(~valid_edges, -1e9)
                attention_weights = torch_module.softmax(attention_logits, dim=-1)
                attention_weights = attention_weights * valid_edges.to(attention_weights.dtype)
                attention_weights = attention_weights / attention_weights.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp(min=1e-8)

                aggregated_message = torch_module.matmul(attention_weights, value)
                fraction_features = torch_module.stack(
                    [
                        token_fractions,
                        torch_module.log1p(token_fractions),
                        torch_module.sqrt(token_fractions.clamp(min=0.0)),
                    ],
                    dim=-1,
                )
                gate = self.message_gate(
                    torch_module.cat(
                        [token_state, aggregated_message, fraction_features],
                        dim=-1,
                    )
                )
                updated_state = self.message_norm(
                    token_state + self.message_proj(gate * aggregated_message)
                )
                updated_state = self.feedforward_norm(
                    updated_state + self.feedforward(updated_state)
                )
                return updated_state * token_mask.unsqueeze(-1).to(updated_state.dtype)

        class RoostLikeFractionalCompositionNetwork(nn_module.Module):
            def __init__(self):
                super().__init__()
                self.padding_index = padding_index
                self.element_embedding = nn_module.Embedding(
                    input_dim + 1,
                    embedding_dim,
                    padding_idx=padding_index,
                )
                self.fraction_encoder = nn_module.Sequential(
                    nn_module.Linear(3, embedding_dim),
                    nn_module.GELU(),
                    nn_module.Linear(embedding_dim, embedding_dim),
                )
                self.input_norm = nn_module.LayerNorm(embedding_dim)
                self.message_passing_blocks = nn_module.ModuleList(
                    RoostLikeMessagePassingBlock()
                    for _ in range(num_message_layers)
                )
                self.composition_norm = nn_module.LayerNorm(embedding_dim)
                self.head = nn_module.Sequential(
                    nn_module.LayerNorm(embedding_dim),
                    nn_module.Linear(embedding_dim, head_hidden_dim),
                    nn_module.GELU(),
                    nn_module.Dropout(dropout),
                    nn_module.Linear(head_hidden_dim, 1),
                )

            def forward(self, x):
                fractions = x.clamp(min=0.0)
                if fractions.ndim != 2:
                    raise ValueError('Expected a 2D fractional composition tensor')

                token_indices, token_fractions, token_mask = _build_sparse_fractional_tokens(
                    fractions,
                    padding_index=self.padding_index,
                    torch_module=torch_module,
                )
                fraction_features = torch_module.stack(
                    [
                        token_fractions,
                        torch_module.log1p(token_fractions),
                        torch_module.sqrt(token_fractions.clamp(min=0.0)),
                    ],
                    dim=-1,
                )
                token_state = self.element_embedding(token_indices)
                token_state = token_state + self.fraction_encoder(fraction_features)
                token_state = token_state * (1.0 + token_fractions.unsqueeze(-1))
                token_state = token_state * token_mask.unsqueeze(-1).to(token_state.dtype)
                token_state = self.input_norm(token_state)

                for message_block in self.message_passing_blocks:
                    token_state = message_block(token_state, token_fractions, token_mask)

                normalized_weights = token_fractions / token_fractions.sum(
                    dim=1,
                    keepdim=True,
                ).clamp(min=1e-8)
                pooled = (token_state * normalized_weights.unsqueeze(-1)).sum(dim=1)
                pooled = self.composition_norm(pooled)
                return self.head(pooled)

        return RoostLikeFractionalCompositionNetwork()


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
