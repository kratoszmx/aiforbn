from __future__ import annotations

import numpy as np

from torch_models.base import *

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

