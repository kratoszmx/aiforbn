from __future__ import annotations

import numpy as np

from torch_models.attention import TorchFractionalAttentionRegressor
from torch_models.base import *
from torch_models.base import _build_sparse_fractional_tokens

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

