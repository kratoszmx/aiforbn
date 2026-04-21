from __future__ import annotations

import numpy as np

from torch_models.attention import TorchFractionalAttentionRegressor
from torch_models.base import *
from torch_models.base import _build_sparse_fractional_tokens

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

