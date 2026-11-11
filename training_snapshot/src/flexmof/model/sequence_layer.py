"""One layer of the invariant MOF sequence encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .attention import MOFAttention


class SequenceEncoderLayer(nn.Module):
    """Self-attention/FFN block that also accumulates pair features."""

    def __init__(self, config, *, update_pair_features=True):
        super().__init__()
        self.self_attn = MOFAttention(
            config.embedding_dim,
            config.num_attention_heads,
            dropout=config.dropout,
            normalize_softmax_probs_after_local_attn_weights_eps=(
                config.normalize_softmax_probs_after_local_attn_weights_eps
            ),
        )
        self.fc1 = nn.Linear(
            config.embedding_dim, config.encoder_ffn_embed_dim, bias=False
        )
        self.fc2 = nn.Linear(
            config.encoder_ffn_embed_dim, config.embedding_dim, bias=False
        )
        self.top_layer_norm = nn.LayerNorm(config.embedding_dim)
        self.mid_layer_norm = nn.LayerNorm(config.embedding_dim)
        self.pair_proj = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(
                config.embedding_dim,
                config.encoder_pair_embed_dim * 2,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(
                config.encoder_pair_embed_dim * 2,
                config.encoder_pair_embed_dim,
                bias=False,
            ),
        ) if update_pair_features else None
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.top_layer_norm.reset_parameters()
        self.mid_layer_norm.reset_parameters()

    def forward(self, x, pair, attention_weights, *, position_ids=None):
        residual = x
        x = self.top_layer_norm(x)
        x, _ = self.self_attn(x, attention_weights, position_ids=position_ids)
        x = residual + x
        residual = x
        x = self.mid_layer_norm(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        # A layer that owns pair_proj is still used by the graph-only path,
        # which has no pair features to accumulate into.
        if self.pair_proj is not None and pair is not None:
            pair_projection = self.pair_proj(x)
            pair += torch.einsum("lbh,kbh->lkbh", pair_projection, pair_projection)
        return x, pair
