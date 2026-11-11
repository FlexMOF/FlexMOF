"""Invariant spatial encoder for periodic MOF diffusion."""

from __future__ import annotations

from torch import nn

from .attention import PrecomputedLogAttentionWeights, collapse_periodic_attention
from .dit import DiTBlock
from .embedding import SpatialEmbedding


class StructureEncoder(nn.Module):
    """Encode noisy coordinates and bond features before equivariant decoding."""

    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            DiTBlock(
                config,
                embedding_dim=config.structure_hidden_dim,
                ffn_embedding_dim=config.structure_ffn_dim,
            )
            for _ in range(config.num_structure_encoder_layer)
        )
        self.embedding = SpatialEmbedding(config)
        self.pair_feat_bias = nn.Sequential(
            nn.Linear(
                config.encoder_pair_embed_dim,
                config.encoder_pair_embed_dim,
                bias=False,
            ),
            nn.SiLU(),
            nn.Linear(
                config.encoder_pair_embed_dim,
                config.num_attention_heads,
                bias=False,
            ),
        )
        self.num_heads = config.num_attention_heads
        self.attention_epsilon = (
            config.normalize_softmax_probs_after_local_attn_weights_eps
        )

    def forward(
        self,
        batch,
        invariant,
        time_embedding,
        padding_mask,
        pair_features,
        periodic,
        clean_mask,
    ):
        invariant = invariant.transpose(0, 1)
        pair_bias = self.pair_feat_bias(pair_features).permute(0, 3, 1, 2)
        if periodic.get("streaming_per_atom", False):
            position_embedding, attention_bias, weights = (
                self.embedding.forward_streaming(
                    batch,
                    padding_mask,
                    clean_mask,
                    periodic,
                    pair_bias,
                )
            )
            log_weights = weights.clamp_min(self.attention_epsilon).log_()
            log_weights.masked_fill_(weights <= 0, float("-inf"))
            log_weights = PrecomputedLogAttentionWeights(log_weights)
            for layer in self.layers:
                position_embedding = layer(
                    position_embedding + time_embedding,
                    invariant,
                    log_weights,
                )
            return position_embedding, attention_bias, log_weights

        position_embedding, position_bias = self.embedding(
            batch, padding_mask, clean_mask, periodic
        )
        attention_bias = pair_bias + position_bias
        weights = collapse_periodic_attention(
            batch["pos"],
            padding_mask,
            periodic,
            self.num_heads,
            attention_bias,
        )
        log_weights = weights.clamp_min(self.attention_epsilon).log_()
        log_weights.masked_fill_(weights <= 0, float("-inf"))
        log_weights = PrecomputedLogAttentionWeights(log_weights)

        for layer in self.layers:
            position_embedding = layer(
                position_embedding + time_embedding,
                invariant,
                log_weights,
            )
        return position_embedding, attention_bias, log_weights
