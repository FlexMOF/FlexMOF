"""Invariant sequence encoder for MOF building-block graphs."""

from __future__ import annotations

import torch
from torch import nn

from .attention import collapse_periodic_attention, PrecomputedLogAttentionWeights
from .sequence_layer import SequenceEncoderLayer


class SequenceEncoder(nn.Module):
    """Transformer encoder over MOF building-block tokens and pair features."""

    def __init__(self, config, *, graph_only=False, graph_norm=False):
        super().__init__()
        self.layers = nn.ModuleList(
            SequenceEncoderLayer(config, update_pair_features=not graph_only)
            for _ in range(config.num_encoder_layers)
        )
        self.ln_pair = nn.LayerNorm(config.encoder_pair_embed_dim) if not graph_only else None
        # ``graph_norm`` also enables the graph path on a full encoder, which is
        # how one trunk serves both the coordinate pass and a graph encoder.
        # It stays absent otherwise, so the coordinate-only state_dict keys are
        # byte-identical to every checkpoint trained before sharing existed.
        self.graph_norm = (
            nn.LayerNorm(config.embedding_dim) if graph_only or graph_norm else None
        )
        self.num_heads = config.num_attention_heads
        self.use_rope = config.sequence_rope

    def forward_graph(self, hidden, mask, attention_bias=None):
        """Encode atom/bond graphs without coordinates or periodic expansion.

        Reuses the same sequence attention and feed-forward blocks. This mode
        disables unused pair accumulation; the standard MOF path is unchanged.
        """
        if self.graph_norm is None:
            raise ValueError("forward_graph requires graph_only=True")
        if attention_bias is None:
            attention_bias = hidden.new_zeros(hidden.size(0), 1, 1, hidden.size(1))
        bias = attention_bias.to(hidden.dtype).masked_fill(~mask[:, None, None, :], float("-inf"))
        weights = PrecomputedLogAttentionWeights(bias)
        hidden = hidden.transpose(0, 1)
        for layer in self.layers:
            hidden, _ = layer(hidden, None, weights)
        return self.graph_norm(hidden.transpose(0, 1)).masked_fill(~mask[..., None], 0)

    @torch.compiler.disable(recursive=False)
    def forward(self, x, pair, batch, periodic, padding_mask):
        weights = collapse_periodic_attention(
            batch["pos"], padding_mask, periodic, self.num_heads
        )
        positions = batch["position_ids"] if self.use_rope else None
        for layer in self.layers:
            x, pair = layer(x, pair, weights, position_ids=positions)

        pair = self.ln_pair(pair)
        if periodic.get("streaming_per_atom", False):
            return x, pair.permute(2, 0, 1, 3)
        outcell_index = periodic["outcell_index"]
        if outcell_index.dim() == 2:
            num_atoms, pair_dim = pair.size(0), pair.size(-1)
            outcell_index = outcell_index.transpose(0, 1)[None, :, :, None].repeat(
                num_atoms, 1, 1, pair_dim
            )
        else:
            pair_dim = pair.size(-1)
            outcell_index = outcell_index.permute(1, 2, 0)[:, :, :, None].repeat(
                1, 1, 1, pair_dim
            )
        expanded_pair = torch.gather(pair, dim=1, index=outcell_index)
        pair = torch.cat([pair, expanded_pair], dim=1)
        return x, pair.permute(2, 0, 1, 3)
