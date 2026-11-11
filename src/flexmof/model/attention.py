"""Memory-efficient attention over precomputed periodic MOF weights."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .periodic import iter_periodic_edge_slices, polynomial_cutoff, unpack_periodic_edges


class PrecomputedLogAttentionWeights(NamedTuple):
    value: Tensor


def apply_rotary_positions(value: Tensor, position_ids: Tensor) -> Tensor:
    """Rotate adjacent Q/K channel pairs, for [batch, heads, length, dim].

    Positions are sequence ranks, not learned IDs or Cartesian coordinates.
    Compute angles in float32 even under mixed precision. No checkpoint tensors.
    """
    dimension = value.size(-1)
    if dimension % 2:
        raise ValueError("RoPE requires an even attention head dimension")
    if position_ids.shape != (value.size(0), value.size(2)):
        raise ValueError("RoPE position_ids must have shape [batch, length]")
    frequencies = 10000.0 ** (
        -torch.arange(0, dimension, 2, device=value.device, dtype=torch.float32)
        / dimension
    )
    angles = position_ids.to(device=value.device, dtype=torch.float32)[..., None] * frequencies
    cosine = angles.cos().to(value.dtype)[:, None]
    sine = angles.sin().to(value.dtype)[:, None]
    even, odd = value[..., 0::2], value[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


def collapse_periodic_attention(
    positions: Tensor,
    padding_mask: Tensor,
    periodic: dict[str, Tensor],
    num_heads: int,
    attention_bias: Tensor | None = None,
) -> Tensor:
    """Collapse expanded periodic neighbors onto atoms in the central cell."""
    if periodic.get("streaming_per_atom", False):
        if attention_bias is not None:
            raise ValueError(
                "Streaming biased attention must be collapsed by SpatialEmbedding"
            )
        batch_size, num_atoms = positions.shape[:2]
        atoms = ~padding_mask
        # LMDB coordinates are float64, while streaming geometry is deliberately
        # cached as float32. Keep both sides of the in-place scatter reduction in
        # the cache dtype and avoid carrying float64 attention weights downstream.
        geometry_positions = positions.to(periodic["edge_distance"].dtype)
        central_dist = (
            geometry_positions[:, :, None] - geometry_positions[:, None]
        ).norm(dim=-1)
        weights = polynomial_cutoff(central_dist, periodic["cutoff"])
        weights *= atoms.unsqueeze(-1) & atoms.unsqueeze(1)
        for edge_offset, edges in iter_periodic_edge_slices(positions, atoms, periodic):
            batch, query, _, source = unpack_periodic_edges(periodic, edges)
            distances = periodic["edge_distance"][
                edge_offset : edge_offset + edges.numel()
            ]
            edge_weights = polynomial_cutoff(distances, periodic["cutoff"])
            flat_index = (batch * num_atoms + query) * num_atoms + source
            weights.view(-1).scatter_add_(0, flat_index, edge_weights)
        return weights[:, None].expand(-1, num_heads, -1, -1)

    batch_size, num_atoms = positions.shape[:2]
    outcell_index = periodic["outcell_index"]
    expanded_count = num_atoms + outcell_index.size(-1)
    if attention_bias is None:
        expanded_weights = torch.ones(
            batch_size,
            num_heads,
            num_atoms,
            expanded_count,
            device=positions.device,
            dtype=positions.dtype,
        )
    else:
        expanded_weights = torch.softmax(attention_bias, dim=-1)

    expand_mask = periodic["expand_mask"]
    local_weights = periodic["local_attention_weight"]
    if expand_mask.dim() == 3:
        full_mask = torch.cat(
            [padding_mask[:, None, :].repeat(1, num_atoms, 1), expand_mask],
            dim=-1,
        )
        central_index = torch.arange(
            num_atoms, device=positions.device, dtype=outcell_index.dtype
        )[None, None, :].repeat(batch_size, num_atoms, 1)
        full_index = torch.cat([central_index, outcell_index], dim=-1)
        full_index = full_index.masked_fill(full_mask, num_atoms)
        full_index = full_index[:, None].repeat(1, num_heads, 1, 1)
    else:
        full_mask = torch.cat([padding_mask, expand_mask], dim=-1)
        central_index = torch.arange(
            num_atoms, device=positions.device, dtype=outcell_index.dtype
        )[None, :].repeat(batch_size, 1)
        full_index = torch.cat([central_index, outcell_index], dim=-1)
        full_index = full_index.masked_fill(full_mask, num_atoms)
        full_index = full_index[:, None, None].repeat(1, num_heads, num_atoms, 1)
        if local_weights is not None:
            local_weights = local_weights.masked_fill(
                full_mask[:, None, :].repeat(1, num_atoms, 1), 0.0
            )

    if local_weights is not None:
        local_weights = local_weights.masked_fill(full_mask, 0.0)
        expanded_weights = expanded_weights * local_weights[:, None].repeat(
            1, num_heads, 1, 1
        )

    weights = torch.zeros(
        batch_size,
        num_heads,
        num_atoms,
        num_atoms + 1,
        dtype=expanded_weights.dtype,
        device=expanded_weights.device,
    )
    weights.scatter_add_(dim=-1, index=full_index, src=expanded_weights)
    return weights[..., :-1]


class MOFAttention(nn.Module):
    """Attention whose normalized neighborhood weights are computed once."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        d_tilde=1,
        normalize_softmax_probs_after_local_attn_weights_eps=1e-10,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.normalize_softmax_probs_after_local_attn_weights_eps = (
            normalize_softmax_probs_after_local_attn_weights_eps
        )
        gain = 1.0 / math.sqrt(2 * d_tilde)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0 / math.sqrt(d_tilde))

    def forward(
        self,
        query,
        softmax_weights,
        *,
        math_kernel=False,
        position_ids=None,
    ):
        if isinstance(softmax_weights, PrecomputedLogAttentionWeights):
            log_weights = softmax_weights.value
        else:
            weights = softmax_weights.to(dtype=query.dtype)
            log_weights = weights.clamp_min(
                self.normalize_softmax_probs_after_local_attn_weights_eps
            ).log_()
            log_weights.masked_fill_(weights <= 0, float("-inf"))

        target_length, batch_size, embedding_dim = query.shape
        source = query
        query = self.q_proj(source)
        key = self.k_proj(source)
        value = self.v_proj(source)
        query = query.view(
            target_length, batch_size, self.num_heads, self.head_dim
        ).permute(1, 2, 0, 3)
        key = key.view(
            target_length, batch_size, self.num_heads, self.head_dim
        ).permute(1, 2, 0, 3)
        value = value.view(
            target_length, batch_size, self.num_heads, self.head_dim
        ).permute(1, 2, 0, 3)

        if position_ids is not None:
            query = apply_rotary_positions(query, position_ids)
            key = apply_rotary_positions(key, position_ids)

        backends = (
            [SDPBackend.MATH]
            if math_kernel or query.device.type == "cpu"
            else [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        )
        with sdpa_kernel(backends):
            output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key,
                value,
                dropout_p=self.dropout if self.training else 0.0,
                attn_mask=log_weights,
                is_causal=False,
            )
        output = (
            output.transpose(1, 2)
            .reshape(batch_size, target_length, embedding_dim)
            .transpose(0, 1)
        )
        return self.out_proj(output), None
