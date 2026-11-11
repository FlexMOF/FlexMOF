"""Gaussian radial features for periodic MOF positions."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .periodic import unpack_periodic_edges


class SpatialBias(nn.Module):
    """Build attention bias and node embeddings from periodic distances."""

    def __init__(self, config, key_prefix: str = ""):
        super().__init__()
        self.gbf = GaussianLayer(config.num_3d_bias_kernel, config.num_edges)
        self.gbf_proj = NonLinear(config.num_3d_bias_kernel, config.num_attention_heads)
        self.pos_embedding_proj = nn.Linear(
            config.num_3d_bias_kernel, config.encoder_embed_dim
        )
        self.key_prefix = key_prefix

    def forward(self, batch, padding_mask, periodic):
        positions = batch[f"{self.key_prefix}pos"]
        _, node_count = positions.shape[:2]
        outcell_index = periodic["outcell_index"]
        if outcell_index.dim() == 3:
            expanded_positions = torch.cat(
                [
                    positions[:, None].repeat(1, node_count, 1, 1),
                    periodic[f"{self.key_prefix}expand_pos"],
                ],
                dim=2,
            )
            relative = positions[:, :, None] - expanded_positions
        else:
            expanded_positions = torch.cat(
                [positions, periodic[f"{self.key_prefix}expand_pos"]], dim=1
            )
            relative = positions.unsqueeze(2) - expanded_positions.unsqueeze(1)
        expanded_count = expanded_positions.size(-2)
        distances = relative.norm(dim=-1).view(-1, node_count, expanded_count)

        edge_features = self.gbf(distances, periodic["expand_node_type_edge"].long())
        attention_bias = self.gbf_proj(edge_features)
        expand_mask = periodic["expand_mask"]
        if expand_mask.dim() == 3:
            full_mask = torch.cat(
                [
                    padding_mask[:, None, :].repeat(1, node_count, 1),
                    expand_mask,
                ],
                dim=-1,
            )
            feature_mask = full_mask[..., None]
        else:
            full_mask = torch.cat([padding_mask, expand_mask], dim=-1)
            feature_mask = full_mask[:, None, :, None]
        attention_bias = attention_bias.masked_fill(feature_mask, float("-inf"))
        edge_features = edge_features.masked_fill(feature_mask, 0.0)
        attention_bias = attention_bias.masked_fill(
            padding_mask[:, :, None, None], 0.0
        ).permute(0, 3, 1, 2)

        local_weights = periodic["local_attention_weight"]
        if local_weights is not None:
            edge_features *= local_weights[..., None].to(edge_features.dtype)
        edge_features = edge_features.sum(dim=-2)
        position_embedding = self.pos_embedding_proj(
            edge_features.to(self.pos_embedding_proj.weight.dtype)
        )
        position_embedding = position_embedding.masked_fill(
            padding_mask[..., None], 0.0
        )
        return attention_bias, position_embedding

    def central_terms(self, batch, padding_mask):
        positions = batch[f"{self.key_prefix}pos"]
        relative = positions[:, :, None] - positions[:, None]
        distances = relative.norm(dim=-1)
        features = self.gbf(distances, batch["node_type_edge"].long())
        bias = self.gbf_proj(features).permute(0, 3, 1, 2)
        pair_mask = padding_mask.unsqueeze(-1) | padding_mask.unsqueeze(1)
        features = features.masked_fill(pair_mask[..., None], 0.0)
        bias = bias.masked_fill(pair_mask[:, None], float("-inf"))
        bias = bias.masked_fill(padding_mask[:, None, :, None], 0.0)
        return bias, features, distances

    def edge_terms(self, batch, periodic, edge_offset, packed_edges):
        batch_index, query, cell, source = unpack_periodic_edges(periodic, packed_edges)
        distance_key = "edge_init_distance" if self.key_prefix else "edge_distance"
        distances = periodic[distance_key][
            edge_offset : edge_offset + packed_edges.numel()
        ]
        edge_types = batch["node_type_edge"][batch_index, query, source].long()
        features = self.gbf(distances, edge_types)
        bias = self.gbf_proj(features)
        return batch_index, query, cell, source, bias, features, distances


def gaussian(x, mean, std):
    normalization = (2 * torch.pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (normalization * std)


class GaussianLayer(nn.Module):
    def __init__(self, kernels=128, edge_types=512 * 3):
        super().__init__()
        self.kernels = kernels
        self.means = nn.Embedding(1, kernels)
        self.stds = nn.Embedding(1, kernels)
        self.mul = nn.Embedding(edge_types, 1, padding_idx=0)
        self.bias = nn.Embedding(edge_types, 1, padding_idx=0)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, distances, edge_types):
        scale = self.mul(edge_types).sum(dim=-2)
        bias = self.bias(edge_types).sum(dim=-2)
        distances = scale * distances.unsqueeze(-1) + bias
        distances = distances.expand(*distances.shape[:-1], self.kernels)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        return gaussian(distances.float(), mean, std).type_as(self.means.weight)


class NonLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, input_size)
        self.layer2 = nn.Linear(input_size, output_size)

    def forward(self, inputs):
        return self.layer2(F.gelu(self.layer1(inputs)))
