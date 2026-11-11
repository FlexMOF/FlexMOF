"""Diffusion-transformer block for invariant coordinate features."""

from __future__ import annotations

from torch import nn

from .attention import MOFAttention


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    """DiT block with adaptive layer-norm-zero conditioning."""

    def __init__(self, config, embedding_dim, ffn_embedding_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            config.embedding_dim, elementwise_affine=False, eps=1e-6
        )
        self.attn = MOFAttention(
            embedding_dim,
            config.num_attention_heads,
            dropout=config.dropout,
            normalize_softmax_probs_after_local_attn_weights_eps=(
                config.normalize_softmax_probs_after_local_attn_weights_eps
            ),
        )
        self.norm2 = nn.LayerNorm(
            config.embedding_dim, elementwise_affine=False, eps=1e-6
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, ffn_embedding_dim, bias=False),
            nn.SiLU(),
            nn.Linear(ffn_embedding_dim, embedding_dim, bias=False),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(embedding_dim, 6 * embedding_dim, bias=False)
        )

    def forward(self, x, conditioning, attention_weights):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(conditioning).chunk(6, dim=2)
        )
        attended, _ = self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa).transpose(0, 1),
            attention_weights,
        )
        x = x + gate_msa * attended.transpose(0, 1)
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
