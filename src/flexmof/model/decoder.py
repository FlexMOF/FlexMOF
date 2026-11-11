"""Equivariant coordinate decoder."""

from torch import nn

from .geometry import MOFGeometry


class EquivariantDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model = MOFGeometry(config)

    def forward(
        self,
        batch,
        invariant,
        attention_bias,
        padding_mask,
        periodic,
        time_embedding,
        attention_weights,
    ):
        return self.model(
            batch,
            invariant,
            batch["pos"],
            attention_bias,
            padding_mask,
            periodic,
            time_embedding,
            attention_weights,
        )
