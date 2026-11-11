"""LG-Flow graph VAE with an additional attached-hydrogen categorical attribute."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .linker_autoencoder import (LinkerAEConfig, LinkerAutoencoder,
                                 reconstruction_loss as base_loss,
                                 reconstruction_counts as base_counts)


@dataclass
class HydrogenAEConfig(LinkerAEConfig):
    # Includes methane's four attached hydrogens, even when absent from a split.
    hydrogen_count_classes: int = 5

    def __post_init__(self):
        super().__post_init__()
        if self.hydrogen_count_classes < 2:
            raise ValueError('The hydrogen-count vocabulary must include zero and positive counts')
        if self.decoder_context_layers:
            raise ValueError('The hydrogen representation uses its own freshly trained codec')


class HydrogenAutoencoder(LinkerAutoencoder):
    def __init__(self, config, trunk=None):
        super().__init__(config, trunk)
        self.hydrogen_embedding = nn.Embedding(config.hydrogen_count_classes, config.width)
        self.hydrogen_out = nn.Linear(config.latent_dim, config.hydrogen_count_classes)

    def node_input(self, batch, spectral_embedding):
        counts = batch['hydrogen_counts']
        hidden = self.atom_embedding(batch['atoms']) + spectral_embedding + self.hydrogen_embedding(counts)
        degree = batch['bonds'].sum(-1, keepdim=True).float() + counts[..., None]
        return hidden + self.degree_embedding(degree)

    def decode(self, latent, mask):
        return dict(super().decode(latent, mask), hydrogen_logits=self.hydrogen_out(latent))


def reconstruction_loss(output, batch, kl_weight):
    loss, details = base_loss(output, batch, kl_weight)
    counts = F.cross_entropy(output['hydrogen_logits'][batch['mask']].float(), batch['hydrogen_counts'][batch['mask']])
    return loss+counts, dict(details, hydrogen_ce=counts.detach())


def reconstruction_counts(output, batch):
    counts = base_counts(output, batch)
    atoms_ok = output['atom_logits'].argmax(-1).eq(batch['atoms']) | ~batch['mask']
    pairs = torch.triu(output['pair_mask'], diagonal=1)
    bonds_ok = output['bond_logits'].argmax(-1).eq(batch['bonds']) | ~pairs
    hydrogen_ok = output['hydrogen_logits'].argmax(-1).eq(batch['hydrogen_counts']) | ~batch['mask']
    # Equality in this representation is sufficient for equality after expansion.
    counts['exact_graphs'] = (atoms_ok.all(-1) & bonds_ok.flatten(1).all(-1) & hydrogen_ok.all(-1)).sum()
    counts['correct_hydrogen_counts'] = (hydrogen_ok & batch['mask']).sum()
    return counts
