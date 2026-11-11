"""LG-Flow-style Laplacian VAE using FlexMOF sequence attention/FFN layers.

Only the encoder sees atoms, bonds and their Laplacian. Decoding requires just
one fixed-width vector per atom and the atom mask. No graph or positional skip
connection crosses the bottleneck. See docs/linker_lgflow.md for differences
from LG-Flow (arXiv:2601.13780) and its reference implementation.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from .sequence import SequenceEncoder


@dataclass
class LinkerAEConfig:
    num_atom_classes: int
    width: int = 256
    layers: int = 6
    heads: int = 8
    latent_dim: int = 32
    frequencies: int = 32
    decoder_width: int = 24
    spectral_modulation: float = 0.005
    kl_weight: float = 1e-6
    dropout: float = 0.0
    decoder_context_layers: int = 0
    decoder_context_width: int = 128

    def __post_init__(self):
        if min(self.width, self.layers, self.heads, self.latent_dim, self.frequencies, self.num_atom_classes) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("Width must be divisible by attention heads")
        if self.decoder_context_layers < 0 or (self.decoder_context_layers and
                (self.decoder_context_width < 1 or self.decoder_context_width % self.heads)):
            raise ValueError("Decoder context needs nonnegative layers and a positive width divisible by heads")


class GraphSequenceStack(SequenceEncoder):
    """The FlexMOF sequence blocks with graph-only attention weights."""

    def __init__(self, width, layers, heads, dropout=0.0):
        config = SimpleNamespace(embedding_dim=width, num_attention_heads=heads,
                                 dropout=dropout, encoder_ffn_embed_dim=width * 4,
                                 encoder_pair_embed_dim=1,
                                 num_encoder_layers=layers, sequence_rope=False,
                                 normalize_softmax_probs_after_local_attn_weights_eps=1e-10)
        super().__init__(config, graph_only=True)

    def forward(self, hidden, mask, bias=None):
        return self.forward_graph(hidden, mask, bias)


def trunk_geometry(trunk):
    """Width and attention-head count of a sequence stack, owned or shared."""
    layer = trunk.layers[0]
    return layer.self_attn.embed_dim, trunk.num_heads


class DecoderLatentContext(nn.Module):
    """Continuous graph context inside the decoder, before any label decisions.

    Only latents and their soft predictions enter attention. A zero-initialized
    latent residual preserves the pretrained decoder exactly at initialization.
    Singleton decoding stays unchanged because it has no neighbor context.
    """

    def __init__(self, config):
        super().__init__()
        self.input = nn.Linear(config.latent_dim+config.num_atom_classes+2, config.decoder_context_width)
        self.edge_bias = nn.Linear(1, config.heads, bias=False)
        nn.init.constant_(self.edge_bias.weight, 4.)
        self.sequence = GraphSequenceStack(config.decoder_context_width, config.decoder_context_layers, config.heads)
        self.output = nn.Linear(config.decoder_context_width, config.latent_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, latent, mask, base):
        atoms = base['atom_logits'].float().softmax(-1)
        bonds = base['bond_logits'].float().softmax(-1)[..., 1].masked_fill(~base['pair_mask'], 0)
        counts = mask.sum(-1).float()
        features = torch.cat([latent, atoms, bonds.sum(-1).log1p()[..., None],
                              counts.log1p()[:, None, None].expand(-1, mask.size(1), -1)], -1)
        bias = self.edge_bias(bonds[..., None]).permute(0, 3, 1, 2)
        hidden = self.sequence(self.input(features), mask, bias)
        delta = self.output(hidden)
        return delta.masked_fill(~mask[..., None] | (counts <= 1)[:, None, None], 0)


class LinkerGraphDecoder(nn.Module):
    """Bilinear edge scores with an endpoint-symmetric DeepSets classifier."""

    def __init__(self, config):
        super().__init__()
        self.latent_dim = config.latent_dim
        self.atom_out = nn.Linear(config.latent_dim, config.num_atom_classes)
        self.edge_qk = nn.Linear(config.latent_dim, 4 * config.latent_dim)
        self.edge_phi = nn.Linear(1, config.decoder_width)
        self.edge_context = nn.Linear(config.decoder_width, config.decoder_width, bias=False)
        self.edge_self = nn.Linear(1, config.decoder_width)
        self.edge_out = nn.Linear(config.decoder_width, 1)
        self.context = DecoderLatentContext(config) if config.decoder_context_layers else None

    def forward(self, latent, mask):
        base = self.decode_base(latent, mask)
        if self.context is None:
            return base
        delta = self.context(latent, mask, base)
        return {**self.decode_base(latent+delta, mask), 'latent_correction': delta}

    def decode_base(self, latent, mask):
        b, n, d = latent.shape
        q, k = self.edge_qk(latent).reshape(b, n, 2, 2, d).unbind(2)
        scores = torch.einsum("bich,bjch->bcij", q, k) / (2 * d) ** 0.5
        scores = (scores + scores.transpose(-1, -2)) * 0.5
        pair_mask = mask[:, :, None] & mask[:, None, :]
        pair_mask &= ~torch.eye(n, device=mask.device, dtype=torch.bool)[None]
        # phi is linear: transform the masked mean instead of materializing
        # B*C*N*N*decoder_width features merely to average them.
        mean = (scores * pair_mask[:, None]).sum(-1) / pair_mask.sum(-1)[:, None].clamp_min(1)
        context = self.edge_context(self.edge_phi(mean[..., None]))
        # Evaluate row chunks to bound decoder memory on the largest linkers.
        rows = []
        for start in range(0, n, 32):
            end = min(start + 32, n)
            endpoints = (context[:, :, start:end, None] + context[:, :, None, :]) * 0.5
            value = self.edge_self(scores[:, :, start:end, :, None]) + endpoints
            rows.append(self.edge_out(F.gelu(value)).squeeze(-1))
        bonds = torch.cat(rows, dim=2).permute(0, 2, 3, 1)
        # Explicit symmetry holds even under floating point roundoff.
        bonds = (bonds + bonds.transpose(1, 2)) * 0.5
        return {"atom_logits": self.atom_out(latent), "bond_logits": bonds,
                "pair_mask": pair_mask}


class LinkerAutoencoder(nn.Module):
    """The graph VAE, optionally encoding through an externally owned trunk.

    With ``trunk=None`` the autoencoder builds and owns its own sequence stack,
    which is what every standalone codec checkpoint contains. With a trunk
    passed in, the stack is *not* registered as a submodule: the trunk's owner
    registers it once, so the shared parameters appear once in ``state_dict``
    and once in an optimizer's parameter list. Everything the autoencoder still
    owns -- the input embeddings, the edge bias, the posterior and the decoder --
    is adapter-specific and stays here.
    """

    def __init__(self, config, trunk=None):
        super().__init__()
        self.config = config
        self.atom_embedding = nn.Embedding(config.num_atom_classes, config.width)
        self.degree_embedding = nn.Linear(1, config.width)
        self.spectral_phi = nn.Sequential(nn.Linear(2, config.width * 2), nn.ReLU(),
                                           nn.Linear(config.width * 2, config.width), nn.ReLU())
        self.spectral_shift = nn.Parameter(torch.arange(config.frequencies).float() * 1e-12)
        self.edge_bias = nn.Embedding(2, config.heads)
        nn.init.zeros_(self.edge_bias.weight)
        with torch.no_grad():
            self.edge_bias.weight[1].fill_(1.0)
        if trunk is None:
            self.sequence = GraphSequenceStack(config.width, config.layers, config.heads, config.dropout)
        else:
            trunk_width, trunk_heads = trunk_geometry(trunk)
            if (trunk_width, trunk_heads) != (config.width, config.heads):
                raise ValueError(
                    f"Shared trunk is {trunk_width} wide over {trunk_heads} heads, but the "
                    f"autoencoder config asks for {config.width} over {config.heads}"
                )
            # A plain list keeps nn.Module from registering the trunk here.
            self._shared_trunk = [trunk]
        self.posterior = nn.Linear(config.width, 2 * config.latent_dim)
        with torch.no_grad():
            self.posterior.bias[config.latent_dim:].fill_(-6.0)
        self.decoder = LinkerGraphDecoder(config)

    @property
    def shares_trunk(self):
        return getattr(self, "_shared_trunk", None) is not None

    @property
    def trunk(self):
        """The sequence stack this encoder runs, owned here or elsewhere."""
        shared = getattr(self, "_shared_trunk", None)
        return self.sequence if shared is None else shared[0]

    def encode_graph(self, hidden, mask, bias):
        trunk = self.trunk
        if isinstance(trunk, GraphSequenceStack):
            return trunk(hidden, mask, bias)
        return trunk.forward_graph(hidden, mask, bias)

    def node_input(self, batch, spectral_embedding):
        hidden = self.atom_embedding(batch['atoms']) + spectral_embedding
        degree = batch['bonds'].sum(-1, keepdim=True).float()
        return hidden + self.degree_embedding(degree)

    def encode(self, batch, *, augment=None):
        if augment is None:
            augment = self.training
        mask = batch["mask"]
        spectral = batch["spectral"][..., :self.config.frequencies, :].clone()
        frequency_mask = batch["spectral_mask"][..., :self.config.frequencies]
        if spectral.size(-2) != self.config.frequencies:
            raise ValueError("Corpus has too few Laplacian frequencies")
        if augment:
            signs = torch.randint(0, 2, (mask.size(0), 1, self.config.frequencies), device=mask.device) * 2 - 1
            spectral[..., 0] *= signs
            if self.config.spectral_modulation:
                spectral[..., 0] *= 1 + self.config.spectral_modulation * torch.randn_like(spectral[..., :1, 0])
        spectral[..., 1] += self.spectral_shift
        embedding = self.spectral_phi(spectral)
        embedding = (embedding * frequency_mask[:, None, :, None]).sum(-2)
        hidden = self.node_input(batch, embedding)
        bias = self.edge_bias(batch["bonds"].long()).permute(0, 3, 1, 2)
        hidden = self.encode_graph(hidden, mask, bias)
        mean, logvar = self.posterior(hidden).chunk(2, dim=-1)
        mean = mean.masked_fill(~mask[..., None], 0)
        logvar = logvar.clamp(-20, 10).masked_fill(~mask[..., None], 0)
        return mean, logvar

    def decode(self, latent, mask):
        return self.decoder(latent, mask)

    def forward(self, batch, *, sample_posterior=True, augment=None):
        mean, logvar = self.encode(batch, augment=augment)
        latent = mean + (logvar * 0.5).exp() * torch.randn_like(mean) if sample_posterior else mean
        latent = latent.masked_fill(~batch["mask"][..., None], 0)
        return {**self.decode(latent, batch["mask"]), "mean": mean, "logvar": logvar, "latent": latent}


def reconstruction_loss(output, batch, kl_weight):
    atom_loss = F.cross_entropy(output["atom_logits"][batch["mask"]].float(), batch["atoms"][batch["mask"]])
    mask = torch.triu(output["pair_mask"], diagonal=1)
    if mask.any():
        bond_loss = F.cross_entropy(output["bond_logits"][mask].float(), batch["bonds"][mask])
    else:
        bond_loss = output["bond_logits"].sum() * 0
    mean, logvar = output["mean"].float(), output["logvar"].float()
    kl = (0.5 * (mean.square() + logvar.exp() - 1 - logvar).sum(-1))[batch["mask"]].mean()
    return atom_loss + bond_loss + kl_weight * kl, {"atom_ce": atom_loss.detach(), "bond_ce": bond_loss.detach(), "kl": kl.detach()}


def reconstruction_counts(output, batch):
    """Raw numerators/denominators for exact, count-weighted evaluation."""
    atoms = output["atom_logits"].argmax(-1)
    bonds = output["bond_logits"].argmax(-1)
    pairs = torch.triu(output["pair_mask"], diagonal=1)
    atom_ok = atoms.eq(batch["atoms"]) | ~batch["mask"]
    bond_ok = bonds.eq(batch["bonds"]) | ~pairs
    exact = atom_ok.all(-1) & bond_ok.flatten(1).all(-1)
    tp = (bonds.eq(1) & batch["bonds"].eq(1) & pairs).sum()
    return {"graphs": batch["mask"].size(0), "exact_graphs": exact.sum(),
            "atoms": batch["mask"].sum(), "correct_atoms": (atom_ok & batch["mask"]).sum(),
            "pairs": pairs.sum(), "correct_pairs": (bond_ok & pairs).sum(),
            "true_positive": tp, "predicted_positive": (bonds.eq(1) & pairs).sum(),
            "target_positive": (batch["bonds"].eq(1) & pairs).sum()}
