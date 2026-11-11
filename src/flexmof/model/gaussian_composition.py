"""Gaussian DDPM on compact integer atom vectors and binary bond matrices."""

import torch
from torch import nn
from torch.nn import functional as F

from .noise import SinusoidalPositionEmbeddings


class AtomVocabulary:
    def __init__(self, atomic_numbers: str):
        numbers = [int(z) for z in atomic_numbers.split(",") if z.strip()]
        if not numbers or len(set(numbers)) != len(numbers):
            raise ValueError("Gaussian composition requires a unique atom vocabulary")
        self.numbers = torch.tensor([0, *numbers], dtype=torch.long)
        self.lookup = torch.zeros(119, dtype=torch.long)
        self.lookup[self.numbers[1:]] = torch.arange(1, len(numbers) + 1)

    def encode(self, atomic_numbers):
        ids = self.lookup.to(atomic_numbers.device)[atomic_numbers.long()]
        if ((atomic_numbers > 0) & (ids == 0)).any():
            raise ValueError("Atom is missing from mof_composition_atom_vocab")
        return ids

    def decode(self, ids):
        return self.numbers.to(ids.device)[ids.long()]


class GaussianGraphDDPM:
    """Standard variance-preserving DDPM, parameterized by predicted clean x0.

    Index 0 is clean; indices 1..T use a cosine beta schedule. Each independent
    upper-triangular bond has unit Gaussian noise, copied to the lower triangle.
    The posterior also supports skipped timesteps, with zero variance at s=0.
    """

    def __init__(self, config):
        self.num_timesteps = config.mof_composition_num_timesteps
        self.vocab = AtomVocabulary(config.mof_composition_atom_vocab)
        self.atom_mean = config.mof_composition_atom_mean
        self.atom_std = config.mof_composition_atom_std
        self.bond_mean = config.mof_composition_bond_mean
        self.bond_std = config.mof_composition_bond_std
        if self.atom_std <= 0 or self.bond_std <= 0:
            raise ValueError("Composition standard deviations must be positive")
        time = torch.linspace(0, 1, self.num_timesteps + 1, dtype=torch.float64)
        cumulative = torch.cos((time + 0.008) / 1.008 * torch.pi / 2).square()
        beta = (1 - cumulative[1:] / cumulative[:-1]).clamp(max=0.999)
        self.alpha_bar = torch.cat([torch.ones(1), (1 - beta).cumprod(0)]).float()

    def alpha(self, time, like):
        value = self.alpha_bar.to(like.device)[time]
        return value.reshape(-1, *([1] * (like.ndim - 1)))

    def normalize_atoms(self, ids):
        return (ids.float() - self.atom_mean) / self.atom_std

    def normalize_bonds(self, bonds):
        return (bonds.float() - self.bond_mean) / self.bond_std

    def atom_ids(self, state):
        return (
            (state * self.atom_std + self.atom_mean)
            .round()
            .long()
            .clamp(1, len(self.vocab.numbers) - 1)
        )

    def bond_labels(self, state):
        return (state * self.bond_std + self.bond_mean >= 0.5).long()

    @staticmethod
    def symmetric_noise(like):
        upper = torch.triu(torch.randn_like(like), diagonal=1)
        return upper + upper.transpose(-1, -2)

    def q_sample(self, clean, time, mask, *, bonds=False):
        noise = self.symmetric_noise(clean) if bonds else torch.randn_like(clean)
        alpha = self.alpha(time, clean)
        return (alpha.sqrt() * clean + (1 - alpha).sqrt() * noise).masked_fill(~mask, 0)

    def sample_prior(self, atom_mask, bond_mask):
        atoms = torch.randn(atom_mask.shape, device=atom_mask.device)
        bonds = self.symmetric_noise(
            torch.empty(bond_mask.shape, device=bond_mask.device)
        )
        return atoms.masked_fill(~atom_mask, 0), bonds.masked_fill(~bond_mask, 0)

    def posterior(self, current, predicted_clean, time, previous):
        alpha_t = self.alpha(time, current)
        alpha_s = self.alpha(previous, current)
        ratio = alpha_t / alpha_s
        denominator = (1 - alpha_t).clamp_min(1e-12)
        mean = (
            alpha_s.sqrt() * (1 - ratio) / denominator * predicted_clean
            + ratio.sqrt() * (1 - alpha_s) / denominator * current
        )
        variance = (1 - alpha_s) * (1 - ratio) / denominator
        return mean, variance.clamp_min(0)

    def p_sample(self, current, predicted_clean, time, previous, mask, *, bonds=False):
        low, high = (
            ((0 - self.bond_mean) / self.bond_std, (1 - self.bond_mean) / self.bond_std)
            if bonds
            else (
                (1 - self.atom_mean) / self.atom_std,
                (len(self.vocab.numbers) - 1 - self.atom_mean) / self.atom_std,
            )
        )
        mean, variance = self.posterior(
            current, predicted_clean.clamp(low, high), time, previous
        )
        noise = self.symmetric_noise(current) if bonds else torch.randn_like(current)
        return (mean + variance.sqrt() * noise).masked_fill(~mask, 0)


def pool_occurrences(context, batch):
    """Average each local atom's features over repeated copies of its block."""
    indices = batch["mof_bb_occ_atom_index"]
    valid = (indices >= 0) & batch["mof_bb_occ_mask"][..., None]
    valid &= batch["mof_bb_atom_mask"][:, :, None]
    batch_index = torch.arange(context.size(0), device=context.device)[
        :, None, None, None
    ]
    gathered = context[batch_index, indices.clamp_min(0)]
    return (gathered * valid[..., None]).sum(2) / valid.sum(2).clamp_min(1)[..., None]


def pool_pair_distances(batch):
    """Pose-invariant mean distances; atom/block reordering permutes the result.

    Use only current noisy coordinates. Linker atoms use their continuous image
    representation, so no reference cell or clean geometry enters these features.
    Reduce repetitions separately to avoid a B*U*R*A*A*3 temporary.
    """
    positions = batch["pos"].float()
    indices = batch["mof_bb_occ_atom_index"]
    valid = indices.ge(0) & indices.lt(positions.size(1))
    valid &= batch["mof_bb_occ_mask"][..., None]
    valid &= batch["mof_bb_atom_mask"][:, :, None]
    b = torch.arange(positions.size(0), device=positions.device)[:, None, None, None]
    gathered = positions[b, indices.clamp(0, positions.size(1) - 1)]
    shape = (*indices.shape[:2], indices.size(-1), indices.size(-1))
    total = positions.new_zeros(shape)
    count = positions.new_zeros(shape)
    for repetition in range(indices.size(2)):
        coordinates = gathered[:, :, repetition]
        pair_valid = valid[:, :, repetition, :, None] & valid[:, :, repetition, None, :]
        distance = (coordinates[..., :, None, :] - coordinates[..., None, :, :]).norm(
            dim=-1
        )
        total = total + distance * pair_valid
        count = count + pair_valid
    return total / count.clamp_min(1)


class CompositionGraphLayer(nn.Module):
    """Shared node attention with continuous noisy bonds as attention bias."""

    def __init__(self, width):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.edge_bias = nn.Linear(1, 4)
        self.out = nn.Linear(width, width)
        self.ffn = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, 2 * width),
            nn.SiLU(),
            nn.Linear(2 * width, width),
        )
        # Initially preserve the previously trained, order-free node features.
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

    def forward(self, hidden, bonds, mask, geometry_bias=None):
        length, width = hidden.shape[-2:]
        q, k, v = (
            self.qkv(self.norm(hidden))
            .reshape(-1, length, 3, 4, width // 4)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        bias = (
            self.edge_bias(bonds[..., None].to(hidden.dtype))
            .reshape(-1, length, length, 4)
            .permute(0, 3, 1, 2)
        )
        if geometry_bias is not None:
            bias = bias + geometry_bias.reshape(-1, length, length, 4).permute(
                0, 3, 1, 2
            )
        bias = bias.masked_fill(~mask.reshape(-1, 1, 1, length), -1e4)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        attended = attended.transpose(1, 2).reshape_as(hidden)
        hidden = hidden + self.out(attended)
        return (hidden + self.ffn(hidden)).masked_fill(~mask[..., None], 0)


class GaussianCompositionHead(nn.Module):
    """Continuous x0 regression, conditioned on noisy states and backbone features.

    All atom/block operations share weights. Occurrence indices only gather the
    corresponding features; neither atom order nor block order is an input.
    """

    def __init__(self, config):
        super().__init__()
        width = min(256, config.embedding_dim)
        self.timesteps = config.mof_composition_num_timesteps
        self.context_proj = nn.Sequential(
            nn.Linear(config.embedding_dim, width), nn.LayerNorm(width)
        )
        self.scaffold_proj = nn.Linear(2, width)
        nn.init.zeros_(self.scaffold_proj.weight)
        nn.init.zeros_(self.scaffold_proj.bias)
        self.graph_layers = nn.ModuleList(
            CompositionGraphLayer(width) for _ in range(2)
        )
        self.use_pair_geometry = config.mof_composition_pair_geometry
        if self.use_pair_geometry:
            self.register_buffer("geometry_centers", torch.arange(16).float())
            self.geometry_attention = nn.Linear(16, 4, bias=False)
            self.geometry_pair_proj = nn.Linear(16, width, bias=False)
            # Preserve transferred head behavior until the new features learn.
            nn.init.zeros_(self.geometry_attention.weight)
            nn.init.zeros_(self.geometry_pair_proj.weight)
        self.atom_state_proj = nn.Linear(1, width)
        self.time_proj = nn.Sequential(
            SinusoidalPositionEmbeddings(32), nn.Linear(32, width), nn.SiLU()
        )
        self.atom_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )
        self.pair_proj = nn.Linear(width, width, bias=False)
        self.bond_state_proj = nn.Linear(1, width)
        self.bond_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1)
        )

    def forward(self, context, batch):
        atoms = batch["mof_comp_atom_state"]
        bonds = batch["mof_comp_bond_state"]
        hidden = pool_occurrences(self.context_proj(context), batch)
        hidden = hidden + self.atom_state_proj(atoms[..., None].to(hidden.dtype))
        hidden = (
            hidden
            + self.time_proj(batch["mof_comp_time"].float() / self.timesteps)[
                :, None, None
            ]
        )
        mask = batch["mof_bb_atom_mask"].bool()
        counts = (
            torch.stack([mask.sum(-1), batch["mof_bb_occ_mask"].sum(-1)], dim=-1)
            .to(hidden.dtype)
            .log1p()
        )
        hidden = hidden + self.scaffold_proj(counts)[:, :, None]
        geometry = geometry_bias = None
        if self.use_pair_geometry:
            distances = pool_pair_distances(batch)
            geometry = torch.exp(
                -0.5 * (distances[..., None] - self.geometry_centers).square()
            ).to(hidden.dtype)
            geometry_bias = self.geometry_attention(geometry)
        for layer in self.graph_layers:
            hidden = layer(hidden, bonds, mask, geometry_bias)
        atom_x0 = self.atom_mlp(hidden).squeeze(-1)
        pair_index = torch.triu(batch["mof_comp_bond_loss_mask"], diagonal=1).nonzero(
            as_tuple=True
        )
        b, u, i, j = pair_index
        pair = self.pair_proj(hidden)
        source, target = pair[b, u, i], pair[b, u, j]
        pair = (
            source
            + target
            + source * target
            + self.bond_state_proj(bonds[pair_index][..., None].to(hidden.dtype))
        )
        if geometry is not None:
            pair = pair + self.geometry_pair_proj(geometry[pair_index])
        values = self.bond_mlp(pair).squeeze(-1)
        upper = bonds.new_zeros(bonds.shape).index_put(
            pair_index, values.to(bonds.dtype)
        )
        return {"atom_x0": atom_x0, "bond_x0": upper + upper.transpose(-1, -2)}
