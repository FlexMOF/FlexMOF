"""Diffusion time embeddings and periodic coordinate noise."""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dimension, max_period=10_000):
        super().__init__()
        if dimension % 2:
            raise ValueError("Sinusoidal embedding dimension must be even")
        self.dimension = dimension
        self.max_period = max_period

    def forward(self, time):
        time = time * self.max_period
        half_dimension = self.dimension // 2
        spacing = math.log(self.max_period) / (half_dimension - 1)
        frequencies = torch.exp(
            torch.arange(half_dimension, device=time.device, dtype=time.dtype)
            * -spacing
        )
        angles = time[:, None] * frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class TimeStepEncoder(nn.Module):
    def __init__(self, num_timesteps, embedding_dim):
        super().__init__()
        self.time_proj = SinusoidalPositionEmbeddings(embedding_dim)
        self.time_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.num_timesteps = num_timesteps

    def forward(self, timesteps, clean_mask):
        graph_count, node_count = timesteps.shape[:2]
        timesteps = timesteps.masked_fill(clean_mask, 0.0)
        embedding = self.time_proj(timesteps.unsqueeze(-1)).view(
            graph_count, node_count, -1
        )
        embedding = embedding.to(self.time_embedding[0].weight.dtype)
        return self.time_embedding(embedding)


class TimeStepSampler:
    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps

    def get_continuous_time_step(self, step, graph_count, device, dtype):
        """Invert training's floor(time * (T - 1)) without rounding down a bin."""
        last = max(1, self.num_timesteps - 1)
        time = torch.full(
            [graph_count],
            step / last,
            device=device,
            dtype=dtype,
        )
        # k / (T - 1) can round just below its bin boundary. Move only those
        # values one representable step up; coefficients use integer k directly.
        return torch.where(
            (time * last).long() < step,
            torch.nextafter(time, torch.ones_like(time)),
            time,
        )

    @staticmethod
    def sample_training(graph_count, device, dtype):
        """Antithetic continuous times used by the original MOF objective."""
        first = torch.rand(graph_count // 2 + 1, device=device)
        return torch.cat([first, 1.0 - first])[:graph_count].to(dtype=dtype)


class DiffNoise(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        (
            self.sqrt_alphas_cumprod,
            self.sqrt_one_minus_alphas_cumprod,
            self.alphas_cumprod,
            self.beta_list,
        ) = self._beta_schedule(
            config.num_timesteps, config.ddpm_beta_start, config.ddpm_beta_end
        )
        self.unit_noise_scale = config.diffusion_noise_std
        self.torch_generator = None

    @staticmethod
    def _beta_schedule(num_timesteps, beta_start, beta_end):
        betas = torch.linspace(-6, 6, num_timesteps)
        beta_list = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        alphas_cumprod = torch.cumprod(1 - beta_list, dim=0)
        return (
            torch.sqrt(alphas_cumprod),
            torch.sqrt(1.0 - alphas_cumprod),
            alphas_cumprod,
            beta_list,
        )

    @staticmethod
    def _extract(values, timesteps, output_shape):
        if timesteps.dim() == 1:
            batch_size = timesteps.shape[0]
            result = values.gather(-1, timesteps.cpu().long())
            shape = (batch_size, *(1,) * (len(output_shape) - 1))
        elif timesteps.dim() == 2:
            batch_size, length = timesteps.shape
            result = torch.gather(
                values.unsqueeze(0).expand(batch_size, -1),
                1,
                timesteps.cpu().long(),
            )
            shape = (batch_size, length, *(1,) * (len(output_shape) - 2))
        else:
            raise ValueError(f"Unsupported timestep shape: {timesteps.shape}")
        return result.reshape(shape).to(timesteps.device)

    def _noise_lattice_vectors(self, positions, non_atom_mask, noise, stable):
        graph_count = positions[stable].size(0)
        lattice_noise = (
            torch.randn(
                graph_count,
                8,
                3,
                device=positions.device,
                dtype=positions.dtype,
                generator=self.torch_generator,
            )
            * self.config.periodic_lattice_diffusion_noise_std
        )
        origin_noise = lattice_noise[:, 0]
        gather_index = positions.new_tensor([0, 4, 2, 1], dtype=torch.long)
        gather_index = gather_index.view(1, 4, 1).repeat(graph_count, 1, 3)
        lattice_vectors = torch.gather(lattice_noise, 1, gather_index)[:, 1:]
        lattice_vectors -= origin_noise[:, None]
        corners = positions.new_tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        corner_noise = torch.matmul(corners, lattice_vectors) + origin_noise[:, None]
        corner_noise -= ((corner_noise[:, 0] + corner_noise[:, -1]) / 2.0)[:, None]
        atom_count = (~non_atom_mask[stable]).sum(dim=-1)
        scatter_index = torch.arange(8, device=positions.device).view(1, 8, 1)
        scatter_index = scatter_index.repeat(graph_count, 1, 3)
        scatter_index += atom_count[:, None, None]
        noise[stable] = noise[stable].scatter(1, scatter_index, corner_noise)
        return noise

    def get_noise(self, positions, non_atom_mask, stable):
        if self.torch_generator is None:
            self.torch_generator = torch.Generator(device=positions.device)
            self.torch_generator.manual_seed(dist.get_rank())
        noise = (
            torch.randn(
                positions.shape,
                device=positions.device,
                dtype=positions.dtype,
                generator=self.torch_generator,
            )
            * self.unit_noise_scale
        )
        noise = noise.masked_fill(non_atom_mask[..., None], 0.0)
        atom_count = (~non_atom_mask).sum(dim=-1)
        center = noise.sum(dim=1, keepdim=True) / atom_count[:, None, None]
        noise[~stable] -= center[~stable]
        noise = noise.masked_fill(non_atom_mask[..., None], 0.0)
        noise = self._noise_lattice_vectors(positions, non_atom_mask, noise, stable)
        if stable.any():
            periodic_noise = torch.randn_like(positions)
            periodic_noise *= self.config.periodic_diffusion_noise_std_factor
            periodic_noise *= atom_count[:, None, None] ** (1 / 3.0)
            mask = stable[:, None] & ~non_atom_mask
            noise = torch.where(mask[..., None], periodic_noise, noise)
        return noise

    def get_sampling_start(self, initial_positions, non_atom_mask, stable):
        return initial_positions + self.get_noise(
            initial_positions, non_atom_mask, stable
        )

    def noise_training_positions(
        self,
        original,
        initial,
        time,
        non_atom_mask,
        stable,
        clean_mask,
    ):
        """Apply the checkpoint's shifted DDPM forward process."""
        indices = (time * (self.config.num_timesteps - 1)).long()
        noise = self.get_noise(original, non_atom_mask, stable)
        alpha = self._extract(self.sqrt_alphas_cumprod, indices, original.shape)
        sigma = self._extract(
            self.sqrt_one_minus_alphas_cumprod, indices, original.shape
        )
        noised = alpha * (original - initial) + sigma * noise + initial
        noised = torch.where(clean_mask[..., None], original, noised)
        return noised, noise, sigma, alpha
