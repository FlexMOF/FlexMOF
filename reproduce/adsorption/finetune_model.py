"""Coordinate fine-tuning with framework, adsorbate and cell loss reporting."""

from __future__ import annotations

import numpy as np
import torch

from flexmof.model.lattice import (  # noqa: E402
    center_periodic,
    complete_lattice,
    initialize_lattice,
)
from flexmof.training import MOFTrainingModel  # noqa: E402

ADSORBATE_CHAIN_ID = 3


def role_masks(batch, loss_mask: torch.Tensor):
    """Split the loss mask into framework atoms, adsorbate atoms and corners."""
    node_count = loss_mask.size(1)
    node_index = torch.arange(node_count, device=loss_mask.device)[None]
    is_atom = node_index < batch["num_atoms"][:, None]
    is_adsorbate = batch["chain_ids"].eq(ADSORBATE_CHAIN_ID)
    return (
        loss_mask & is_atom & ~is_adsorbate,
        loss_mask & is_atom & is_adsorbate,
        loss_mask & ~is_atom,
    )


def _masked_square_sum(prediction, target, mask):
    squared = (prediction.to(target.dtype) - target).square()
    squared = squared.masked_fill(~mask[..., None], 0.0)
    return squared.sum(dim=(1, 2)), 3.0 * mask.sum(dim=-1)


def component_rms(prediction, target, mask):
    """Per-structure RMS, returning NaN where a component has no atoms."""
    total, denominator = _masked_square_sum(prediction, target, mask)
    value = torch.sqrt(total / denominator.clamp_min(1) + 1e-5)
    return value.masked_fill(denominator.eq(0), float("nan"))


class AdsorptionTrainingModel(MOFTrainingModel):
    def __init__(self, config, loss_mode: str = "e1"):
        super().__init__(config)
        if loss_mode != "e1":
            raise ValueError("The adsorption experiment uses the coordinate RMS objective")

    @torch.no_grad()
    def evaluate_fixed(self, batch, time_value: float, seed: int = 0):
        """Loss components at one fixed diffusion time with reproducible noise."""
        return self.fixed_time_losses(batch, time_value, seed)

    def fixed_time_losses(self, batch, time_value: float, seed: int = 0):
        """Loss components at one fixed diffusion time with reproducible noise.

        Training samples a new t every step, so its loss curve is dominated by
        which t came up rather than by learning. Pinning t and the noise makes
        successive evaluations comparable.

        This keeps gradients, so the pinned target can also be optimised
        directly -- the strictest check that the model is able to fit anything
        at all.
        """
        was_training = self.training
        self.eval()
        generator = self.diffusion_noise.torch_generator
        saved = generator.get_state() if generator is not None else None
        if generator is not None:
            generator.manual_seed(seed)
        # For periodic structures DiffusionNoise.get_noise discards the
        # generator-backed draw and overwrites it with a plain randn_like, so
        # seeding the generator alone leaves the target moving by ~20% between
        # identical calls. Pin the global RNG too, and restore it so callers
        # that interleave this with training keep their own stream.
        device = batch["pos"].device
        cpu_state = torch.get_rng_state()
        cuda_state = (torch.cuda.get_rng_state(device)
                      if device.type == "cuda" else None)
        torch.manual_seed(seed)
        if cuda_state is not None:
            torch.cuda.manual_seed(seed)

        try:
            padding_mask, non_atom_mask, loss_mask = self._masks(batch)
            original = center_periodic(batch, padding_mask).float().clone()
            initial = initialize_lattice(
                batch, self.config.diff_init_lattice_size_factor
            )
            graph_time = torch.full(
                (batch["pos"].size(0),), time_value,
                device=batch["pos"].device, dtype=batch["pos"].dtype,
            )
            time = graph_time[:, None].repeat(1, batch["pos"].size(1))
            time = time.masked_fill(padding_mask, 0.0)
            noised, noise, sigma, _ = self.diffusion_noise.noise_training_positions(
                original, initial, time, non_atom_mask,
                batch["is_stable_periodic"], padding_mask,
            )
            batch["pos"] = complete_lattice(noised, batch)
            batch["sqrt_one_minus_alphas_cumprod_t"] = sigma
            prediction = self.net(batch, time, padding_mask, compute_joint_objectives=False)["noise_pred"]

            overall, _ = self.coordinate_loss(prediction, noise, loss_mask)
            framework_mask, adsorbate_mask, cell_mask = role_masks(batch, loss_mask)
            return {
                "coord": overall,
                "framework": component_rms(prediction, noise, framework_mask),
                "adsorbate": component_rms(prediction, noise, adsorbate_mask),
                "cell": component_rms(prediction, noise, cell_mask),
            }
        finally:
            if generator is not None and saved is not None:
                generator.set_state(saved)
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, device)
            self.train(was_training)

    def forward(self, batch):
        padding_mask, non_atom_mask, loss_mask = self._masks(batch)

        # Same draw order as MOFTrainingModel.forward so a finetune resumes the
        # pretrained stochastic stream rather than a shifted one.
        torch.rand_like(batch["token_id"][..., None], dtype=torch.float)
        torch.rand_like(batch["token_id"], dtype=torch.float)
        original = center_periodic(batch, padding_mask).float().clone()
        initial = initialize_lattice(batch, self.config.diff_init_lattice_size_factor)
        graph_time = self.time_sampler.sample_training(
            batch["pos"].size(0), batch["pos"].device, batch["pos"].dtype
        )
        np.random.rand(batch["pos"].size(0))
        np.random.choice(3, batch["pos"].size(0), p=(0.0, 1.0, 0.0))
        torch.rand(batch["pos"].size(0), device=batch["pos"].device)
        time = graph_time[:, None].repeat(1, batch["pos"].size(1))
        time = time.masked_fill(padding_mask, 0.0)
        noised, noise, sigma, _ = self.diffusion_noise.noise_training_positions(
            original,
            initial,
            time,
            non_atom_mask,
            batch["is_stable_periodic"],
            padding_mask,
        )
        batch["pos"] = complete_lattice(noised, batch)
        batch["sqrt_one_minus_alphas_cumprod_t"] = sigma

        network_output = self.net(batch, time, padding_mask, compute_joint_objectives=False)
        prediction = network_output["noise_pred"]

        overall, per_structure = self.coordinate_loss(prediction, noise, loss_mask)
        framework_mask, adsorbate_mask, cell_mask = role_masks(batch, loss_mask)

        with torch.no_grad():
            components = {
                "rms_framework": component_rms(prediction, noise, framework_mask),
                "rms_adsorbate": component_rms(prediction, noise, adsorbate_mask),
                "rms_cell": component_rms(prediction, noise, cell_mask),
            }

        return {
            "loss": overall,
            "coordinate_loss": overall.detach(),
            "composition_loss": torch.zeros((), device=overall.device),
            "per_structure_loss": per_structure.detach(),
            "time": graph_time.detach(),
            **{name: value for name, value in components.items()},
        }
