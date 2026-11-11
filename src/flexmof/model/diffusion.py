"""Shifted coordinate DDPM sampling."""

from __future__ import annotations

import torch


def _sampling_timesteps(alphas_cumprod, config):
    if config.num_timesteps < 1 or config.num_timesteps_stepsize >= 0:
        raise ValueError("Sampling requires positive num_timesteps and a negative stride")
    if alphas_cumprod.ndim != 1 or alphas_cumprod.numel() != config.num_timesteps:
        raise ValueError("The noise schedule must contain num_timesteps coefficients")
    return torch.arange(
        config.num_timesteps - 1, -1, config.num_timesteps_stepsize,
        device=alphas_cumprod.device,
    )


class DDPM:
    """Ancestral DDPM with fixed posterior variance and epsilon prediction.

    A stride of -1 gives the ordinary adjacent-step reverse chain. Larger
    strides use q(x_s | x_t, x_0) for the selected pair of noise levels. The
    supplied epsilon must come from DiffNoise so that its atom scales and
    correlated lattice corners match the forward process. Index -1 denotes
    the clean endpoint, after the final (still noisy) schedule entry at 0.
    """

    def __init__(self, alphas_cumprod: torch.Tensor, config):
        self.config = config
        self.timesteps = _sampling_timesteps(alphas_cumprod, config)
        # Compute small posterior variances without float32 cancellation.
        alpha_t = alphas_cumprod[self.timesteps].double()
        alpha_s = torch.cat((alpha_t[1:], alpha_t.new_ones(1)))
        relative_alpha = alpha_t / alpha_s
        self.alpha = alpha_t.sqrt()
        self.sigma = (1 - alpha_t).sqrt()
        self.clean_coefficient = alpha_s.sqrt() * (1 - relative_alpha) / (1 - alpha_t)
        self.current_coefficient = relative_alpha.sqrt() * (1 - alpha_s) / (1 - alpha_t)
        self.posterior_std = (
            (1 - alpha_s) * (1 - relative_alpha) / (1 - alpha_t)
        ).clamp_min(0).sqrt()

    def reset(self):
        """DDPM has no multistep history to reset."""

    def sample_step(self, current, initial, model_output, epsilon, timestep):
        index = (self.config.num_timesteps - 1 - int(timestep)) // (
            -self.config.num_timesteps_stepsize
        )
        shifted = current - initial
        clean = (
            shifted - self.sigma[index].to(current) * model_output
        ) / self.alpha[index].to(current)
        if index == len(self.timesteps) - 1:
            return initial + clean
        return (
            initial
            + self.clean_coefficient[index].to(current) * clean
            + self.current_coefficient[index].to(current) * shifted
            + self.posterior_std[index].to(current) * epsilon
        )
