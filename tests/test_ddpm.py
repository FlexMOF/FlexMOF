"""Numerical invariants for the released coordinate sampler."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

source = Path(__file__).resolve().parents[1] / 'src/flexmof/model/diffusion.py'
spec = importlib.util.spec_from_file_location('flexmof_diffusion_test', source)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DDPMTest(unittest.TestCase):
    def setUp(self):
        self.schedule = torch.cumprod(1 - torch.linspace(1e-4, 0.002, 5000, dtype=torch.float64), 0)
        self.sampler = module.DDPM(self.schedule, SimpleNamespace(num_timesteps=5000, num_timesteps_stepsize=-1))

    def test_full_adjacent_schedule(self):
        self.assertEqual(self.sampler.timesteps.numel(), 5000)
        self.assertEqual(self.sampler.timesteps[0].item(), 4999)
        self.assertEqual(self.sampler.timesteps[-1].item(), 0)
        self.assertTrue(torch.all(self.sampler.timesteps.diff() == -1))
        self.assertTrue(torch.isfinite(self.sampler.posterior_std).all())

    def test_clean_endpoint_and_shift(self):
        clean = torch.tensor([0.4, -0.1], dtype=torch.float64)
        eps = torch.tensor([0.2, 0.7], dtype=torch.float64)
        shift = torch.tensor([4.0, -3.0], dtype=torch.float64)
        a = self.schedule[0]
        current = shift + a.sqrt() * clean + (1-a).sqrt() * eps
        result = self.sampler.sample_step(current, shift, eps, torch.ones_like(eps), 0)
        torch.testing.assert_close(result, shift+clean)

    def test_intermediate_posterior_mean(self):
        t = 1000
        clean = torch.tensor([0.4, -0.1], dtype=torch.float64)
        eps = torch.tensor([0.2, 0.7], dtype=torch.float64)
        shift = torch.tensor([4.0, -3.0], dtype=torch.float64)
        a_t, a_prev = self.schedule[t], self.schedule[t-1]
        x_t = a_t.sqrt() * clean + (1-a_t).sqrt() * eps
        beta = 1 - a_t/a_prev
        expected = (a_prev.sqrt()*beta*clean + (1-beta).sqrt()*(1-a_prev)*x_t) / (1-a_t)
        result = self.sampler.sample_step(shift+x_t, shift, eps, torch.zeros_like(eps), t)
        torch.testing.assert_close(result, shift+expected)


if __name__ == '__main__':
    unittest.main()
