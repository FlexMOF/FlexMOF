"""EMA of trainable weights, with exact restoration and resumable state."""

from contextlib import contextmanager

import torch


class ModelEMA:
    def __init__(self, model, decay):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be between zero and one")
        self.decay = decay
        self.updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters() if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model):
        parameters = dict(model.named_parameters())
        for name, value in self.shadow.items():
            value.lerp_(parameters[name].detach(), 1 - self.decay)
        self.updates += 1

    def state_dict(self):
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, state):
        if state["decay"] != self.decay or set(state["shadow"]) != set(self.shadow):
            raise ValueError("EMA configuration/parameter set differs from checkpoint")
        for name, value in self.shadow.items():
            value.copy_(state["shadow"][name])
        self.updates = int(state["updates"])

    @contextmanager
    def average_parameters(self, model):
        parameters = dict(model.named_parameters())
        original = {name: parameters[name].detach().clone() for name in self.shadow}
        try:
            with torch.no_grad():
                for name, value in self.shadow.items():
                    parameters[name].copy_(value)
            yield
        finally:
            with torch.no_grad():
                for name, value in original.items():
                    parameters[name].copy_(value)

    def model_state_dict(self, model):
        # Frozen parameters and buffers stay exactly at their loaded values.
        return {name: self.shadow.get(name, value) for name, value in model.state_dict().items()}
