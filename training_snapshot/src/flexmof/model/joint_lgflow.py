"""LG-Flow on a whole MOF's set of unique building-block graph latents.

Local DiT weights initialize from the trained linker prior. Periodic exchanges
between block summaries make the reverse process joint. Neither atom IDs nor
block indices are embedded; indexing tensors only pack independent MOF sets.
"""

from dataclasses import dataclass
import math

import torch
from torch import nn

from .lgflow_composition import LatentDiTBlock, LatentDiTHead
from flexmof.amp import is_autocast_enabled
from flexmof.random_streams import auxiliary_noise_seed


@dataclass
class JointLGFlowConfig:
    latent_dim: int = 32
    context_dim: int = 768
    width: int = 768
    layers: int = 12
    exchange_every: int = 3
    time_logit_std: float = 1.0
    time_logit_mean: float = 0.0
    atom_exchange: bool = False
    carbon_loss_weight: float = 0.0
    carbon_loss_max_time: float = 0.6
    carbon_loss_straight_through: bool = False
    carbon_loss_temperature: float = 1.0
    atom_noise_transport: bool = False
    mof_noise_transport_group_size: int = 0

    def __post_init__(self):
        if self.width % 8 or min(self.width, self.layers, self.exchange_every) < 1 or self.time_logit_std <= 0 or not math.isfinite(self.time_logit_mean):
            raise ValueError('Positive model dimensions and a width divisible by eight are required')
        if not math.isfinite(self.carbon_loss_weight) or self.carbon_loss_weight < 0 or not 0 < self.carbon_loss_max_time <= 1:
            raise ValueError('Invalid carbon supervision weight or time cutoff')
        if not math.isfinite(self.carbon_loss_temperature) or self.carbon_loss_temperature <= 0:
            raise ValueError('Carbon gradient temperature must be positive and finite')
        if self.carbon_loss_temperature != 1 and not self.carbon_loss_straight_through:
            raise ValueError('Temperature smoothing requires hard-forward carbon supervision')
        if self.mof_noise_transport_group_size < 0 or self.mof_noise_transport_group_size == 1:
            raise ValueError('MOF transport group size must be zero or at least two')
        if self.mof_noise_transport_group_size and not self.atom_noise_transport:
            raise ValueError('MOF transport requires atom transport too')


class JointLGFlow(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.local = LatentDiTHead(config.latent_dim, config.context_dim, config.width, config.layers)
        # This model exchanges block summaries instead of receiving a coordinate backbone.
        self.local.context_proj.requires_grad_(False)
        self.exchange_layers = [i for i in range(config.layers) if (i+1) % config.exchange_every == 0]
        if not self.exchange_layers:
            self.exchange_layers = [config.layers-1]
        self.exchanges = nn.ModuleList([LatentDiTBlock(config.width) for _ in self.exchange_layers])
        self.exchange_out = nn.ModuleList([nn.Linear(config.width, config.width, bias=False) for _ in self.exchange_layers])
        self.global_counts = nn.Sequential(nn.Linear(3, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.role_embedding = nn.Embedding(2, config.width)
        nn.init.zeros_(self.role_embedding.weight)
        for output in self.exchange_out:
            nn.init.zeros_(output.weight)

        if config.atom_exchange:
            self.atom_exchanges = nn.ModuleList([LatentDiTBlock(config.width) for _ in self.exchange_layers])
            self.atom_exchange_out = nn.ModuleList([nn.Linear(config.width, config.width, bias=False) for _ in self.exchange_layers])
            for output in self.atom_exchange_out:
                nn.init.zeros_(output.weight)

    def initialize_local(self, state):
        self.local.load_state_dict(state, strict=True)

    def predict_clean(self, state, batch, time, self_condition=None):
        mask = batch['mask']
        owner, slot = batch['owner'], batch['slot']
        if self_condition is None:
            self_condition = torch.zeros_like(state)
        hidden = self.local.state_proj(torch.cat([state, self_condition], -1))
        frequency = torch.exp(-torch.arange(128, device=time.device, dtype=torch.float32) * math.log(10000) / 128)
        angle = time.float()[:, None] * frequency
        time_embedding = self.local.time_proj(torch.cat([angle.cos(), angle.sin()], -1))
        counts = torch.stack([mask.sum(-1).float(), batch['repetitions'].float()], -1).log1p()
        condition = time_embedding[owner] + self.local.count_proj(counts)
        condition = condition + self.role_embedding(batch['is_linker'].long())
        global_condition = time_embedding + self.global_counts(batch['mof_counts'].float().log1p())
        exchange_index = 0
        for index, layer in enumerate(self.local.blocks):
            hidden = layer(hidden, condition, mask)
            if index in self.exchange_layers:
                pooled = hidden.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(-1).clamp_min(1)[:, None]
                packed = pooled.new_zeros(*batch['block_mask'].shape, pooled.size(-1))
                packed[owner, slot] = pooled
                messages = self.exchanges[exchange_index](packed, global_condition, batch['block_mask'])
                update = self.exchange_out[exchange_index](messages[owner, slot])
                hidden = (hidden + update[:, None]).masked_fill(~mask[..., None], 0)
                if self.config.atom_exchange:
                    # Direct atom messages supplement the existing block summaries.
                    atom_owner, atom_slot = batch['atom_owner'], batch['atom_slot']
                    packed_atoms = hidden.new_zeros(*batch['joint_atom_mask'].shape, hidden.size(-1))
                    packed_atoms[atom_owner, atom_slot] = hidden[mask]
                    atom_messages = self.atom_exchanges[exchange_index](packed_atoms, global_condition, batch['joint_atom_mask'])
                    atom_update = torch.zeros_like(hidden)
                    atom_update[mask] = self.atom_exchange_out[exchange_index](atom_messages[atom_owner, atom_slot])
                    hidden = hidden + atom_update
                exchange_index += 1
        return self.local.output(self.local.norm(hidden)).masked_fill(~mask[..., None], 0)

    def forward(self, clean, batch, regularizer=None):
        """One time per MOF; independent node noise; equal MOF/block weighting."""
        time = (torch.randn(batch['block_mask'].size(0), device=clean.device)*self.config.time_logit_std + self.config.time_logit_mean).sigmoid()
        atom_time = time[batch['owner'], None, None]
        noise = torch.randn_like(clean)
        if self.config.mof_noise_transport_group_size:
            from .latent_transport import align_mof_noise
            noise = align_mof_noise(clean, noise, batch, self.config.mof_noise_transport_group_size)
        if self.config.atom_noise_transport:
            from .latent_transport import align_atom_noise
            noise = align_atom_noise(clean, noise, batch['mask'])
        state = ((1-atom_time)*clean + atom_time*noise).masked_fill(~batch['mask'][..., None], 0)
        condition = None
        if torch.rand((), device=clean.device) < 0.5:
            # Do not share no-grad AMP weight casts with the differentiable pass.
            with torch.no_grad(), torch.autocast(clean.device.type, enabled=is_autocast_enabled(clean.device.type), cache_enabled=False):
                condition = self.predict_clean(state, batch, time)
        prediction = self.predict_clean(state, batch, time, condition).float()
        error = ((prediction-clean.float()) / atom_time.clamp_min(0.1)).square()
        per_block = (error * batch['mask'][..., None]).sum((1, 2)) / (batch['mask'].sum(-1) * clean.size(-1))
        per_mof = per_block.new_zeros(time.size(0)).index_add(0, batch['owner'], per_block)
        flow_loss = (per_mof / batch['block_mask'].sum(-1)).mean()
        if not self.config.carbon_loss_weight:
            return flow_loss
        if regularizer is None:
            raise ValueError('Carbon-supervised training requires the frozen-decoder regularizer')
        chemical = regularizer(prediction, batch, time)
        chemical_mof = chemical.new_zeros(time.size(0)).index_add(0, batch['owner'], chemical)
        chemical_loss = (chemical_mof/batch['block_mask'].sum(-1)).mean()
        self.last_loss_components = torch.stack([flow_loss.detach(), chemical_loss.detach()])
        return flow_loss + self.config.carbon_loss_weight*chemical_loss


@torch.inference_mode()
def sample_joint_latents(model, batch, *, steps=100, churn=4., seed=614, guide=None, guidance=1., clean_transform=None, noise_stream_version=1):
    """Heun plus the existing LG-Flow marginal-preserving noise refresh."""
    if steps < 1 or churn < 0:
        raise ValueError('Steps must be positive and churn nonnegative')
    if getattr(model.config, 'atom_noise_transport', False) and churn:
        raise ValueError('Atom-transport coupling requires churn=0; Gaussian refresh assumes independent noise-target pairing')
    if not math.isfinite(guidance) or guidance < 1 or (guidance != 1 and guide is None):
        raise ValueError('Guidance must be finite, at least one, and requires a guide when non-unit')
    model.eval()
    if guide is not None:
        guide.eval()
        if guide.config.latent_dim != model.config.latent_dim:
            raise ValueError('Guide and primary model need the same latent dimensions')
    use_guide = guide is not None and guidance != 1

    def predict(state, time, condition, guide_condition):
        clean = model.predict_clean(state, batch, time, condition).float()
        if not use_guide:
            adjusted = clean if clean_transform is None else clean_transform(clean, batch, time)
            return adjusted, clean, None
        weak = guide.predict_clean(state, batch, time, guide_condition).float()
        adjusted = weak + guidance*(clean-weak)
        if clean_transform is not None:
            adjusted = clean_transform(adjusted, batch, time)
        return adjusted, clean, weak

    device = batch['mask'].device
    initial_rng = torch.Generator(device=device).manual_seed(seed)
    refresh_rng = torch.Generator(device=device).manual_seed(auxiliary_noise_seed(seed, 'lgflow_refresh', noise_stream_version))
    mask = batch['mask'][..., None]
    state = torch.randn(*batch['mask'].shape, model.config.latent_dim, device=device, generator=initial_rng).masked_fill(~mask, 0)
    condition = guide_condition = None
    for index in range(steps, 0, -1):
        t, previous, delta = index/steps, (index-1)/steps, 1/steps
        if churn and 0.1 <= t <= 0.8:
            raised = min(t*(1+churn/steps), 0.99)
            ratio = (1-raised)/(1-t)
            deviation = math.sqrt(max(0., raised**2-(ratio*t)**2))
            state = ratio*state + deviation*torch.randn(state.shape, device=device, generator=refresh_rng)
            state = state.masked_fill(~mask, 0)
            delta += raised-t
            t = raised
        time = torch.full((batch['block_mask'].size(0),), t, device=device)
        clean, own_clean, weak_clean = predict(state, time, condition, guide_condition)
        velocity = (state-clean)/t
        proposal = state-delta*velocity
        if index > 1:
            second_clean, own_second, weak_second = predict(
                proposal, torch.full_like(time, previous), own_clean, weak_clean)
            second_velocity = (proposal-second_clean)/previous
            state = state-delta*(velocity+second_velocity)/2
            # Self-conditioning stays inside each model's own prediction range.
            condition, guide_condition = own_second, weak_second
        else:
            state = clean
    if not torch.isfinite(state).all():
        raise FloatingPointError('Joint latent integration produced nonfinite values')
    return state.masked_fill(~mask, 0)
