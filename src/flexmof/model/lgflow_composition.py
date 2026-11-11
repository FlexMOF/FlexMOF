"""Learned latent composition flow and its frozen sequence graph codec."""

import torch
from torch import nn
from torch.nn import functional as F

from .gaussian_composition import AtomVocabulary, pool_occurrences
from .linker_autoencoder import LinkerAEConfig, LinkerAutoencoder, GraphSequenceStack
from .noise import SinusoidalPositionEmbeddings
from ..linker_data import laplacian_features


class LatentFlowHead(nn.Module):
    """Velocity on per-atom latents, optionally conditioned on backbone context.

    Input time increases from clean (0) to Gaussian noise (1). The predicted
    velocity therefore points toward noise; reverse integration uses dt < 0.
    No clean graph, eigenvectors, positions or node indices enter this head.
    """

    def __init__(self, latent_dim, context_dim, width=256, layers=4):
        super().__init__()
        self.state_proj = nn.Linear(latent_dim, width)
        self.context_proj = nn.Linear(context_dim, width, bias=False)
        nn.init.zeros_(self.context_proj.weight)
        self.time_proj = nn.Sequential(SinusoidalPositionEmbeddings(32), nn.Linear(32, width), nn.SiLU(), nn.Linear(width, width))
        self.count_proj = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
        with torch.no_grad():
            self.count_proj[0].weight[:, 1].zero_()
        self.sequence = GraphSequenceStack(width, layers, 8)
        self.output = nn.Linear(width, latent_dim)

    def forward(self, state, mask, time, context=None, repetitions=None):
        hidden = self.state_proj(state)
        if context is not None:
            hidden = hidden + self.context_proj(context.to(hidden.dtype))
        repeats = torch.zeros_like(time) if repetitions is None else repetitions
        counts = torch.stack([mask.sum(-1).float(), repeats.float()], -1).log1p()
        hidden = hidden + self.time_proj(time.float())[:, None] + self.count_proj(counts)[:, None]
        return self.output(self.sequence(hidden, mask)).masked_fill(~mask[..., None], 0)


class LatentDiTBlock(nn.Module):
    """LG-Flow-style adaptive normalization in the latent denoiser only."""

    def __init__(self, width, heads=8):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.attention_out = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, 4*width), nn.GELU(), nn.Linear(4*width, width))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6*width))

    def forward(self, hidden, condition, mask):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(condition)[:,None].chunk(6, -1)
        normalized = self.norm1(hidden)*(1+scale1)+shift1
        b,n,d = hidden.shape
        q,k,v = self.qkv(normalized).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        attended = F.scaled_dot_product_attention(q,k,v,attn_mask=mask[:,None,None,:])
        hidden = hidden + gate1*self.attention_out(attended.transpose(1,2).reshape(b,n,d))
        hidden = hidden + gate2*self.mlp(self.norm2(hidden)*(1+scale2)+shift2)
        return hidden.masked_fill(~mask[...,None],0)


class LatentDiTHead(nn.Module):
    """Clean-latent prediction with adaptive time conditioning and self-conditioning.

    Follows the LG-Flow reference denoiser (MIT, see docs/licenses/LG-Flow.txt).
    The public forward still returns clean-to-noise velocity for joint Euler
    integration. No positions or atom indices are used.
    """

    def __init__(self, latent_dim, context_dim, width=256, layers=4):
        super().__init__()
        if width % 8:
            raise ValueError("Latent DiT width must be divisible by 8")
        self.state_proj = nn.Linear(2*latent_dim,width)
        self.context_proj = nn.Linear(context_dim,width,bias=False)
        self.time_proj = nn.Sequential(nn.Linear(256,width),nn.SiLU(),nn.Linear(width,width))
        self.count_proj = nn.Sequential(nn.Linear(2,width),nn.SiLU(),nn.Linear(width,width))
        self.blocks = nn.ModuleList([LatentDiTBlock(width) for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width,latent_dim)
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.normal_(module.weight,std=0.02)
                if module.bias is not None:nn.init.zeros_(module.bias)
        nn.init.zeros_(self.context_proj.weight)
        with torch.no_grad():self.count_proj[0].weight[:,1].zero_()

    def predict_clean(self,state,mask,time,context=None,repetitions=None,self_condition=None):
        if self_condition is None:self_condition=torch.zeros_like(state)
        hidden=self.state_proj(torch.cat([state,self_condition],-1))
        if context is not None:hidden=hidden+self.context_proj(context.to(hidden.dtype))
        # Match the reference's continuous-time Fourier embedding without
        # adding high-frequency oscillations to t in [0,1].
        frequency=torch.exp(-torch.arange(128,device=time.device,dtype=torch.float32)*9.210340371976184/128)
        angle=time.float()[:,None]*frequency
        condition=self.time_proj(torch.cat([angle.cos(),angle.sin()],-1))
        repeat=torch.zeros_like(time) if repetitions is None else repetitions.float()
        counts=torch.stack([mask.sum(-1).float(),repeat],-1).log1p()
        condition=condition+self.count_proj(counts)
        for block in self.blocks:hidden=block(hidden,condition,mask)
        return self.output(self.norm(hidden)).masked_fill(~mask[...,None],0)

    def forward(self,state,mask,time,context=None,repetitions=None,self_condition=None):
        clean=self.predict_clean(state,mask,time,context,repetitions,self_condition)
        return ((state-clean)/time[:,None,None].clamp_min(1e-5)).masked_fill(~mask[...,None],0)

    @torch.no_grad()
    def detached_clean(self,state,mask,time,context=None,repetitions=None):
        # Casts created without gradients must not enter an outer AMP cache:
        # reusing them in the next pass silently drops linear-weight gradients.
        enabled=(torch.is_autocast_enabled() if state.device.type=="cuda"
                 else torch.is_autocast_cpu_enabled())
        with torch.autocast(state.device.type,enabled=enabled,cache_enabled=False):
            return self.predict_clean(state,mask,time,context,repetitions)


def make_latent_flow_head(*,architecture="sequence",**config):
    if architecture not in {"sequence","dit"}:
        raise ValueError(f"Unknown latent prior architecture: {architecture}")
    return (LatentDiTHead if architecture=="dit" else LatentFlowHead)(**config)


class LGFlowComposition(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_timesteps = config.mof_composition_num_timesteps
        self.vocab = AtomVocabulary(config.mof_composition_atom_vocab)
        ae_config = LinkerAEConfig(num_atom_classes=len(self.vocab.numbers) - 1,
            width=config.mof_lgflow_ae_width, layers=config.mof_lgflow_ae_layers,
            latent_dim=config.mof_lgflow_latent_dim, frequencies=config.mof_lgflow_frequencies,
            decoder_width=config.mof_lgflow_decoder_width)
        self.autoencoder = LinkerAutoencoder(ae_config).requires_grad_(False).eval()
        self.register_buffer("latent_center", torch.zeros(ae_config.latent_dim))
        self.register_buffer("latent_scale", torch.ones(ae_config.latent_dim))
        self.register_buffer("codec_initialized", torch.tensor(False))

    def train(self, mode=True):
        super().train(mode)
        self.autoencoder.eval()
        return self

    def load_pretrained(self, path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["vocabulary"] != self.vocab.numbers[1:].tolist():
            raise ValueError("LG-Flow autoencoder vocabulary does not match MOF configuration")
        self.autoencoder.load_state_dict(checkpoint["model"], strict=True)
        if "latent_center" not in checkpoint or "latent_scale" not in checkpoint:
            raise ValueError("Autoencoder checkpoint lacks calibrated latent normalization")
        center, scale = checkpoint["latent_center"], checkpoint["latent_scale"]
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Invalid latent normalization in autoencoder checkpoint")
        self.latent_center.copy_(center)
        self.latent_scale.copy_(scale)
        self.codec_initialized.fill_(True)

    @torch.no_grad()
    def encode_clean(self, batch):
        if not self.codec_initialized:
            raise RuntimeError("LG-Flow requires a pretrained graph autoencoder")
        atoms, bonds, mask = batch["mof_bb_atom_types"], batch["mof_bb_bond_adj"], batch["mof_bb_atom_mask"].bool()
        shape = atoms.shape
        atoms, bonds, mask = atoms.flatten(0,1), bonds.flatten(0,1), mask.flatten(0,1)
        valid = mask.any(-1)
        atoms, bonds, selected_mask = atoms[valid], bonds[valid], mask[valid]
        f = self.autoencoder.config.frequencies
        spectral = bonds.new_zeros((*atoms.shape, f, 2), dtype=torch.float)
        spectral_mask = torch.zeros(atoms.size(0), f, dtype=torch.bool, device=atoms.device)
        for i in range(atoms.size(0)):
            indices = selected_mask[i].nonzero().flatten()
            local = bonds[i][indices][:,indices].cpu()
            # Match the corpus's CPU eigensolver. Repeated eigenvalues can
            # otherwise acquire different bases under CUDA eigendecomposition.
            local_spectral, local_mask = laplacian_features(local, f)
            spectral[i,indices] = local_spectral.to(atoms.device)
            spectral_mask[i] = local_mask.to(atoms.device)
        inputs = dict(atoms=(atoms - 1).clamp_min(0), bonds=bonds, mask=selected_mask,
                      spectral=spectral, spectral_mask=spectral_mask)
        mean, _ = self.autoencoder.encode(inputs, augment=False)
        normalized = (mean - self.latent_center) / self.latent_scale
        result = mean.new_zeros(mask.size(0), mask.size(1), mean.size(-1))
        result[valid] = normalized.masked_fill(~selected_mask[...,None], 0)
        return result.reshape(*shape, mean.size(-1))

    @torch.no_grad()
    def decode(self, state, mask):
        if not self.codec_initialized:
            raise RuntimeError("LG-Flow requires a pretrained graph autoencoder")
        shape = mask.shape
        latent = state.flatten(0,1) * self.latent_scale + self.latent_center
        output = self.autoencoder.decode(latent, mask.flatten(0,1))
        atoms = (output["atom_logits"].argmax(-1) + 1).reshape(shape).masked_fill(~mask, 0)
        bonds = output["bond_logits"].argmax(-1).masked_fill(~output["pair_mask"], 0)
        return atoms, bonds.reshape(*shape, shape[-1])

    def q_sample(self, clean, time, mask):
        noise = torch.randn_like(clean)
        t = (time.float() / self.num_timesteps).reshape(-1, 1, 1, 1)
        return ((1-t)*clean + t*noise).masked_fill(~mask[...,None], 0), (noise-clean).masked_fill(~mask[...,None], 0)

    def sample_prior(self, mask):
        return torch.randn(*mask.shape, self.autoencoder.config.latent_dim, device=mask.device).masked_fill(~mask[...,None], 0)

    def p_sample(self, state, velocity, time, previous, mask):
        dt = ((previous-time).float() / self.num_timesteps).reshape(-1, 1, 1, 1)
        return (state + dt * velocity).masked_fill(~mask[...,None], 0)


class MOFLatentFlowHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.timesteps = config.mof_composition_num_timesteps
        self.flow = make_latent_flow_head(architecture=config.mof_lgflow_head_architecture,
            latent_dim=config.mof_lgflow_latent_dim, context_dim=config.embedding_dim,
            width=config.mof_lgflow_head_width, layers=config.mof_lgflow_head_layers)

    def forward(self, context, batch):
        state = batch["mof_comp_latent_state"]
        mask = batch["mof_bb_atom_mask"].bool()
        features = pool_occurrences(context, batch).flatten(0,1)
        time = batch["mof_comp_time"][:,None].expand(mask.shape[:2]).flatten().float() / self.timesteps
        repetitions = batch["mof_bb_occ_mask"].sum(-1).flatten()
        valid = mask.flatten(0,1).any(-1)
        velocity = torch.zeros_like(state.flatten(0,1))
        inputs = (state.flatten(0,1)[valid], mask.flatten(0,1)[valid],
                  time[valid], features[valid], repetitions[valid])
        if hasattr(self.flow,"predict_clean"):
            previous = batch.get("mof_comp_self_condition")
            condition = previous.flatten(0,1)[valid] if previous is not None else None
            if self.training and torch.rand((),device=state.device)<0.5:
                condition=self.flow.detached_clean(*inputs)
            velocity[valid] = self.flow(*inputs,self_condition=condition)
        else:
            velocity[valid] = self.flow(*inputs)
        return {"latent_velocity": velocity.reshape_as(state)}
