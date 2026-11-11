"""Native PyTorch/DDP training for the FlexMOF coordinate denoiser."""

from __future__ import annotations

import math
import os
import random
import time
import gc
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, load_model_weights
from .config import MOFConfig, architecture_values
from .data import DistributedMOFBatchSampler, MOFDataset, collate_mofs
from .ordering import CANONICAL_ORDER_VERSION
from .logging import log_info, log_metrics
from .model.lattice import center_periodic, complete_lattice, initialize_lattice
from .model.composition import masked_classification_loss
from .model.noise import DiffNoise, TimeStepSampler
from .network import MOFNetwork
from .performance import DistributedThroughputMeter, Throughput, global_sample_count
from .tracking import WandbTracker
from .ema import ModelEMA


_COMPOSITION_METRIC_GROUPS = (
    ("seq", "atom"),
    ("seq", "bond"),
    ("3d", "atom"),
    ("3d", "bond"),
    ("seq", "latent"),
    ("3d", "latent"),
)

# The joint model's scalars. They are averaged rather than count-weighted,
# because each one is already a per-block or per-node mean. Reading
# reconstruction and flow separately is the only way to tell an unstable
# denoiser apart from an unstable autoencoder in a log line.
_JOINT_METRIC_NAMES = (
    "mof_joint_reconstruction_loss",
    "mof_joint_flow_loss",
    "mof_joint_atom_ce",
    "mof_joint_bond_ce",
    "mof_joint_hydrogen_ce",
    "mof_joint_kl",
    "mof_joint_latent_scale",
)


@dataclass
class TrainingConfig:
    data_root: str
    output_dir: str
    dataset_subdir: str = "bw_db"
    init_checkpoint: str = ""
    resume: str = ""
    micro_batch_size: int = 4
    accumulation_steps: int = 32
    max_atoms: int = 768
    max_steps: int = 4_375_000
    epochs: int = 1000
    learning_rate: float = 1e-4
    composition_lr_multiplier: float = 1.0
    ema_decay: float = 0.0
    warmup_steps: int = 262_500
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    precision: str = "fp32"
    log_interval: int = 10
    save_interval: int = 2500
    validate_interval: int = 0
    validation_batches: int = -1
    validation_log_interval: int = 1
    validation_seed: int = -1
    validate_at_start: bool = False
    validate_at_epoch_end: bool = True
    save_at_epoch_end: bool = True
    num_workers: int = 0
    seed: int = 12347
    wandb_project: str = ""
    wandb_entity: str = ""
    wandb_name: str = ""
    wandb_group: str = ""
    wandb_tags: str = ""
    wandb_mode: str = "online"
    wandb_log_checkpoints: bool = False
    composition_diffusion: bool = False

    def __post_init__(self):
        if self.init_checkpoint and self.resume:
            raise ValueError("Use either init_checkpoint or resume, not both")
        if self.accumulation_steps < 1 or self.micro_batch_size < 1:
            raise ValueError("Batch and accumulation sizes must be positive")
        if self.max_steps < 1 or self.epochs < 1:
            raise ValueError("max_steps and epochs must be positive")
        if self.composition_lr_multiplier <= 0:
            raise ValueError("composition_lr_multiplier must be positive")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if (
            self.log_interval < 0
            or self.save_interval < 0
            or self.validation_log_interval < 0
        ):
            raise ValueError("Logging and saving intervals cannot be negative")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"Unsupported precision: {self.precision}")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError(f"Unsupported W&B mode: {self.wandb_mode}")


class MOFTrainingModel(nn.Module):
    """MOF network plus its periodic DDPM corruption and training loss."""

    def __init__(self, config: MOFConfig):
        super().__init__()
        self.config = config
        self.net = MOFNetwork(config)
        self.diffusion_noise = DiffNoise(config)
        self.time_sampler = TimeStepSampler(config.num_timesteps)
        if config.mof_joint_denoiser_only:
            # Preserve intentionally frozen denoiser parameters (e.g. its
            # unused context projection); never blanket-unfreeze the module.
            for name, parameter in self.named_parameters():
                if not name.startswith("net.mof_joint.denoiser."):
                    parameter.requires_grad_(False)
            self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.config.mof_joint_denoiser_only:
            self.net.eval()
            self.diffusion_noise.eval()
            self.net.mof_joint.denoiser.train(mode)
        return self

    @staticmethod
    def _masks(batch):
        token_id = batch["token_id"]
        node_count = token_id.size(1)
        node_index = torch.arange(node_count, device=token_id.device)[None]
        non_atom_mask = node_index >= batch["num_atoms"][:, None]
        loss_mask = node_index < batch["num_atoms"][:, None]
        batch_index = torch.arange(token_id.size(0), device=token_id.device)
        for corner in (0, 1, 2, 4):
            loss_mask[batch_index, batch["num_atoms"] + corner] = True
        return token_id.eq(0), non_atom_mask, loss_mask

    @staticmethod
    def coordinate_loss(prediction, target, loss_mask):
        """Mean per-structure RMS error over atoms and independent corners."""
        squared_error = (prediction.to(target.dtype) - target).square()
        squared_error = squared_error.masked_fill(~loss_mask[..., None], 0.0)
        denominator = 3.0 * loss_mask.sum(dim=-1).clamp_min(1)
        per_structure = torch.sqrt(squared_error.sum(dim=(1, 2)) / denominator + 1e-5)
        return per_structure.mean(), per_structure

    def forward(self, batch):
        if self.config.mof_joint_denoiser_only:
            joint = self.net.mof_joint
            with torch.no_grad():
                encoded = joint.encode(batch)
            flow, _ = joint.flow_loss(encoded, batch)
            loss = self.config.mof_joint_flow_weight * flow
            # Coordinate corruption, CSP and reconstruction are not evaluated.
            # Zero is an internal reducer placeholder, not a measured CSP loss.
            zero = loss.detach().new_zeros(())
            return {
                "loss": loss,
                "coordinate_only_loss": zero,
                "coordinate_loss": zero,
                "composition_loss": loss.detach(),
                "per_structure_loss": loss.detach().expand(batch["token_id"].size(0)),
                "mof_joint_flow_loss": flow.detach(),
                "mof_joint_latent_scale": joint.latent_normalization.scale.mean().detach(),
            }
        padding_mask, non_atom_mask, loss_mask = self._masks(batch)

        # Retain the historical ordering of stochastic training operations.
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
        self.net.prepare_composition_training(batch, graph_time)
        network_output = self.net(batch, time, padding_mask)
        prediction = network_output["noise_pred"]
        loss, per_structure = self.coordinate_loss(prediction, noise, loss_mask)
        composition_loss, composition_metrics = self.composition_loss(
            network_output, batch
        )
        coordinate_only = loss
        loss = loss + composition_loss
        return {
            "loss": loss,
            # Differentiable and separate, so the coordinate term can be
            # back-propagated on its own to show which parameters it reaches.
            "coordinate_only_loss": coordinate_only,
            "coordinate_loss": (loss - composition_loss).detach(),
            "composition_loss": composition_loss.detach(),
            "per_structure_loss": (per_structure + composition_loss.detach()),
            "time": graph_time.detach(),
            **composition_metrics,
        }

    def composition_loss(self, output, batch):
        if not self.config.use_mof_composition_diffusion:
            zero = output["noise_pred"].sum() * 0.0
            return zero, {}
        if getattr(self.config, "mof_joint_composition", False):
            return self.joint_composition_loss(output)
        if getattr(self.config, "mof_composition_diffusion_strategy", "") == "lgflow":
            return self.lgflow_composition_loss(output, batch)
        if (
            getattr(self.config, "mof_composition_diffusion_strategy", "")
            == "gaussian_ddpm"
        ):
            return self.gaussian_composition_loss(output, batch)
        atom_labels = (
            batch["mof_bb_atom_types"]
            .long()
            .clamp(min=0, max=self.config.mof_composition_num_atom_classes - 1)
        )
        atom_mask = batch["mof_comp_atom_loss_mask"].bool()
        atom_mask &= batch["mof_has_composition_targets"][:, None, None]
        bond_labels = (
            batch["mof_bb_bond_adj"]
            .long()
            .clamp(min=0, max=self.config.mof_composition_num_bond_classes - 1)
        )
        bond_mask = batch["mof_comp_bond_loss_mask"].bool()
        bond_mask &= batch["mof_has_composition_targets"][:, None, None, None]
        total = output["noise_pred"].sum() * 0.0
        metrics = {}
        for prefix, head_weight in (
            ("seq", self.config.mof_composition_seq_loss_weight),
            ("3d", self.config.mof_composition_3d_loss_weight),
        ):
            atom_loss, atom_accuracy, atom_count = masked_classification_loss(
                output[f"mof_comp_{prefix}_atom_logits"],
                atom_labels,
                atom_mask,
            )
            bond_loss, bond_accuracy, bond_count = masked_classification_loss(
                output[f"mof_comp_{prefix}_bond_logits"],
                bond_labels,
                bond_mask,
            )
            head_loss = (
                self.config.mof_composition_atom_loss_weight * atom_loss
                + self.config.mof_composition_bond_loss_weight * bond_loss
            )
            total = total + head_weight * head_loss
            metrics.update(
                {
                    f"mof_comp_{prefix}_atom_loss": atom_loss.detach(),
                    f"mof_comp_{prefix}_bond_loss": bond_loss.detach(),
                    f"mof_comp_{prefix}_atom_acc": atom_accuracy.detach(),
                    f"mof_comp_{prefix}_bond_acc": bond_accuracy.detach(),
                    f"mof_comp_{prefix}_atom_count": torch.tensor(
                        atom_count, device=atom_loss.device, dtype=torch.float32
                    ),
                    f"mof_comp_{prefix}_bond_count": torch.tensor(
                        bond_count, device=bond_loss.device, dtype=torch.float32
                    ),
                }
            )
        return total, metrics

    def joint_composition_loss(self, output):
        """Weighted sum of the two composition terms the network computed.

        The network builds them, not this method, because both need the same
        forward encode that the coordinate branch also reads. The coordinate
        term is added by the caller, so the three losses share one backward
        pass and one optimizer step over one set of shared parameters.
        """
        reconstruction = output["mof_joint_reconstruction_loss"]
        flow = output["mof_joint_flow_loss"]
        total = (
            self.config.mof_joint_reconstruction_weight * reconstruction
            + self.config.mof_joint_flow_weight * flow
        )
        metrics = {
            "mof_joint_reconstruction_loss": reconstruction.detach(),
            "mof_joint_flow_loss": flow.detach(),
            **{
                key: value.detach()
                for key, value in output.items()
                if key.startswith("mof_joint_")
                and key not in {"mof_joint_reconstruction_loss", "mof_joint_flow_loss"}
            },
        }
        return total, metrics

    def lgflow_composition_loss(self, output, batch):
        mask = batch["mof_comp_atom_loss_mask"] & batch["mof_comp_time"].gt(0)[:, None, None]
        target = batch["mof_comp_latent_target"].float()
        total = output["noise_pred"].sum() * 0
        metrics = {}
        for prefix, weight in (("seq", self.config.mof_composition_seq_loss_weight),
                               ("3d", self.config.mof_composition_3d_loss_weight)):
            prediction = output[f"mof_comp_{prefix}_latent_velocity"].float()
            count = mask.sum() * target.size(-1)
            error = (prediction-target).square()
            if self.config.mof_lgflow_head_architecture == "dit":
                t = batch["mof_comp_time"].float().view(-1,1,1,1) / self.config.mof_composition_num_timesteps
                # Equivalent to the reference's clean-latent loss divided by
                # max(t, 0.1)^2, with our clean=0 time convention.
                error = error * (t/t.clamp_min(0.1)).square()
            loss = (error * mask[...,None]).sum() / count.clamp_min(1)
            total = total + weight * loss
            # Decode predicted clean latents for chemistry accuracy diagnostics.
            # This is detached; the supervised objective is latent velocity MSE.
            with torch.no_grad():
                time = batch["mof_comp_time"].float().view(-1,1,1,1) / self.config.mof_composition_num_timesteps
                clean = batch["mof_comp_latent_state"] - time * prediction
                atoms, bonds = self.net.mof_lgflow.decode(clean, batch["mof_comp_atom_loss_mask"])
                bond_mask = torch.triu(batch["mof_comp_bond_loss_mask"], diagonal=1)
                bond_mask &= batch["mof_comp_time"].gt(0)[:, None, None, None]
                atom_acc = ((atoms == batch["mof_bb_atom_types"]) & mask).sum() / mask.sum().clamp_min(1)
                bond_acc = ((bonds == batch["mof_bb_bond_adj"]) & bond_mask).sum() / bond_mask.sum().clamp_min(1)
            metrics.update({f"mof_comp_{prefix}_latent_loss": loss.detach(),
                            f"mof_comp_{prefix}_latent_count": count.float(),
                            f"mof_comp_{prefix}_atom_acc": atom_acc,
                            f"mof_comp_{prefix}_atom_count": mask.sum().float(),
                            f"mof_comp_{prefix}_bond_acc": bond_acc,
                            f"mof_comp_{prefix}_bond_count": bond_mask.sum().float()})
        return total, metrics

    def gaussian_composition_loss(self, output, batch):
        process = self.net.mof_gaussian_diffusion
        total = output["noise_pred"].sum() * 0.0
        metrics = {}
        for prefix, head_weight in (
            ("seq", self.config.mof_composition_seq_loss_weight),
            ("3d", self.config.mof_composition_3d_loss_weight),
        ):
            for kind, labels, normalize, decode, weight in (
                (
                    "atom", batch["mof_bb_atom_types"], process.normalize_atoms,
                    process.atom_ids, self.config.mof_composition_atom_loss_weight,
                ),
                (
                    "bond", batch["mof_bb_bond_adj"], process.normalize_bonds,
                    process.bond_labels, self.config.mof_composition_bond_loss_weight,
                ),
            ):
                mask = batch[f"mof_comp_{kind}_loss_mask"]
                # Composition is frozen at t=0 during the coordinate-only tail.
                # Train its denoiser on t>0, rather than rewarding clean copying
                # on the 80% of coordinate steps that require no composition update.
                active = batch["mof_comp_time"].gt(0)
                mask = mask & active.view(-1, *([1] * (mask.ndim - 1)))
                if kind == "bond":
                    mask = torch.triu(mask, diagonal=1)
                prediction = output[f"mof_comp_{prefix}_{kind}_x0"].float()
                count = mask.sum().clamp_min(1)
                loss = ((prediction - normalize(labels)).square() * mask).sum() / count
                accuracy = ((decode(prediction) == labels) * mask).sum() / count
                total = total + head_weight * weight * loss
                base = f"mof_comp_{prefix}_{kind}"
                metrics.update({
                    f"{base}_loss": loss.detach(),
                    f"{base}_acc": accuracy.detach(),
                    f"{base}_count": mask.sum().float(),
                })
        return total, metrics


def learning_rate(step: int, config: TrainingConfig) -> float:
    """Linear warmup followed by the legacy cosine decay."""
    if config.warmup_steps and step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    decay_steps = max(1, config.max_steps - config.warmup_steps)
    progress = min(1.0, (step - config.warmup_steps) / decay_steps)
    return config.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def optimizer_parameter_groups(model: nn.Module, composition_lr_multiplier: float):
    """Let newly initialized Gaussian heads learn faster than the backbone.

    Both branches deduplicate by parameter identity. In the joint model the
    autoencoder encodes through the CSP trunk, so the same tensors are reachable
    under two names; handing them to an optimizer twice would double their
    effective step and their weight decay.
    """
    if composition_lr_multiplier == 1.0:
        return [{"params": _unique_parameters(p for p in model.parameters() if p.requires_grad), "lr_scale": 1.0}]
    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            target = heads if name.startswith((
                "net.mof_gaussian_",
                "net.mof_lgflow_seq_head.",
                "net.mof_lgflow_3d_head.",
                "net.mof_joint.denoiser.",
            )) else backbone
            target.append(parameter)
    return [
        {"params": _unique_parameters(backbone), "lr_scale": 1.0},
        {"params": _unique_parameters(heads), "lr_scale": composition_lr_multiplier},
    ]


def _unique_parameters(parameters):
    seen, unique = set(), []
    for parameter in parameters:
        if id(parameter) not in seen:
            seen.add(id(parameter))
            unique.append(parameter)
    return unique


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _groups(loader: DataLoader, size: int) -> Iterator[list[dict]]:
    """Keep the final partial update; the training loop scales by len(group)."""
    iterator = iter(loader)
    while group := list(islice(iterator, size)):
        yield group


def _make_data_loader(dataset, batch_sampler, num_workers: int) -> DataLoader:
    # Iterators are created after CUDA/NCCL and tracker initialization. Forking
    # their threads/locks can deadlock a worker before it yields its first batch.
    # MOFDataset's pickle hooks reopen LMDB in each spawned process.
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_mofs,
        num_workers=num_workers,
        multiprocessing_context="spawn" if num_workers else None,
        pin_memory=True,
    )


def _architecture_config(
    checkpoint: dict | None,
    composition_diffusion: bool = False,
    overrides: dict | None = None,
) -> MOFConfig:
    if checkpoint is None:
        values = {}
    else:
        if "args" not in checkpoint:
            raise ValueError("Checkpoint has no architecture metadata under 'args'")
        values = architecture_values(dict(checkpoint["args"]))
    # Per-atom expansion changes only runtime graph construction, not learned
    # tensor shapes, so old coordinate checkpoints can use it directly.
    values["use_per_atom_outcell_index"] = True
    values["pbc_cutoff"] = MOFConfig.pbc_cutoff
    if overrides:
        values.update(overrides)
    if composition_diffusion:
        values["use_mof_composition_diffusion"] = True
    return MOFConfig(**values)


def _check_finetune_source(source, architecture, *, resume=False):
    if architecture.mof_joint_denoiser_only:
        if source is None or not source.get("args", {}).get("mof_joint_composition", False):
            raise ValueError("Denoiser-only finetuning requires an existing joint checkpoint")
    if resume:
        for name in ("mof_joint_freeze_codec", "mof_joint_denoiser_only",
                     "mof_disable_graphormer_2d",
                     "mof_joint_canonical_blocks", "mof_joint_target_augment",
                     "mof_joint_time_logit_mean", "mof_joint_time_logit_std"):
            previous = source["args"].get(name, getattr(MOFConfig, name))
            if previous != getattr(architecture, name):
                raise ValueError(f"Resume cannot change {name}; use a new finetune instead")


def _rng_state(model: MOFTrainingModel) -> dict:
    generator = model.diffusion_noise.torch_generator
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "cpu": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(),
        "diffusion": None if generator is None else generator.get_state(),
    }


def _restore_rng_state(model: MOFTrainingModel, state: dict, device) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state(state["cuda"], device)
    if state["diffusion"] is not None:
        if model.diffusion_noise.torch_generator is None:
            model.diffusion_noise.torch_generator = torch.Generator(device=device)
        model.diffusion_noise.torch_generator.set_state(state["diffusion"])


def _save_checkpoint(
    path: Path,
    model: MOFTrainingModel,
    optimizer,
    scaler,
    architecture: MOFConfig,
    training: TrainingConfig,
    state: dict,
    ema: ModelEMA | None = None,
) -> None:
    local_rng = _rng_state(model)
    rng_states: list[dict | None] = [None] * dist.get_world_size()
    dist.all_gather_object(rng_states, local_rng)
    if dist.get_rank() != 0:
        return
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": {
            name: getattr(architecture, name)
            for name in architecture.checkpoint_fields()
        },
        "training_config": asdict(training),
        "trainer_state": state,
        "rng_states": rng_states,
        "implementation": {
            "graphormer_2d_topology_enabled": not architecture.mof_disable_graphormer_2d,
            "periodic_graph_prefix": model.net.structure_encoder.embedding.graph_2d_attention_bias.PERIODIC_GRAPH_PREFIX,
            "canonical_order": CANONICAL_ORDER_VERSION if architecture.mof_canonical_order else None,
            "sequence_rope": architecture.sequence_rope,
            "learned_block_or_slot_ids": False,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    if ema is not None:
        checkpoint["ema"] = ema.state_dict()
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)
    if ema is not None:
        ema_path = path.with_suffix(".ema.pt")
        ema_temporary = ema_path.with_suffix(".pt.tmp")
        torch.save({
            "model": ema.model_state_dict(model), "args": checkpoint["args"],
            "training_config": checkpoint["training_config"],
            "trainer_state": state, "ema_updates": ema.updates,
            "weights": "ema", "source_training_checkpoint": str(path),
        }, ema_temporary)
        os.replace(ema_temporary, ema_path)
        (path.parent / "latest_ema_checkpoint.txt").write_text(ema_path.name + "\n")
    (path.parent / "latest_checkpoint.txt").write_text(path.name + "\n")


def _checkpoint_world_size(checkpoint: dict) -> int:
    """How many ranks wrote the checkpoint, or 0 if it did not record that."""
    return len(checkpoint.get("rng_states", []))


def _load_resume_state(checkpoint, optimizer, scaler) -> tuple[dict, dict | None]:
    if "optimizer" not in checkpoint or "trainer_state" not in checkpoint:
        raise ValueError("Resume requires a FlexMOF training checkpoint")
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))
    states = checkpoint.get("rng_states", [])
    state = dict(checkpoint["trainer_state"])
    if len(states) == dist.get_world_size():
        return state, states[dist.get_rank()]
    if dist.get_rank() == 0:
        log_info(
            f"Resuming on {dist.get_world_size()} ranks from a checkpoint written by "
            f"{len(states)}. Weights, optimizer state and the step count carry over "
            "unchanged; each rank keeps the stream seeded from seed+rank instead of a "
            "saved one, so the sample order from here on differs from an uninterrupted "
            "run."
        )
    return state, None


def _check_resume_batch(checkpoint: dict, saved: dict, current: TrainingConfig) -> None:
    """An optimizer step must still see the number of samples it used to.

    The device count is allowed to change, so ``micro_batch_size`` and
    ``accumulation_steps`` are free to move with it as long as their product
    with the world size holds: that product is what the learning-rate schedule,
    the step count and ``total_samples`` are denominated in.
    """
    ranks = _checkpoint_world_size(checkpoint)
    layout = ("micro_batch_size", "accumulation_steps")
    if not ranks:
        changed = {
            name: (saved[name], getattr(current, name))
            for name in layout
            if saved.get(name) != getattr(current, name)
        }
        if changed:
            raise ValueError(
                "Resume from a checkpoint that recorded no RNG states cannot infer "
                f"its device count, so it requires the same batch layout: {changed}"
            )
        return
    before = saved["micro_batch_size"] * saved["accumulation_steps"] * ranks
    after = current.micro_batch_size * current.accumulation_steps * dist.get_world_size()
    if before != after:
        raise ValueError(
            "Resume requires the same number of samples per optimizer step: the "
            f"checkpoint used {saved['micro_batch_size']} x "
            f"{saved['accumulation_steps']} x {ranks} ranks = {before}, this run asks "
            f"for {current.micro_batch_size} x {current.accumulation_steps} x "
            f"{dist.get_world_size()} ranks = {after}"
        )


def _check_resume_config(checkpoint: dict, current: TrainingConfig) -> None:
    saved = checkpoint.get("training_config")
    if not saved:
        return
    fixed_fields = (
        "max_atoms",
        "learning_rate",
        "warmup_steps",
        "weight_decay",
        "gradient_clip",
        "precision",
        "seed",
    )
    changed = {
        name: (saved[name], getattr(current, name))
        for name in fixed_fields
        if saved.get(name) != getattr(current, name)
    }
    if changed:
        raise ValueError(f"Resume configuration does not match checkpoint: {changed}")
    if saved.get("composition_lr_multiplier", 1.0) != current.composition_lr_multiplier:
        raise ValueError("Resume requires the same composition_lr_multiplier")
    if saved.get("ema_decay", 0.0) != current.ema_decay:
        raise ValueError("Resume requires the same ema_decay")
    _check_resume_batch(checkpoint, saved, current)


def _reduce_mean(value: torch.Tensor) -> float:
    reduced = value.detach().float()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return float(reduced / dist.get_world_size())


def _composition_metric_totals(outputs: list[dict]) -> torch.Tensor:
    """Pack count-weighted loss, correct predictions, and counts per head."""
    device = outputs[0]["loss"].device
    zero = torch.zeros((), device=device)
    rows = []
    for prefix, target in _COMPOSITION_METRIC_GROUPS:
        base = f"mof_comp_{prefix}_{target}"
        counts = [
            output.get(f"{base}_count", zero).detach().float() for output in outputs
        ]
        count = torch.stack(counts).sum()
        loss_sum = torch.stack(
            [
                output.get(f"{base}_loss", zero).detach().float() * item_count
                for output, item_count in zip(outputs, counts)
            ]
        ).sum()
        correct = torch.stack(
            [
                output.get(f"{base}_acc", zero).detach().float() * item_count
                for output, item_count in zip(outputs, counts)
            ]
        ).sum()
        rows.append(torch.stack((loss_sum, correct, count)))
    return torch.stack(rows)


def _joint_metric_totals(outputs: list[dict]) -> torch.Tensor:
    """Sum each joint scalar and its presence count, for a later mean."""
    device = outputs[0]["loss"].device
    rows = []
    for name in _JOINT_METRIC_NAMES:
        values = [
            output[name].detach().float() for output in outputs if name in output
        ]
        total = torch.stack(values).sum() if values else torch.zeros((), device=device)
        rows.append(torch.stack((total, torch.tensor(float(len(values)), device=device))))
    return torch.stack(rows)


def _joint_metrics(totals: torch.Tensor) -> dict[str, float]:
    metrics = {}
    for name, row in zip(_JOINT_METRIC_NAMES, totals):
        count = float(row[1])
        if count > 0:
            metrics[name] = float(row[0] / count)
    return metrics


def _composition_metrics(totals: torch.Tensor) -> dict[str, float]:
    """Convert globally reduced composition totals to old-repo metric names."""
    metrics = {}
    latent_objective = any(float(row[2]) > 0 for row, (_, target) in zip(totals, _COMPOSITION_METRIC_GROUPS) if target == "latent")
    for row, (prefix, target) in zip(totals, _COMPOSITION_METRIC_GROUPS):
        count = float(row[2])
        if count <= 0:
            continue
        base = f"mof_comp_{prefix}_{target}"
        if target == "latent" or not latent_objective:
            metrics[f"{base}_loss"] = float(row[0] / count)
        if target != "latent":
            metrics[f"{base}_acc"] = float(row[1] / count)
    return metrics


@torch.no_grad()
def validate(
    model, loader, training, device, *, epoch, global_step, rank, ema=None,
):
    context = ema.average_parameters(model) if ema is not None else nullcontext()
    rng = _rng_state(model) if training.validation_seed >= 0 else None
    original_generator = model.diffusion_noise.torch_generator
    try:
        if rng is not None:
            _seed_everything(training.validation_seed + rank)
            model.diffusion_noise.torch_generator = torch.Generator(device=device).manual_seed(
                training.validation_seed + 1_000_000 + rank
            )
        with context:
            result = _validate(model, loader, training, device, epoch=epoch,
                               global_step=global_step, rank=rank)
    finally:
        if rng is not None:
            model.diffusion_noise.torch_generator = original_generator
            _restore_rng_state(model, rng, device)
    if ema is not None:
        result[2]["ema_updates"] = ema.updates
    return result


@torch.no_grad()
def _validate(
    model,
    loader,
    training: TrainingConfig,
    device,
    *,
    epoch: int,
    global_step: int,
    rank: int,
) -> tuple[float, Throughput, dict[str, float]]:
    model.eval()
    total = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    composition_totals = torch.zeros(len(_COMPOSITION_METRIC_GROUPS), 3, device=device)
    joint_totals = torch.zeros(len(_JOINT_METRIC_NAMES), 2, device=device)
    coordinate_total = torch.zeros((), device=device)
    meter = DistributedThroughputMeter(device)
    total_batches = len(loader)
    if training.validation_batches >= 0:
        total_batches = min(total_batches, training.validation_batches)
    if rank == 0:
        log_info(f"Start validation for epoch: {epoch}, global step: {global_step}")
    for index, batch in enumerate(loader):
        if 0 <= training.validation_batches <= index:
            break
        output = model(_move_to_device(batch, device))
        total += output["per_structure_loss"].sum()
        count += output["per_structure_loss"].numel()
        composition_totals += _composition_metric_totals([output])
        examples = output["per_structure_loss"].numel()
        joint_totals += _joint_metric_totals([output]) * examples
        coordinate_total += output["coordinate_loss"] * examples
        meter.update(output["per_structure_loss"].numel())
        should_log = (
            training.validation_log_interval > 0
            and (index + 1) % training.validation_log_interval == 0
        )
        if should_log:
            progress_total = total.clone()
            progress_count = count.clone()
            progress_composition = composition_totals.clone()
            dist.all_reduce(progress_total)
            dist.all_reduce(progress_count)
            dist.all_reduce(progress_composition)
            throughput = meter.measure()
            if rank == 0:
                log_metrics(
                    {
                        "valid_loss": float(
                            progress_total / progress_count.clamp_min(1)
                        ),
                        "num_examples": int(progress_count),
                        "epoch": epoch,
                        "global_step": global_step,
                        "validation_batch": index + 1,
                        "validation_batches": total_batches,
                        "samples_per_second": throughput.samples_per_second,
                        **_composition_metrics(progress_composition),
                    }
                )
    dist.all_reduce(total)
    dist.all_reduce(count)
    dist.all_reduce(composition_totals)
    dist.all_reduce(joint_totals)
    dist.all_reduce(coordinate_total)
    throughput = meter.measure()
    model.train()
    valid_loss = float(total / count.clamp_min(1))
    composition_metrics = _composition_metrics(composition_totals)
    composition_metrics.update(_joint_metrics(joint_totals))
    if not model.config.mof_joint_denoiser_only:
        composition_metrics["coordinate_loss"] = float(coordinate_total / count.clamp_min(1))
    if rank == 0:
        log_metrics(
            {
                "valid_loss": valid_loss,
                "num_examples": int(count),
                "epoch": epoch,
                "global_step": global_step,
                "samples_per_second": throughput.samples_per_second,
                "seconds": throughput.seconds,
                **composition_metrics,
            }
        )
    return valid_loss, throughput, composition_metrics


def train(training: TrainingConfig, mof_overrides: dict | None = None) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", device_id=device)
    output_dir = Path(training.output_dir).resolve()
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    _seed_everything(training.seed + rank)

    source_path = training.resume or training.init_checkpoint
    source = load_checkpoint(source_path) if source_path else None
    architecture = _architecture_config(
        source, training.composition_diffusion, mof_overrides
    )
    _check_finetune_source(source, architecture, resume=bool(training.resume))
    model = MOFTrainingModel(architecture)
    if source is not None:
        source_has_composition = bool(
            dict(source.get("args", {})).get("use_mof_composition_diffusion", False)
        )
        initialize_composition = (
            bool(training.init_checkpoint)
            and architecture.use_mof_composition_diffusion
            and (
                not source_has_composition
                or dict(source.get("args", {})).get(
                    "mof_composition_diffusion_strategy", "digress_uniform"
                ) != architecture.mof_composition_diffusion_strategy
            )
        )
        missing_prefixes = (
            (
                "net.embedding.mof_bb_id_embed.",
                "net.mof_composition_seq_head.",
                "net.mof_composition_3d_head.",
                "net.mof_gaussian_seq_head.",
                "net.mof_gaussian_3d_head.",
                "net.mof_lgflow.",
                "net.mof_lgflow_seq_head.",
                "net.mof_lgflow_3d_head.",
                "net.mof_joint.",
                "net.cell_laplacian.",
                "net.seq_encoder.graph_norm.",
            )
            if initialize_composition
            or (training.init_checkpoint and model.net.gaussian_composition)
            or (training.init_checkpoint and model.net.joint_composition
                and not dict(source.get("args", {})).get("mof_joint_composition", False))
            else ()
        )
        load_model_weights(
            model,
            source,
            source_path,
            allow_missing_prefixes=missing_prefixes,
        )
    if model.net.joint_composition:
        # Joint training starts from scratch by construction. The autoencoder
        # is trained here rather than loaded, and its latent normalization is
        # estimated online, so there is no pretrained-codec requirement to
        # satisfy. Optionally warm-starting the denoiser is still allowed.
        if architecture.mof_lgflow_prior_checkpoint:
            prior = torch.load(
                architecture.mof_lgflow_prior_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            model.net.mof_joint.denoiser.load_state_dict(prior["model"], strict=True)
            if rank == 0:
                print(
                    "initialized the joint denoiser from "
                    f"{architecture.mof_lgflow_prior_checkpoint}",
                    flush=True,
                )
    elif model.net.lgflow_composition and not model.net.mof_lgflow.codec_initialized:
        if not architecture.mof_lgflow_autoencoder_checkpoint:
            raise ValueError("Initialize LG-Flow with mof_lgflow_autoencoder_checkpoint")
        model.net.mof_lgflow.load_pretrained(architecture.mof_lgflow_autoencoder_checkpoint)
        if architecture.mof_lgflow_prior_checkpoint:
            prior = torch.load(architecture.mof_lgflow_prior_checkpoint, map_location="cpu", weights_only=False)
            if prior["config"].get("architecture", "sequence") != architecture.mof_lgflow_head_architecture:
                raise ValueError("LG-Flow prior and joint head architectures differ")
            if prior["vocabulary"] != model.net.mof_lgflow.vocab.numbers[1:].tolist():
                raise ValueError("LG-Flow prior vocabulary mismatch")
            if not torch.equal(prior["latent_center"], model.net.mof_lgflow.latent_center.cpu()) or not torch.equal(prior["latent_scale"], model.net.mof_lgflow.latent_scale.cpu()):
                raise ValueError("LG-Flow prior was trained in a different autoencoder latent space")
            for head in (model.net.mof_lgflow_seq_head, model.net.mof_lgflow_3d_head):
                head.flow.load_state_dict(prior["model"], strict=True)
    model.to(device)
    if architecture.mof_joint_freeze_codec and not model.net.mof_joint.latent_normalization.ready:
        raise ValueError("Frozen-codec finetuning requires calibrated checkpoint weights")
    saved_ema = None
    if training.resume and training.ema_decay:
        if "ema" not in source:
            raise ValueError("EMA resume requires saved EMA state")
        saved_ema = source["ema"]
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, training.composition_lr_multiplier),
        lr=0.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=training.weight_decay,
        fused=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=training.precision == "fp16")
    state = {
        "epoch": 0,
        "groups_in_epoch": 0,
        "global_step": 0,
        "total_samples": 0,
    }
    resume_rng = None
    if training.init_checkpoint:
        state["finetune_source"] = training.init_checkpoint
        state["source_global_step"] = source.get("trainer_state", {}).get("global_step")
    if training.resume:
        _check_resume_config(source, training)
        state, resume_rng = _load_resume_state(source, optimizer, scaler)
        state.setdefault(
            "total_samples",
            state["global_step"]
            * training.micro_batch_size
            * training.accumulation_steps
            * world_size,
        )
        if state["global_step"] >= training.max_steps:
            return
    tracker = WandbTracker(training, architecture, state, output_dir, rank)
    if rank == 0:
        (output_dir / "run_config.json").write_text(json.dumps({
            "training": asdict(training), "architecture": asdict(architecture),
            "initial_state": state,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "frozen_parameters": sum(p.numel() for p in model.parameters() if not p.requires_grad),
            "validation_weights": "ema" if training.ema_decay else "live",
            "training_scope": "composition_denoiser_only" if architecture.mof_joint_denoiser_only else "joint_or_csp",
            "coordinate_objective_enabled": not architecture.mof_joint_denoiser_only,
            "graphormer_2d_topology_enabled": not architecture.mof_disable_graphormer_2d,
        }, indent=2) + "\n")
        if architecture.mof_disable_graphormer_2d:
            log_info("Graphormer 2D ablation: invariant structure encoder has no shortest-path, "
                     "multi-hop bond or disconnected-graph bias.")
        if architecture.mof_joint_denoiser_only:
            log_info("Training only net.mof_joint.denoiser; CSP, shared encoder, codec, "
                     "latent normalization and injection are frozen. Validation measures "
                     "composition flow loss; coordinate/reconstruction objectives are disabled.")
    del source
    gc.collect()

    train_dataset = MOFDataset(
        Path(training.data_root) / training.dataset_subdir,
        split="train",
        max_atoms=training.max_atoms,
        composition_config=architecture,
    )
    valid_dataset = MOFDataset(
        Path(training.data_root) / training.dataset_subdir,
        split="valid",
        max_atoms=training.max_atoms,
        composition_config=architecture,
    )
    if rank == 0:
        log_info(
            f"Dataset: train={len(train_dataset)}, valid={len(valid_dataset)}, "
            f"overfit_key={architecture.mof_overfit_key!r}, "
            f"canonical_order={architecture.mof_canonical_order}, "
            f"sequence_rope={architecture.sequence_rope}"
        )
    train_sampler = DistributedMOFBatchSampler(
        train_dataset, training.micro_batch_size, world_size, rank, training.seed
    )
    valid_sampler = DistributedMOFBatchSampler(
        valid_dataset, training.micro_batch_size, world_size, rank,
        training.validation_seed if training.validation_seed >= 0 else training.seed,
    )
    train_loader = _make_data_loader(
        train_dataset, train_sampler, training.num_workers
    )
    valid_loader = _make_data_loader(
        valid_dataset, valid_sampler, training.num_workers
    )

    ddp = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        gradient_as_bucket_view=True,
        # This PyTorch reducer's static mode is incompatible with no_sync().
        static_graph=training.accumulation_steps == 1,
    )
    # DDP first broadcasts rank zero's initialization, including scratch runs.
    ema = ModelEMA(model, training.ema_decay) if training.ema_decay else None
    if saved_ema is not None:
        ema.load_state_dict(saved_ema)
    del saved_ema
    autocast_dtype = torch.bfloat16 if training.precision == "bf16" else torch.float16
    optimizer.zero_grad(set_to_none=True)
    started = time.monotonic()
    train_meter = DistributedThroughputMeter(device)
    interval_loss = 0.0
    interval_coordinate_loss = 0.0
    interval_composition_loss = 0.0
    interval_composition_totals = torch.zeros(
        len(_COMPOSITION_METRIC_GROUPS), 3, device=device
    )
    interval_joint_totals = torch.zeros(len(_JOINT_METRIC_NAMES), 2, device=device)
    interval_steps = 0
    if rank == 0:
        log_info("Start training")

    if training.validate_at_start and not training.resume:
        paused = time.monotonic()
        valid_loss, throughput, metrics = validate(
            model, valid_loader, training, device, epoch=state["epoch"],
            global_step=state["global_step"], rank=rank, ema=ema,
        )
        if rank == 0:
            tracker.log({"validation/loss": valid_loss,
                         "validation/samples_per_second": throughput.samples_per_second,
                         **{f"validation/{k}": v for k, v in metrics.items()}},
                        step=state["global_step"])
        train_meter.exclude(time.monotonic() - paused)

    for epoch in range(state["epoch"], training.epochs):
        train_sampler.set_epoch(epoch)
        valid_sampler.set_epoch(0 if training.validation_seed >= 0 else epoch)
        groups = _groups(train_loader, training.accumulation_steps)
        skip = state["groups_in_epoch"] if epoch == state["epoch"] else 0
        for _ in range(skip):
            next(groups, None)
        if resume_rng is not None:
            # Iterator construction consumes the global RNG even with zero
            # workers. Restore only after recreating/skipping the iterator.
            _restore_rng_state(model, resume_rng, device)
            resume_rng = None
        if rank == 0:
            log_info(f"Start Training for epoch: {epoch}")
        for group_index, group in enumerate(groups, start=skip):
            lr = learning_rate(state["global_step"], training)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = lr * parameter_group.get("lr_scale", 1.0)
            losses = []
            coordinate_losses = []
            composition_losses = []
            step_composition_totals = torch.zeros_like(interval_composition_totals)
            step_joint_totals = torch.zeros_like(interval_joint_totals)
            for micro_index, batch in enumerate(group):
                synchronize = micro_index + 1 == len(group)
                sync_context = nullcontext() if synchronize else ddp.no_sync()
                amp_context = torch.autocast(
                    "cuda",
                    dtype=autocast_dtype,
                    enabled=training.precision != "fp32",
                )
                try:
                    with sync_context, amp_context:
                        output = ddp(_move_to_device(batch, device))
                        loss = output["loss"] / len(group)
                    scaler.scale(loss).backward()
                except Exception:
                    # CPU batch metadata remains accessible even after a CUDA
                    # failure; emit it before any distributed cleanup.
                    print("TRAIN_BATCH_FAILURE " + json.dumps({
                        "rank": rank, "epoch": epoch,
                        "global_step": state["global_step"],
                        "micro_index": micro_index,
                        "dataset_indices": batch["idx"].tolist(),
                        "num_atoms": batch["num_atoms"].tolist(),
                    }), file=sys.stderr, flush=True)
                    raise
                losses.append(output["loss"].detach())
                coordinate_losses.append(output["coordinate_loss"])
                composition_losses.append(output["composition_loss"])
                if architecture.use_mof_composition_diffusion:
                    step_composition_totals += _composition_metric_totals([output])
                if architecture.mof_joint_composition:
                    step_joint_totals += _joint_metric_totals([output])
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), training.gradient_clip
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError("Nonfinite training gradient")
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)

            state["global_step"] += 1
            state["groups_in_epoch"] = group_index + 1
            local_samples = sum(batch["num_atoms"].size(0) for batch in group)
            state["total_samples"] += global_sample_count(local_samples, device)
            train_meter.update(local_samples)
            mean_loss = _reduce_mean(torch.stack(losses).mean())
            mean_coordinate_loss = _reduce_mean(torch.stack(coordinate_losses).mean())
            mean_composition_loss = _reduce_mean(torch.stack(composition_losses).mean())
            if architecture.use_mof_composition_diffusion:
                dist.all_reduce(step_composition_totals)
                interval_composition_totals += step_composition_totals
            if architecture.mof_joint_composition:
                dist.all_reduce(step_joint_totals)
                interval_joint_totals += step_joint_totals
            interval_loss += mean_loss
            interval_coordinate_loss += mean_coordinate_loss
            interval_composition_loss += mean_composition_loss
            interval_steps += 1
            if training.log_interval > 0 and (
                state["global_step"] % training.log_interval == 0
                or state["global_step"] >= training.max_steps
            ):
                throughput = train_meter.measure(reset=True)
                elapsed = time.monotonic() - started
                log_loss = interval_loss / interval_steps
                log_coordinate_loss = interval_coordinate_loss / interval_steps
                log_composition_loss = interval_composition_loss / interval_steps
                log_composition_metrics = _composition_metrics(
                    interval_composition_totals
                )
                log_composition_metrics.update(_joint_metrics(interval_joint_totals))
                if rank == 0:
                    log_metrics(
                        {
                            "loss": log_loss,
                            "grad_scale": scaler.get_scale(),
                            "lr": lr,
                            "epoch": epoch,
                            "batch": state["groups_in_epoch"],
                            "total_samples": state["total_samples"],
                            "global_step": state["global_step"],
                            "samples_per_second": throughput.samples_per_second,
                            "grad_norm": float(grad_norm),
                            **({} if architecture.mof_joint_denoiser_only else
                               {"coordinate_loss": log_coordinate_loss}),
                            "composition_loss": log_composition_loss,
                            **log_composition_metrics,
                        }
                    )
                    tracker.log(
                        {
                            "train/loss": log_loss,
                            **({} if architecture.mof_joint_denoiser_only else
                               {"train/coordinate_loss": log_coordinate_loss}),
                            "train/composition_loss": log_composition_loss,
                            "train/learning_rate": lr,
                            "train/composition_learning_rate": (
                                lr * training.composition_lr_multiplier
                            ),
                            "train/grad_norm": float(grad_norm),
                            "train/epoch": epoch,
                            "train/samples_per_second": throughput.samples_per_second,
                            "train/total_samples": state["total_samples"],
                            "time/elapsed_seconds": elapsed,
                            **{
                                f"train/{key}": value
                                for key, value in log_composition_metrics.items()
                            },
                        },
                        step=state["global_step"],
                    )
                interval_loss = 0.0
                interval_coordinate_loss = 0.0
                interval_composition_loss = 0.0
                interval_composition_totals.zero_()
                interval_joint_totals.zero_()
                interval_steps = 0
            should_validate = (
                training.validate_interval > 0
                and state["global_step"] % training.validate_interval == 0
            )
            if should_validate:
                paused = time.monotonic()
                valid_loss, validation_throughput, validation_composition = validate(
                    model,
                    valid_loader,
                    training,
                    device,
                    epoch=epoch,
                    global_step=state["global_step"],
                    rank=rank,
                    ema=ema,
                )
                if rank == 0:
                    tracker.log(
                        {
                            "validation/loss": valid_loss,
                            "validation/samples_per_second": (
                                validation_throughput.samples_per_second
                            ),
                            "train/epoch": epoch,
                            **{
                                f"validation/{key}": value
                                for key, value in validation_composition.items()
                            },
                        },
                        step=state["global_step"],
                    )
                train_meter.exclude(time.monotonic() - paused)
            should_save = (
                training.save_interval > 0
                and state["global_step"] % training.save_interval == 0
            )
            if should_save or state["global_step"] >= training.max_steps:
                paused = time.monotonic()
                path = output_dir / f"checkpoint_step{state['global_step']:08d}.pt"
                _save_checkpoint(
                    path, model, optimizer, scaler, architecture, training, state, ema
                )
                tracker.log_checkpoint(path, state["global_step"])
                dist.barrier()
                train_meter.exclude(time.monotonic() - paused)
            if state["global_step"] >= training.max_steps:
                if rank == 0:
                    log_info("Finished Training")
                tracker.finish()
                return
        state["epoch"] = epoch + 1
        state["groups_in_epoch"] = 0
        if training.validate_at_epoch_end:
            valid_loss, validation_throughput, validation_composition = validate(
                model,
                valid_loader,
                training,
                device,
                epoch=epoch,
                global_step=state["global_step"],
                rank=rank,
                ema=ema,
            )
            if rank == 0:
                tracker.log(
                    {
                        "validation/loss": valid_loss,
                        "validation/samples_per_second": (
                            validation_throughput.samples_per_second
                        ),
                        "train/epoch": epoch,
                        **{
                            f"validation/{key}": value
                            for key, value in validation_composition.items()
                        },
                    },
                    step=state["global_step"],
                )
        if training.save_at_epoch_end:
            path = output_dir / f"checkpoint_epoch{epoch:04d}.pt"
            _save_checkpoint(
                path, model, optimizer, scaler, architecture, training, state, ema
            )
            tracker.log_checkpoint(path, state["global_step"])
            dist.barrier()
    tracker.finish()
    if rank == 0:
        log_info("Finished Training")
