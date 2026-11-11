"""MOF coordinate diffusion and checkpoint loading."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .checkpoint import load_checkpoint, load_model_weights
from .model.diffusion import DDPM
from .model.lattice import center_periodic, complete_lattice, initialize_lattice
from .model.noise import DiffNoise, TimeStepSampler
from .network import MOFNetwork
from .structures import MOFStructureEvaluator, compact_graphs_match


class MOFSampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.net = MOFNetwork(config)
        self.diffusion_noise = DiffNoise(config)
        self.diffusion_process = DDPM(self.diffusion_noise.alphas_cumprod, config)
        self.time_sampler = TimeStepSampler(config.num_timesteps)
        self.converter = MOFStructureEvaluator(config.output_dir)
        self._sample_dump_batch_idx = 0
        self._load_checkpoint(config.checkpoint_path)

    def _load_checkpoint(self, checkpoint_path):
        load_model_weights(self, load_checkpoint(checkpoint_path), checkpoint_path)

    @staticmethod
    def _prepare_mof_masks(batch):
        # Preserve the checkpoint launcher's RNG stream: it made two unused
        # random draws before initializing every all-MOF batch.
        token_id = batch["token_id"]
        torch.rand_like(token_id.unsqueeze(-1), dtype=torch.float)
        torch.rand_like(token_id, dtype=torch.float)

        node_count = token_id.size(1)
        batch["non_atom_mask"] = (
            torch.arange(node_count, device=token_id.device)[None]
            >= batch["num_atoms"][:, None]
        )

    def _initialize_positions(self, batch):
        initialize_lattice(batch, self.config.diff_init_lattice_size_factor)

    @torch.inference_mode()
    def sample(self, batch):
        """Draw a sample, then optionally denoise that sampled state again.

        Recycling restarts the clocks, not the state: positions (including the
        lattice) and sampled composition are passed directly into the next
        reverse trajectory. Gaussian atom/bond states retain their continuous
        normalized values; neither targets nor fresh forward noise are used.
        """
        self._prepare_mof_masks(batch)
        padding_mask = batch["token_id"].eq(0)
        self._initialize_positions(batch)
        clean_mask = padding_mask.clone()
        original = center_periodic(batch, padding_mask).clone()
        batch["pos"] = self.diffusion_noise.get_sampling_start(
            batch["init_pos"], batch["non_atom_mask"], batch["is_stable_periodic"]
        )
        batch["pos"] = torch.where(clean_mask.unsqueeze(-1), original, batch["pos"])
        batch["pos"] = complete_lattice(batch["pos"], batch)
        batch["pos"] = center_periodic(batch, padding_mask)
        self.net.initialize_composition_sampling(batch)

        for recycle_index in range(self.config.num_recycles + 1):
            self.diffusion_process.reset()
            self._denoise(batch, clean_mask, original, padding_mask, recycle_index)

        predicted = batch["pos"].clone()
        return {
            "pred_pos": predicted,
            "orig_pos": original,
            "pred_mof_cell_atom_types": batch.get("mof_comp_sampled_cell_atom_types"),
            "pred_mof_bb_atom_ids": batch.get("mof_comp_atom_ids"),
            "pred_mof_bb_bond_adj": batch.get("mof_comp_noised_bond_adj"),
        }

    def _denoise(self, batch, clean_mask, original, padding_mask, recycle_index):
        """Run one full reverse schedule on the current joint sampled state."""
        graph_count = batch["pos"].size(0)
        for step in tqdm(
            self.diffusion_process.timesteps.tolist(),
            disable=torch.distributed.get_rank() != 0,
            desc=(
                f"Denoising pass {recycle_index + 1}/{self.config.num_recycles + 1}"
                if self.config.num_recycles else None
            ),
        ):
            self.net.set_composition_sampling_time(batch, step)
            time = self.time_sampler.get_continuous_time_step(
                step, graph_count, device=batch["pos"].device, dtype=batch["pos"].dtype
            )
            time = time[:, None].repeat(1, batch["pos"].size(1))
            time = time.masked_fill(clean_mask, 0.0)
            current = batch["pos"].clone()
            indices = torch.full_like(batch["token_id"], step).masked_fill(clean_mask, 0)
            batch["sqrt_one_minus_alphas_cumprod_t"] = self.diffusion_noise._extract(
                self.diffusion_noise.sqrt_one_minus_alphas_cumprod,
                indices,
                batch["pos"].shape,
            )
            if getattr(self.net, "joint_composition", False):
                network_output = self.net(batch, time, clean_mask, compute_joint_objectives=False)
            else:
                network_output = self.net(batch, time, clean_mask)
            prediction = network_output["noise_pred"]
            self.net.composition_sample_step(batch, network_output, step)
            epsilon = self.diffusion_noise.get_noise(
                batch["pos"], batch["non_atom_mask"], batch["is_stable_periodic"]
            )
            batch["pos"] = self.diffusion_process.sample_step(
                current,
                batch["init_pos"],
                prediction,
                epsilon,
                step,
            )
            batch["pos"] = torch.where(clean_mask.unsqueeze(-1), original, batch["pos"])
            batch["pos"] = complete_lattice(batch["pos"], batch)
            batch["pos"] = center_periodic(batch, padding_mask).detach()

    @torch.inference_mode()
    def sample_and_match(self, batch, sample_time_index: int = 0):
        original_positions = batch["pos"].clone()
        original_cell = batch["cell"].clone()
        batch["pos"] = torch.zeros_like(batch["pos"])
        batch["cell"] = torch.zeros_like(batch["cell"])
        sampled_output = self.sample(batch)
        results = self.converter.convert_and_match(
            batch,
            original_positions,
            sampled_output.get("pred_mof_cell_atom_types"),
            sample_time_index=sample_time_index,
        )
        if self.net.gaussian_composition or self.net.lgflow_composition:
            atoms = sampled_output["pred_mof_bb_atom_ids"]
            bonds = sampled_output["pred_mof_bb_bond_adj"]
            self._dump_composition(batch, atoms, bonds, sample_time_index)
            for i, result in enumerate(results):
                atom_mask = batch["mof_comp_atom_loss_mask"][i]
                bond_mask = torch.triu(batch["mof_comp_bond_loss_mask"][i], diagonal=1)
                atom_correct = atoms[i][atom_mask].eq(batch["mof_bb_atom_types"][i][atom_mask])
                predicted_bonds = bonds[i][bond_mask].bool()
                target_bonds = batch["mof_bb_bond_adj"][i][bond_mask].bool()
                bond_correct = predicted_bonds.eq(target_bonds)
                true_positive = (predicted_bonds & target_bonds).sum()
                result.update(
                    composition_atom_accuracy=float(atom_correct.float().mean()),
                    composition_bond_accuracy=float(bond_correct.float().mean()),
                    composition_bond_precision=float(true_positive / predicted_bonds.sum().clamp_min(1)),
                    composition_bond_recall=float(true_positive / target_bonds.sum().clamp_min(1)),
                    composition_exact=int(atom_correct.all() & bond_correct.all()),
                    composition_graph_match=int(compact_graphs_match(
                        atoms[i], bonds[i], batch["mof_bb_atom_types"][i],
                        batch["mof_bb_bond_adj"][i], atom_mask,
                    )),
                )
        self._dump_metrics(batch, results, sample_time_index)
        batch["pos"] = original_positions
        batch["cell"] = original_cell
        metrics: dict[str, list] = {}
        for result in results:
            for key, value in result.items():
                metrics.setdefault(key, []).append(value)
        return {
            key: torch.tensor(value, device=batch["pos"].device)[:, None]
            for key, value in metrics.items()
        }

    def _dump_composition(self, batch, atoms, bonds, sample_time_index):
        """Retain generated block graphs for independent validity/diversity checks."""
        from .structures import compact_composition_record

        process = self.net.mof_lgflow if self.net.lgflow_composition else self.net.mof_gaussian_diffusion
        numbers = process.vocab.decode(atoms)
        for index, sample_id in enumerate(batch["idx"].tolist()):
            record = compact_composition_record(
                numbers[index], bonds[index], batch["mof_bb_atom_mask"][index],
                batch["mof_bb_occ_atom_index"][index], batch["mof_bb_occ_mask"][index],
            )
            record.update(
                sample_id=sample_id, sample_time_index=sample_time_index,
                num_recycles=self.config.num_recycles,
            )
            path = Path(self.config.output_dir) / f"composition_{sample_id}_{sample_time_index}.json"
            path.write_text(json.dumps(record) + "\n")

    def _dump_metrics(self, batch, results, sample_time_index: int = 0):
        rank = torch.distributed.get_rank()
        output = Path(f"{self.config.metric_path}.rank{rank}.jsonl")
        output.parent.mkdir(parents=True, exist_ok=True)
        positions = batch["pos"].to(torch.float64).cpu()
        cells = batch["cell"].to(torch.float64).cpu()
        with output.open("a") as handle:
            for index, result in enumerate(results):
                atom_count = int(batch["num_atoms"][index])
                record = {
                    "sample_id": int(batch["idx"][index]),
                    "rank": rank,
                    "batch_idx": self._sample_dump_batch_idx,
                    "sample_time_index": sample_time_index,
                    "num_recycles": self.config.num_recycles,
                    "num_atoms": atom_count,
                    "pos_hash": hashlib.md5(
                        np.ascontiguousarray(
                            positions[index, :atom_count].numpy()
                        ).tobytes()
                    ).hexdigest(),
                    "cell_hash": hashlib.md5(
                        np.ascontiguousarray(cells[index].numpy()).tobytes()
                    ).hexdigest(),
                }
                record.update(
                    {
                        key: float(value) if np.isscalar(value) else value
                        for key, value in result.items()
                    }
                )
                handle.write(json.dumps(record, default=str) + "\n")
        self._sample_dump_batch_idx += 1
