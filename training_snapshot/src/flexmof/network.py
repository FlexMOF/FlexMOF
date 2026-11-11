"""The single checkpoint-compatible neural network used by FlexMOF."""

from __future__ import annotations

import torch
from torch import nn

from .model.composition import (
    DiGressUniformGraphDiffusion,
    GraphDiffusionInput,
    MOFCompositionHead,
)
from .model.cell_laplacian import CellLaplacianEncoding
from .model.decoder import EquivariantDecoder
from .model.joint_composition import JointComposition
from .model.gaussian_composition import AtomVocabulary, GaussianGraphDDPM, GaussianCompositionHead
from .model.lgflow_composition import LGFlowComposition, MOFLatentFlowHead
from .model.embedding import SequenceEmbedding
from .model.geometry import EquivariantVectorOutput
from .model.invariant import StructureEncoder
from .model.periodic import CellExpander
from .model.sequence import SequenceEncoder


class MOFNetwork(nn.Module):
    """The ``seq-dit-geom`` coordinate denoiser.

    Attribute names match the checkpoint tensors. Only coordinate-generation
    modules are constructed.
    """

    lgflow_composition = False
    joint_composition = False

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cell_expander = CellExpander(
            config.pbc_cutoff,
            config.pbc_expanded_token_cutoff,
            config.pbc_expanded_num_cell_per_direction,
            distance_chunk_size=config.pbc_distance_chunk_size,
            sparse_edge_chunk_size=config.pbc_sparse_edge_chunk_size,
            activation_checkpoint_edges=config.pbc_activation_checkpoint_edges,
        )
        self.joint_composition = bool(
            getattr(config, "mof_joint_composition", False)
        )
        self.embedding = SequenceEmbedding(config)
        # The joint model's composition autoencoder encodes through this same
        # trunk, which needs the graph-path norm. Without joint composition the
        # encoder is built exactly as before, so its state_dict keys are
        # unchanged and every pre-fusion checkpoint still loads strictly.
        self.seq_encoder = SequenceEncoder(config, graph_norm=self.joint_composition)
        self.structure_encoder = StructureEncoder(config)
        self.structure_decoder = EquivariantDecoder(config)
        self.periodic_noise_head = EquivariantVectorOutput(config.embedding_dim)
        self.mlp_w = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim, bias=False),
            nn.SiLU(),
            nn.Linear(config.embedding_dim, 1, bias=False),
        )
        self.layer_norm = nn.LayerNorm(config.embedding_dim)
        self.gaussian_composition = (
            config.use_mof_composition_diffusion
            and config.mof_composition_diffusion_strategy == "gaussian_ddpm"
            and not self.joint_composition
        )
        self.lgflow_composition = (
            config.use_mof_composition_diffusion
            and config.mof_composition_diffusion_strategy == "lgflow"
            and not self.joint_composition
        )
        if self.joint_composition:
            if not config.use_mof_composition_diffusion:
                raise ValueError(
                    "Joint composition requires use_mof_composition_diffusion"
                )
            if config.mof_composition_diffusion_strategy != "joint_lgflow":
                raise ValueError(
                    "Joint composition requires mof_composition_diffusion_strategy=joint_lgflow"
                )
            self.mof_joint = JointComposition(config, self.seq_encoder)
            self.cell_laplacian = (
                CellLaplacianEncoding(config)
                if config.mof_joint_csp_laplacian_pe
                else None
            )
        elif self.lgflow_composition:
            self.mof_lgflow = LGFlowComposition(config)
            self.mof_lgflow_seq_head = MOFLatentFlowHead(config)
            self.mof_lgflow_3d_head = MOFLatentFlowHead(config)
        elif self.gaussian_composition:
            self.mof_gaussian_diffusion = GaussianGraphDDPM(config)
            self.mof_gaussian_seq_head = GaussianCompositionHead(config)
            self.mof_gaussian_3d_head = GaussianCompositionHead(config)
        elif config.use_mof_composition_diffusion:
            if config.mof_composition_diffusion_strategy != "digress_uniform":
                raise NotImplementedError(
                    "Composition strategy must be digress_uniform, gaussian_ddpm or lgflow"
                )
            self.mof_graph_diffusion = DiGressUniformGraphDiffusion(
                config.mof_composition_num_atom_classes,
                config.mof_composition_num_bond_classes,
                config.mof_composition_num_timesteps,
                list(
                    range(
                        1,
                        min(config.mof_composition_num_atom_classes, 119),
                    )
                ),
            )
            self.mof_composition_seq_head = MOFCompositionHead(
                config, config.embedding_dim
            )
            self.mof_composition_3d_head = MOFCompositionHead(
                config, config.embedding_dim
            )

    @staticmethod
    def _set_masked_types(batch):
        batch["masked_token_type"] = batch["token_id"]
        MOFNetwork._set_node_type_edges(batch)

    @staticmethod
    def _set_node_type_edges(batch):
        token_i = batch["masked_token_type"][:, :, None, None]
        token_j = batch["masked_token_type"][:, None, :, None]
        batch["node_type_edge"] = token_i * 160 + token_j

    def _expand_periodic_cells(self, batch):
        graph_options = {}
        if not self.config.mof_disable_graphormer_2d:
            graph_options = dict(
                edge_index=batch.get("mof_comp_edge_index", batch["edge_index"]).transpose(1, 2),
                target_cell_index=batch.get("mof_comp_target_cell_index", batch["target_cell_index"]),
                edge_attr=batch.get("mof_comp_edge_attr", batch["edge_attr"]),
                num_edges=batch.get("mof_comp_num_edges", batch["num_edges"]),
            )
        expansion = (
            self.cell_expander.expand_per_atom_streaming
            if self.config.use_per_atom_outcell_index
            else self.cell_expander.expand
        )
        expansion_options = {}
        if self.config.use_per_atom_outcell_index:
            expansion_options["central_cell_graph"] = self.config.mof_unwrap_linkers_and_center
        else:
            expansion_options["use_local_attention"] = False
        pbc = batch["pbc"]
        if self.config.mof_disable_pbc:
            pbc = torch.zeros_like(pbc)
        return expansion(
            batch["pos"],
            batch["init_pos"],
            pbc,
            batch["num_atoms"],
            batch["masked_token_type"],
            batch["cell"],
            batch["node_type_edge"],
            use_grad=False,
            **graph_options,
            **expansion_options,
        )

    def _composition_edge_mask(self, batch):
        atom_mask = batch["mof_bb_atom_mask"].bool()
        edge_mask = atom_mask.unsqueeze(-1) & atom_mask.unsqueeze(-2)
        edge_mask &= batch["mof_bb_mask"].bool().unsqueeze(-1).unsqueeze(-1)
        diagonal = torch.eye(
            atom_mask.size(-1), device=atom_mask.device, dtype=torch.bool
        ).view(1, 1, atom_mask.size(-1), atom_mask.size(-1))
        return edge_mask & ~diagonal

    def _position_to_composition_index(self, position_index):
        offset = max(
            0, self.config.num_timesteps - self.config.mof_composition_num_timesteps
        )
        if self.gaussian_composition or self.lgflow_composition:
            return (position_index - offset + 1).clamp(
                min=0, max=self.config.mof_composition_num_timesteps
            )
        return (position_index - offset).clamp(
            min=0, max=self.config.mof_composition_num_timesteps - 1
        )

    def prepare_composition_training(self, batch, position_time, *, augment_time=True):
        if not self.config.use_mof_composition_diffusion:
            return
        if self.joint_composition:
            # Nothing is noised into the batch. The latent flow draws its own
            # time and its own noise inside the forward pass, over latents the
            # encoder produces there, so no noisy composition ever reaches the
            # CSP input or the autoencoder's encoder.
            batch["mof_comp_atom_loss_mask"] = batch["mof_bb_atom_mask"].bool()
            batch["mof_comp_bond_loss_mask"] = self._composition_edge_mask(batch)
            return
        position_index = (
            (
                position_time.float().clamp(0.0, 1.0)
                * max(1, self.config.num_timesteps - 1)
            )
            .round()
            .long()
        )
        composition_index = self._position_to_composition_index(position_index)
        if self.lgflow_composition:
            batch.pop("mof_comp_self_condition", None)
            probability = self.config.mof_composition_independent_time_probability
            if augment_time and probability > 0:
                independent = torch.randint(1, self.config.mof_composition_num_timesteps + 1,
                    composition_index.shape, device=composition_index.device)
                composition_index = torch.where(torch.rand_like(position_time) < probability, independent, composition_index)
            batch["mof_comp_time"] = composition_index
            batch["mof_comp_atom_loss_mask"] = batch["mof_bb_atom_mask"].bool()
            batch["mof_comp_bond_loss_mask"] = self._composition_edge_mask(batch)
            clean = self.mof_lgflow.encode_clean(batch)
            state, velocity = self.mof_lgflow.q_sample(clean, composition_index, batch["mof_comp_atom_loss_mask"])
            batch["mof_comp_latent_state"] = state
            batch["mof_comp_latent_target"] = velocity
            self._update_lgflow_tokens(batch)
            return
        if self.gaussian_composition:
            probability = self.config.mof_composition_independent_time_probability
            if augment_time and probability > 0:
                independent = torch.randint(
                    1, self.config.mof_composition_num_timesteps + 1,
                    composition_index.shape, device=composition_index.device,
                )
                selected = torch.rand(composition_index.shape, device=composition_index.device) < probability
                composition_index = torch.where(selected, independent, composition_index)
            process = self.mof_gaussian_diffusion
            batch["mof_comp_time"] = composition_index
            batch["mof_comp_atom_loss_mask"] = batch["mof_bb_atom_mask"].bool()
            batch["mof_comp_bond_loss_mask"] = self._composition_edge_mask(batch)
            batch["mof_comp_atom_state"] = process.q_sample(
                process.normalize_atoms(batch["mof_bb_atom_types"]),
                composition_index, batch["mof_comp_atom_loss_mask"],
            )
            batch["mof_comp_bond_state"] = process.q_sample(
                process.normalize_bonds(batch["mof_bb_bond_adj"]),
                composition_index, batch["mof_comp_bond_loss_mask"], bonds=True,
            )
            self._update_gaussian_tokens(batch)
            return
        last = max(1, self.config.mof_composition_num_timesteps - 1)
        composition_time = composition_index.float() / last
        alpha_bar = 1.0 - composition_time
        state = self.mof_graph_diffusion.q_sample(
            GraphDiffusionInput(
                node_labels=batch["mof_bb_atom_types"]
                .long()
                .clamp(
                    min=0,
                    max=self.config.mof_composition_num_atom_classes - 1,
                ),
                node_mask=batch["mof_bb_atom_mask"].bool(),
                edge_labels=batch["mof_bb_bond_adj"]
                .long()
                .clamp(
                    min=0,
                    max=self.config.mof_composition_num_bond_classes - 1,
                ),
                edge_mask=self._composition_edge_mask(batch),
                time_step=composition_time,
                alpha_bar=alpha_bar,
            )
        )
        batch["mof_comp_noised_atom_types"] = state.node_t
        batch["mof_comp_noised_bond_adj"] = state.edge_t
        batch["mof_comp_time"] = state.time_index
        batch["mof_comp_alpha_bar"] = state.alpha_bar
        batch["mof_comp_atom_loss_mask"] = state.node_loss_mask
        batch["mof_comp_bond_loss_mask"] = state.edge_loss_mask

    def initialize_composition_sampling(self, batch):
        if not self.config.use_mof_composition_diffusion:
            return False
        if self.joint_composition:
            # Joint generation is strictly two-stage and runs from the sampler,
            # not from the coordinate loop's per-step composition hooks. The
            # composition is already clean and fixed by the time the CSP runs.
            return False
        node_mask = batch["mof_bb_atom_mask"].bool()
        edge_mask = self._composition_edge_mask(batch)
        if self.lgflow_composition:
            batch.pop("mof_comp_self_condition", None)
            batch["mof_comp_latent_state"] = self.mof_lgflow.sample_prior(node_mask)
            batch["mof_comp_atom_loss_mask"] = node_mask
            batch["mof_comp_bond_loss_mask"] = edge_mask
            batch["mof_comp_time"] = torch.full((node_mask.size(0),), self.config.mof_composition_num_timesteps,
                                                device=node_mask.device, dtype=torch.long)
            batch["mof_comp_sampling_mode"] = True
            self._update_lgflow_tokens(batch)
            return True
        if self.gaussian_composition:
            node_t, edge_t = self.mof_gaussian_diffusion.sample_prior(node_mask, edge_mask)
            batch["mof_comp_atom_state"] = node_t
            batch["mof_comp_bond_state"] = edge_t
        else:
            node_t, edge_t = self.mof_graph_diffusion.sample_prior(node_mask, edge_mask)
        batch["mof_comp_noised_atom_types"] = node_t
        batch["mof_comp_noised_bond_adj"] = edge_t
        batch["mof_comp_atom_loss_mask"] = node_mask
        batch["mof_comp_bond_loss_mask"] = edge_mask
        batch["mof_comp_time"] = torch.full(
            (node_mask.size(0),),
            self.config.mof_composition_num_timesteps - (not self.gaussian_composition),
            device=node_mask.device,
            dtype=torch.long,
        )
        batch["mof_comp_alpha_bar"] = torch.zeros(
            node_mask.size(0), device=node_mask.device
        )
        batch["mof_comp_sampling_mode"] = True
        if self.gaussian_composition:
            self._update_gaussian_tokens(batch)
        return True

    def _update_lgflow_tokens(self, batch):
        mask = batch["mof_comp_atom_loss_mask"]
        ids, bonds = self.mof_lgflow.decode(batch["mof_comp_latent_state"], mask)
        batch["mof_comp_atom_ids"] = ids
        batch["mof_comp_noised_atom_types"] = (self.mof_lgflow.vocab.decode(ids) + 1).masked_fill(~mask, 0)
        batch["mof_comp_noised_bond_adj"] = bonds

    def _update_gaussian_tokens(self, batch):
        """Discrete proxies for the unchanged backbone; keep DDPM states continuous."""
        process = self.mof_gaussian_diffusion
        ids = process.atom_ids(batch["mof_comp_atom_state"])
        batch["mof_comp_atom_ids"] = ids.masked_fill(~batch["mof_comp_atom_loss_mask"], 0)
        batch["mof_comp_noised_atom_types"] = (process.vocab.decode(ids) + 1).masked_fill(
            ~batch["mof_comp_atom_loss_mask"], 0
        )
        batch["mof_comp_noised_bond_adj"] = process.bond_labels(
            batch["mof_comp_bond_state"]
        ).masked_fill(~batch["mof_comp_bond_loss_mask"], 0)

    def set_composition_sampling_time(self, batch, position_step):
        if not self.config.use_mof_composition_diffusion:
            return
        if self.joint_composition:
            return
        composition_step = self._position_to_composition_index(
            torch.tensor(position_step, device=batch["token_id"].device)
        )
        batch["mof_comp_time"] = composition_step.expand(batch["token_id"].size(0))
        if self.gaussian_composition or self.lgflow_composition:
            return
        last = max(1, self.config.mof_composition_num_timesteps - 1)
        batch["mof_comp_alpha_bar"] = 1.0 - batch["mof_comp_time"].float() / last

    def composition_sample_step(self, batch, output, position_step):
        if not self.config.use_mof_composition_diffusion:
            return
        if self.joint_composition:
            return
        stride = -self.config.num_timesteps_stepsize
        composition_t = int(
            self._position_to_composition_index(torch.tensor(position_step))
        )
        composition_s = int(
            self._position_to_composition_index(torch.tensor(position_step - stride))
        )
        if composition_t == composition_s:
            self._refresh_sampled_cell_atoms(batch)
            return
        prefix = self.config.mof_composition_sampling_head
        if self.lgflow_composition:
            time = torch.full_like(batch["mof_comp_time"], composition_t)
            previous = torch.full_like(time, composition_s)
            if self.config.mof_lgflow_head_architecture == "dit":
                scaled_time = time.float().view(-1,1,1,1) / self.config.mof_composition_num_timesteps
                batch["mof_comp_self_condition"] = (batch["mof_comp_latent_state"] -
                    scaled_time * output[f"mof_comp_{prefix}_latent_velocity"]).detach()
            batch["mof_comp_latent_state"] = self.mof_lgflow.p_sample(
                batch["mof_comp_latent_state"], output[f"mof_comp_{prefix}_latent_velocity"],
                time, previous, batch["mof_comp_atom_loss_mask"])
            self._update_lgflow_tokens(batch)
            self._refresh_sampled_cell_atoms(batch)
            return
        if self.gaussian_composition:
            process = self.mof_gaussian_diffusion
            time = torch.full_like(batch["mof_comp_time"], composition_t)
            previous = torch.full_like(time, composition_s)
            for kind in ("atom", "bond"):
                batch[f"mof_comp_{kind}_state"] = process.p_sample(
                    batch[f"mof_comp_{kind}_state"], output[f"mof_comp_{prefix}_{kind}_x0"],
                    time, previous, batch[f"mof_comp_{kind}_loss_mask"], bonds=kind == "bond",
                )
            self._update_gaussian_tokens(batch)
            self._refresh_sampled_cell_atoms(batch)
            return
        node_logits = output[f"mof_comp_{prefix}_atom_logits"]
        edge_logits = output[f"mof_comp_{prefix}_bond_logits"]
        last = max(1, self.config.mof_composition_num_timesteps - 1)
        alpha_t = torch.full(
            (batch["token_id"].size(0),),
            1.0 - composition_t / last,
            device=batch["token_id"].device,
        )
        alpha_s = torch.full_like(alpha_t, 1.0 - composition_s / last)
        node_s, edge_s, _, _ = self.mof_graph_diffusion.p_sample(
            batch["mof_comp_noised_atom_types"],
            batch["mof_comp_noised_bond_adj"],
            node_logits,
            edge_logits,
            batch["mof_comp_atom_loss_mask"],
            batch["mof_comp_bond_loss_mask"],
            alpha_t,
            alpha_s,
        )
        batch["mof_comp_noised_atom_types"] = node_s.detach()
        batch["mof_comp_noised_bond_adj"] = edge_s.detach()
        batch["mof_comp_alpha_bar"] = alpha_s
        self._refresh_sampled_cell_atoms(batch)

    @staticmethod
    def _cell_occurrence_maps(batch):
        cache = batch.get("_mof_cell_occurrence_maps")
        if cache is not None:
            return cache
        occurrence = batch["mof_bb_occ_atom_index"].long() # batch_size x num_unique_bbs x num_repetitions x num_atoms
        occurrence_mask = batch["mof_bb_occ_mask"].bool() # batch_size x num_unique_bbs x num_repetitions
        atom_mask = batch["mof_bb_atom_mask"].bool() # batch_size x num_unique_bbs x num_atoms
        batch_size, num_bbs, num_repetitions, num_local_atoms = occurrence.shape
        max_nodes = batch["token_id"].size(1)
        valid = (
            occurrence_mask.unsqueeze(-1)
            & atom_mask.unsqueeze(2)
            & occurrence.ge(0)
            & occurrence.lt(max_nodes)
        )
        batch_index = torch.arange(batch_size, device=occurrence.device).view(
            -1, 1, 1, 1
        )
        bb_index = torch.arange(num_bbs, device=occurrence.device).view(1, -1, 1, 1)
        repetition = torch.arange(num_repetitions, device=occurrence.device).view(
            1, 1, -1, 1
        )
        local = torch.arange(num_local_atoms, device=occurrence.device).view(
            1, 1, 1, -1
        )
        selected_batch = batch_index.expand_as(occurrence)[valid]
        destination = occurrence[valid]
        cell_bb = torch.full(
            (batch_size, max_nodes), -1, device=occurrence.device, dtype=torch.long
        )
        cell_occurrence = torch.full_like(cell_bb, -1)
        cell_local = torch.full_like(cell_bb, -1)
        cell_bb[selected_batch, destination] = bb_index.expand_as(occurrence)[valid] # batch_size x num_all_atoms -> bb_index 
        cell_occurrence[selected_batch, destination] = (
            bb_index * num_repetitions + repetition
        ).expand_as(occurrence)[valid] # batch_size x num_all_atoms -> flatten_repetition_id
        cell_local[selected_batch, destination] = local.expand_as(occurrence)[valid] # batch_size x num_all_atoms -> atom_id_in_bb
        cache = (cell_bb, cell_occurrence, cell_local)
        batch["_mof_cell_occurrence_maps"] = cache
        return cache

    def _refresh_sampled_cell_atoms(self, batch, atoms=None):
        cell_bb, _, cell_local = self._cell_occurrence_maps(batch)
        real_atoms = (
            torch.arange(batch["token_id"].size(1), device=cell_bb.device)[None]
            < batch["num_atoms"][:, None]
        )
        covered = cell_bb.ge(0) & real_atoms
        batch_index = torch.arange(cell_bb.size(0), device=cell_bb.device)[:, None]
        source = batch["mof_comp_noised_atom_types"] if atoms is None else atoms
        gathered = source[
            batch_index, cell_bb.clamp_min(0), cell_local.clamp_min(0)
        ]
        sampled = torch.zeros_like(batch["token_id"])
        sampled = torch.where(covered, gathered.to(sampled.dtype), sampled)
        if not (covered | ~real_atoms).all():
            raise ValueError("Composition targets do not cover every MOF atom")
        batch["mof_comp_sampled_cell_atom_types"] = sampled

    @staticmethod
    def _target_cell_offsets(positions, cell, edges):
        # Composition bonds join atoms of the same linker occurrence in the
        # original cell. The previous nearest-image search discarded its
        # result with zero_(); no lattice solve or image search is needed.
        return torch.zeros(edges.size(0), 3, device=positions.device, dtype=torch.long)

    def _build_composition_edges(self, batch, cell_occurrence, cell_local, bonds=None):
        edge_lists, edge_offsets = [], []
        max_edges = 0
        noised_bonds = batch["mof_comp_noised_bond_adj"] if bonds is None else bonds
        cell_bb, _, _ = self._cell_occurrence_maps(batch)
        for sample_index in range(batch["token_id"].size(0)):
            atom_count = int(batch["num_atoms"][sample_index])
            occurrence = cell_occurrence[sample_index, :atom_count]
            local = cell_local[sample_index, :atom_count]
            bb = cell_bb[sample_index, :atom_count]
            same = occurrence[:, None].eq(occurrence[None]) & occurrence[:, None].ge(0)
            local_bonds = noised_bonds[
                sample_index,
                bb[:, None].clamp_min(0),
                local[:, None].clamp_min(0),
                local[None].clamp_min(0),
            ]
            adjacency = torch.triu(same & local_bonds.gt(0), diagonal=1)
            edges = adjacency.nonzero(as_tuple=False)
            edge_lists.append(edges)
            edge_offsets.append(
                self._target_cell_offsets(
                    batch["pos"][sample_index, :atom_count],
                    batch["cell"][sample_index],
                    edges,
                )
            )
            max_edges = max(max_edges, edges.size(0))

        edge_index = torch.zeros(
            batch["token_id"].size(0),
            max_edges,
            2,
            device=batch["token_id"].device,
            dtype=torch.long,
        )
        target_cell = torch.zeros(
            batch["token_id"].size(0),
            max_edges,
            3,
            device=batch["token_id"].device,
            dtype=torch.long,
        )
        edge_attr = torch.zeros(
            batch["token_id"].size(0),
            max_edges,
            1,
            device=batch["token_id"].device,
            dtype=torch.long,
        )
        edge_counts = torch.zeros(
            batch["token_id"].size(0),
            device=batch["token_id"].device,
            dtype=torch.long,
        )
        for sample_index, edges in enumerate(edge_lists):
            count = edges.size(0)
            edge_counts[sample_index] = count
            edge_index[sample_index, :count] = edges
            target_cell[sample_index, :count] = edge_offsets[sample_index]
            edge_attr[sample_index, :count, 0] = 1
        batch["mof_comp_edge_index"] = edge_index
        batch["mof_comp_target_cell_index"] = target_cell
        batch["mof_comp_edge_attr"] = edge_attr
        batch["mof_comp_num_edges"] = edge_counts

    def _apply_composition_to_backbone(self, batch):
        for key in (
            "mof_comp_edge_index",
            "mof_comp_target_cell_index",
            "mof_comp_edge_attr",
            "mof_comp_num_edges",
        ):
            batch.pop(key, None)
        if self.joint_composition:
            # The CSP never sees a noisy composition in the joint model. During
            # training it reads the target composition; during generation it
            # reads the graph stage one already decoded, which the sampler has
            # written into the batch. Either way the tokens are clean and only
            # the bond edges are rebuilt from the occurrence maps, so training
            # and sampling construct the same CSP input.
            _, cell_occurrence, cell_local = self._cell_occurrence_maps(batch)
            self._build_composition_edges(
                batch, cell_occurrence, cell_local, bonds=batch["mof_bb_bond_adj"]
            )
            self._refresh_sampled_cell_atoms(
                batch, atoms=AtomVocabulary(self.config.mof_composition_atom_vocab).decode(
                    batch["mof_bb_atom_types"]
                ) + 1
            )
            return
        if "mof_comp_time" not in batch or (
            batch["mof_comp_time"].eq(0).all()
            and not batch.get("mof_comp_sampling_mode", False)
            and not self.lgflow_composition
        ):
            self._refresh_sampled_cell_atoms(batch)
            return
        cell_bb, cell_occurrence, cell_local = self._cell_occurrence_maps(batch)
        real_atoms = (
            torch.arange(batch["token_id"].size(1), device=cell_bb.device)[None]
            < batch["num_atoms"][:, None]
        )
        covered = cell_bb.ge(0) & real_atoms
        if not (covered | ~real_atoms).all():
            raise ValueError("Composition targets do not cover every MOF atom")
        batch_index = torch.arange(cell_bb.size(0), device=cell_bb.device)[:, None]
        sampled_atoms = batch["mof_comp_noised_atom_types"][
            batch_index, cell_bb.clamp_min(0), cell_local.clamp_min(0)
        ]
        remapped = self.gaussian_composition or self.lgflow_composition
        valid = sampled_atoms.ge(2 if remapped else 1) & sampled_atoms.le(
            119 if remapped else 118
        )
        sampled_tokens = torch.where(
            valid,
            sampled_atoms,
            torch.full_like(
                sampled_atoms, self.config.mof_composition_backbone_mask_token
            ),
        )
        batch["masked_token_type"] = torch.where(
            covered, sampled_tokens, batch["masked_token_type"]
        )
        self._build_composition_edges(batch, cell_occurrence, cell_local)
        self._refresh_sampled_cell_atoms(batch)

    def _composition_outputs(self, batch, padding_mask, sequence, scalar):
        if (
            not self.config.use_mof_composition_diffusion
            or "mof_comp_noised_atom_types" not in batch
        ):
            return {}
        if self.lgflow_composition:
            return {
                f"mof_comp_{name}_{key}": value
                for name, head, features in (
                    ("seq", self.mof_lgflow_seq_head, sequence.transpose(0, 1)),
                    ("3d", self.mof_lgflow_3d_head, scalar),
                )
                for key, value in head(features, batch).items()
            }
        if self.gaussian_composition:
            return {
                f"mof_comp_{name}_{key}": value
                for name, head, features in (
                    ("seq", self.mof_gaussian_seq_head, sequence.transpose(0, 1)),
                    ("3d", self.mof_gaussian_3d_head, scalar),
                )
                for key, value in head(features, batch).items()
            }
        inputs = (
            padding_mask,
            batch["mof_comp_noised_atom_types"],
            batch["mof_comp_noised_bond_adj"],
            batch["mof_comp_time"],
            batch,
        )
        sequence_output = self.mof_composition_seq_head(sequence, *inputs)
        final_output = self.mof_composition_3d_head(scalar, *inputs)
        result = {}
        result.update(
            {f"mof_comp_seq_{key}": value for key, value in sequence_output.items()}
        )
        result.update(
            {f"mof_comp_3d_{key}": value for key, value in final_output.items()}
        )
        return result

    def forward(self, batch, time_step, clean_mask, *, compute_joint_objectives=True):
        self._set_masked_types(batch)
        if self.config.use_mof_composition_diffusion:
            self._apply_composition_to_backbone(batch)
            self._set_node_type_edges(batch)
        periodic = self._expand_periodic_cells(batch)

        token_embedding, padding_mask, time_embedding, pair_features = self.embedding(
            batch, time_step, clean_mask
        )
        joint = None
        if self.joint_composition:
            joint = self.mof_joint.encode(batch)
            injection = self.mof_joint.node_injection(
                joint, batch, token_embedding.shape[1]
            )
            if injection is not None:
                token_embedding = token_embedding + injection.masked_fill(
                    padding_mask[..., None], 0
                )
            if self.cell_laplacian is not None:
                token_embedding = token_embedding + self.cell_laplacian(
                    batch, padding_mask
                )
        invariant, pair_features = self.seq_encoder(
            token_embedding.transpose(0, 1),
            pair_features,
            batch,
            periodic,
            padding_mask,
        )
        invariant = self.layer_norm(invariant)
        sequence_features = invariant
        invariant, attention_bias, attention_weights = self.structure_encoder(
            batch,
            invariant,
            time_embedding,
            padding_mask,
            pair_features,
            periodic,
            clean_mask,
        )
        scalar, vector = self.structure_decoder(
            batch,
            invariant.transpose(0, 1),
            attention_bias,
            padding_mask,
            periodic,
            time_embedding,
            attention_weights,
        )
        predicted = self.periodic_noise_head(scalar, vector)
        scale_shift = self.mlp_w(time_embedding)
        logit_bias = torch.logit(batch["sqrt_one_minus_alphas_cumprod_t"])
        scale = torch.sigmoid(scale_shift + logit_bias)
        predicted = scale * (batch["pos"] - batch["init_pos"]) + (1 - scale) * predicted
        result = {"noise_pred": predicted, "decoder_x_output": scalar}
        if joint is not None:
            if compute_joint_objectives:
                result.update(self._joint_composition_outputs(joint, batch))
        else:
            result.update(
                self._composition_outputs(batch, padding_mask, sequence_features, scalar)
            )
        return result

    def _joint_composition_outputs(self, joint, batch):
        """Reconstruction and latent-flow terms, with their gradient boundaries.

        Reconstruction keeps the differentiable logits path, so it trains the
        autoencoder adapters, the posterior, the decoder and the shared trunk.
        The flow term trains only the denoiser: its target is detached, and its
        conditioning never includes a clean graph or a trunk activation.
        """
        reconstruction, details, _ = self.mof_joint.reconstruction(joint)
        flow, _ = self.mof_joint.flow_loss(joint, batch)
        return {
            "mof_joint_reconstruction_loss": reconstruction,
            "mof_joint_flow_loss": flow,
            # The online latent scale is what makes the flow target comparable
            # across steps; logging it is how a run shows its normalization
            # settled rather than still drifting.
            "mof_joint_latent_scale": self.mof_joint.latent_normalization.scale.mean(),
            **{f"mof_joint_{name}": value for name, value in details.items()},
        }
