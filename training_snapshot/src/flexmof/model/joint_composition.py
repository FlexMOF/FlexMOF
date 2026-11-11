"""The fused composition stack: one trunk, three losses, two generation stages.

This is the joint model. It puts together the pieces the standalone runs
established -- a ``HydrogenAutoencoder`` over ``terminal_h_counts_v1`` graphs
and a ``JointLGFlow`` denoiser over that autoencoder's latents -- and encodes
through the CSP sequence trunk instead of a separate stack of its own.

Three losses meet here, and the gradient each one is allowed to carry is the
whole point of the design:

* **Reconstruction** runs the differentiable logits path, so it updates the
  autoencoder's input adapters, its posterior, its decoder, *and* the shared
  trunk.
* **Coordinate** loss reaches the shared trunk twice: directly, because the
  CSP pass runs through the same parameters, and through the autoencoder's
  posterior, because the latent injected into the CSP node features is not
  detached. The second path is what makes the composition encoder answerable
  for geometry rather than only for reconstruction.
* **Latent flow** trains the denoiser against a detached latent target. It
  never sees a clean graph, a coordinate, or any trunk activation, because at
  generation time none of those exist yet.

The denoiser's conditioning is deliberately narrow: the current noisy latent,
its own self-conditioning, and the sizes, repetition counts and linker roles
that are given as the generation request. Detaching a richer context would
only stop gradients, not stop the leak.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..hydrogen_fold import (
    cell_atom_core_index,
    expand_terminal_hydrogen_counts,
    fold_terminal_hydrogens,
)
from ..linker_data import laplacian_features
from .hydrogen_autoencoder import (
    HydrogenAEConfig,
    HydrogenAutoencoder,
    reconstruction_loss as hydrogen_reconstruction_loss,
)
from .joint_lgflow import JointLGFlow, JointLGFlowConfig

HYDROGEN_NUMBER = 1


def core_spectral_features(core_bonds, core_mask, frequencies):
    """Laplacian eigenpairs of each unpadded core graph.

    Solved per graph on the CPU, like the corpus preparation. A batched CUDA
    eigendecomposition can pick a different basis for repeated eigenvalues,
    which would make a latent depend on where its graph sat in a batch.
    """
    device = core_bonds.device
    graphs, cores = core_mask.shape
    spectral = torch.zeros(graphs, cores, frequencies, 2, device=device)
    spectral_mask = torch.zeros(graphs, frequencies, dtype=torch.bool, device=device)
    bonds_cpu = (core_bonds > 0).long().cpu()
    mask_cpu = core_mask.cpu()
    for index in range(graphs):
        present = mask_cpu[index].nonzero().flatten()
        if not present.numel():
            continue
        local = bonds_cpu[index][present][:, present]
        values, value_mask = laplacian_features(local, frequencies)
        spectral[index, present.to(device)] = values.to(device)
        spectral_mask[index] = value_mask.to(device)
    return spectral, spectral_mask


class LatentNormalization(nn.Module):
    """Latent centering/scaling that can be estimated from scratch.

    A pretrained codec ships calibrated statistics. Joint training from scratch
    has none, so the statistics are tracked as an exponential moving average
    over the latents the encoder actually produces, reduced across ranks so
    every replica normalizes identically. The statistics are always detached:
    they shape the denoiser's target, they are not something to learn.
    """

    def __init__(self, latent_dim, momentum=0.99, warmup=100):
        super().__init__()
        if not 0 < momentum < 1:
            raise ValueError("Latent normalization momentum must lie in (0, 1)")
        self.momentum = momentum
        self.warmup = warmup
        self.register_buffer("center", torch.zeros(latent_dim))
        self.register_buffer("scale", torch.ones(latent_dim))
        self.register_buffer("observations", torch.zeros((), dtype=torch.long))
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def observe(self, latent, mask):
        """Fold one batch of encoder means into the running statistics."""
        values = latent.detach()[mask]
        count = torch.tensor(
            [float(values.size(0))], device=latent.device, dtype=torch.float32
        )
        total = values.float().sum(0)
        squares = values.float().square().sum(0)
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            packed = torch.cat([count, total, squares])
            torch.distributed.all_reduce(packed, op=torch.distributed.ReduceOp.SUM)
            count, total, squares = packed[:1], packed[1 : 1 + total.numel()], packed[1 + total.numel() :]
        if count.item() < 2:
            return
        mean = total / count
        variance = (squares / count - mean.square()).clamp_min(1e-8)
        # Start from the first real batch rather than decaying away from 0/1.
        weight = 0.0 if not bool(self.calibrated) else self.momentum
        self.center.mul_(weight).add_(mean, alpha=1 - weight)
        self.scale.mul_(weight).add_(variance.sqrt(), alpha=1 - weight)
        self.scale.clamp_min_(1e-4)
        self.observations.add_(int(count.item()))
        self.calibrated.fill_(True)

    def load_calibration(self, center, scale):
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("Latent normalization must be finite")
        if bool((scale <= 0).any()):
            raise ValueError("Latent normalization scale must be positive")
        with torch.no_grad():
            self.center.copy_(center)
            self.scale.copy_(scale)
            self.calibrated.fill_(True)

    @property
    def ready(self):
        return bool(self.calibrated)

    @property
    def warming_up(self):
        """True while the running statistics have seen too little to trust.

        The standalone pipeline calibrated its latent space offline over the
        whole corpus before the denoiser saw a single batch. Training from
        scratch cannot, so the denoiser's target is held back until the moving
        average has accumulated a comparable number of node observations.
        """
        return not self.ready or int(self.observations) < self.warmup

    def normalize(self, latent):
        return (latent - self.center) / self.scale

    def denormalize(self, latent):
        return latent * self.scale + self.center


class JointComposition(nn.Module):
    """Autoencoder plus latent denoiser, encoding through a shared trunk."""

    def __init__(self, config, trunk):
        super().__init__()
        self.config = config
        vocabulary = [int(z) for z in config.mof_composition_atom_vocab.split(",") if z.strip()]
        if HYDROGEN_NUMBER not in vocabulary:
            raise ValueError("Joint composition needs hydrogen in the atom vocabulary")
        self.register_buffer(
            "vocabulary", torch.tensor(vocabulary, dtype=torch.long), persistent=True
        )
        self.hydrogen_label = vocabulary.index(HYDROGEN_NUMBER)

        ae_config = HydrogenAEConfig(
            num_atom_classes=len(vocabulary),
            width=config.embedding_dim,
            layers=config.num_encoder_layers,
            heads=config.num_attention_heads,
            latent_dim=config.mof_joint_latent_dim,
            frequencies=config.mof_joint_ae_frequencies,
            decoder_width=config.mof_joint_ae_decoder_width,
            kl_weight=config.mof_joint_ae_kl_weight,
            dropout=config.dropout,
            hydrogen_count_classes=config.mof_joint_hydrogen_count_classes,
        )
        # The trunk is passed in, so it is registered by the network that owns
        # it and appears exactly once in state_dict and in the optimizer.
        self.autoencoder = HydrogenAutoencoder(ae_config, trunk=trunk)
        self.latent_normalization = LatentNormalization(
            ae_config.latent_dim,
            momentum=config.mof_joint_latent_momentum,
            warmup=config.mof_joint_latent_warmup,
        )
        self.denoiser = JointLGFlow(
            JointLGFlowConfig(
                latent_dim=ae_config.latent_dim,
                context_dim=config.embedding_dim,
                width=config.mof_joint_denoiser_width,
                layers=config.mof_joint_denoiser_layers,
                exchange_every=config.mof_joint_denoiser_exchange_every,
                time_logit_mean=config.mof_joint_time_logit_mean,
                time_logit_std=config.mof_joint_time_logit_std,
            )
        )
        self.sampling_steps = config.mof_joint_sampling_steps
        self.sampling_churn = config.mof_joint_sampling_churn
        self.sampling_seed = config.mof_joint_sampling_seed
        self.sampling_noise_stream = config.mof_joint_sampling_noise_stream
        self.inject_latent = config.mof_joint_latent_injection
        if self.inject_latent:
            # Deliberately not zero-initialized. Zero weights would make the
            # coordinate loss's gradient into the posterior exactly zero on the
            # first step, which is the thing this path exists to provide and the
            # thing the gradient test checks. The scale is small enough that the
            # injection starts as a perturbation of the node features.
            self.latent_to_node = nn.Linear(ae_config.latent_dim, config.embedding_dim, bias=False)
            nn.init.normal_(self.latent_to_node.weight, std=config.mof_joint_latent_injection_scale)

        if config.mof_joint_freeze_codec:
            self.autoencoder.requires_grad_(False)
            self.autoencoder.trunk.requires_grad_(False)
            self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.config.mof_joint_freeze_codec:
            # The trunk belongs to the network, so freezing just the codec
            # would otherwise leave it trainable and its dropout active.
            self.autoencoder.eval()
            self.autoencoder.trunk.eval()
            self.latent_normalization.eval()
        return self

    @property
    def latent_dim(self):
        return self.autoencoder.config.latent_dim

    def trunk_parameter_ids(self):
        return {id(parameter) for parameter in self.autoencoder.trunk.parameters()}

    def own_parameters(self):
        """Every parameter this module is responsible for, trunk excluded.

        The trunk belongs to the network. Deduplicating here is what keeps an
        optimizer from being handed the shared weights twice.
        """
        shared = self.trunk_parameter_ids()
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in shared
        ]

    # ------------------------------------------------------------------
    # Representation conversion

    def fold_batch(self, batch):
        """Flatten the padded per-MOF block tensors and fold terminal H."""
        atoms = batch["mof_bb_atom_types"]
        bonds = batch["mof_bb_bond_adj"]
        mask = batch["mof_bb_atom_mask"].bool()
        mofs, blocks = mask.shape[:2]
        flat_atoms = atoms.flatten(0, 1)
        flat_bonds = bonds.flatten(0, 1)
        flat_mask = mask.flatten(0, 1)
        present = flat_mask.any(-1)
        if not bool(present.any()):
            raise ValueError("Joint composition batch has no building-block atoms")
        # Dataset labels are AtomVocabulary ids with 0 reserved for padding;
        # the codec indexes its vocabulary from zero.
        codec_atoms = (flat_atoms[present] - 1).clamp_min(0)
        folded = fold_terminal_hydrogens(
            codec_atoms,
            flat_bonds[present],
            flat_mask[present],
            hydrogen_label=self.hydrogen_label,
            count_classes=self.autoencoder.config.hydrogen_count_classes,
        )
        folded["present"] = present
        folded["shape"] = (mofs, blocks)
        return folded

    def autoencoder_inputs(self, folded, batch=None):
        frequencies = self.autoencoder.config.frequencies
        spectral = None
        if batch is not None:
            cached = batch.get("mof_bb_core_spectral")
            cached_mask = batch.get("mof_bb_core_spectral_mask")
            if cached is not None and cached_mask is not None:
                present = folded["present"]
                spectral = cached.flatten(0, 1)[present][..., :frequencies, :]
                spectral_mask = cached_mask.flatten(0, 1)[present][..., :frequencies]
                cores = folded["core_mask"].shape[-1]
                if spectral.shape[1] < cores:
                    padding = spectral.new_zeros(
                        spectral.size(0), cores - spectral.shape[1], frequencies, 2
                    )
                    spectral = torch.cat([spectral, padding], dim=1)
                else:
                    spectral = spectral[:, :cores]
        if spectral is None:
            spectral, spectral_mask = core_spectral_features(
                folded["core_bonds"], folded["core_mask"], frequencies
            )
        return {
            "atoms": folded["core_atoms"],
            "bonds": folded["core_bonds"],
            "mask": folded["core_mask"],
            "hydrogen_counts": folded["core_hydrogen_counts"],
            "spectral": spectral,
            "spectral_mask": spectral_mask,
        }

    # ------------------------------------------------------------------
    # Training

    def encode(self, batch, *, augment=None, folded=None):
        """Differentiable encode of the clean composition through the trunk."""
        folded = folded if folded is not None else self.fold_batch(batch)
        inputs = self.autoencoder_inputs(folded, batch)
        frozen = self.config.mof_joint_freeze_codec
        if frozen and not self.latent_normalization.ready:
            raise RuntimeError("Frozen joint codec requires checkpoint latent calibration")
        if frozen:
            augment = False
        elif augment is None:
            augment = self.training and self.config.mof_joint_target_augment
        mean, logvar = self.autoencoder.encode(inputs, augment=augment)
        if self.training and not frozen:
            self.latent_normalization.observe(mean, inputs["mask"])
        elif not self.latent_normalization.ready:
            raise RuntimeError(
                "Joint composition latents are not calibrated; train or load "
                "a checkpoint before encoding in eval mode"
            )
        return {"mean": mean, "logvar": logvar, "inputs": inputs, "folded": folded}

    def reconstruction(self, encoded, *, sample_posterior=True):
        """Reconstruct from the posterior, keeping the logits differentiable."""
        mean, logvar = encoded["mean"], encoded["logvar"]
        mask = encoded["inputs"]["mask"]
        if sample_posterior and not self.config.mof_joint_freeze_codec:
            latent = mean + (logvar * 0.5).exp() * torch.randn_like(mean)
        else:
            latent = mean
        latent = latent.masked_fill(~mask[..., None], 0)
        decoded = self.autoencoder.decode(latent, mask)
        output = {**decoded, "mean": mean, "logvar": logvar, "latent": latent}
        loss, details = hydrogen_reconstruction_loss(
            output, encoded["inputs"], self.autoencoder.config.kl_weight
        )
        return loss, details, output

    def composition_request(self, batch, folded=None):
        """The stage-one conditioning carried by a batch.

        Size-conditioned generation, which is the setting this version targets:
        the request holds per-block core sizes, repetition counts and linker
        roles, matching what ``collate_joint`` accepts at inference time. The
        folded graph is read only to count cores; its atom types, bonds and
        hydrogen counts never reach the denoiser.
        """
        folded = folded if folded is not None else self.fold_batch(batch)
        present = folded["present"]
        roles = batch.get("mof_bb_is_linker")
        if roles is None:
            raise KeyError(
                "Joint composition needs per-block linker roles; re-collate the "
                "batch with a MOFDataset that emits mof_bb_is_linker"
            )
        return {
            "shape": folded["shape"],
            "present": present,
            "core_sizes": folded["core_mask"].sum(-1),
            "repetitions": batch["mof_bb_repetitions"].flatten(0, 1)[present],
            "is_linker": roles.flatten(0, 1)[present].bool(),
        }

    def pack_request(self, request):
        """Build the denoiser's per-MOF block packing from a request alone."""
        mofs, blocks = request["shape"]
        present = request["present"]
        sizes = request["core_sizes"]
        device = sizes.device
        repetitions = request["repetitions"]
        width = int(sizes.max().item()) if sizes.numel() else 1
        mask = torch.arange(max(width, 1), device=device)[None] < sizes[:, None]
        block_index = present.nonzero().flatten()
        owner = block_index.div(blocks, rounding_mode="floor")
        slot = block_index.remainder(blocks)
        block_mask = present.reshape(mofs, blocks)
        cell_atoms = torch.zeros(mofs, device=device, dtype=torch.long).index_add(
            0, owner, sizes * repetitions
        )
        total_repetitions = torch.zeros(mofs, device=device, dtype=torch.long).index_add(
            0, owner, repetitions
        )
        return {
            "mask": mask,
            "owner": owner,
            "slot": slot,
            "repetitions": repetitions,
            "is_linker": request["is_linker"],
            "block_mask": block_mask,
            "mof_counts": torch.stack(
                [block_mask.sum(-1), cell_atoms, total_repetitions], dim=-1
            ),
        }

    def pack_blocks(self, encoded, batch):
        """Training-time packing; identical in construction to generation's."""
        return self.pack_request(
            self.composition_request(batch, folded=encoded["folded"])
        )

    def flow_loss(self, encoded, batch):
        """Latent flow matching against a detached target.

        Detaching the target keeps the denoiser from reshaping the latent space
        to be easy to denoise. Reconstruction and the coordinate branch still
        train the encoder, since they read ``encoded["mean"]`` directly.
        """
        packed = self.pack_blocks(encoded, batch)
        target = self.latent_normalization.normalize(encoded["mean"].detach())
        target = target.masked_fill(~packed["mask"][..., None], 0)
        loss = self.denoiser(target, packed)
        if self.training and self.latent_normalization.warming_up:
            # Zero the term without skipping the forward pass: the denoiser's
            # parameters still receive a (zero) gradient, which is what DDP
            # requires, and no step is taken against a target whose scale is
            # still moving by orders of magnitude.
            loss = loss * 0.0
        return loss, packed

    # ------------------------------------------------------------------
    # Node-feature injection

    def node_injection(self, encoded, batch, total_atoms):
        """Per-cell-atom features from the (undetached) composition latent.

        The coordinate loss flows back through this into the posterior and the
        shared trunk. At generation time the latent comes from re-encoding the
        graph stage one decoded, so this feature is available then too.
        """
        if not self.inject_latent:
            return None
        folded = encoded["folded"]
        mofs, blocks = folded["shape"]
        present = folded["present"]
        latent = self.latent_normalization.normalize(encoded["mean"])
        latent = latent.masked_fill(~folded["core_mask"][..., None], 0)

        occurrence = batch["mof_bb_occ_atom_index"].flatten(0, 1)[present]
        occurrence_mask = batch["mof_bb_occ_mask"].flatten(0, 1)[present].bool()
        explicit_mask = batch["mof_bb_atom_mask"].bool().flatten(0, 1)[present]
        block_owner = present.nonzero().flatten().div(blocks, rounding_mode="floor")

        features = latent.new_zeros(mofs, total_atoms, latent.size(-1))
        for mof in range(mofs):
            rows = (block_owner == mof).nonzero().flatten()
            if not rows.numel():
                continue
            bb, core = cell_atom_core_index(
                occurrence[rows],
                folded["explicit_latent_core"][rows],
                occurrence_mask[rows],
                explicit_mask[rows],
                total_atoms,
            )
            claimed = core >= 0
            if not bool(claimed.any()):
                continue
            features[mof, claimed] = latent[rows][bb[claimed], core[claimed]]
        return self.latent_to_node(features)

    # ------------------------------------------------------------------
    # Generation

    @torch.no_grad()
    def decode_latents(self, latent, mask, *, hydrogen_totals=None):
        """Stage-one decode: latents to a clean core graph and H counts."""
        decoded = self.autoencoder.decode(
            self.latent_normalization.denormalize(latent).masked_fill(~mask[..., None], 0),
            mask,
        )
        core_atoms = decoded["atom_logits"].argmax(-1).masked_fill(~mask, 0)
        core_bonds = (
            decoded["bond_logits"].argmax(-1).masked_fill(~decoded["pair_mask"], 0)
        )
        counts = decoded["hydrogen_logits"].argmax(-1).masked_fill(~mask, 0)
        if hydrogen_totals is not None:
            from ..hydrogen_budget import decode_hydrogen_budget
            core_atoms, counts = decode_hydrogen_budget(
                decoded["atom_logits"], decoded["hydrogen_logits"], mask,
                hydrogen_totals, self.hydrogen_label,
            )
        return {
            "core_atoms": core_atoms,
            "core_bonds": core_bonds,
            "core_mask": mask,
            "core_hydrogen_counts": counts,
        }

    @torch.no_grad()
    def generate_composition(self, batch, *, steps=None, churn=None, seed=None):
        """Stage one, start to finish: noise to an explicit-hydrogen graph.

        Nothing here sees a clean latent, a clean graph or any coordinate: the
        trajectory starts from Gaussian noise over the requested block shapes,
        self-conditions on its own predictions, and only decodes once the
        reverse process has finished.
        """
        return self.generate_request(self.composition_request(batch), steps=steps, churn=churn, seed=seed)

    @torch.no_grad()
    def generate_request(self, request, *, steps=None, churn=None, seed=None):
        """Generate from metadata alone; no dataset or reference graph is read."""
        from .joint_lgflow import sample_joint_latents

        packed = self.pack_request(request)
        latent = sample_joint_latents(
            self.denoiser,
            packed,
            steps=self.sampling_steps if steps is None else steps,
            churn=self.sampling_churn if churn is None else churn,
            seed=self.sampling_seed if seed is None else seed,
            noise_stream_version=self.sampling_noise_stream,
        )
        decoded = self.decode_latents(latent, packed["mask"],
                                      hydrogen_totals=request.get("hydrogen_totals"))
        return {"packed": packed, "latent": latent, **self.expand_decoded(decoded)}

    @torch.no_grad()
    def expand_decoded(self, decoded):
        """Stage-one to stage-two handoff: explicit-H graphs in atomic numbers."""
        expanded = expand_terminal_hydrogen_counts(
            decoded["core_atoms"],
            decoded["core_bonds"],
            decoded["core_mask"],
            decoded["core_hydrogen_counts"],
            hydrogen_label=self.hydrogen_label,
        )
        numbers = self.vocabulary[expanded["atom_types"].clamp_min(0)]
        expanded["atomic_numbers"] = numbers.masked_fill(~expanded["atom_mask"], 0)
        return expanded
