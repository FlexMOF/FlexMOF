"""Small, explicit configuration surface for checkpoint-compatible sampling."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path


@dataclass
class MOFConfig:
    """Values consumed by the MOF architecture and sampler."""

    # Transformer architecture
    encoder_layers: int = 14
    encoder_attention_heads: int = 32
    encoder_embed_dim: int = 768
    encoder_ffn_embed_dim: int = 768
    encoder_pair_embed_dim: int = 64
    num_structure_encoder_layer: int = 12
    structure_hidden_dim: int = 768
    structure_ffn_dim: int = 768
    decoder_ffn_dim: int = 768
    num_pred_attn_layer: int = 12
    dropout: float = 0.1
    attention_dropout: float = 0.0
    act_dropout: float = 0.1
    # Parameter-free rotary Q/K positions in the sequence encoder only.
    sequence_rope: bool = False
    mof_canonical_order: bool = False
    # Empty selects the genuine requested split; a key explicitly enables overfit.
    mof_overfit_key: str = ""
    mof_overfit_repeats: int = 100000

    # Graph features
    num_edges: int = 25600
    num_edge_dis: int = 128
    num_3d_bias_kernel: int = 32
    multi_hop_max_dist: int = 5
    mof_max_shortest_path_distance: int = 1024
    # Ablate the invariant structure encoder's Graphormer shortest-path and
    # multi-hop bond biases. The joint composition codec remains graph-aware.
    # Persisted so training and subsequent CSP use the same mode.
    mof_disable_graphormer_2d: bool = False

    # Periodic geometry
    pbc_cutoff: float = 40.0
    pbc_expanded_token_cutoff: int = 896
    pbc_expanded_num_cell_per_direction: int = 5
    # Bound candidate distance and differentiable radial-feature temporaries.
    pbc_distance_chunk_size: int = 4096
    pbc_sparse_edge_chunk_size: int = 262144
    pbc_activation_checkpoint_edges: int = 8_000_000
    use_per_atom_outcell_index: bool = True
    normalize_softmax_probs_after_local_attn_weights_eps: float = 1e-10
    # Ablation: hide every periodic image from the network, so atoms only see
    # the primary cell. The dataset, lattice diffusion, and emitted structures
    # stay periodic, which keeps a disabled run comparable to an enabled one.
    mof_disable_pbc: bool = False
    # Represent every linker as one continuous periodic image, then translate
    # all real atoms together so that their arithmetic center is the cell center.
    # Linker atoms may consequently lie outside the primary cell.
    mof_unwrap_linkers_and_center: bool = False

    # Equivariant architecture
    equivar_use_linear_bias: bool = True
    use_smooth_equviariant_norm: bool = True

    # Diffusion
    num_timesteps: int = 5000
    num_timesteps_stepsize: int = -100
    ddpm_beta_start: float = 1e-7
    ddpm_beta_end: float = 2e-3
    diffusion_noise_std: float = 1.0
    periodic_diffusion_noise_std_factor: float = 1.0531306506190654
    periodic_lattice_diffusion_noise_std: float = 0.5
    diff_init_lattice_size_factor: float = 2.859496852322873

    # Compact building-block composition diffusion
    use_mof_composition_diffusion: bool = False
    mof_composition_diffusion_strategy: str = "digress_uniform"
    # Gaussian DDPM vocabulary is atomic numbers in contiguous ID order (1..K).
    # Keep this metadata in checkpoints so decoding never depends on a dataset.
    mof_composition_atom_vocab: str = ""
    mof_composition_atom_mean: float = 0.0
    mof_composition_atom_std: float = 1.0
    mof_composition_bond_mean: float = 0.5
    mof_composition_bond_std: float = 0.5
    mof_composition_sampling_head: str = "3d"
    mof_composition_num_timesteps: int = 1000
    mof_composition_num_atom_classes: int = 160
    mof_composition_num_bond_classes: int = 2
    mof_composition_atom_mask_token: int = 160
    mof_composition_bond_mask_token: int = 2
    mof_composition_backbone_mask_token: int = 157
    mof_composition_max_total_atoms: int = 768
    mof_composition_max_unique_bbs: int = 32
    mof_composition_max_atoms_per_bb: int = 256
    mof_composition_max_repetitions: int = 128
    mof_composition_max_building_blocks: int = 768
    mof_composition_seq_loss_weight: float = 1.0
    mof_composition_3d_loss_weight: float = 1.0
    mof_composition_atom_loss_weight: float = 1.0
    mof_composition_bond_loss_weight: float = 1.0
    mof_composition_sparse_bond_head: bool = True
    # Optional invariant distances between atoms in the same noisy block.
    mof_composition_pair_geometry: bool = False
    # Training augmentation only; sampling still uses the configured joint clock.
    mof_composition_independent_time_probability: float = 0.0
    # LG-Flow's frozen graph codec and additive latent prediction heads.
    mof_lgflow_autoencoder_checkpoint: str = ""
    mof_lgflow_prior_checkpoint: str = ""
    mof_lgflow_ae_width: int = 256
    mof_lgflow_ae_layers: int = 6
    mof_lgflow_latent_dim: int = 32
    mof_lgflow_frequencies: int = 32
    mof_lgflow_decoder_width: int = 24
    mof_lgflow_head_width: int = 256
    mof_lgflow_head_layers: int = 4
    mof_lgflow_head_architecture: str = "sequence"

    # Joint composition-and-structure training. The composition autoencoder
    # encodes through the CSP sequence trunk, so its width and head count are
    # not configurable here: they are the trunk's. Everything below is either
    # a latent-space dimension, a denoiser dimension, or a loss weight.
    mof_joint_composition: bool = False
    mof_joint_latent_dim: int = 32
    mof_joint_ae_frequencies: int = 32
    mof_joint_ae_decoder_width: int = 24
    mof_joint_ae_kl_weight: float = 1e-6
    mof_joint_hydrogen_count_classes: int = 5
    mof_joint_denoiser_width: int = 768
    mof_joint_denoiser_layers: int = 12
    mof_joint_denoiser_exchange_every: int = 3
    # Defaults preserve historical checkpoints; the alignment finetune opts in.
    mof_joint_time_logit_mean: float = 0.0
    mof_joint_time_logit_std: float = 1.0
    mof_joint_canonical_blocks: bool = False
    mof_joint_freeze_codec: bool = False
    # Finetune only the composition flow; preserve every CSP/codec tensor.
    mof_joint_denoiser_only: bool = False
    mof_joint_target_augment: bool = True
    # Latent features added to the CSP node inputs. This is the path that lets
    # the coordinate loss reach the autoencoder's posterior; the trunk is
    # already reached directly by the CSP pass.
    mof_joint_latent_injection: bool = True
    mof_joint_latent_injection_scale: float = 0.02
    mof_joint_reconstruction_weight: float = 1.0
    mof_joint_flow_weight: float = 1.0
    mof_joint_latent_momentum: float = 0.99
    # Node observations the online latent statistics collect before the flow
    # term is allowed to contribute. Until then the denoiser would be chasing a
    # target whose scale is still moving by orders of magnitude.
    mof_joint_latent_warmup: int = 20000
    # Composition sampling settings.
    mof_joint_sampling_steps: int = 100
    mof_joint_sampling_churn: float = 8.0
    mof_joint_sampling_seed: int = 9013
    mof_joint_sampling_noise_stream: int = 2
    # Cell-level Laplacian positional encoding in front of the trunk. This is
    # new in the fused model and is distinct from the autoencoder's own
    # building-block spectral features, which are a required encoder input and
    # are never disabled. Regression against the pre-fusion checkpoints runs
    # with this off.
    mof_joint_csp_laplacian_pe: bool = False
    mof_joint_csp_laplacian_frequencies: int = 8
    use_building_block_id: bool = False

    # Runtime-only settings
    checkpoint_path: str = ""
    data_root: str = ""
    dataset_subdir: str = "bw_db"
    output_dir: str = ""
    metric_path: str = ""
    micro_batch_size: int = 4
    max_batches: int = -1
    num_samples: int = 1
    # Additional full denoising passes, starting directly from the last sample.
    # No forward noising or fresh prior is applied between passes.
    num_recycles: int = 0
    seed: int = 12347
    local_rank: int = 0
    rank: int = 0
    world_size: int = 1

    def __post_init__(self):
        if (
            isinstance(self.num_recycles, bool)
            or not isinstance(self.num_recycles, int)
            or self.num_recycles < 0
        ):
            raise ValueError("num_recycles must be a non-negative integer")
        if self.mof_overfit_repeats < 1:
            raise ValueError("Overfit repeats must be positive")
        if self.sequence_rope and (
            self.encoder_embed_dim % self.encoder_attention_heads
            or (self.encoder_embed_dim // self.encoder_attention_heads) % 2
        ):
            raise ValueError("Sequence RoPE requires an even attention head dimension")
        if not 0.0 <= self.mof_composition_independent_time_probability <= 1.0:
            raise ValueError("Independent composition time probability must be in [0, 1]")
        if self.mof_composition_pair_geometry and not self.mof_unwrap_linkers_and_center:
            raise ValueError("Composition pair geometry requires unwrapped linkers")
        if self.use_building_block_id:
            raise ValueError("The model cannot use order-based building-block IDs")
        if self.mof_composition_diffusion_strategy == "lgflow":
            if self.mof_lgflow_head_architecture not in {"sequence", "dit"}:
                raise ValueError("LG-Flow head architecture must be sequence or dit")
            if not self.mof_composition_atom_vocab:
                raise ValueError("LG-Flow requires a configured atom vocabulary")
            if self.mof_lgflow_ae_width % 8 or self.mof_lgflow_head_width % 8:
                raise ValueError("LG-Flow widths must be divisible by eight heads")
        if self.mof_joint_composition:
            if self.mof_composition_diffusion_strategy != "joint_lgflow":
                raise ValueError(
                    "Joint composition requires "
                    "mof_composition_diffusion_strategy=joint_lgflow"
                )
            if not self.mof_composition_atom_vocab:
                raise ValueError("Joint composition requires a configured atom vocabulary")
            if self.mof_joint_denoiser_width % 8:
                raise ValueError("The joint denoiser width must be divisible by eight")
            if self.mof_joint_latent_dim < 1 or self.mof_joint_ae_frequencies < 1:
                raise ValueError("Joint latent and spectral dimensions must be positive")
            if not 0.0 <= self.mof_joint_latent_momentum < 1.0:
                raise ValueError("The latent normalization momentum must be in [0, 1)")
            if self.mof_joint_latent_warmup < 0:
                raise ValueError("The latent normalization warmup cannot be negative")
            if self.mof_joint_sampling_steps < 1 or self.mof_joint_sampling_churn < 0:
                raise ValueError("Stage-one sampling needs positive steps, nonnegative churn")
        elif self.mof_composition_diffusion_strategy == "joint_lgflow":
            raise ValueError(
                "mof_composition_diffusion_strategy=joint_lgflow is only meaningful "
                "with mof_joint_composition enabled"
            )
        if self.mof_joint_freeze_codec and not self.mof_joint_composition:
            raise ValueError("Freezing the joint codec requires joint composition")
        if self.mof_joint_denoiser_only:
            if not (self.mof_joint_composition and self.use_mof_composition_diffusion
                    and self.mof_joint_freeze_codec):
                raise ValueError("Denoiser-only finetuning requires joint composition and a frozen codec")
            if not self.mof_joint_flow_weight > 0 or self.mof_joint_reconstruction_weight != 0:
                raise ValueError("Denoiser-only finetuning requires positive flow weight and zero reconstruction weight")
        # Canonical internal names formerly created by GraphormerConfig.
        self.num_attention_heads = self.encoder_attention_heads
        self.num_encoder_layers = self.encoder_layers
        self.embedding_dim = self.encoder_embed_dim
        self.ffn_embedding_dim = self.encoder_ffn_embed_dim
        self.activation_dropout = self.act_dropout
        self.max_seq_len = 1024

    @classmethod
    def checkpoint_fields(cls) -> set[str]:
        runtime = {
            "checkpoint_path",
            "data_root",
            "dataset_subdir",
            "output_dir",
            "metric_path",
            "micro_batch_size",
            "max_batches",
            "num_samples",
            "num_recycles",
            "seed",
            "local_rank",
            "rank",
            "world_size",
        }
        return {field.name for field in fields(cls)} - runtime


def config_from_checkpoint(
    checkpoint_args: dict,
    checkpoint_path: str,
    data_root: str,
    output_dir: str,
    dataset_subdir: str,
    micro_batch_size: int,
    max_batches: int,
    seed: int,
    overrides: dict | None = None,
    num_samples: int = 1,
) -> MOFConfig:
    """Load architecture values through a deliberate MOF-only allowlist."""
    values = architecture_values(checkpoint_args)
    # These are sampling semantics from scripts/pretrain_psm.sh rather than
    # learned architecture values.  A training checkpoint may store its
    # pretraining defaults here, so sampling must set them deliberately.
    values.update(
        num_timesteps=5000,
        num_timesteps_stepsize=-100,
        ddpm_beta_start=1e-7,
        ddpm_beta_end=2e-3,
        diffusion_noise_std=1.0,
        diff_init_lattice_size_factor=2.859496852322873,
        periodic_diffusion_noise_std_factor=1.0531306506190654,
        periodic_lattice_diffusion_noise_std=0.5,
        pbc_cutoff=40.0,
        pbc_expanded_num_cell_per_direction=5,
        pbc_expanded_token_cutoff=896,
        pbc_distance_chunk_size=4096,
        pbc_sparse_edge_chunk_size=262144,
        pbc_activation_checkpoint_edges=8_000_000,
        use_per_atom_outcell_index=True,
        normalize_softmax_probs_after_local_attn_weights_eps=1e-10,
        mof_max_shortest_path_distance=1024,
    )
    output = str(Path(output_dir).resolve())
    values.update(
        checkpoint_path=str(Path(checkpoint_path).resolve()),
        data_root=str(Path(data_root).resolve()),
        dataset_subdir=dataset_subdir,
        output_dir=output,
        metric_path=str(Path(output) / "metrics"),
        micro_batch_size=micro_batch_size,
        max_batches=max_batches,
        num_samples=num_samples,
        seed=seed,
    )
    if overrides:
        values.update(overrides)
    return MOFConfig(**values)


def architecture_values(checkpoint_args: dict) -> dict:
    """Return MOF architecture/diffusion values present in a checkpoint."""
    return {
        key: checkpoint_args[key]
        for key in MOFConfig.checkpoint_fields()
        if key in checkpoint_args
    }
