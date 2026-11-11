"""Optional rank-zero Weights & Biases tracking for FlexMOF training."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import MOFConfig
    from .training import TrainingConfig


class WandbTracker:
    """Small lazy-importing adapter that keeps W&B out of non-tracking runs."""

    def __init__(
        self,
        training: TrainingConfig,
        architecture: MOFConfig,
        trainer_state: dict[str, Any],
        output_dir: Path,
        rank: int,
    ) -> None:
        self.run = None
        self.log_checkpoints = training.wandb_log_checkpoints
        if rank != 0 or not training.wandb_project:
            return
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError(
                "W&B tracking requires the 'wandb' package; install FlexMOF "
                "normally or run `pip install wandb`."
            ) from error

        run_id = trainer_state.get("wandb_run_id") or wandb.util.generate_id()
        trainer_state["wandb_run_id"] = run_id
        tags = [tag.strip() for tag in training.wandb_tags.split(",") if tag.strip()]
        self.run = wandb.init(
            project=training.wandb_project,
            entity=training.wandb_entity or None,
            name=training.wandb_name or None,
            group=training.wandb_group or None,
            tags=tags or None,
            id=run_id,
            resume="allow",
            mode=training.wandb_mode,
            dir=str(output_dir),
            config={
                "training": asdict(training),
                "architecture": {
                    name: getattr(architecture, name)
                    for name in architecture.checkpoint_fields()
                },
            },
        )

    def log(self, metrics: dict[str, Any], step: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def log_checkpoint(self, path: Path, step: int) -> None:
        if self.run is None or not self.log_checkpoints:
            return
        import wandb

        artifact = wandb.Artifact(
            name=f"{self.run.project}-{self.run.id}-checkpoint",
            type="model",
            metadata={"global_step": step},
        )
        artifact.add_file(str(path), name=path.name)
        self.run.log_artifact(artifact, aliases=["latest", f"step-{step}"])

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
            self.run = None
