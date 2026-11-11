"""Continuous, checkpointed DDP ODAC25 E1 finetuning from the joint model."""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime
import hashlib
import json
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / 'atommof_compat'))

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from flexmof.training import _architecture_config
from finetune_model import AdsorptionTrainingModel
from native_batching import ddp_loss_scale, epoch_batches, epoch_fingerprint, update_window
from odac_adapter import zero_new_role_embeddings
from odac_multigraph import JointOdacDataset, collate_odac, load_expanded_weights, make_config


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rng_state(model, device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device),
            "diffusion": model.diffusion_noise.torch_generator.get_state()}


def restore_rng(model, state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu(), device)
    model.diffusion_noise.torch_generator.set_state(state["diffusion"].cpu())


def move(batch, device):
    return {k: v.to(device).clone() if torch.is_tensor(v) else v for k, v in batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-updates", type=int, default=0)
    parser.add_argument("--resume-source", type=Path)
    parser.add_argument("--expected-resume-update", type=int)
    parser.add_argument("--revalidate-on-resume", action="store_true")
    parser.add_argument("--recovery-evidence", type=Path)
    args = parser.parse_args()
    if (args.resume_source or args.revalidate_on_resume or args.expected_resume_update is not None) and not args.resume:
        parser.error("recovery options require --resume")
    spec = json.loads(Path(args.spec).read_text())
    root = Path(spec["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    rank, local_rank, world = (int(os.environ.get(x, d)) for x, d in
                               (("RANK", "0"), ("LOCAL_RANK", "0"), ("WORLD_SIZE", "1")))
    if world != int(spec["world_size"]):
        raise ValueError(f"world size {world} != requested {spec['world_size']}")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    stop = {"requested": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(requested=True))

    def any_flag(value):
        flag = torch.tensor(int(bool(value)), device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        return bool(flag.item())

    model = optimizer = dataset = None
    state = None
    started = time.monotonic()
    try:
        latest = root / "latest_checkpoint.pt"
        previous = root / "previous_checkpoint.pt"
        resume_path = args.resume_source or (latest if latest.is_file() else previous)
        if args.resume and not resume_path.is_file():
            raise FileNotFoundError("resume requested without a complete checkpoint")
        if not args.resume and (latest.exists() or previous.exists() or (root / "training.jsonl").exists()):
            raise FileExistsError("existing output requires --resume; refusing to restart")
        source_path = resume_path if args.resume else Path(spec["checkpoint"])
        if not args.resume:
            digest = hashlib.sha256()
            with source_path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != spec["checkpoint_sha256"]:
                raise ValueError("pretrained checkpoint hash mismatch")
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        if args.resume:
            if source.get("format") != "odac_joint_e1_v1" or source["spec_sha256"] != fingerprint(spec) or source["world_size"] != world:
                raise ValueError("resume identity/spec/world-size mismatch")
            if args.expected_resume_update is not None and source["trainer_state"]["update"] != args.expected_resume_update:
                raise ValueError("checkpoint update does not match requested recovery point")
            config = _architecture_config(source, False, None)
        else:
            config = make_config(source, _architecture_config)
        dataset = JointOdacDataset(spec["train_lmdb"], spec["train_metadata"],
                                   max_atoms=677, composition_config=config)
        if len(dataset) != 153249 or max(dataset.sizes) != 677:
            raise ValueError("ODAC training population differs from the fixed submission split")
        validation = JointOdacDataset(spec["val_lmdb"], spec["val_metadata"],
                                      max_atoms=1000, composition_config=config)
        if len(validation) != 3130:
            raise ValueError("ODAC validation population differs from 3130 records")
        model = AdsorptionTrainingModel(config, loss_mode="e1")
        if args.resume:
            model.load_state_dict(source["model"], strict=True)
        else:
            load_expanded_weights(model, source)
            zero_new_role_embeddings(model)
        model.to(device).float()
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(spec["lr"]), weight_decay=0.0)
        if args.resume:
            optimizer.load_state_dict(source["optimizer"])
            state = source["trainer_state"]
        else:
            state = {"epoch": 0, "batch_cursor": 0, "update": 0,
                     "data_order_fingerprint": None, "presentations": 0}
        ddp = DDP(model, device_ids=[local_rank], output_device=local_rank,
                  gradient_as_bucket_view=True, find_unused_parameters=True)
        torch.manual_seed(int(spec["seed"]) + rank)
        np.random.seed((int(spec["seed"]) + rank) % (2 ** 32))
        random.seed(int(spec["seed"]) + rank)
        model.diffusion_noise.torch_generator = torch.Generator(device=device).manual_seed(int(spec["seed"]) + rank)
        if args.resume:
            restore_rng(model, source["rng_states"][rank], device)
        del source
        model.train()
        if rank == 0:
            manifest_path = (args.recovery_evidence / "resume_manifest.json") if args.recovery_evidence else (root / "run_manifest.json")
            atomic_json(manifest_path, {"spec": spec, "spec_sha256": fingerprint(spec),
                "world_size": world, "training_records": len(dataset), "validation_records": len(validation),
                "parameters": sum(p.numel() for p in model.parameters()), "loss": "E1 coordinate RMS",
                "pretrained_source": spec["checkpoint"], "expanded_vocabulary": config.mof_composition_atom_vocab,
                "started_unix": time.time(), "resumed": bool(args.resume),
                "resume_source": str(resume_path) if args.resume else None,
                "resume_update": state["update"] if args.resume else None})

        def save(reason):
            all_rng = [None] * world
            dist.all_gather_object(all_rng, rng_state(model, device))
            error = None
            if rank == 0:
                try:
                    payload = {"format": "odac_joint_e1_v1", "model": model.state_dict(),
                               "args": dataclasses.asdict(config), "optimizer": optimizer.state_dict(),
                               "trainer_state": state, "rng_states": all_rng, "world_size": world,
                               "spec_sha256": fingerprint(spec), "reason": reason, "saved_unix": time.time()}
                    tmp = root / f"checkpoint.tmp.{os.getpid()}"
                    with tmp.open("wb") as f:
                        torch.save(payload, f)
                        f.flush()
                        os.fsync(f.fileno())
                    # If interrupted between renames, the previous checkpoint
                    # is still valid and the launcher selects it on restart.
                    if latest.exists():
                        os.replace(latest, previous)
                    os.replace(tmp, latest)
                    atomic_json(root / "checkpoint_status.json", {"update": state["update"],
                        "epoch": state["epoch"], "batch_cursor": state["batch_cursor"],
                        "reason": reason, "path": str(latest), "saved_unix": time.time(),
                        "all_rank_rng_saved": len(all_rng), "optimizer_saved": True})
                except Exception as exc:
                    error = repr(exc)
            if any_flag(error is not None):
                raise RuntimeError(error or "rank 0 failed to save checkpoint")

        @torch.no_grad()
        def validate():
            # Every rank evaluates one fixed disjoint group; no target geometry
            # is used as a generative input. Fixed RNG/time makes epochs comparable.
            saved_rng = rng_state(model, device)
            model.eval()
            torch.cuda.reset_peak_memory_stats(device)
            try:
                chosen = list(range(rank * 2, rank * 2 + 2))
                totals = torch.zeros(8, dtype=torch.float64, device=device)
                for j, index in enumerate(chosen):
                    for t in (0.2, 0.5, 0.8):
                        batch = move(collate_odac([validation[index]]), device)
                        values = model.fixed_time_losses(batch, t, seed=2026110900 + rank * 2 + j)
                        totals[0] += values["coord"].double()
                        totals[1] += 1
                        for offset, key in zip((2, 4, 6), ("framework", "adsorbate", "cell")):
                            value = values[key].double()
                            if torch.isfinite(value).all():
                                totals[offset] += value.sum()
                                totals[offset + 1] += value.numel()
                dist.all_reduce(totals)
                metrics = {name: float(totals[i] / totals[i + 1]) if totals[i + 1] else None
                           for i, name in zip((0, 2, 4, 6), ("coord", "framework", "adsorbate", "cell"))}
                local_memory = {"rank": rank,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
                memories = [None] * world
                dist.all_gather_object(memories, local_memory)
                return metrics, memories
            finally:
                restore_rng(model, saved_rng, device)
                model.train()

        def record_validation(metrics, memories, recovery=False):
            if rank == 0:
                row = {"phase": "validation", "update": state["update"], "epoch": state["epoch"],
                       "records": 2 * world, "times": [0.2, 0.5, 0.8], **metrics, "time_unix": time.time()}
                with (root / "training.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                    f.flush()
                atomic_json(root / "validation_latest.json", row)
                if args.recovery_evidence:
                    atomic_json(args.recovery_evidence / f"validation_update{state['update']}.json", {
                        "status": "passed", "recovery": recovery, "metrics": row,
                        "per_rank_memory": memories, "gradient_tracking": False})

        if args.revalidate_on_resume:
            if state["batch_cursor"] != 0 or state["epoch"] < 1:
                raise ValueError("revalidation requires a completed-epoch checkpoint")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            metrics, memories = validate()
            record_validation(metrics, memories, recovery=True)
            dist.barrier()

        if not args.resume and not args.smoke_updates:
            save("initial")
        while True:
            batches = epoch_batches(dataset.sizes, spec["seed"], state["epoch"],
                                    int(spec["max_square"]), int(spec["max_batch"]))
            order = epoch_fingerprint(batches)
            if state["data_order_fingerprint"] not in (None, order):
                raise ValueError("training order differs from saved checkpoint")
            state["data_order_fingerprint"] = order
            while state["batch_cursor"] < len(batches):
                if any_flag(stop["requested"] or (root / "STOP").exists()):
                    save("requested_stop")
                    if rank == 0:
                        atomic_json(root / "progress.json", {**state, "phase": "stopped"})
                    return
                rounds, next_cursor, count = update_window(batches, state["batch_cursor"], world, 1)
                optimizer.zero_grad(set_to_none=True)
                metrics = torch.zeros(8, dtype=torch.float64, device=device)
                step_started = time.monotonic()
                for group in rounds:
                    micro = group[rank]
                    batch = move(collate_odac([dataset[i] for i in micro.indices]), device)
                    values = ddp(batch)
                    loss = values["loss"]
                    if any_flag(not bool(torch.isfinite(loss))):
                        raise FloatingPointError("nonfinite loss on at least one rank")
                    (loss * ddp_loss_scale(micro.objective_count, count, world)).backward()
                    if not micro.padding:
                        metrics[0] += loss.detach().double() * len(micro.indices)
                        metrics[1] += len(micro.indices)
                        for offset, key in zip((2, 4, 6), ("rms_framework", "rms_adsorbate", "rms_cell")):
                            values_finite = values[key].detach().double()
                            values_finite = values_finite[torch.isfinite(values_finite)]
                            metrics[offset] += values_finite.sum()
                            metrics[offset + 1] += values_finite.numel()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec["grad_clip"]), error_if_nonfinite=True)
                next_update = state["update"] + 1
                lr = float(spec["lr"]) * min(next_update / int(spec["warmup"]), 1.0)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.step()
                state.update(update=next_update, batch_cursor=next_cursor,
                             presentations=state["presentations"] + count)
                dist.all_reduce(metrics)
                if int(metrics[1].item()) != count:
                    raise AssertionError("DDP weighted structure count mismatch")
                if rank == 0:
                    row = {"phase": "training", **state, "epoch_microbatches": len(batches),
                           "global_structures": count, "loss": float(metrics[0] / metrics[1]),
                           "grad_norm_before_clip": float(grad), "learning_rate": lr,
                           "update_seconds": time.monotonic() - step_started,
                           "elapsed_seconds_this_launch": time.monotonic() - started, "time_unix": time.time()}
                    for offset, key in zip((2, 4, 6), ("rms_framework", "rms_adsorbate", "rms_cell")):
                        row[key] = float(metrics[offset] / metrics[offset + 1]) if metrics[offset + 1] else None
                    with (root / "training.jsonl").open("a") as f:
                        f.write(json.dumps(row, allow_nan=False) + "\n")
                        f.flush()
                    atomic_json(root / "progress.json", row)
                if next_update % int(spec["save_every"]) == 0:
                    save("periodic")
                if args.smoke_updates and next_update >= args.smoke_updates:
                    dist.barrier()
                    if rank == 0:
                        atomic_json(root / "pilot_complete.json", {"status": "passed", "updates": next_update,
                            "world_size": world, "last_global_structures": count, "time_unix": time.time()})
                    return
            expected = (state["epoch"] + 1) * len(dataset)
            if state["presentations"] != expected:
                raise AssertionError(f"epoch exposure mismatch: {state['presentations']} != {expected}")
            state.update(epoch=state["epoch"] + 1, batch_cursor=0, data_order_fingerprint=None)
            save("epoch_completed")
            # Drop references to the last training batch and gradients before
            # validation; the next training update already starts with zero_grad.
            del batch, values, loss
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            metrics, memories = validate()
            record_validation(metrics, memories)
    except Exception as exc:
        failure_dir = args.recovery_evidence or root
        atomic_json(failure_dir / f"rank{rank}_failure.json", {"rank": rank, "error": repr(exc),
                    "traceback": traceback.format_exc(), "state": state, "time_unix": time.time()})
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
