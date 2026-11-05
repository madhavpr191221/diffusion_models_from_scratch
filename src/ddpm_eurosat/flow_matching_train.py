"""Train the unconditional EuroSAT image-space conditional flow-matching model."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from ddpm_eurosat.data import EXPECTED_SPLIT_COUNTS, UnconditionalEuroSATSplit, load_eurosat_split
from ddpm_eurosat.flow_matching import (
    BEST_FLOW_MATCHING_CHECKPOINT,
    BEST_FLOW_MATCHING_EMA_CHECKPOINT,
    LATEST_FLOW_MATCHING_CHECKPOINT,
    ContinuousTimeVelocityUNet,
    dequantize_normalized_pixels,
    flow_matching_loss,
)
from ddpm_eurosat.train import (
    append_metric,
    atomic_json,
    choose_device,
    create_run_dir,
    last_event_id,
    load_config,
    now_iso,
    save_checkpoint,
    seed_everything,
    status_update,
    restore_rng_state,
    update_ema,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def validate_config(config: dict[str, Any]) -> None:
    if config.get("process") != "flow_matching":
        raise ValueError("This trainer requires process = 'flow_matching'")
    if config.get("model", {}).get("architecture") != ContinuousTimeVelocityUNet.architecture_id:
        raise ValueError("Flow matching requires architecture = 'flow_matching_compact_v1'")
    if config.get("model", {}).get("prediction") != "velocity":
        raise ValueError("Flow matching model output must be velocity")
    if config.get("flow_matching", {}).get("time_convention") != "noise_to_data":
        raise ValueError("Flow matching time must run from noise at 0 to data at 1")
    if config.get("flow_matching", {}).get("dequantization") != "centered_uniform_width_2_over_255":
        raise ValueError("Unexpected flow-matching dequantization convention")
    training = config["training"]
    if training["optimizer"].lower() not in {"adam", "adamw"}:
        raise ValueError("training.optimizer must be Adam or AdamW")
    if int(training["epochs"]) < 1 or int(training["effective_batch_size"]) < 1:
        raise ValueError("epochs and effective_batch_size must be positive")
    micro = int(training["micro_batch_size"])
    effective = int(training["effective_batch_size"])
    if micro < 1 or micro > effective:
        raise ValueError("micro_batch_size must be in [1, effective_batch_size]")
    if int(training["warmup_updates"]) < 0 or float(training["learning_rate"]) <= 0:
        raise ValueError("warmup_updates must be nonnegative and learning_rate positive")
    if not 0 < float(training["ema_decay"]) < 1:
        raise ValueError("ema_decay must lie in (0,1)")


def _load_flow_config(path: Path) -> dict[str, Any]:
    config = load_config(path)
    validate_config(config)
    return config


def _model_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def run_preflight(config: dict[str, Any], device: torch.device, *, config_path: Path) -> dict[str, Any]:
    """Measure one real-image microbatch backward pass without an optimizer update."""
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    train_source = load_eurosat_split("train", PROJECT_ROOT / config["dataset"]["cache_dir"])
    if len(train_source) != EXPECTED_SPLIT_COUNTS["train"]:
        raise ValueError(f"Expected {EXPECTED_SPLIT_COUNTS['train']} train images, got {len(train_source)}")
    dataset = UnconditionalEuroSATSplit(train_source)
    batch_size = int(config["training"]["micro_batch_size"])
    images = torch.stack([dataset[index] for index in range(batch_size)]).to(device)
    model = ContinuousTimeVelocityUNet().to(device)
    model.train()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss = flow_matching_loss(model, dequantize_normalized_pixels(images))
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mb: float | None = torch.cuda.max_memory_allocated(device) / (1024**2)
    else:
        peak_memory_mb = None
    backward_seconds = time.perf_counter() - started
    batch_per_epoch = (len(dataset) + int(config["training"]["effective_batch_size"]) - 1) // int(config["training"]["effective_batch_size"])
    update_count = batch_per_epoch * int(config["training"]["epochs"])
    micro_batches_per_update = (int(config["training"]["effective_batch_size"]) + batch_size - 1) // batch_size
    estimated_training_seconds = backward_seconds * micro_batches_per_update * update_count
    result = {
        "kind": "flow_matching_resource_preflight",
        "created_at": now_iso(),
        "config_path": str(config_path),
        "process": "flow_matching",
        "device": str(device),
        "model_architecture": ContinuousTimeVelocityUNet.architecture_id,
        "parameter_count": _model_parameter_count(model),
        "micro_batch_size": batch_size,
        "effective_batch_size": int(config["training"]["effective_batch_size"]),
        "epochs": int(config["training"]["epochs"]),
        "estimated_optimizer_updates": update_count,
        "measured_forward_backward_seconds": round(backward_seconds, 4),
        "measured_peak_allocated_gpu_memory_mb": (
            round(peak_memory_mb, 1) if peak_memory_mb is not None else None
        ),
        "linear_training_time_estimate_hours": round(estimated_training_seconds / 3600, 2),
        "training_time_estimate_note": (
            "Linear estimate from one microbatch backward pass; excludes optimizer, data loading, "
            "validation, checkpointing, and possible throughput warm-up."
        ),
        "loss": float(loss.detach().cpu()),
        "optimizer_updated": False,
        "split_loaded": "train",
        "test_split_loaded": False,
    }
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"flow-matching-preflight-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    atomic_json(output_path, result)
    result["artifact"] = str(output_path)
    print(json.dumps(result, indent=2), flush=True)
    return result


@torch.no_grad()
def _validate(model: nn.Module, images: list[Tensor], device: torch.device, seed: int) -> float:
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    weighted_loss = 0.0
    count = 0
    for offset in range(0, len(images), 16):
        batch = torch.stack(images[offset : offset + 16]).to(device)
        batch = dequantize_normalized_pixels(batch, generator=generator)
        loss = flow_matching_loss(model, batch, generator=generator)
        weighted_loss += float(loss) * batch.shape[0]
        count += batch.shape[0]
    model.train(was_training)
    return weighted_loss / max(count, 1)


def _checkpoint_manifest_without(run_dir: Path, name: str) -> None:
    manifest_path = run_dir / "checkpoint_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    if isinstance(manifest, list):
        atomic_json(manifest_path, [item for item in manifest if not isinstance(item, dict) or item.get("name") != name])


def _remove_checkpoint(run_dir: Path, name: str) -> None:
    (run_dir / "checkpoints" / name).unlink(missing_ok=True)
    _checkpoint_manifest_without(run_dir, name)


def _restore_flow_checkpoint(
    checkpoint_path: Path,
    config: dict[str, Any],
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("config") != config:
        raise ValueError("Resume configuration differs from the checkpoint configuration")
    required = ("model_state_dict", "ema_model_state_dict", "optimizer_state_dict", "rng_state")
    if any(checkpoint.get(name) is None for name in required):
        raise ValueError("Resume checkpoint is missing model, EMA, optimizer, or RNG state")
    if int(checkpoint.get("batch_in_epoch", 0)) != 0:
        raise ValueError("Flow Matching resumes only from epoch-boundary checkpoints")
    model.load_state_dict(checkpoint.get("training_model_state_dict") or checkpoint["model_state_dict"])
    ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def run_training(
    config: dict[str, Any], device: torch.device, *, resume_path: Path | None = None
) -> Path:
    validate_config(config)
    seed = int(config.get("seed", 42))
    if resume_path is None:
        seed_everything(seed)
    training = config["training"]
    dataset_config = config["dataset"]
    train_source = load_eurosat_split("train", PROJECT_ROOT / dataset_config["cache_dir"])
    validation_source = load_eurosat_split("validation", PROJECT_ROOT / dataset_config["cache_dir"])
    if len(train_source) != EXPECTED_SPLIT_COUNTS["train"]:
        raise ValueError(f"Expected {EXPECTED_SPLIT_COUNTS['train']} train images, got {len(train_source)}")
    if len(validation_source) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise ValueError(f"Expected {EXPECTED_SPLIT_COUNTS['validation']} validation images, got {len(validation_source)}")
    train_images = UnconditionalEuroSATSplit(train_source)
    validation_images = UnconditionalEuroSATSplit(validation_source)
    validation_rng = torch.Generator(device="cpu").manual_seed(seed + 20_000)
    validation_indices = torch.randperm(len(validation_images), generator=validation_rng)[
        : min(int(training["validation_images"]), len(validation_images))
    ].tolist()
    fixed_validation = [validation_images[index] for index in validation_indices]

    model = ContinuousTimeVelocityUNet().to(device)
    ema_model = copy.deepcopy(model).to(device).eval()
    optimizer_type = torch.optim.Adam if training["optimizer"].lower() == "adam" else torch.optim.AdamW
    optimizer = optimizer_type(model.parameters(), lr=float(training["learning_rate"]))
    epochs = int(training["epochs"])
    effective_batch = int(training["effective_batch_size"])
    micro_batch = int(training["micro_batch_size"])
    batches_per_epoch = (len(train_images) + effective_batch - 1) // effective_batch
    planned_updates = epochs * batches_per_epoch
    if resume_path is None:
        run_dir = create_run_dir(PROJECT_ROOT / "runs", seed, "flow-matching")
        atomic_json(run_dir / "config.json", {
            "run_id": run_dir.name,
            "created_at": now_iso(),
            "config": config,
            "parameter_count": _model_parameter_count(model),
            "git_commit": _git_commit(),
        })
        global_step = 0
        start_epoch = 0
        best_validation: float | None = None
        best_ema_validation: float | None = None
        elapsed_before_resume = 0.0
    else:
        resume_path = resume_path.resolve()
        if resume_path.name != LATEST_FLOW_MATCHING_CHECKPOINT or resume_path.parent.name != "checkpoints":
            raise ValueError(f"Resume from the rolling {LATEST_FLOW_MATCHING_CHECKPOINT} checkpoint only")
        run_dir = resume_path.parent.parent
        checkpoint = _restore_flow_checkpoint(resume_path, config, model, ema_model, optimizer)
        global_step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint["epoch"])
        best_validation = checkpoint.get("best_validation_loss")
        best_ema_validation = checkpoint.get("best_ema_validation_loss")
        previous_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        if previous_status.get("state") == "completed":
            raise ValueError("A completed Flow Matching run cannot be resumed")
        elapsed_before_resume = float(previous_status.get("elapsed_seconds", 0.0))
        if start_epoch >= epochs:
            raise ValueError("Resume checkpoint already reached the configured epoch count")
        restore_rng_state(checkpoint["rng_state"])
    status_fields: dict[str, Any] = {
        "state": "running", "mode": "train", "process": "flow_matching",
        "objective": "conditional_flow_matching", "device": str(device),
        "epoch": start_epoch, "total_epochs": epochs, "global_step": global_step,
        "planned_steps": planned_updates, "effective_batch_size": effective_batch,
        "micro_batch_size": micro_batch, "error": None,
        "elapsed_seconds": elapsed_before_resume,
    }
    if resume_path is None:
        status_fields["started_at"] = now_iso()
    else:
        status_fields["resumed_at"] = now_iso()
    status_update(run_dir, **status_fields)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    event_id = last_event_id(run_dir / "metrics.jsonl")
    elapsed_before_resume = float(elapsed_before_resume)
    if resume_path is None:
        save_checkpoint(
            run_dir, LATEST_FLOW_MATCHING_CHECKPOINT, "latest-flow-matching", model, optimizer, config,
            epoch=0, batch_in_epoch=0, global_step=0, best_validation_loss=None,
            best_ema_validation_loss=None, ema_model=ema_model,
        )
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            order_generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
            indices = torch.randperm(len(train_images), generator=order_generator).tolist()
            total_loss = 0.0
            for batch_number, start in enumerate(range(0, len(indices), effective_batch), start=1):
                batch_indices = indices[start : start + effective_batch]
                batch_cpu = torch.stack([train_images[index] for index in batch_indices])
                if bool(training.get("horizontal_flip", False)):
                    flip_mask = torch.rand(batch_cpu.shape[0]) < 0.5
                    batch_cpu[flip_mask] = batch_cpu[flip_mask].flip(-1)
                optimizer.zero_grad(set_to_none=True)
                batch_loss = 0.0
                for offset in range(0, batch_cpu.shape[0], micro_batch):
                    images = batch_cpu[offset : offset + micro_batch].to(device, non_blocking=device.type == "cuda")
                    images = dequantize_normalized_pixels(images)
                    loss = flow_matching_loss(model, images)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite FM loss at update {global_step + 1}")
                    fraction = images.shape[0] / batch_cpu.shape[0]
                    (loss * fraction).backward()
                    batch_loss += float(loss.detach()) * fraction
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"Non-finite FM gradient at update {global_step + 1}")
                learning_rate = float(training["learning_rate"])
                warmup_updates = int(training["warmup_updates"])
                if warmup_updates:
                    learning_rate *= min((global_step + 1) / warmup_updates, 1.0)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimizer.step()
                update_ema(ema_model, model, float(training["ema_decay"]))
                global_step += 1
                total_loss += batch_loss
                event_id = append_metric(run_dir, {
                    "kind": "train", "epoch": epoch + 1, "global_step": global_step,
                    "loss": batch_loss, "learning_rate": learning_rate,
                    "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started,
                }, event_id)
                if global_step % 10 == 0 or global_step == planned_updates:
                    peak = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
                    status_update(
                        run_dir, state="running", epoch=epoch + 1, total_epochs=epochs,
                        global_step=global_step, planned_steps=planned_updates,
                        train_loss=batch_loss,
                        elapsed_seconds=elapsed_before_resume + time.perf_counter() - started,
                        steps_per_second=global_step / max(elapsed_before_resume + time.perf_counter() - started, 1e-9),
                        peak_gpu_memory_mb=peak,
                    )

            validation_seed = seed + 30_000 + epoch
            validation_loss = _validate(model, fixed_validation, device, validation_seed)
            ema_validation_loss = _validate(ema_model, fixed_validation, device, validation_seed)
            event_id = append_metric(run_dir, {
                "kind": "validation", "epoch": epoch + 1, "global_step": global_step,
                "loss": validation_loss, "ema_loss": ema_validation_loss,
                "num_images": len(fixed_validation),
                "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started,
            }, event_id)
            if best_validation is None or validation_loss < best_validation:
                best_validation = validation_loss
                save_checkpoint(
                    run_dir, BEST_FLOW_MATCHING_CHECKPOINT, "best-validation-flow-matching", model, optimizer, config,
                    epoch=epoch + 1, batch_in_epoch=0, global_step=global_step,
                    best_validation_loss=best_validation, best_ema_validation_loss=best_ema_validation,
                    validation_loss=validation_loss, ema_model=ema_model, include_optimizer=False,
                )
            if best_ema_validation is None or ema_validation_loss < best_ema_validation:
                best_ema_validation = ema_validation_loss
                save_checkpoint(
                    run_dir, BEST_FLOW_MATCHING_EMA_CHECKPOINT, "best-validation-ema-flow-matching", model, optimizer, config,
                    epoch=epoch + 1, batch_in_epoch=0, global_step=global_step,
                    best_validation_loss=best_validation, best_ema_validation_loss=best_ema_validation,
                    validation_loss=ema_validation_loss, ema_model=ema_model,
                    weights_model=ema_model, include_optimizer=False,
                )
            save_checkpoint(
                run_dir, LATEST_FLOW_MATCHING_CHECKPOINT, "latest-flow-matching", model, optimizer, config,
                epoch=epoch + 1, batch_in_epoch=0, global_step=global_step,
                best_validation_loss=best_validation,
                best_ema_validation_loss=best_ema_validation, ema_model=ema_model,
            )
            status_update(
                run_dir, state="running", epoch=epoch + 1, total_epochs=epochs,
                global_step=global_step, planned_steps=planned_updates,
                epoch_train_loss=total_loss / batches_per_epoch,
                validation_loss=validation_loss, best_validation_loss=best_validation,
                ema_validation_loss=ema_validation_loss, best_ema_validation_loss=best_ema_validation,
                elapsed_seconds=elapsed_before_resume + time.perf_counter() - started,
                peak_gpu_memory_mb=(torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None),
            )
        status_update(
            run_dir, state="completed", mode="train", process="flow_matching",
            epoch=epochs, total_epochs=epochs, global_step=global_step, planned_steps=planned_updates,
            elapsed_seconds=elapsed_before_resume + time.perf_counter() - started,
            peak_gpu_memory_mb=(torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None),
            completed_at=now_iso(),
        )
        _remove_checkpoint(run_dir, LATEST_FLOW_MATCHING_CHECKPOINT)
        print(f"Flow-matching run artifacts: {run_dir}", flush=True)
        return run_dir
    except KeyboardInterrupt:
        status_update(run_dir, state="interrupted", global_step=global_step, elapsed_seconds=elapsed_before_resume + time.perf_counter() - started)
        print(f"Training interrupted. Resume with --resume {run_dir / 'checkpoints' / LATEST_FLOW_MATCHING_CHECKPOINT}", file=sys.stderr)
        raise
    except Exception as error:
        status_update(run_dir, state="failed", global_step=global_step, error=f"{type(error).__name__}: {error}")
        raise


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
            text=True, check=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the unconditional EuroSAT flow-matching model")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "eurosat_flow_matching_50_epoch.toml")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight", action="store_true", help="Measure one forward/backward batch without updating weights")
    actions.add_argument("--run-training", action="store_true", help="Run the configured full training job")
    actions.add_argument("--resume", type=Path, help="Resume a Flow Matching run from its rolling latest checkpoint")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    config = _load_flow_config(args.config)
    device = choose_device(args.device)
    if args.preflight:
        run_preflight(config, device, config_path=args.config)
    else:
        run_training(config, device, resume_path=args.resume)


if __name__ == "__main__":
    main()
