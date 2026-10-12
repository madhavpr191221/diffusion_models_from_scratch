"""CLI training loop for the unconditional EuroSAT DDPM."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from ddpm_eurosat.data import EXPECTED_SPLIT_COUNTS, UnconditionalEuroSATSplit, load_eurosat_split
from ddpm_eurosat.diffusion import DiffusionSchedule, q_sample, sample_timesteps
from ddpm_eurosat.model import build_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PILOT_IMAGES = 64
PILOT_UPDATES = 200


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def replace_with_retry(source: Path, destination: Path) -> None:
    for attempt in range(20):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.025 * (attempt + 1))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    replace_with_retry(temporary, path)


def append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
        stream.flush()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    replace_with_retry(temporary, path)


def make_schedule(config: dict[str, Any], device: torch.device) -> DiffusionSchedule:
    diffusion = config["diffusion"]
    return DiffusionSchedule(
        timesteps=int(diffusion["timesteps"]),
        beta_start=float(diffusion["beta_start"]),
        beta_end=float(diffusion["beta_end"]),
    ).to(device)


def make_model(config: dict[str, Any]) -> nn.Module:
    return build_model(config)


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """Update the inference shadow weights after one optimizer update."""
    model_state = model.state_dict()
    for name, ema_value in ema_model.state_dict().items():
        source = model_state[name].detach()
        if ema_value.is_floating_point():
            ema_value.lerp_(source, 1.0 - decay)
        else:
            ema_value.copy_(source)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if config["training"]["optimizer"].lower() not in {"adam", "adamw"}:
        raise ValueError("training.optimizer must be Adam or AdamW")
    checkpoint_every_epochs = int(config["training"].get("checkpoint_every_epochs", 0))
    if checkpoint_every_epochs < 0:
        raise ValueError("checkpoint_every_epochs must be zero or greater")
    return config


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch environment")
    return torch.device(requested)


def status_update(run_dir: Path, **fields: Any) -> None:
    status_path = run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    status.update(fields)
    status["updated_at"] = now_iso()
    atomic_json(status_path, status)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    *,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    best_validation_loss: float | None,
    best_ema_validation_loss: float | None = None,
    ema_model: nn.Module | None = None,
    weights_model: nn.Module | None = None,
    include_optimizer: bool = True,
) -> dict[str, Any]:
    payload = {
        "format_version": 1,
        "model_state_dict": (weights_model if weights_model is not None else model).state_dict(),
        "training_model_state_dict": model.state_dict() if weights_model is not None and include_optimizer else None,
        "config": config,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "best_ema_validation_loss": best_ema_validation_loss,
        "saved_at": now_iso(),
    }
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["rng_state"] = capture_rng_state()
    if ema_model is not None:
        payload["ema_model_state_dict"] = ema_model.state_dict()
    return payload


def update_checkpoint_manifest(run_dir: Path, item: dict[str, Any]) -> None:
    path = run_dir / "checkpoint_manifest.json"
    items = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    items = [existing for existing in items if existing["name"] != item["name"]]
    items.append(item)
    items.sort(key=lambda entry: (entry["global_step"], entry["name"]))
    atomic_json(path, items)


def save_checkpoint(
    run_dir: Path,
    name: str,
    kind: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    *,
    epoch: int,
    batch_in_epoch: int,
    global_step: int,
    best_validation_loss: float | None,
    validation_loss: float | None = None,
    best_ema_validation_loss: float | None = None,
    ema_model: nn.Module | None = None,
    weights_model: nn.Module | None = None,
    include_optimizer: bool = True,
) -> None:
    path = run_dir / "checkpoints" / name
    payload = checkpoint_payload(
        model,
        optimizer,
        config,
        epoch=epoch,
        batch_in_epoch=batch_in_epoch,
        global_step=global_step,
        best_validation_loss=best_validation_loss,
        best_ema_validation_loss=best_ema_validation_loss,
        ema_model=ema_model,
        weights_model=weights_model,
        include_optimizer=include_optimizer,
    )
    atomic_torch_save(path, payload)
    update_checkpoint_manifest(
        run_dir,
        {
            "name": name,
            "kind": kind,
            "epoch": epoch + 1 if batch_in_epoch else epoch,
            "global_step": global_step,
            "validation_loss": validation_loss,
            "bytes": path.stat().st_size,
            "saved_at": payload["saved_at"],
        },
    )


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                # Ignore a partial final line left by an interrupted write.
                continue
    return events


def append_metric(run_dir: Path, event: dict[str, Any], previous_event_id: int) -> int:
    metrics_path = run_dir / "metrics.jsonl"
    event["event_id"] = previous_event_id + 1
    event["recorded_at"] = now_iso()
    append_event(metrics_path, event)
    return int(event["event_id"])


def evaluate(
    model: nn.Module,
    schedule: DiffusionSchedule,
    loader: DataLoader[torch.Tensor],
    device: torch.device,
    validation_seed: int,
) -> float:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(validation_seed)
    total_loss = 0.0
    total_images = 0
    with torch.no_grad():
        for x0 in loader:
            x0 = x0.to(device, non_blocking=device.type == "cuda")
            t = torch.randint(0, schedule.timesteps, (x0.shape[0],), generator=generator)
            noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype)
            t = t.to(device)
            noise = noise.to(device)
            noisy, target_noise = q_sample(schedule, x0, t, noise)
            prediction = model(noisy, t)
            loss = nn.functional.mse_loss(prediction, target_noise)
            total_loss += float(loss.item()) * x0.shape[0]
            total_images += x0.shape[0]
    model.train()
    return total_loss / total_images


def create_run_dir(runs_root: Path, seed: int, mode: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = runs_root / f"{stamp}-{mode}-seed-{seed}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def last_event_id(path: Path) -> int:
    last_id = 0
    if path.is_file():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    last_id = max(last_id, int(json.loads(line).get("event_id", 0)))
                except (ValueError, json.JSONDecodeError):
                    continue
    return last_id


def run_update_training(
    args: argparse.Namespace,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
    pilot: bool,
) -> int:
    """Train the paper-scale configuration by optimizer update, with accumulation."""
    training = config["training"]
    seed = int(config.get("seed", 42))
    target_updates = int(training["target_updates"])
    pilot_updates = int(training.get("pilot_updates", 3))
    planned_steps = pilot_updates if pilot else target_updates
    effective_batch = int(training["effective_batch_size"])
    micro_batch = int(training["micro_batch_size"])
    if effective_batch < 1 or micro_batch < 1 or micro_batch > effective_batch:
        raise ValueError("effective_batch_size and micro_batch_size are incompatible")
    if int(training["warmup_updates"]) < 0:
        raise ValueError("warmup_updates must be zero or greater")
    if int(training["checkpoint_every_updates"]) < 1 or int(training["validation_every_updates"]) < 1:
        raise ValueError("checkpoint and validation intervals must be positive")

    seed_everything(seed)
    model = make_model(config).to(device)
    ema_decay = float(training["ema_decay"])
    if not 0.0 < ema_decay < 1.0:
        raise ValueError("training.ema_decay must be between zero and one")
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    schedule = make_schedule(config, device)
    optimizer_class = torch.optim.Adam if training["optimizer"].lower() == "adam" else torch.optim.AdamW
    optimizer = optimizer_class(model.parameters(), lr=float(training["learning_rate"]))

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint.get("config", {}).get("model", {}).get("architecture") != config["model"]["architecture"]:
            raise ValueError("Resume checkpoint architecture does not match the saved run configuration")
        model.load_state_dict(checkpoint.get("training_model_state_dict") or checkpoint["model_state_dict"])
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
        global_step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint["batch_in_epoch"])
        best_validation_loss = checkpoint.get("best_validation_loss")
        best_ema_validation_loss = checkpoint.get("best_ema_validation_loss")
        previous_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        elapsed_before_resume = float(previous_status.get("elapsed_seconds", 0.0))
    else:
        global_step = start_epoch = start_batch = 0
        best_validation_loss = best_ema_validation_loss = None
        elapsed_before_resume = 0.0
        atomic_json(
            run_dir / "config.json",
            {
                "config": config,
                "pilot": pilot,
                "seed": seed,
                "device": str(device),
                "dataset": {
                    "repository": config["dataset"]["repository"],
                    "revision": config["dataset"]["revision"],
                    "train_split": "train",
                    "validation_split": "validation",
                    "test_split_loaded": False,
                },
                "created_at": now_iso(),
                "software": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda_runtime": torch.version.cuda},
            },
        )

    train_source = load_eurosat_split("train", PROJECT_ROOT / config["dataset"]["cache_dir"])
    if len(train_source) != EXPECTED_SPLIT_COUNTS["train"]:
        raise ValueError(f"Expected {EXPECTED_SPLIT_COUNTS['train']} train examples, received {len(train_source)}")
    train_dataset = UnconditionalEuroSATSplit(train_source)
    validation_source = load_eurosat_split("validation", PROJECT_ROOT / config["dataset"]["cache_dir"])
    if len(validation_source) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise ValueError(f"Expected {EXPECTED_SPLIT_COUNTS['validation']} validation examples, received {len(validation_source)}")
    validation_dataset = UnconditionalEuroSATSplit(validation_source)
    if not pilot:
        validation_dataset = Subset(validation_dataset, range(min(int(training["validation_images"]), len(validation_dataset))))
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=micro_batch,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    batches_per_epoch = (len(train_dataset) + effective_batch - 1) // effective_batch
    total_epochs = (target_updates + batches_per_epoch - 1) // batches_per_epoch
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def elapsed_seconds() -> float:
        return elapsed_before_resume + time.perf_counter() - started

    raw_metric_path = run_dir / "metrics.jsonl"
    monitor_metric_path = run_dir / "monitor_metrics.jsonl"
    event_id = last_event_id(raw_metric_path)
    monitor_event_id = last_event_id(monitor_metric_path)
    initial_status = {
        "state": "running", "mode": "pilot" if pilot else "train", "device": str(device),
        "epoch": start_epoch, "total_epochs": 1 if pilot else total_epochs,
        "global_step": global_step, "planned_steps": planned_steps,
        "effective_batch_size": effective_batch, "micro_batch_size": micro_batch, "error": None,
    }
    if not args.resume:
        initial_status["started_at"] = now_iso()
    status_update(run_dir, **initial_status)

    current_epoch = start_epoch
    current_batch = start_batch
    pending_loss = 0.0
    pending_count = 0

    def save_latest(epoch: int, batch_position: int, step: int) -> None:
        save_checkpoint(
            run_dir,
            "latest.pt",
            "latest",
            model,
            optimizer,
            config,
            epoch=epoch,
            batch_in_epoch=batch_position,
            global_step=step,
            best_validation_loss=best_validation_loss,
            best_ema_validation_loss=best_ema_validation_loss,
            ema_model=ema_model,
        )

    order_generator = torch.Generator(device="cpu").manual_seed(seed + current_epoch)
    order = torch.randperm(len(train_dataset), generator=order_generator).tolist()
    batch_count = (len(order) + effective_batch - 1) // effective_batch

    try:
        if pilot:
            # A few effective-size updates measure the real architecture, accumulation,
            # optimizer, and EMA path without committing to the long schedule.
            source_indices = [index % len(train_dataset) for index in range(effective_batch)]
            batches = [source_indices]
            update_limit = pilot_updates
        else:
            batches = []
            update_limit = target_updates

        while global_step < update_limit:
            if pilot:
                batch_indices = batches[0]
                epoch_index, batch_index = 0, global_step
            else:
                if current_batch >= batch_count:
                    current_epoch += 1
                    current_batch = 0
                    order_generator = torch.Generator(device="cpu").manual_seed(seed + current_epoch)
                    order = torch.randperm(len(train_dataset), generator=order_generator).tolist()
                    batch_count = (len(order) + effective_batch - 1) // effective_batch
                    continue
                begin = current_batch * effective_batch
                batch_indices = order[begin : begin + effective_batch]
                epoch_index, batch_index = current_epoch, current_batch

            x0_cpu = torch.stack([train_dataset[index] for index in batch_indices])
            if bool(training.get("horizontal_flip", False)):
                flip_mask = torch.rand(x0_cpu.shape[0]) < 0.5
                x0_cpu[flip_mask] = x0_cpu[flip_mask].flip(-1)
            optimizer.zero_grad(set_to_none=True)
            weighted_loss = 0.0
            for offset in range(0, x0_cpu.shape[0], micro_batch):
                x0 = x0_cpu[offset : offset + micro_batch].to(device, non_blocking=device.type == "cuda")
                timesteps = sample_timesteps(x0.shape[0], schedule.timesteps, device=device)
                noisy, noise = q_sample(schedule, x0, timesteps)
                micro_loss = nn.functional.mse_loss(model(noisy, timesteps), noise)
                if not torch.isfinite(micro_loss):
                    raise FloatingPointError(f"Non-finite training loss at update {global_step + 1}")
                fraction = x0.shape[0] / x0_cpu.shape[0]
                (micro_loss * fraction).backward()
                weighted_loss += float(micro_loss.detach()) * fraction

            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite gradient norm at update {global_step + 1}")
            warmup = int(training["warmup_updates"])
            learning_rate = float(training["learning_rate"])
            if warmup:
                learning_rate *= min((global_step + 1) / warmup, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            update_ema(ema_model, model, ema_decay)
            global_step += 1
            if not pilot:
                current_batch += 1

            loss_value = weighted_loss
            event = {
                "kind": "train",
                "epoch": 1 if pilot else epoch_index + 1,
                "global_step": global_step,
                "loss": loss_value,
                "batch_size": int(x0_cpu.shape[0]),
                "learning_rate": learning_rate,
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": elapsed_seconds(),
            }
            event_id = append_metric(run_dir, event, event_id)
            pending_loss += loss_value
            pending_count += 1
            if global_step % int(training["monitor_interval_updates"]) == 0 or global_step == update_limit:
                summary_event = {
                    "kind": "train",
                    "epoch": 1 if pilot else epoch_index + 1,
                    "global_step": global_step,
                    "loss": pending_loss / max(pending_count, 1),
                    "window_updates": pending_count,
                    "elapsed_seconds": elapsed_seconds(),
                }
                monitor_event_id = append_metric_to(monitor_metric_path, summary_event, monitor_event_id)
                pending_loss = 0.0
                pending_count = 0

            if not pilot and global_step % int(training["validation_every_updates"]) == 0:
                raw_validation_loss = evaluate(model, schedule, validation_loader, device, seed + 10_000)
                ema_validation_loss = evaluate(ema_model, schedule, validation_loader, device, seed + 10_000)
                validation_event = {
                    "kind": "validation",
                    "epoch": epoch_index + 1,
                    "global_step": global_step,
                    "loss": raw_validation_loss,
                    "ema_loss": ema_validation_loss,
                    "elapsed_seconds": elapsed_seconds(),
                }
                event_id = append_metric(run_dir, validation_event, event_id)
                monitor_event_id = append_metric_to(monitor_metric_path, validation_event.copy(), monitor_event_id)
                improved = best_validation_loss is None or raw_validation_loss < best_validation_loss
                improved_ema = best_ema_validation_loss is None or ema_validation_loss < best_ema_validation_loss
                if improved:
                    best_validation_loss = raw_validation_loss
                    save_checkpoint(
                        run_dir, "best-val.pt", "best-validation", model, optimizer, config,
                        epoch=epoch_index + 1, batch_in_epoch=current_batch, global_step=global_step,
                        best_validation_loss=best_validation_loss, best_ema_validation_loss=best_ema_validation_loss,
                        include_optimizer=False,
                    )
                if improved_ema:
                    best_ema_validation_loss = ema_validation_loss
                    save_checkpoint(
                        run_dir, "best-val-ema.pt", "best-validation-ema", model, optimizer, config,
                        epoch=epoch_index + 1, batch_in_epoch=current_batch, global_step=global_step,
                        best_validation_loss=best_validation_loss, best_ema_validation_loss=best_ema_validation_loss,
                        weights_model=ema_model, include_optimizer=False,
                    )

            if not pilot and global_step % int(training["model_checkpoint_every_updates"]) == 0:
                save_checkpoint(
                    run_dir,
                    f"ema-step-{global_step:09d}.pt",
                    "ema-progress",
                    model,
                    optimizer,
                    config,
                    epoch=epoch_index + 1,
                    batch_in_epoch=current_batch,
                    global_step=global_step,
                    best_validation_loss=best_validation_loss,
                    best_ema_validation_loss=best_ema_validation_loss,
                    weights_model=ema_model,
                    include_optimizer=False,
                )
            if not pilot and global_step % int(training["checkpoint_every_updates"]) == 0:
                save_latest(epoch_index, current_batch, global_step)

            memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
            if (
                global_step % int(training["status_interval_updates"]) == 0
                or global_step % int(training["validation_every_updates"]) == 0
                or global_step % int(training["checkpoint_every_updates"]) == 0
                or global_step == update_limit
            ):
                status_update(
                    run_dir,
                    state="running",
                    epoch=1 if pilot else epoch_index + 1,
                    total_epochs=1 if pilot else total_epochs,
                    batch_in_epoch=0 if pilot else current_batch,
                    global_step=global_step,
                    planned_steps=planned_steps,
                    train_loss=loss_value,
                    best_validation_loss=best_validation_loss,
                    best_ema_validation_loss=best_ema_validation_loss,
                    elapsed_seconds=elapsed_seconds(),
                    steps_per_second=global_step / max(elapsed_seconds(), 1e-9),
                    peak_gpu_memory_mb=memory_mb,
                )
            if global_step % max(1, int(training["monitor_interval_updates"])) == 0:
                print(
                    f"update {global_step}/{planned_steps} loss={loss_value:.6f} lr={learning_rate:.7g} "
                    f"elapsed={elapsed_seconds()/60:.1f} min peak_gpu_mb={memory_mb}",
                    flush=True,
                )

        if pilot:
            save_checkpoint(
                run_dir, "pilot-final.pt", "pilot", model, optimizer, config,
                epoch=1, batch_in_epoch=0, global_step=global_step,
                best_validation_loss=None, best_ema_validation_loss=None, ema_model=ema_model,
            )
        else:
            if global_step % int(training["validation_every_updates"]) != 0:
                raw_validation_loss = evaluate(model, schedule, validation_loader, device, seed + 10_000)
                ema_validation_loss = evaluate(ema_model, schedule, validation_loader, device, seed + 10_000)
                final_validation = {
                    "kind": "validation", "epoch": current_epoch + 1, "global_step": global_step,
                    "loss": raw_validation_loss, "ema_loss": ema_validation_loss, "elapsed_seconds": elapsed_seconds(),
                }
                event_id = append_metric(run_dir, final_validation, event_id)
                monitor_event_id = append_metric_to(monitor_metric_path, final_validation.copy(), monitor_event_id)
                if best_validation_loss is None or raw_validation_loss < best_validation_loss:
                    best_validation_loss = raw_validation_loss
                    save_checkpoint(
                        run_dir, "best-val.pt", "best-validation", model, optimizer, config,
                        epoch=current_epoch + 1, batch_in_epoch=current_batch, global_step=global_step,
                        best_validation_loss=best_validation_loss, best_ema_validation_loss=best_ema_validation_loss,
                        include_optimizer=False,
                    )
                if best_ema_validation_loss is None or ema_validation_loss < best_ema_validation_loss:
                    best_ema_validation_loss = ema_validation_loss
                    save_checkpoint(
                        run_dir, "best-val-ema.pt", "best-validation-ema", model, optimizer, config,
                        epoch=current_epoch + 1, batch_in_epoch=current_batch, global_step=global_step,
                        best_validation_loss=best_validation_loss, best_ema_validation_loss=best_ema_validation_loss,
                        weights_model=ema_model, include_optimizer=False,
                    )
            full_validation_loader = DataLoader(
                UnconditionalEuroSATSplit(validation_source),
                batch_size=micro_batch,
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            full_raw_loss = evaluate(model, schedule, full_validation_loader, device, seed + 20_000)
            full_ema_loss = evaluate(ema_model, schedule, full_validation_loader, device, seed + 20_000)
            full_event = {
                "kind": "validation_full", "epoch": current_epoch + 1, "global_step": global_step,
                "loss": full_raw_loss, "ema_loss": full_ema_loss, "elapsed_seconds": elapsed_seconds(),
            }
            event_id = append_metric(run_dir, full_event, event_id)
            monitor_event_id = append_metric_to(monitor_metric_path, full_event.copy(), monitor_event_id)
            if global_step % int(training["checkpoint_every_updates"]) != 0:
                save_latest(current_epoch, current_batch, global_step)
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
        status_update(
            run_dir,
            state="completed",
            epoch=1 if pilot else current_epoch + 1,
            total_epochs=1 if pilot else total_epochs,
            global_step=global_step,
            planned_steps=planned_steps,
            elapsed_seconds=elapsed_seconds(),
            steps_per_second=global_step / max(elapsed_seconds(), 1e-9),
            peak_gpu_memory_mb=memory_mb,
            completed_at=now_iso(),
        )
        return 0
    except KeyboardInterrupt:
        if not pilot:
            save_latest(current_epoch, current_batch, global_step)
        status_update(run_dir, state="interrupted", global_step=global_step, elapsed_seconds=elapsed_seconds())
        print(f"Training interrupted. Resume with --resume {run_dir / 'checkpoints' / 'latest.pt'}", file=sys.stderr)
        return 130
    except Exception as error:
        status_update(run_dir, state="failed", global_step=global_step, error=f"{type(error).__name__}: {error}", elapsed_seconds=elapsed_seconds())
        raise


def append_metric_to(path: Path, event: dict[str, Any], previous_event_id: int) -> int:
    event["event_id"] = previous_event_id + 1
    event["recorded_at"] = now_iso()
    append_event(path, event)
    return int(event["event_id"])


def run_training(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = load_config(config_path)
    training = config["training"]
    seed = int(config.get("seed", 42))
    pilot = bool(args.pilot)
    device = choose_device(args.device)

    if args.resume:
        resume_path = args.resume.resolve()
        run_dir = resume_path.parent.parent
        if not (run_dir / "config.json").is_file():
            raise ValueError("Resume checkpoint must be inside a run with config.json")
        saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        config = saved_config["config"]
        training = config["training"]
        seed = int(config.get("seed", seed))
        if bool(saved_config.get("pilot")):
            pilot = True
        if pilot:
            raise ValueError("Pilot checkpoints are diagnostic and cannot resume a baseline run")
    else:
        if args.run_dir:
            run_dir = args.run_dir.resolve()
            run_dir.mkdir(parents=True, exist_ok=False)
        else:
            run_dir = create_run_dir(PROJECT_ROOT / "runs", seed, "pilot" if pilot else "train")

    training = config["training"]
    checkpoint_every_epochs = int(training.get("checkpoint_every_epochs", 0))
    if checkpoint_every_epochs < 0:
        raise ValueError("checkpoint_every_epochs must be zero or greater")

    if "target_updates" in training:
        return run_update_training(args, config, run_dir, choose_device(args.device), pilot)

    seed_everything(seed)
    model = make_model(config).to(device)
    ema_decay = training.get("ema_decay")
    if ema_decay is not None:
        ema_decay = float(ema_decay)
        if not 0.0 < ema_decay < 1.0:
            raise ValueError("training.ema_decay must be between zero and one")
        if pilot:
            raise ValueError("EMA is intended for full training runs, not the diagnostic pilot")
        ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    else:
        ema_model = None
    schedule = make_schedule(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]))
    batch_size = int(training["batch_size"])
    epochs = int(training["epochs"])

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint.get("training_model_state_dict") or checkpoint["model_state_dict"])
        if ema_model is not None:
            if "ema_model_state_dict" not in checkpoint:
                raise ValueError("EMA training cannot resume from a checkpoint without EMA state; start a new run")
            ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint["batch_in_epoch"])
        global_step = int(checkpoint["global_step"])
        best_validation_loss = checkpoint.get("best_validation_loss")
        best_ema_validation_loss = checkpoint.get("best_ema_validation_loss")
        previous_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        elapsed_before_resume = float(previous_status.get("elapsed_seconds", 0.0))
    else:
        start_epoch = start_batch = global_step = 0
        best_validation_loss = None
        best_ema_validation_loss = None
        elapsed_before_resume = 0.0
        atomic_json(
            run_dir / "config.json",
            {
                "config": config,
                "pilot": pilot,
                "seed": seed,
                "device": str(device),
                "dataset": {
                    "repository": "blanchon/EuroSAT_RGB",
                    "train_split": "train",
                    "validation_split": "validation",
                    "test_split_loaded": False,
                },
                "created_at": now_iso(),
                "software": {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                },
            },
        )

    train_source = load_eurosat_split("train", PROJECT_ROOT / config["dataset"]["cache_dir"])
    if len(train_source) != EXPECTED_SPLIT_COUNTS["train"]:
        raise ValueError(
            f"Expected {EXPECTED_SPLIT_COUNTS['train']} train examples, received {len(train_source)}"
        )
    train_dataset = UnconditionalEuroSATSplit(train_source)
    if pilot:
        subset_indices = random.Random(seed).sample(range(len(train_dataset)), PILOT_IMAGES)
        train_dataset = Subset(train_dataset, subset_indices)
        planned_steps = PILOT_UPDATES
        total_epochs = 1
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=device.type == "cuda")
        validation_loader = None
    else:
        planned_steps = ((len(train_dataset) + batch_size - 1) // batch_size) * epochs
        total_epochs = epochs
        train_loader = None
        validation_source = load_eurosat_split("validation", PROJECT_ROOT / config["dataset"]["cache_dir"])
        if len(validation_source) != EXPECTED_SPLIT_COUNTS["validation"]:
            raise ValueError(
                f"Expected {EXPECTED_SPLIT_COUNTS['validation']} validation examples, received {len(validation_source)}"
            )
        validation_dataset = UnconditionalEuroSATSplit(validation_source)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def elapsed_seconds() -> float:
        return elapsed_before_resume + time.perf_counter() - started
    all_events = read_events(run_dir / "metrics.jsonl")
    event_id = max((item["event_id"] for item in all_events), default=0)
    # A failed or interrupted attempt can have metric rows beyond its last
    # durable checkpoint. Keep those rows in the append-only log, but exclude
    # them from resumed epoch aggregates because training will replay them.
    existing_events = [item for item in all_events if int(item.get("global_step", 0)) <= global_step]
    initial_status = {
        "state": "running",
        "mode": "pilot" if pilot else "train",
        "device": str(device),
        "epoch": start_epoch,
        "total_epochs": total_epochs,
        "global_step": global_step,
        "planned_steps": planned_steps,
        "error": None,
    }
    if not args.resume:
        initial_status["started_at"] = now_iso()
    status_update(run_dir, **initial_status)

    checkpoint_steps = (
        set()
        if checkpoint_every_epochs
        else {
            max(1, round(planned_steps * fraction))
            for fraction in (0.25, 0.50, 0.75, 1.0)
        }
    )
    event_id = max(event_id, 0)
    epoch_index = start_epoch
    batch_index = max(start_batch - 1, -1)
    next_batch_in_epoch = start_batch
    try:
        if pilot:
            model.train()
            batch_iter = iter(train_loader)
            recent_start = None
            for update in range(1, PILOT_UPDATES + 1):
                try:
                    x0 = next(batch_iter)
                except StopIteration:
                    batch_iter = iter(train_loader)
                    x0 = next(batch_iter)
                x0 = x0.to(device, non_blocking=device.type == "cuda")
                t = sample_timesteps(x0.shape[0], schedule.timesteps, device=device)
                noisy, noise = q_sample(schedule, x0, t)
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(noisy, t), noise)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite pilot loss at update {update}")
                loss.backward()
                optimizer.step()
                if ema_model is not None:
                    update_ema(ema_model, model, ema_decay)
                global_step += 1
                elapsed = elapsed_seconds()
                event = {"kind": "train", "epoch": 1, "global_step": global_step, "loss": float(loss.item()), "batch_size": int(x0.shape[0]), "elapsed_seconds": elapsed}
                event_id = append_metric(run_dir, event, event_id)
                existing_events.append(event)
                status_update(run_dir, global_step=global_step, epoch=1, elapsed_seconds=elapsed)
                if update % 50 == 0:
                    print(f"pilot update {update}/{PILOT_UPDATES} loss={loss.item():.6f}", flush=True)
            save_checkpoint(run_dir, "pilot-final.pt", "pilot", model, optimizer, config, epoch=1, batch_in_epoch=0, global_step=global_step, best_validation_loss=None)
        else:
            model.train()
            for epoch_index in range(start_epoch, epochs):
                order_generator = torch.Generator(device="cpu").manual_seed(seed + epoch_index)
                order = torch.randperm(len(train_dataset), generator=order_generator).tolist()
                batch_count = (len(order) + batch_size - 1) // batch_size
                first_batch = start_batch if epoch_index == start_epoch else 0
                next_batch_in_epoch = first_batch
                prior_epoch_events = [item for item in existing_events if item.get("kind") == "train" and item.get("epoch") == epoch_index + 1]
                epoch_loss_sum = sum(float(item["loss"]) * int(item.get("batch_size", batch_size)) for item in prior_epoch_events)
                epoch_images = sum(int(item.get("batch_size", batch_size)) for item in prior_epoch_events)
                last_loss = float("nan")
                for batch_index in range(first_batch, batch_count):
                    begin = batch_index * batch_size
                    indices = order[begin : begin + batch_size]
                    x0 = torch.stack([train_dataset[index] for index in indices])
                    x0 = x0.to(device, non_blocking=device.type == "cuda")
                    t = sample_timesteps(x0.shape[0], schedule.timesteps, device=device)
                    noisy, noise = q_sample(schedule, x0, t)
                    optimizer.zero_grad(set_to_none=True)
                    loss = nn.functional.mse_loss(model(noisy, t), noise)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite training loss at step {global_step + 1}")
                    loss.backward()
                    optimizer.step()
                    if ema_model is not None:
                        update_ema(ema_model, model, ema_decay)
                    next_batch_in_epoch = batch_index + 1
                    global_step += 1
                    last_loss = float(loss.item())
                    epoch_loss_sum += last_loss * x0.shape[0]
                    epoch_images += x0.shape[0]
                    elapsed = elapsed_seconds()
                    event = {"kind": "train", "epoch": epoch_index + 1, "global_step": global_step, "loss": last_loss, "batch_size": int(x0.shape[0]), "elapsed_seconds": elapsed}
                    event_id = append_metric(run_dir, event, event_id)
                    existing_events.append(event)
                    status_update(run_dir, global_step=global_step, epoch=epoch_index + 1, batch_in_epoch=batch_index + 1, elapsed_seconds=elapsed)
                    if global_step in checkpoint_steps:
                        fraction = global_step / planned_steps
                        milestone = min((25, 50, 75, 100), key=lambda value: abs(value / 100 - fraction))
                        save_checkpoint(run_dir, f"step-{global_step:07d}.pt", f"milestone-{milestone}", model, optimizer, config, epoch=epoch_index, batch_in_epoch=batch_index + 1, global_step=global_step, best_validation_loss=best_validation_loss, best_ema_validation_loss=best_ema_validation_loss, ema_model=ema_model)
                        save_checkpoint(run_dir, "latest.pt", "latest", model, optimizer, config, epoch=epoch_index, batch_in_epoch=batch_index + 1, global_step=global_step, best_validation_loss=best_validation_loss, best_ema_validation_loss=best_ema_validation_loss, ema_model=ema_model)
                    if global_step % 250 == 0:
                        print(f"epoch {epoch_index + 1}/{epochs} step {global_step}/{planned_steps} loss={last_loss:.6f}", flush=True)

                train_loss = epoch_loss_sum / max(epoch_images, 1)
                validation_loss = evaluate(model, schedule, validation_loader, device, seed + 10_000)
                ema_validation_loss = (
                    evaluate(ema_model, schedule, validation_loader, device, seed + 10_000)
                    if ema_model is not None
                    else None
                )
                event = {"kind": "validation", "epoch": epoch_index + 1, "global_step": global_step, "loss": validation_loss, "ema_loss": ema_validation_loss, "train_epoch_loss": train_loss, "elapsed_seconds": elapsed_seconds()}
                event_id = append_metric(run_dir, event, event_id)
                existing_events.append(event)
                improved = best_validation_loss is None or validation_loss < best_validation_loss
                improved_ema = (
                    ema_model is not None
                    and (best_ema_validation_loss is None or ema_validation_loss < best_ema_validation_loss)
                )
                if improved:
                    best_validation_loss = validation_loss
                if improved_ema:
                    best_ema_validation_loss = ema_validation_loss
                if improved:
                    save_checkpoint(run_dir, "best-val.pt", "best-validation", model, optimizer, config, epoch=epoch_index + 1, batch_in_epoch=0, global_step=global_step, best_validation_loss=best_validation_loss, best_ema_validation_loss=best_ema_validation_loss, validation_loss=validation_loss, ema_model=ema_model)
                if improved_ema:
                    save_checkpoint(run_dir, "best-val-ema.pt", "best-validation-ema", model, optimizer, config, epoch=epoch_index + 1, batch_in_epoch=0, global_step=global_step, best_validation_loss=best_validation_loss, best_ema_validation_loss=best_ema_validation_loss, validation_loss=ema_validation_loss, ema_model=ema_model, weights_model=ema_model)
                save_checkpoint(run_dir, "latest.pt", "latest", model, optimizer, config, epoch=epoch_index + 1, batch_in_epoch=0, global_step=global_step, best_validation_loss=best_validation_loss, best_ema_validation_loss=best_ema_validation_loss, validation_loss=validation_loss, ema_model=ema_model)
                if checkpoint_every_epochs and (
                    (epoch_index + 1) % checkpoint_every_epochs == 0
                    or epoch_index + 1 == epochs
                ):
                    save_checkpoint(
                        run_dir,
                        f"epoch-{epoch_index + 1:05d}.pt",
                        f"epoch-{epoch_index + 1}",
                        model,
                        optimizer,
                        config,
                        epoch=epoch_index + 1,
                        batch_in_epoch=0,
                        global_step=global_step,
                        best_validation_loss=best_validation_loss,
                        best_ema_validation_loss=best_ema_validation_loss,
                        validation_loss=validation_loss,
                        ema_model=ema_model,
                    )
                elapsed = elapsed_seconds()
                memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
                status_update(run_dir, global_step=global_step, epoch=epoch_index + 1, batch_in_epoch=0, train_loss=train_loss, validation_loss=validation_loss, ema_validation_loss=ema_validation_loss, best_validation_loss=best_validation_loss, best_ema_validation_loss=best_ema_validation_loss, elapsed_seconds=elapsed, steps_per_second=global_step / max(elapsed, 1e-9), peak_gpu_memory_mb=memory_mb)
                ema_text = f" ema_val={ema_validation_loss:.6f}" if ema_validation_loss is not None else ""
                print(f"epoch {epoch_index + 1}/{epochs} train_mse={train_loss:.6f} validation_mse={validation_loss:.6f}{ema_text} elapsed={elapsed / 60:.1f} min", flush=True)

            if not checkpoint_every_epochs and global_step not in checkpoint_steps:
                save_checkpoint(run_dir, f"step-{global_step:07d}.pt", "milestone-100", model, optimizer, config, epoch=epochs, batch_in_epoch=0, global_step=global_step, best_validation_loss=best_validation_loss, best_ema_validation_loss=best_ema_validation_loss, ema_model=ema_model)
            memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
            elapsed = elapsed_seconds()
            status_update(run_dir, state="completed", epoch=epochs, global_step=global_step, elapsed_seconds=elapsed, steps_per_second=global_step / max(elapsed, 1e-9), peak_gpu_memory_mb=memory_mb, completed_at=now_iso())
        if pilot:
            memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
            elapsed = elapsed_seconds()
            status_update(run_dir, state="completed", epoch=1, global_step=global_step, elapsed_seconds=elapsed, steps_per_second=global_step / max(elapsed, 1e-9), peak_gpu_memory_mb=memory_mb, completed_at=now_iso())
    except KeyboardInterrupt:
        if not pilot:
            save_checkpoint(run_dir, "latest.pt", "latest", model, optimizer, config, epoch=epoch_index, batch_in_epoch=next_batch_in_epoch, global_step=global_step, best_validation_loss=best_validation_loss, best_ema_validation_loss=best_ema_validation_loss, ema_model=ema_model)
        status_update(run_dir, state="interrupted", global_step=global_step, elapsed_seconds=elapsed_seconds())
        print(f"Training interrupted. Resume with --resume {run_dir / 'checkpoints' / 'latest.pt'}", file=sys.stderr)
        return 130
    except Exception as error:
        status_update(run_dir, state="failed", global_step=global_step, error=f"{type(error).__name__}: {error}", elapsed_seconds=elapsed_seconds())
        raise

    print(f"Run artifacts: {run_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the unconditional EuroSAT DDPM")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "base.toml")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--pilot", action="store_true", help="Run the fixed 64-image, 200-update pilot")
    parser.add_argument("--run-dir", type=Path, help="Optional explicit output directory for a new run")
    parser.add_argument("--resume", type=Path, help="Resume from a checkpoint in an existing run")
    args = parser.parse_args()
    if args.pilot and args.resume:
        parser.error("--pilot and --resume cannot be used together")
    return args


def main() -> None:
    raise SystemExit(run_training(parse_args()))


if __name__ == "__main__":
    main()
