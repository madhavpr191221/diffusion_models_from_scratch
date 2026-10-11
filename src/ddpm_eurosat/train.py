"""CLI training loop for the unconditional EuroSAT DDPM."""

from __future__ import annotations

import argparse
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
from ddpm_eurosat.model import TimeConditionedUNet

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PILOT_IMAGES = 64
PILOT_UPDATES = 200


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


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
    os.replace(temporary, path)


def make_schedule(config: dict[str, Any], device: torch.device) -> DiffusionSchedule:
    diffusion = config["diffusion"]
    return DiffusionSchedule(
        timesteps=int(diffusion["timesteps"]),
        beta_start=float(diffusion["beta_start"]),
        beta_end=float(diffusion["beta_end"]),
    ).to(device)


def make_model() -> TimeConditionedUNet:
    return TimeConditionedUNet()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    if config["training"]["optimizer"].lower() != "adamw":
        raise ValueError("Stage 4 currently supports the configured AdamW optimizer only")
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
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "epoch": epoch,
        "batch_in_epoch": batch_in_epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "rng_state": capture_rng_state(),
        "saved_at": now_iso(),
    }


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

    seed_everything(seed)
    model = make_model().to(device)
    schedule = make_schedule(config, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]))
    batch_size = int(training["batch_size"])
    epochs = int(training["epochs"])

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
        start_epoch = int(checkpoint["epoch"])
        start_batch = int(checkpoint["batch_in_epoch"])
        global_step = int(checkpoint["global_step"])
        best_validation_loss = checkpoint.get("best_validation_loss")
        previous_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        elapsed_before_resume = float(previous_status.get("elapsed_seconds", 0.0))
    else:
        start_epoch = start_batch = global_step = 0
        best_validation_loss = None
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
    existing_events = read_events(run_dir / "metrics.jsonl")
    event_id = max((item["event_id"] for item in existing_events), default=0)
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

    checkpoint_steps = {
        max(1, round(planned_steps * fraction))
        for fraction in (0.25, 0.50, 0.75, 1.0)
    }
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
                        save_checkpoint(run_dir, f"step-{global_step:07d}.pt", f"milestone-{milestone}", model, optimizer, config, epoch=epoch_index, batch_in_epoch=batch_index + 1, global_step=global_step, best_validation_loss=best_validation_loss)
                        save_checkpoint(run_dir, "latest.pt", "latest", model, optimizer, config, epoch=epoch_index, batch_in_epoch=batch_index + 1, global_step=global_step, best_validation_loss=best_validation_loss)
                    if global_step % 250 == 0:
                        print(f"epoch {epoch_index + 1}/{epochs} step {global_step}/{planned_steps} loss={last_loss:.6f}", flush=True)

                train_loss = epoch_loss_sum / max(epoch_images, 1)
                validation_loss = evaluate(model, schedule, validation_loader, device, seed + 10_000)
                event = {"kind": "validation", "epoch": epoch_index + 1, "global_step": global_step, "loss": validation_loss, "train_epoch_loss": train_loss, "elapsed_seconds": elapsed_seconds()}
                event_id = append_metric(run_dir, event, event_id)
                existing_events.append(event)
                improved = best_validation_loss is None or validation_loss < best_validation_loss
                if improved:
                    best_validation_loss = validation_loss
                    save_checkpoint(run_dir, "best-val.pt", "best-validation", model, optimizer, config, epoch=epoch_index + 1, batch_in_epoch=0, global_step=global_step, best_validation_loss=best_validation_loss, validation_loss=validation_loss)
                save_checkpoint(run_dir, "latest.pt", "latest", model, optimizer, config, epoch=epoch_index + 1, batch_in_epoch=0, global_step=global_step, best_validation_loss=best_validation_loss, validation_loss=validation_loss)
                elapsed = elapsed_seconds()
                memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
                status_update(run_dir, global_step=global_step, epoch=epoch_index + 1, batch_in_epoch=0, train_loss=train_loss, validation_loss=validation_loss, best_validation_loss=best_validation_loss, elapsed_seconds=elapsed, steps_per_second=global_step / max(elapsed, 1e-9), peak_gpu_memory_mb=memory_mb)
                print(f"epoch {epoch_index + 1}/{epochs} train_mse={train_loss:.6f} validation_mse={validation_loss:.6f} elapsed={elapsed / 60:.1f} min", flush=True)

            if global_step not in checkpoint_steps:
                save_checkpoint(run_dir, f"step-{global_step:07d}.pt", "milestone-100", model, optimizer, config, epoch=epochs, batch_in_epoch=0, global_step=global_step, best_validation_loss=best_validation_loss)
            memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
            elapsed = elapsed_seconds()
            status_update(run_dir, state="completed", epoch=epochs, global_step=global_step, elapsed_seconds=elapsed, steps_per_second=global_step / max(elapsed, 1e-9), peak_gpu_memory_mb=memory_mb, completed_at=now_iso())
        if pilot:
            memory_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
            elapsed = elapsed_seconds()
            status_update(run_dir, state="completed", epoch=1, global_step=global_step, elapsed_seconds=elapsed, steps_per_second=global_step / max(elapsed, 1e-9), peak_gpu_memory_mb=memory_mb, completed_at=now_iso())
    except KeyboardInterrupt:
        if not pilot:
            save_checkpoint(run_dir, "latest.pt", "latest", model, optimizer, config, epoch=epoch_index, batch_in_epoch=next_batch_in_epoch, global_step=global_step, best_validation_loss=best_validation_loss)
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
