"""Train an unconditional EuroSAT VP-SDE epsilon or direct-score model."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset

from ddpm_eurosat.data import EXPECTED_SPLIT_COUNTS, UnconditionalEuroSATSplit, load_eurosat_split
from ddpm_eurosat.model import ContinuousTimeDirectScoreUNet, ContinuousTimeOriginalDDPMUNet, build_model
from ddpm_eurosat.train import (
    append_metric,
    append_metric_to,
    atomic_json,
    choose_device,
    create_run_dir,
    last_event_id,
    load_config,
    now_iso,
    restore_rng_state,
    save_checkpoint,
    seed_everything,
    status_update,
    update_ema,
)
from ddpm_eurosat.vp_sde import (
    VPSDESchedule,
    epsilon_prediction_loss,
    q_sample_vp,
    sample_continuous_times,
    weighted_dsm_loss,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def validate_config(config: dict[str, Any]) -> None:
    """Reject configs that would silently select a different process or model."""
    if config.get("process") != "vp_sde":
        raise ValueError("This trainer requires process = 'vp_sde'")
    model_config = config.get("model", {})
    prediction = model_config.get("prediction")
    expected_architecture = {
        "epsilon": "vp_sde_ddpm_cifar_v1",
        "score": "vp_sde_direct_score_ddpm_cifar_v1",
    }.get(prediction)
    if expected_architecture is None or model_config.get("architecture") != expected_architecture:
        raise ValueError("VP-SDE model prediction must be epsilon or score and match its architecture ID")
    if "diffusion" in config:
        raise ValueError("VP-SDE configs must use [vp_sde], not a discrete [diffusion] schedule")

    sde = config["vp_sde"]
    VPSDESchedule(float(sde["beta_min"]), float(sde["beta_max"]))
    if not 0.0 < float(sde["training_time_min"]) < 1.0:
        raise ValueError("vp_sde.training_time_min must be in (0, 1)")

    training = config["training"]
    if str(training["optimizer"]).lower() not in {"adam", "adamw"}:
        raise ValueError("training.optimizer must be 'adam' or 'adamw'")
    for key in ("target_updates", "pilot_updates", "effective_batch_size", "micro_batch_size"):
        if int(training[key]) < 1:
            raise ValueError(f"training.{key} must be positive")
    effective_batch = int(training["effective_batch_size"])
    micro_batch = int(training["micro_batch_size"])
    if effective_batch % micro_batch != 0:
        raise ValueError("micro_batch_size must divide effective_batch_size")
    for key in ("validation_every_updates", "monitor_interval_updates", "status_interval_updates", "checkpoint_every_updates", "model_checkpoint_every_updates"):
        if int(training[key]) < 1:
            raise ValueError(f"training.{key} must be positive")
    if int(training["warmup_updates"]) < 0:
        raise ValueError("training.warmup_updates must be zero or greater")
    if float(training["learning_rate"]) <= 0.0:
        raise ValueError("training.learning_rate must be positive")
    if not 0.0 < float(training["ema_decay"]) < 1.0:
        raise ValueError("training.ema_decay must be between zero and one")


def epsilon_loss_for_batch(
    model: nn.Module,
    schedule: VPSDESchedule,
    x0: Tensor,
    t: Tensor,
    noise: Tensor | None = None,
) -> Tensor:
    """Noise an image batch and return epsilon MSE (the weighted DSM objective)."""
    x_t, target_noise = q_sample_vp(schedule, x0, t, noise)
    return epsilon_prediction_loss(model(x_t, t), target_noise)


def score_loss_for_batch(
    model: nn.Module,
    schedule: VPSDESchedule,
    x0: Tensor,
    t: Tensor,
    noise: Tensor | None = None,
) -> Tensor:
    """Train the direct score output against ``-epsilon / sigma(t)``.

    Sigma-squared weighting makes this DSM loss equal epsilon MSE, but the
    network itself emits the score vector rather than epsilon.
    """
    x_t, target_noise = q_sample_vp(schedule, x0, t, noise)
    return weighted_dsm_loss(schedule, model(x_t, t), target_noise, t)


def loss_for_batch(
    model: nn.Module,
    schedule: VPSDESchedule,
    x0: Tensor,
    t: Tensor,
    noise: Tensor | None = None,
    *,
    prediction: str = "epsilon",
) -> Tensor:
    """Dispatch to the explicit epsilon-MSE or direct-score DSM objective."""
    if prediction == "epsilon":
        return epsilon_loss_for_batch(model, schedule, x0, t, noise)
    if prediction == "score":
        return score_loss_for_batch(model, schedule, x0, t, noise)
    raise ValueError("prediction must be 'epsilon' or 'score'")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    schedule: VPSDESchedule,
    loader: Iterable[Tensor],
    device: torch.device,
    *,
    validation_seed: int,
    training_time_min: float,
    prediction: str = "epsilon",
) -> float:
    """Measure epsilon MSE with fixed per-device times/noise on each call."""
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(validation_seed)
    total_loss = 0.0
    total_images = 0
    try:
        for x0 in loader:
            x0 = x0.to(device, non_blocking=device.type == "cuda")
            t = sample_continuous_times(
                x0.shape[0],
                device=device,
                t_min=training_time_min,
                generator=generator,
            )
            noise = torch.randn(x0.shape, device=device, dtype=x0.dtype, generator=generator)
            loss = loss_for_batch(model, schedule, x0, t, noise, prediction=prediction)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite VP-SDE validation loss")
            total_loss += float(loss) * x0.shape[0]
            total_images += x0.shape[0]
    finally:
        model.train(was_training)
    if total_images == 0:
        raise ValueError("Validation loader is empty")
    return total_loss / total_images


def _run_metadata(config: dict[str, Any], pilot: bool, device: torch.device) -> dict[str, Any]:
    prediction = config["model"]["prediction"]
    return {
        "config": config,
        "pilot": pilot,
        "seed": int(config["seed"]),
        "device": str(device),
        "dataset": {
            "repository": config["dataset"]["repository"],
            "revision": config["dataset"]["revision"],
            "train_split": "train",
            "validation_split": "validation",
            "validation_split_loaded": not pilot,
            "test_split_loaded": False,
        },
        "created_at": now_iso(),
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "loss": "sigma_squared_weighted_dsm_epsilon_mse" if prediction == "epsilon" else "sigma_squared_weighted_dsm_direct_score",
    }


def _append_monitor_event(path: Path, event: dict[str, Any], previous_event_id: int) -> int:
    return append_metric_to(path, event, previous_event_id)


def run_training(args: argparse.Namespace) -> int:
    pilot = bool(args.pilot)
    if args.resume and pilot:
        raise ValueError("A diagnostic pilot cannot resume a training run")

    if args.resume:
        resume_path = args.resume.resolve()
        run_dir = resume_path.parent.parent
        run_metadata_path = run_dir / "config.json"
        if not run_metadata_path.is_file():
            raise ValueError("Resume checkpoint must be inside a run with config.json")
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        config = run_metadata["config"]
        if bool(run_metadata.get("pilot")):
            raise ValueError("Pilot checkpoints are diagnostic and cannot resume a baseline run")
    else:
        config = load_config(args.config.resolve())

    validate_config(config)
    seed = int(config.get("seed", 42))
    device = choose_device(args.device)
    if args.resume:
        checkpoint_path = args.resume.resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
    elif args.run_dir:
        run_dir = args.run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=False)
    else:
        run_dir = create_run_dir(PROJECT_ROOT / "runs", seed, "pilot" if pilot else "train")

    training = config["training"]
    sde_config = config["vp_sde"]
    schedule = VPSDESchedule(float(sde_config["beta_min"]), float(sde_config["beta_max"]))
    target_updates = int(training["pilot_updates"] if pilot else training["target_updates"])
    effective_batch = int(training["effective_batch_size"])
    micro_batch = int(training["micro_batch_size"])
    ema_decay = float(training["ema_decay"])
    validation_count = int(training["validation_images"])
    training_time_min = float(sde_config["training_time_min"])
    prediction = str(config["model"]["prediction"])
    objective = "sigma_squared_weighted_dsm_epsilon_mse" if prediction == "epsilon" else "sigma_squared_weighted_dsm_direct_score"

    seed_everything(seed)
    model = build_model(config)
    if not isinstance(model, (ContinuousTimeOriginalDDPMUNet, ContinuousTimeDirectScoreUNet)):
        raise TypeError("VP-SDE config did not construct the continuous-time DDPM U-Net")
    model = model.to(device)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer_type = torch.optim.Adam if training["optimizer"].lower() == "adam" else torch.optim.AdamW
    optimizer = optimizer_type(model.parameters(), lr=float(training["learning_rate"]))

    global_step = 0
    current_epoch = 0
    current_batch = 0
    best_validation_loss: float | None = None
    best_ema_validation_loss: float | None = None
    elapsed_before_resume = 0.0
    if args.resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_config = checkpoint.get("config", {})
        if saved_config.get("process") != "vp_sde" or saved_config.get("model", {}).get("architecture") != config["model"]["architecture"]:
            raise ValueError("Resume checkpoint process or architecture does not match VP-SDE")
        model.load_state_dict(checkpoint.get("training_model_state_dict") or checkpoint["model_state_dict"])
        if "ema_model_state_dict" not in checkpoint or "optimizer_state_dict" not in checkpoint or "rng_state" not in checkpoint:
            raise ValueError("Resume checkpoint is missing EMA, optimizer, or RNG state")
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        restore_rng_state(checkpoint["rng_state"])
        global_step = int(checkpoint["global_step"])
        current_epoch = int(checkpoint["epoch"])
        current_batch = int(checkpoint["batch_in_epoch"])
        best_validation_loss = checkpoint.get("best_validation_loss")
        best_ema_validation_loss = checkpoint.get("best_ema_validation_loss")
        previous_status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        elapsed_before_resume = float(previous_status.get("elapsed_seconds", 0.0))
    else:
        atomic_json(run_dir / "config.json", _run_metadata(config, pilot, device))

    train_source = load_eurosat_split("train", PROJECT_ROOT / config["dataset"]["cache_dir"])
    if len(train_source) != EXPECTED_SPLIT_COUNTS["train"]:
        raise ValueError(f"Expected {EXPECTED_SPLIT_COUNTS['train']} train examples, received {len(train_source)}")
    train_dataset = UnconditionalEuroSATSplit(train_source)

    validation_loader: DataLoader[Tensor] | None = None
    validation_source = None
    if not pilot:
        validation_source = load_eurosat_split("validation", PROJECT_ROOT / config["dataset"]["cache_dir"])
        if len(validation_source) != EXPECTED_SPLIT_COUNTS["validation"]:
            raise ValueError(
                f"Expected {EXPECTED_SPLIT_COUNTS['validation']} validation examples, received {len(validation_source)}"
            )
        validation_dataset = UnconditionalEuroSATSplit(validation_source)
        validation_dataset = Subset(validation_dataset, range(min(validation_count, len(validation_dataset))))
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=micro_batch,
            shuffle=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

    batches_per_epoch = (len(train_dataset) + effective_batch - 1) // effective_batch
    total_epochs = (int(training["target_updates"]) + batches_per_epoch - 1) // batches_per_epoch
    if pilot and int(training["pilot_updates"]) * effective_batch > len(train_dataset):
        raise ValueError("Pilot updates times effective batch size cannot exceed the train split size")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    def elapsed_seconds() -> float:
        return elapsed_before_resume + time.perf_counter() - started

    metrics_path = run_dir / "metrics.jsonl"
    monitor_metrics_path = run_dir / "monitor_metrics.jsonl"
    event_id = last_event_id(metrics_path)
    monitor_event_id = last_event_id(monitor_metrics_path)
    planned_steps = int(training["pilot_updates"] if pilot else training["target_updates"])
    status_update(
        run_dir,
        state="running",
        mode="pilot" if pilot else "train",
        process="vp_sde",
        objective=objective,
        device=str(device),
        epoch=1 if pilot else current_epoch + 1,
        total_epochs=1 if pilot else total_epochs,
        global_step=global_step,
        planned_steps=planned_steps,
        effective_batch_size=effective_batch,
        micro_batch_size=micro_batch,
        beta_min=schedule.beta_min,
        beta_max=schedule.beta_max,
        error=None,
        **({"started_at": now_iso()} if not args.resume else {}),
    )

    pending_loss = 0.0
    pending_count = 0
    pilot_order = torch.randperm(
        len(train_dataset), generator=torch.Generator(device="cpu").manual_seed(seed + 50_000)
    ).tolist() if pilot else []
    order_generator = torch.Generator(device="cpu").manual_seed(seed + current_epoch)
    order = torch.randperm(len(train_dataset), generator=order_generator).tolist()
    batch_count = (len(order) + effective_batch - 1) // effective_batch

    def append_training_metrics(loss_value: float, epoch: int, learning_rate: float, batch_size: int) -> None:
        nonlocal event_id, monitor_event_id, pending_loss, pending_count
        event = {
            "kind": "train",
            "epoch": epoch,
            "global_step": global_step,
            "loss": loss_value,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "elapsed_seconds": elapsed_seconds(),
        }
        event_id = append_metric(run_dir, event, event_id)
        pending_loss += loss_value
        pending_count += 1
        if global_step % int(training["monitor_interval_updates"]) == 0 or global_step == planned_steps:
            monitor_event_id = _append_monitor_event(
                monitor_metrics_path,
                {
                    "kind": "train",
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": pending_loss / pending_count,
                    "window_updates": pending_count,
                    "elapsed_seconds": elapsed_seconds(),
                },
                monitor_event_id,
            )
            pending_loss = 0.0
            pending_count = 0

    def append_validation_metrics(epoch: int, raw_loss: float, ema_loss: float, kind: str = "validation") -> None:
        nonlocal event_id, monitor_event_id
        event = {
            "kind": kind,
            "epoch": epoch,
            "global_step": global_step,
            "loss": raw_loss,
            "ema_loss": ema_loss,
            "elapsed_seconds": elapsed_seconds(),
        }
        event_id = append_metric(run_dir, event, event_id)
        monitor_event_id = _append_monitor_event(monitor_metrics_path, event.copy(), monitor_event_id)

    def save_latest(epoch: int, batch_position: int) -> None:
        save_checkpoint(
            run_dir,
            "latest.pt",
            "latest",
            model,
            optimizer,
            config,
            epoch=epoch,
            batch_in_epoch=batch_position,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            best_ema_validation_loss=best_ema_validation_loss,
            ema_model=ema_model,
        )

    try:
        while global_step < planned_steps:
            if pilot:
                start = global_step * effective_batch
                batch_indices = pilot_order[start : start + effective_batch]
                epoch_index = 0
            else:
                if current_batch >= batch_count:
                    current_epoch += 1
                    current_batch = 0
                    order_generator = torch.Generator(device="cpu").manual_seed(seed + current_epoch)
                    order = torch.randperm(len(train_dataset), generator=order_generator).tolist()
                    batch_count = (len(order) + effective_batch - 1) // effective_batch
                    continue
                start = current_batch * effective_batch
                batch_indices = order[start : start + effective_batch]
                epoch_index = current_epoch

            x0_cpu = torch.stack([train_dataset[index] for index in batch_indices])
            if bool(training.get("horizontal_flip", False)):
                flip_mask = torch.rand(x0_cpu.shape[0]) < 0.5
                x0_cpu[flip_mask] = x0_cpu[flip_mask].flip(-1)

            optimizer.zero_grad(set_to_none=True)
            weighted_loss = 0.0
            for offset in range(0, x0_cpu.shape[0], micro_batch):
                x0 = x0_cpu[offset : offset + micro_batch].to(device, non_blocking=device.type == "cuda")
                t = sample_continuous_times(
                    x0.shape[0], device=device, t_min=training_time_min
                )
                noise = torch.randn_like(x0)
                loss = loss_for_batch(model, schedule, x0, t, noise, prediction=prediction)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite VP-SDE training loss at update {global_step + 1}")
                fraction = x0.shape[0] / x0_cpu.shape[0]
                (loss * fraction).backward()
                weighted_loss += float(loss.detach()) * fraction

            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite gradient norm at update {global_step + 1}")
            learning_rate = float(training["learning_rate"])
            warmup = int(training["warmup_updates"])
            if warmup:
                learning_rate *= min((global_step + 1) / warmup, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            update_ema(ema_model, model, ema_decay)
            global_step += 1
            if not pilot:
                current_batch += 1

            append_training_metrics(
                weighted_loss,
                1 if pilot else epoch_index + 1,
                learning_rate,
                int(x0_cpu.shape[0]),
            )

            if not pilot and global_step % int(training["validation_every_updates"]) == 0:
                assert validation_loader is not None
                validation_seed = seed + 10_000
                raw_validation_loss = evaluate(
                    model, schedule, validation_loader, device,
                    validation_seed=validation_seed, training_time_min=training_time_min,
                    prediction=prediction,
                )
                ema_validation_loss = evaluate(
                    ema_model, schedule, validation_loader, device,
                    validation_seed=validation_seed, training_time_min=training_time_min,
                    prediction=prediction,
                )
                append_validation_metrics(epoch_index + 1, raw_validation_loss, ema_validation_loss)
                if best_validation_loss is None or raw_validation_loss < best_validation_loss:
                    best_validation_loss = raw_validation_loss
                    save_checkpoint(
                        run_dir, "best-val.pt", "best-validation", model, optimizer, config,
                        epoch=epoch_index + 1, batch_in_epoch=current_batch, global_step=global_step,
                        best_validation_loss=best_validation_loss,
                        best_ema_validation_loss=best_ema_validation_loss,
                        validation_loss=raw_validation_loss, include_optimizer=False,
                    )
                if best_ema_validation_loss is None or ema_validation_loss < best_ema_validation_loss:
                    best_ema_validation_loss = ema_validation_loss
                    save_checkpoint(
                        run_dir, "best-val-ema.pt", "best-validation-ema", model, optimizer, config,
                        epoch=epoch_index + 1, batch_in_epoch=current_batch, global_step=global_step,
                        best_validation_loss=best_validation_loss,
                        best_ema_validation_loss=best_ema_validation_loss,
                        validation_loss=ema_validation_loss, ema_model=ema_model,
                        weights_model=ema_model, include_optimizer=False,
                    )

            if not pilot and global_step % int(training["model_checkpoint_every_updates"]) == 0:
                save_checkpoint(
                    run_dir, f"ema-step-{global_step:09d}.pt", "ema-progress", model, optimizer, config,
                    epoch=epoch_index + 1, batch_in_epoch=current_batch, global_step=global_step,
                    best_validation_loss=best_validation_loss,
                    best_ema_validation_loss=best_ema_validation_loss,
                    ema_model=ema_model, weights_model=ema_model, include_optimizer=False,
                )
            if not pilot and global_step % int(training["checkpoint_every_updates"]) == 0:
                save_latest(epoch_index, current_batch)

            memory_mb = (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else None
            )
            if (
                global_step % int(training["status_interval_updates"]) == 0
                or global_step % int(training["validation_every_updates"]) == 0
                or global_step % int(training["checkpoint_every_updates"]) == 0
                or global_step == planned_steps
            ):
                status_update(
                    run_dir,
                    state="running",
                    epoch=1 if pilot else epoch_index + 1,
                    total_epochs=1 if pilot else total_epochs,
                    batch_in_epoch=0 if pilot else current_batch,
                    global_step=global_step,
                    planned_steps=planned_steps,
                    train_loss=weighted_loss,
                    best_validation_loss=best_validation_loss,
                    best_ema_validation_loss=best_ema_validation_loss,
                    elapsed_seconds=elapsed_seconds(),
                    steps_per_second=global_step / max(elapsed_seconds(), 1e-9),
                    peak_gpu_memory_mb=memory_mb,
                )
            if global_step % int(training["monitor_interval_updates"]) == 0 or global_step == planned_steps:
                print(
                    f"VP-SDE update {global_step}/{planned_steps} "
                    f"{prediction}_loss={weighted_loss:.6f} lr={learning_rate:.7g} "
                    f"elapsed={elapsed_seconds() / 60:.1f} min peak_gpu_mb={memory_mb}",
                    flush=True,
                )

        if pilot:
            save_checkpoint(
                run_dir, "pilot-final.pt", "pilot", model, optimizer, config,
                epoch=1, batch_in_epoch=0, global_step=global_step,
                best_validation_loss=None, best_ema_validation_loss=None, ema_model=ema_model,
            )
        else:
            assert validation_source is not None
            full_validation_loader = DataLoader(
                UnconditionalEuroSATSplit(validation_source),
                batch_size=micro_batch,
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            full_seed = seed + 20_000
            full_raw_loss = evaluate(
                model, schedule, full_validation_loader, device,
                validation_seed=full_seed, training_time_min=training_time_min,
                prediction=prediction,
            )
            full_ema_loss = evaluate(
                ema_model, schedule, full_validation_loader, device,
                validation_seed=full_seed, training_time_min=training_time_min,
                prediction=prediction,
            )
            append_validation_metrics(current_epoch + 1, full_raw_loss, full_ema_loss, "validation_full")
            save_latest(current_epoch, current_batch)

        memory_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
            if device.type == "cuda"
            else None
        )
        status_update(
            run_dir,
            state="completed",
            mode="pilot" if pilot else "train",
            epoch=1 if pilot else current_epoch + 1,
            total_epochs=1 if pilot else total_epochs,
            global_step=global_step,
            planned_steps=planned_steps,
            elapsed_seconds=elapsed_seconds(),
            steps_per_second=global_step / max(elapsed_seconds(), 1e-9),
            peak_gpu_memory_mb=memory_mb,
            completed_at=now_iso(),
        )
        print(f"Run artifacts: {run_dir}", flush=True)
        return 0
    except KeyboardInterrupt:
        if not pilot:
            save_latest(current_epoch, current_batch)
        status_update(run_dir, state="interrupted", global_step=global_step, elapsed_seconds=elapsed_seconds())
        print(f"Training interrupted. Resume with --resume {run_dir / 'checkpoints' / 'latest.pt'}", file=sys.stderr)
        return 130
    except Exception as error:
        status_update(
            run_dir, state="failed", global_step=global_step,
            error=f"{type(error).__name__}: {error}", elapsed_seconds=elapsed_seconds(),
        )
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the unconditional EuroSAT VP-SDE score model")
    parser.add_argument(
        "--config", type=Path,
        default=PROJECT_ROOT / "configs" / "eurosat_vp_sde_50_epoch.toml",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--pilot", action="store_true", help="Run the configured short GPU pilot")
    parser.add_argument("--run-dir", type=Path, help="Optional explicit output directory for a new run")
    parser.add_argument("--resume", type=Path, help="Resume a full VP-SDE run from latest.pt")
    args = parser.parse_args(argv)
    if args.pilot and args.resume:
        parser.error("--pilot and --resume cannot be used together")
    return args


def main() -> None:
    raise SystemExit(run_training(parse_args()))


if __name__ == "__main__":
    main()
