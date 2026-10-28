import json
from pathlib import Path

import pytest
import torch
from fastapi import BackgroundTasks, HTTPException

from ddpm_eurosat import api


def _write_vp_sde_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"config": {"process": "vp_sde", "vp_sde": {"sampling_steps": 1000}}}), encoding="utf-8"
    )
    (run_dir / "status.json").write_text(
        json.dumps({"state": "completed", "mode": "train"}), encoding="utf-8"
    )
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "best-val.pt").write_bytes(b"checkpoint")
    (checkpoints / "best-val-ema.pt").write_bytes(b"ema-checkpoint")
    (run_dir / "checkpoint_manifest.json").write_text(
        json.dumps([{"name": "best-val.pt"}, {"name": "best-val-ema.pt"}]), encoding="utf-8"
    )


def test_completed_vp_sde_ema_run_is_advertised_for_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "20260101-train-seed-42"
    _write_vp_sde_run(run_dir)
    monkeypatch.setattr(api, "RUNS_ROOT", tmp_path)

    result = api.list_runs()

    assert result["items"][0]["process"] == "vp_sde"
    assert result["items"][0]["can_generate"] is True


def test_direct_score_run_is_identified_for_ui_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "20260101-direct-score-seed-42"
    _write_vp_sde_run(run_dir)
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    config["config"]["model"] = {
        "architecture": "vp_sde_direct_score_ddpm_cifar_v1", "prediction": "score",
    }
    (run_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(api, "RUNS_ROOT", tmp_path)

    result = api.list_runs()

    assert result["items"][0]["prediction"] == "score"
    assert result["items"][0]["can_generate"] is True


def test_vp_sde_generation_uses_sde_steps_and_keeps_studies_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "20260101-train-seed-42"
    _write_vp_sde_run(run_dir)
    monkeypatch.setattr(api, "get_run_dir", lambda run_id: run_dir)
    monkeypatch.setattr(api, "_active_sampling_job", None)

    background = BackgroundTasks()
    result = api.create_generation(
        run_dir.name,
        api.CreateGenerationRequest(checkpoint="best-val-ema.pt"),
        background,
    )
    assert result["state"] == "queued"
    progress = api.read_generation_progress(run_dir / "generation" / str(result["generation_id"]))
    assert progress["process"] == "vp_sde"
    assert progress["total_steps"] == 1000
    assert len(background.tasks) == 1

    with pytest.raises(HTTPException, match="VP-SDE cohort generation") as cohort_error:
        api.create_generation_study(
            run_dir.name,
            api.CreateGenerationStudyRequest(checkpoint="best-val.pt"),
            BackgroundTasks(),
        )
    assert cohort_error.value.status_code == 409


def test_vp_sde_generation_accepts_annealed_langevin_sampler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "20260101-train-seed-42"
    _write_vp_sde_run(run_dir)
    monkeypatch.setattr(api, "get_run_dir", lambda run_id: run_dir)
    monkeypatch.setattr(api, "_active_sampling_job", None)
    background = BackgroundTasks()

    result = api.create_generation(
        run_dir.name,
        api.CreateGenerationRequest(checkpoint="best-val-ema.pt", sampler="annealed_langevin"),
        background,
    )
    progress = api.read_generation_progress(run_dir / "generation" / str(result["generation_id"]))
    assert progress["sampler"] == "annealed_langevin"
    assert progress["total_steps"] == 1000
    assert len(background.tasks) == 1


def test_vp_sde_frame_exposes_continuous_time_header(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "20260101-train-seed-42"
    _write_vp_sde_run(run_dir)
    output_dir = run_dir / "generation" / ("a" * 32)
    (output_dir / "frames").mkdir(parents=True)
    (output_dir / "frames" / "frame-00010.png").write_bytes(b"png")
    api.atomic_json(output_dir / "progress.json", {
        "process": "vp_sde", "total_steps": 1000, "frame_stride": 10, "latest_frame_index": 10,
    })
    monkeypatch.setattr(api, "get_run_dir", lambda run_id: run_dir)

    response = api.get_generation_frame(run_dir.name, "a" * 32, 10)

    assert response.headers["x-process"] == "vp_sde"
    assert float(response.headers["x-time"]) == 0.9


def test_annealed_langevin_frame_time_tracks_noise_levels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "20260101-train-seed-42"
    _write_vp_sde_run(run_dir)
    output_dir = run_dir / "generation" / ("c" * 32)
    (output_dir / "frames").mkdir(parents=True)
    (output_dir / "frames" / "frame-00050.png").write_bytes(b"png")
    (output_dir / "frames" / "frame-00100.png").write_bytes(b"png")
    api.atomic_json(output_dir / "progress.json", {
        "process": "vp_sde", "sampler": "annealed_langevin", "total_steps": 1000, "latest_frame_index": 100,
    })
    monkeypatch.setattr(api, "get_run_dir", lambda run_id: run_dir)
    response = api.get_generation_frame(run_dir.name, "c" * 32, 50)
    expected = 1.0 - (49 / 99) * (1.0 - 1e-5)
    assert abs(float(response.headers["x-time"]) - expected) < 1e-7
    final_response = api.get_generation_frame(run_dir.name, "c" * 32, 100)
    assert abs(float(final_response.headers["x-time"]) - 1e-5) < 1e-7


def test_vp_sde_generation_job_dispatches_sampler_and_publishes_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "20260101-train-seed-42"
    _write_vp_sde_run(run_dir)
    output_dir = run_dir / "generation" / ("b" * 32)
    output_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"config": {"process": "vp_sde", "vp_sde": {"sampling_steps": 10}}}), encoding="utf-8"
    )
    monkeypatch.setattr(api, "get_run_dir", lambda run_id: run_dir)
    monkeypatch.setattr(api, "choose_device", lambda name: torch.device("cpu"))
    monkeypatch.setattr(api, "validation_images", lambda: torch.zeros((2, 3, 64, 64)))
    observed: dict[str, object] = {}

    def fake_generate(checkpoint_path, output_path, **kwargs):
        observed["steps"] = kwargs["sampling_steps"]
        kwargs["on_frame"](0, 0, 1.0, torch.zeros((3, 64, 64)))
        kwargs["on_frame"](1, 10, 0.0, torch.ones((3, 64, 64)))
        return {"process": "vp_sde", "network_evaluations": kwargs["sampling_steps"]}

    monkeypatch.setattr(api, "generate_from_vp_sde_checkpoint", fake_generate)

    api.run_generation_job(
        run_dir.name, "b" * 32, run_dir / "checkpoints" / "best-val-ema.pt",
        output_dir, 1, 44, "best-val-ema.pt", "vp_sde",
    )

    progress = api.read_generation_progress(output_dir)
    assert observed["steps"] == 10
    assert progress["state"] == "completed"
    assert progress["process"] == "vp_sde"
    assert progress["current_t"] == 0.0
    assert progress["completed_steps"] == 10
    assert (output_dir / "frames" / "frame-00000.png").is_file()
    assert (output_dir / "frames" / "frame-00001.png").is_file()
