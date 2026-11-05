import json
from pathlib import Path

import pytest
from fastapi import BackgroundTasks, HTTPException

from ddpm_eurosat import api
from ddpm_eurosat.flow_matching import BEST_FLOW_MATCHING_CHECKPOINT, BEST_FLOW_MATCHING_EMA_CHECKPOINT


def _write_flow_run(run_dir: Path, *, process: str = "flow_matching") -> None:
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    (checkpoints / BEST_FLOW_MATCHING_CHECKPOINT).write_bytes(b"checkpoint")
    (checkpoints / BEST_FLOW_MATCHING_EMA_CHECKPOINT).write_bytes(b"ema-checkpoint")
    (run_dir / "config.json").write_text(
        json.dumps({"created_at": "2026-11-05T10:00:00Z", "config": {"process": process}}),
        encoding="utf-8",
    )
    (run_dir / "status.json").write_text(
        json.dumps({"state": "completed", "mode": "train", "epoch": 50, "total_epochs": 50}),
        encoding="utf-8",
    )


def test_flow_runs_are_listed_separately_and_only_when_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed = tmp_path / "20261105-train-seed-42"
    _write_flow_run(completed)
    incomplete = tmp_path / "20261104-train-seed-42"
    _write_flow_run(incomplete)
    status = json.loads((incomplete / "status.json").read_text(encoding="utf-8"))
    status["state"] = "running"
    (incomplete / "status.json").write_text(json.dumps(status), encoding="utf-8")
    _write_flow_run(tmp_path / "not-a-flow-run", process="vp_sde")
    monkeypatch.setattr(api, "RUNS_ROOT", tmp_path)

    result = api.list_flow_runs()

    assert len(result["items"]) == 2
    by_id = {item["run_id"]: item for item in result["items"]}
    assert by_id[completed.name]["can_generate"] is True
    assert by_id[incomplete.name]["can_generate"] is False


def test_flow_study_queues_only_completed_flow_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "20261105-train-seed-42"
    _write_flow_run(run_dir)
    monkeypatch.setattr(api, "get_run_dir", lambda run_id: run_dir)
    monkeypatch.setattr(api, "_active_sampling_job", None)
    tasks = BackgroundTasks()

    result = api.create_flow_study(
        run_dir.name,
        api.CreateFlowStudyRequest(num_samples=1, seed=42, solver="heun", steps=4),
        tasks,
    )

    assert result["run_id"] == run_dir.name
    assert len(result["study_id"]) == 32
    assert len(tasks.tasks) == 1
    status = api.read_run_json(run_dir / "flow-studies" / result["study_id"], "status.json", {})
    assert status["generation_state"] == "queued"
    assert status["checkpoint"] == BEST_FLOW_MATCHING_EMA_CHECKPOINT


def test_flow_study_accepts_raw_best_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "20261105-train-seed-42"
    _write_flow_run(run_dir)
    monkeypatch.setattr(api, "get_run_dir", lambda run_id: run_dir)
    monkeypatch.setattr(api, "_active_sampling_job", None)

    result = api.create_flow_study(
        run_dir.name,
        api.CreateFlowStudyRequest(
            num_samples=1, seed=42, solver="euler", steps=4, checkpoint=BEST_FLOW_MATCHING_CHECKPOINT,
        ),
        BackgroundTasks(),
    )

    status = api.read_run_json(run_dir / "flow-studies" / result["study_id"], "status.json", {})
    assert status["checkpoint"] == BEST_FLOW_MATCHING_CHECKPOINT


def test_flow_study_rejects_unapproved_checkpoint_filename(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "20261105-train-seed-42"
    _write_flow_run(run_dir)
    monkeypatch.setattr(api, "get_run_dir", lambda run_id: run_dir)
    monkeypatch.setattr(api, "_active_sampling_job", None)
    with pytest.raises(HTTPException, match="Unsupported Flow Matching checkpoint") as error:
        api.create_flow_study(
            run_dir.name,
            api.CreateFlowStudyRequest(num_samples=1, checkpoint="../../best-val.pt"),
            BackgroundTasks(),
        )
    assert error.value.status_code == 422


def test_flow_study_rejects_non_flow_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "20261105-train-seed-42"
    _write_flow_run(run_dir, process="vp_sde")
    monkeypatch.setattr(api, "get_run_dir", lambda run_id: run_dir)

    with pytest.raises(HTTPException, match="not a Flow Matching model") as error:
        api.create_flow_study(
            run_dir.name,
            api.CreateFlowStudyRequest(num_samples=1, seed=42, solver="euler", steps=2),
            BackgroundTasks(),
        )
    assert error.value.status_code == 409
