import json
from pathlib import Path

import pytest
import torch
from torch import nn

from ddpm_eurosat.flow_matching_train import (
    BEST_FLOW_MATCHING_CHECKPOINT,
    LATEST_FLOW_MATCHING_CHECKPOINT,
    _remove_checkpoint,
    _restore_flow_checkpoint,
)
from ddpm_eurosat.train import capture_rng_state


def _make_checkpoint(path: Path, config: dict[str, object], *, batch_in_epoch: int = 0) -> tuple[nn.Module, nn.Module]:
    model = nn.Linear(3, 2)
    ema = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    checkpoint = {
        "config": config,
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rng_state": capture_rng_state(),
        "epoch": 4,
        "batch_in_epoch": batch_in_epoch,
        "global_step": 40,
    }
    torch.save(checkpoint, path)
    return model, ema


def test_resume_restores_model_ema_optimizer_and_progress(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / LATEST_FLOW_MATCHING_CHECKPOINT
    config = {"process": "flow_matching", "training": {"epochs": 10}}
    source_model, source_ema = _make_checkpoint(checkpoint_path, config)
    model = nn.Linear(3, 2)
    ema = nn.Linear(3, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    result = _restore_flow_checkpoint(checkpoint_path, config, model, ema, optimizer)

    assert result["epoch"] == 4
    assert result["global_step"] == 40
    for target, source in zip(model.parameters(), source_model.parameters(), strict=True):
        torch.testing.assert_close(target, source)
    for target, source in zip(ema.parameters(), source_ema.parameters(), strict=True):
        torch.testing.assert_close(target, source)
    assert optimizer.param_groups[0]["lr"] == 0.01


def test_resume_rejects_config_mismatch_and_mid_epoch_checkpoint(tmp_path: Path) -> None:
    config = {"process": "flow_matching", "training": {"epochs": 10}}
    path = tmp_path / LATEST_FLOW_MATCHING_CHECKPOINT
    _make_checkpoint(path, config)
    with pytest.raises(ValueError, match="configuration differs"):
        _restore_flow_checkpoint(path, {"process": "flow_matching"}, nn.Linear(3, 2), nn.Linear(3, 2), torch.optim.Adam(nn.Linear(3, 2).parameters()))

    _make_checkpoint(path, config, batch_in_epoch=2)
    model = nn.Linear(3, 2)
    with pytest.raises(ValueError, match="epoch-boundary"):
        _restore_flow_checkpoint(path, config, model, nn.Linear(3, 2), torch.optim.Adam(model.parameters()))


def test_success_cleanup_removes_only_rolling_checkpoint(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / LATEST_FLOW_MATCHING_CHECKPOINT).write_bytes(b"latest")
    (checkpoints / BEST_FLOW_MATCHING_CHECKPOINT).write_bytes(b"best")
    manifest = [
        {"name": LATEST_FLOW_MATCHING_CHECKPOINT},
        {"name": BEST_FLOW_MATCHING_CHECKPOINT},
    ]
    (tmp_path / "checkpoint_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    _remove_checkpoint(tmp_path, LATEST_FLOW_MATCHING_CHECKPOINT)

    assert not (checkpoints / LATEST_FLOW_MATCHING_CHECKPOINT).exists()
    assert (checkpoints / BEST_FLOW_MATCHING_CHECKPOINT).is_file()
    retained = json.loads((tmp_path / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert [item["name"] for item in retained] == [BEST_FLOW_MATCHING_CHECKPOINT]
