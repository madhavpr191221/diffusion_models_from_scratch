from pathlib import Path

import pytest
import torch
from torch import nn

from ddpm_eurosat.train import save_checkpoint
from ddpm_eurosat.score_sde_train import (
    epsilon_loss_for_batch,
    evaluate,
    loss_for_batch,
    load_config,
    score_loss_for_batch,
    validate_config,
)
from ddpm_eurosat.model import build_model
from ddpm_eurosat.vp_sde import VPSDESchedule, epsilon_to_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScalarEpsilonModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.25))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.value.expand_as(x_t)


def test_vp_sde_config_selects_only_the_continuous_process() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "eurosat_vp_sde_50_epoch.toml")
    validate_config(config)
    assert config["process"] == "vp_sde"
    assert config["model"]["architecture"] == "vp_sde_ddpm_cifar_v1"
    assert config["training"]["pilot_updates"] == 50
    assert "diffusion" not in config


def test_direct_score_config_selects_its_own_output_parameterization() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "eurosat_vp_sde_direct_score_50_epoch.toml")
    validate_config(config)
    model = build_model(config)
    assert config["model"]["prediction"] == "score"
    assert type(model).__name__ == "ContinuousTimeDirectScoreUNet"


def test_trainer_rejects_a_discrete_ddpm_config() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "eurosat_original_ddpm_50_epoch.toml")
    with pytest.raises(ValueError, match="process = 'vp_sde'"):
        validate_config(config)


def test_epsilon_objective_backpropagates_finite_gradients() -> None:
    model = ScalarEpsilonModel()
    schedule = VPSDESchedule()
    x0 = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(2))
    t = torch.tensor([0.05, 0.7])
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(3))

    loss = epsilon_loss_for_batch(model, schedule, x0, t, noise)
    loss.backward()

    assert torch.isfinite(loss)
    assert model.value.grad is not None
    assert torch.isfinite(model.value.grad)


def test_direct_score_weighted_dsm_matches_epsilon_mse_for_equivalent_output() -> None:
    schedule = VPSDESchedule()
    x0 = torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(21))
    t = torch.tensor([0.05, 0.7])
    noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(22))
    epsilon_model = ScalarEpsilonModel()

    class FixedScoreModel(nn.Module):
        def forward(self, x: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
            return epsilon_to_score(schedule, epsilon_model(x, times), times)

    direct_score_model = FixedScoreModel()
    epsilon_loss = epsilon_loss_for_batch(epsilon_model, schedule, x0, t, noise)
    score_loss = score_loss_for_batch(direct_score_model, schedule, x0, t, noise)

    assert torch.allclose(score_loss, epsilon_loss, atol=2e-6, rtol=2e-6)
    assert torch.allclose(loss_for_batch(direct_score_model, schedule, x0, t, noise, prediction="score"), score_loss)

    trainable_score = ScalarEpsilonModel()
    direct_loss = score_loss_for_batch(trainable_score, schedule, x0, t, noise)
    direct_loss.backward()
    assert torch.isfinite(direct_loss)
    assert trainable_score.value.grad is not None and torch.isfinite(trainable_score.value.grad)


def test_validation_reuses_the_same_seeded_corruptions() -> None:
    model = ScalarEpsilonModel()
    model.train()
    schedule = VPSDESchedule()
    validation_batches = [
        torch.linspace(-1.0, 1.0, 2 * 3 * 8 * 8).reshape(2, 3, 8, 8),
        torch.linspace(1.0, -1.0, 1 * 3 * 8 * 8).reshape(1, 3, 8, 8),
    ]

    first = evaluate(
        model, schedule, validation_batches, torch.device("cpu"),
        validation_seed=10042, training_time_min=1e-5,
    )
    repeated = evaluate(
        model, schedule, validation_batches, torch.device("cpu"),
        validation_seed=10042, training_time_min=1e-5,
    )

    assert first == repeated
    assert model.training


def test_score_run_checkpoint_round_trips_resume_state(tmp_path: Path) -> None:
    model = ScalarEpsilonModel()
    ema_model = ScalarEpsilonModel().eval().requires_grad_(False)
    ema_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    (model.value.square()).backward()
    optimizer.step()
    config = {
        "process": "vp_sde",
        "model": {"architecture": "vp_sde_ddpm_cifar_v1"},
    }

    save_checkpoint(
        tmp_path,
        "latest.pt",
        "latest",
        model,
        optimizer,
        config,
        epoch=3,
        batch_in_epoch=4,
        global_step=17,
        best_validation_loss=0.4,
        best_ema_validation_loss=0.35,
        ema_model=ema_model,
    )
    saved = torch.load(tmp_path / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False)

    resumed_model = ScalarEpsilonModel()
    resumed_ema = ScalarEpsilonModel().eval().requires_grad_(False)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=2e-4)
    resumed_model.load_state_dict(saved["training_model_state_dict"] or saved["model_state_dict"])
    resumed_ema.load_state_dict(saved["ema_model_state_dict"])
    resumed_optimizer.load_state_dict(saved["optimizer_state_dict"])

    assert saved["config"]["process"] == "vp_sde"
    assert saved["global_step"] == 17
    assert saved["epoch"] == 3
    assert saved["batch_in_epoch"] == 4
    assert saved["rng_state"]["torch_cpu"] is not None
    assert torch.equal(resumed_model.value, model.value)
    assert torch.equal(resumed_ema.value, ema_model.value)
    assert resumed_optimizer.state_dict()["state"]
