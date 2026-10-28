import pytest
import torch

from ddpm_eurosat.model import (
    ContinuousTimeOriginalDDPMUNet,
    OriginalDDPMUNet,
    build_model,
)


def test_continuous_architecture_reuses_original_unet_capacity_and_io_shape() -> None:
    config = {"model": {"architecture": "vp_sde_ddpm_cifar_v1", "dropout": 0.1}}
    model = build_model(config)
    assert isinstance(model, ContinuousTimeOriginalDDPMUNet)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 35_700_000 < parameter_count < 35_800_000

    model.eval()
    with torch.inference_mode():
        prediction = model(torch.randn(1, 3, 64, 64), torch.tensor([0.5]))
    assert prediction.shape == (1, 3, 64, 64)
    assert torch.isfinite(prediction).all()


def test_continuous_architecture_adds_no_parameters_or_state_keys() -> None:
    discrete = OriginalDDPMUNet()
    continuous = ContinuousTimeOriginalDDPMUNet()
    assert list(discrete.state_dict()) == list(continuous.state_dict())
    assert {
        name: tuple(value.shape) for name, value in discrete.state_dict().items()
    } == {
        name: tuple(value.shape) for name, value in continuous.state_dict().items()
    }


def test_continuous_model_rejects_integer_time() -> None:
    model = ContinuousTimeOriginalDDPMUNet()
    with pytest.raises(TypeError, match="t must be floating point"):
        model(torch.zeros(1, 3, 64, 64), torch.tensor([500]))


def test_model_factory_keeps_discrete_ddpm_architecture_separate() -> None:
    discrete = build_model({"model": {"architecture": "ddpm_cifar_v1"}})
    continuous = build_model({"model": {"architecture": "vp_sde_ddpm_cifar_v1"}})
    assert type(discrete) is OriginalDDPMUNet
    assert isinstance(continuous, ContinuousTimeOriginalDDPMUNet)
