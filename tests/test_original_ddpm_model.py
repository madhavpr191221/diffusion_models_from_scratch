import torch

from ddpm_eurosat.model import OriginalDDPMUNet, TimeConditionedUNet, build_model


def test_original_ddpm_unet_matches_expected_parameter_scale_and_io_shape() -> None:
    model = OriginalDDPMUNet()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    assert 35_700_000 < parameter_count < 35_800_000

    model.eval()
    with torch.inference_mode():
        prediction = model(torch.randn(1, 3, 64, 64), torch.tensor([500]))
    assert prediction.shape == (1, 3, 64, 64)
    assert torch.isfinite(prediction).all()


def test_model_factory_keeps_legacy_checkpoints_on_compact_unet() -> None:
    assert isinstance(build_model({}), TimeConditionedUNet)
    assert isinstance(build_model({"model": {"architecture": "ddpm_cifar_v1"}}), OriginalDDPMUNet)
