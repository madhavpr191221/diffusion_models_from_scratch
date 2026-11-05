import torch
from torch import nn

from ddpm_eurosat.flow_matching import (
    ContinuousTimeVelocityUNet,
    dequantize_normalized_pixels,
    flow_matching_loss,
    flow_matching_path,
    sample_flow_matching,
)


def test_linear_conditional_path_endpoints_and_velocity_target():
    noise = torch.randn(3, 3, 64, 64)
    data = torch.randn_like(noise)
    at_noise, target = flow_matching_path(noise, data, torch.zeros(3))
    at_data, _ = flow_matching_path(noise, data, torch.ones(3))
    torch.testing.assert_close(at_noise, noise)
    torch.testing.assert_close(at_data, data)
    torch.testing.assert_close(target, data - noise)


def test_conditional_path_derivative_matches_velocity_target():
    noise = torch.randn(2, 3, 64, 64, dtype=torch.float64)
    data = torch.randn_like(noise)
    time = torch.tensor([0.31, 0.72], dtype=torch.float64)
    delta = 1e-6
    left, target = flow_matching_path(noise, data, time)
    right, _ = flow_matching_path(noise, data, time + delta)
    torch.testing.assert_close((right - left) / delta, target, atol=2e-9, rtol=2e-8)


def test_dequantization_is_seeded_and_uses_centered_cell_width():
    images = torch.zeros(2, 3, 64, 64)
    first = dequantize_normalized_pixels(images, generator=torch.Generator().manual_seed(5))
    second = dequantize_normalized_pixels(images, generator=torch.Generator().manual_seed(5))
    torch.testing.assert_close(first, second)
    assert torch.all(first >= -1.0 / 255.0)
    assert torch.all(first <= 1.0 / 255.0)


class ConstantTrainableField(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(0.0))

    def forward(self, x, time):
        return self.value.expand_as(x)


def test_flow_matching_loss_is_finite_and_backpropagates():
    model = ConstantTrainableField()
    images = torch.randn(2, 3, 64, 64)
    loss = flow_matching_loss(
        model, images, noise=torch.zeros_like(images), time=torch.full((2,), 0.5)
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.value.grad is not None and torch.isfinite(model.value.grad)


def test_velocity_unet_has_image_shaped_output_and_gradients():
    model = ContinuousTimeVelocityUNet()
    images = torch.randn(1, 3, 64, 64)
    time = torch.tensor([0.45])
    velocity = model(images, time)
    assert velocity.shape == images.shape
    velocity.square().mean().backward()
    assert model.output_conv.weight.grad is not None


def test_sampler_replays_seed_and_records_selected_trajectory():
    model = ConstantTrainableField()
    first, trajectory = sample_flow_matching(model, count=2, seed=9, steps=8, method="heun", save_every=2)
    second, _ = sample_flow_matching(model, count=2, seed=9, steps=8, method="heun", save_every=2)
    torch.testing.assert_close(first, second)
    assert trajectory.nfe == 16
    assert trajectory.states.shape == (5, 2, 3, 64, 64)
    torch.testing.assert_close(trajectory.times, torch.linspace(0, 1, 5))
