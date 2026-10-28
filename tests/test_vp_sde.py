import pytest
import torch

from ddpm_eurosat.vp_sde import (
    VPSDESchedule,
    conditional_score_target,
    epsilon_prediction_loss,
    epsilon_to_score,
    q_sample_vp,
    sample_continuous_times,
    weighted_dsm_loss,
)


def test_linear_schedule_and_marginal_coefficients_match_closed_forms() -> None:
    schedule = VPSDESchedule(beta_min=0.1, beta_max=20.0)
    t = torch.tensor([0.0, 0.25, 1.0], dtype=torch.float64)
    expected_integral = 0.1 * t + 0.5 * 19.9 * t.square()

    assert torch.allclose(schedule.beta(t), 0.1 + 19.9 * t)
    assert torch.allclose(schedule.integrated_beta(t), expected_integral)
    assert torch.allclose(schedule.marginal_mean_coefficient(t), torch.exp(-0.5 * expected_integral))
    assert torch.allclose(schedule.marginal_std(t).square(), 1.0 - torch.exp(-expected_integral))
    assert schedule.marginal_mean_coefficient(t)[0].item() == 1.0
    assert schedule.marginal_std(t)[0].item() == 0.0


def test_exact_forward_marginal_uses_per_example_broadcasting() -> None:
    schedule = VPSDESchedule()
    x0 = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    noise = torch.full_like(x0, 0.25)
    t = torch.tensor([0.0, 0.6])
    phi = schedule.marginal_mean_coefficient(t).reshape(2, 1, 1, 1)
    sigma = schedule.marginal_std(t).reshape(2, 1, 1, 1)

    actual, returned_noise = q_sample_vp(schedule, x0, t, noise)

    assert actual.shape == x0.shape
    assert torch.equal(returned_noise, noise)
    assert torch.allclose(actual, phi * x0 + sigma * noise)
    assert torch.equal(actual[0], x0[0])


def test_continuous_time_sampling_is_bounded_and_seed_reproducible() -> None:
    first = sample_continuous_times(
        32, device="cpu", t_min=1e-5, generator=torch.Generator().manual_seed(17)
    )
    repeated = sample_continuous_times(
        32, device="cpu", t_min=1e-5, generator=torch.Generator().manual_seed(17)
    )

    assert first.shape == (32,)
    assert first.dtype == torch.float32
    assert torch.all(first >= 1e-5)
    assert torch.all(first < 1.0)
    assert torch.equal(first, repeated)


def test_epsilon_score_relation_and_weighted_dsm_equal_epsilon_mse() -> None:
    schedule = VPSDESchedule()
    t = torch.tensor([0.03, 0.4, 0.95])
    noise = torch.randn(3, 3, 8, 8, generator=torch.Generator().manual_seed(7))
    epsilon_prediction = torch.randn(noise.shape, generator=torch.Generator().manual_seed(8))

    predicted_score = epsilon_to_score(schedule, epsilon_prediction, t)
    target_score = conditional_score_target(schedule, noise, t)
    score_loss = weighted_dsm_loss(schedule, predicted_score, noise, t)
    epsilon_loss = epsilon_prediction_loss(epsilon_prediction, noise)

    assert predicted_score.shape == noise.shape
    assert torch.isfinite(target_score).all()
    assert torch.allclose(predicted_score, -epsilon_prediction / schedule.marginal_std(t).view(3, 1, 1, 1))
    assert torch.allclose(score_loss, epsilon_loss, atol=2e-6, rtol=2e-6)


def test_discrete_retention_euler_product_converges_to_vp_marginal() -> None:
    """Small discrete intervals approximate exp(-integral beta), not DDPM betas."""
    schedule = VPSDESchedule()
    target_alpha_bar = schedule.marginal_mean_coefficient(torch.tensor([1.0])).square()[0]

    def euler_alpha_bar(intervals: int) -> Tensor:
        dt = 1.0 / intervals
        right_endpoints = torch.arange(1, intervals + 1, dtype=torch.float64) * dt
        interval_beta = schedule.beta(right_endpoints) * dt
        return torch.prod(1.0 - interval_beta)

    coarse_error = abs(euler_alpha_bar(100).item() - target_alpha_bar.item())
    fine_error = abs(euler_alpha_bar(1_000).item() - target_alpha_bar.item())
    assert fine_error < coarse_error
    assert fine_error < 0.01


def test_schedule_rejects_invalid_parameters_and_time_shapes() -> None:
    with pytest.raises(ValueError, match="beta_min"):
        VPSDESchedule(beta_min=0.0)
    with pytest.raises(ValueError, match=r"shape \[B\]"):
        VPSDESchedule().beta(torch.tensor([[0.5]]))
    with pytest.raises(TypeError, match="floating point"):
        VPSDESchedule().beta(torch.tensor([1]))
    with pytest.raises(ValueError, match="t_min"):
        sample_continuous_times(2, device="cpu", t_min=1.0)
