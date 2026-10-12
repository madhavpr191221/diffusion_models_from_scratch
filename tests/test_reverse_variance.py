import torch

from ddpm_eurosat.diffusion import DiffusionSchedule, ddpm_reverse_step


def test_fixed_large_uses_beta_variance_and_keeps_final_step_deterministic() -> None:
    schedule = DiffusionSchedule(timesteps=10)
    x_t = torch.zeros(1, 3, 4, 4)
    predicted_noise = torch.zeros_like(x_t)
    unit_noise = torch.ones_like(x_t)
    no_noise = torch.zeros_like(x_t)

    with_noise = ddpm_reverse_step(schedule, x_t, predicted_noise, 9, noise=unit_noise, variance_type="fixed_large")
    without_noise = ddpm_reverse_step(schedule, x_t, predicted_noise, 9, noise=no_noise, variance_type="fixed_large")
    expected = torch.sqrt(schedule.betas[9])
    assert torch.allclose(with_noise - without_noise, torch.full_like(x_t, expected))

    final_with_noise = ddpm_reverse_step(schedule, x_t, predicted_noise, 0, noise=unit_noise, variance_type="fixed_large")
    final_without_noise = ddpm_reverse_step(schedule, x_t, predicted_noise, 0, noise=no_noise, variance_type="fixed_large")
    assert torch.equal(final_with_noise, final_without_noise)
