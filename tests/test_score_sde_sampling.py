import torch
from torch import nn

from ddpm_eurosat.score_sde_sampling import sample_annealed_langevin, sample_reverse_sde
from ddpm_eurosat.vp_sde import VPSDESchedule


class ZeroEpsilonModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.times: list[float] = []

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        self.times.extend(t.detach().cpu().tolist())
        return torch.zeros_like(x)


class ZeroScoreModel(ZeroEpsilonModel):
    pass


def test_reverse_sde_uses_decreasing_positive_times_and_saves_full_trajectory() -> None:
    model = ZeroEpsilonModel()
    samples, trajectory = sample_reverse_sde(
        model, VPSDESchedule(), count=2, seed=19, device=torch.device("cpu"),
        sampling_steps=4, trajectory_index=1,
    )

    assert samples.shape == (2, 3, 64, 64)
    assert trajectory is not None and trajectory.shape == (5, 3, 64, 64)
    assert model.times == [1.0, 1.0, 0.75, 0.75, 0.5, 0.5, 0.25, 0.25]
    assert torch.isfinite(samples).all()
    assert torch.equal(samples[1], trajectory[-1])


def test_reverse_sde_repeats_for_same_seed_and_restores_model_mode() -> None:
    first_model = ZeroEpsilonModel().train()
    second_model = ZeroEpsilonModel().eval()
    first, first_path = sample_reverse_sde(
        first_model, VPSDESchedule(), count=1, seed=2026, device=torch.device("cpu"), sampling_steps=10,
    )
    second, second_path = sample_reverse_sde(
        second_model, VPSDESchedule(), count=1, seed=2026, device=torch.device("cpu"), sampling_steps=10,
    )

    assert torch.equal(first, second)
    assert first_path is not None and second_path is not None and torch.equal(first_path, second_path)
    assert first_model.training
    assert not second_model.training


def test_reverse_sde_adds_brownian_noise_on_final_interval() -> None:
    schedule = VPSDESchedule()
    seed = 101
    generator = torch.Generator(device="cpu").manual_seed(seed)
    initial = torch.randn((1, 3, 64, 64), generator=generator)
    final_increment = torch.randn(initial.shape, generator=generator)
    beta = schedule.beta(torch.ones(1)).item()
    expected = initial + 0.5 * beta * initial + beta**0.5 * final_increment

    actual, _ = sample_reverse_sde(
        ZeroEpsilonModel(), schedule, count=1, seed=seed, device=torch.device("cpu"),
        sampling_steps=1, trajectory_index=None,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_reverse_sde_reports_initial_and_final_frames() -> None:
    frames: list[tuple[int, int, float]] = []
    sample_reverse_sde(
        ZeroEpsilonModel(), VPSDESchedule(), count=1, seed=5, device=torch.device("cpu"),
        sampling_steps=10, on_frame=lambda index, step, t, image: frames.append((index, step, t)),
    )

    assert frames[0] == (0, 0, 1.0)
    assert frames[-1] == (1, 10, 0.0)


def test_annealed_langevin_is_seeded_and_saves_one_state_per_update() -> None:
    model = ZeroScoreModel().train()
    first, path = sample_annealed_langevin(
        model, VPSDESchedule(), count=2, seed=77, device=torch.device("cpu"),
        prediction="score", levels=3, steps_per_level=2, trajectory_index=1,
    )
    second, repeated_path = sample_annealed_langevin(
        ZeroScoreModel(), VPSDESchedule(), count=2, seed=77, device=torch.device("cpu"),
        prediction="score", levels=3, steps_per_level=2, trajectory_index=1,
    )
    assert torch.equal(first, second)
    assert path is not None and repeated_path is not None
    assert path.shape == (7, 3, 64, 64)
    assert torch.equal(path, repeated_path)
    assert model.training
    assert torch.isfinite(first).all()


def test_annealed_langevin_emits_frames_with_noise_level_progress() -> None:
    frames: list[tuple[int, int, float]] = []
    sample_annealed_langevin(
        ZeroScoreModel(), VPSDESchedule(), count=1, seed=8, device=torch.device("cpu"),
        prediction="score", levels=2, steps_per_level=5,
        on_frame=lambda index, step, t, image: frames.append((index, step, t)),
    )
    assert frames[0] == (0, 0, 1.0)
    assert frames[1][0:2] == (1, 10)
    assert abs(frames[1][2] - 1e-5) < 1e-10


def test_annealed_langevin_applies_the_sigma_scaled_update_equation() -> None:
    schedule = VPSDESchedule()
    seed = 913
    base_step_size = 2e-5
    generator = torch.Generator(device="cpu").manual_seed(seed)
    expected = torch.randn((1, 3, 64, 64), generator=generator)
    times = torch.linspace(1.0, 1e-5, 2)
    sigma_one = schedule.marginal_std(torch.ones(1))[0]
    for t_value in times:
        eta = base_step_size * schedule.marginal_std(t_value.reshape(1))[0].square() / sigma_one.square()
        noise = torch.randn(expected.shape, generator=generator)
        expected = expected + torch.sqrt(eta) * noise  # ZeroScoreModel removes the score term.

    actual, trajectory = sample_annealed_langevin(
        ZeroScoreModel(), schedule, count=1, seed=seed, device=torch.device("cpu"),
        prediction="score", levels=2, steps_per_level=1, base_step_size=base_step_size,
    )
    assert trajectory is not None
    assert torch.allclose(actual, expected[0], atol=1e-6, rtol=1e-6)
