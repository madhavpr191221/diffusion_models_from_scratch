import pytest
import torch

from ddpm_eurosat.ode import solve_ode


def test_euler_constant_field_solution_and_nfe():
    initial = torch.zeros(2, 3)
    times = torch.linspace(0, 1, 5)
    solution = solve_ode(lambda t, y: torch.ones_like(y) * 2, initial, times, method="euler")
    torch.testing.assert_close(solution.states[-1], torch.full_like(initial, 2))
    assert solution.nfe == 4
    assert solution.states.shape == (5, 2, 3)


@pytest.mark.parametrize("method, expected_nfe", [("euler", 10), ("heun", 20)])
def test_backward_constant_field_and_snapshot_order(method, expected_nfe):
    initial = torch.full((2, 1), 3.0)
    times = torch.linspace(1, 0, 11)
    solution = solve_ode(lambda t, y: torch.ones_like(y), initial, times, method=method, save_every=3)
    torch.testing.assert_close(solution.states[-1], torch.full_like(initial, 2.0))
    assert solution.nfe == expected_nfe
    torch.testing.assert_close(solution.times, torch.tensor([1.0, 0.7, 0.4, 0.1, 0.0]))


def test_heun_converges_for_linear_field_under_step_refinement():
    errors = []
    for steps in (8, 16):
        times = torch.linspace(0, 1, steps + 1)
        result = solve_ode(lambda t, y: y, torch.ones(1, 1), times, method="heun")
        errors.append(abs(float(result.states[-1, 0, 0]) - torch.e))
    assert errors[1] < errors[0] / 3


def test_rejects_non_monotone_times_and_wrong_field_shape():
    initial = torch.zeros(1, 2)
    with pytest.raises(ValueError, match="strictly monotone"):
        solve_ode(lambda t, y: y, initial, torch.tensor([0.0, 1.0, 0.5]))
    with pytest.raises(ValueError, match="state shape"):
        solve_ode(lambda t, y: y[:, :1], initial, torch.tensor([0.0, 1.0]))


def test_seeded_initial_state_is_deterministic_for_pure_field():
    initial_a = torch.randn((2, 3), generator=torch.Generator().manual_seed(19))
    initial_b = torch.randn((2, 3), generator=torch.Generator().manual_seed(19))
    times = torch.linspace(0, 1, 6)
    field = lambda t, y: -0.2 * y
    first = solve_ode(field, initial_a, times, method="heun")
    second = solve_ode(field, initial_b, times, method="heun")
    torch.testing.assert_close(first.states, second.states)
