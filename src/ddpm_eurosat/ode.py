"""Small deterministic fixed-step solvers for continuous image transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch
from torch import Tensor

VectorField = Callable[[Tensor, Tensor], Tensor]
SolverMethod = Literal["euler", "heun"]


@dataclass(frozen=True)
class ODESolution:
    """Selected states along an ODE solution, with actual field-call count."""

    times: Tensor
    states: Tensor
    nfe: int
    method: SolverMethod


def solve_ode(
    field: VectorField,
    initial_state: Tensor,
    time_grid: Tensor,
    *,
    method: SolverMethod = "heun",
    save_every: int = 1,
) -> ODESolution:
    """Integrate ``dy/dt = field(t, y)`` over a monotone grid.

    ``time_grid`` may increase or decrease. Euler uses
    ``y[n+1] = y[n] + h*f(t[n], y[n])``. Heun averages the field at the
    interval endpoints. Snapshots include the initial and final state; an
    intermediate state is retained every ``save_every`` intervals.
    """
    if initial_state.ndim < 1 or initial_state.shape[0] < 1:
        raise ValueError("initial_state must include a non-empty batch dimension")
    if time_grid.ndim != 1 or time_grid.numel() < 2:
        raise ValueError("time_grid must be a one-dimensional tensor with at least two values")
    if not time_grid.is_floating_point() or not torch.isfinite(time_grid).all():
        raise ValueError("time_grid must contain finite floating-point values")
    if time_grid.device != initial_state.device:
        raise ValueError("time_grid and initial_state must be on the same device")
    if save_every < 1:
        raise ValueError("save_every must be at least one")
    if method not in ("euler", "heun"):
        raise ValueError(f"Unsupported ODE method: {method}")

    intervals = time_grid[1:] - time_grid[:-1]
    increasing = bool(torch.all(intervals > 0))
    decreasing = bool(torch.all(intervals < 0))
    if not (increasing or decreasing):
        raise ValueError("time_grid must be strictly monotone")

    state = initial_state
    saved_times = [time_grid[0]]
    saved_states = [state]
    nfe = 0
    for index, step in enumerate(intervals):
        t = time_grid[index]
        first = field(t, state)
        nfe += 1
        if first.shape != state.shape:
            raise ValueError("vector field must return a tensor with the state shape")
        if not torch.isfinite(first).all():
            raise FloatingPointError(f"non-finite vector field at interval {index}")
        if method == "euler":
            state = state + step * first
        else:
            candidate = state + step * first
            second = field(time_grid[index + 1], candidate)
            nfe += 1
            if second.shape != state.shape:
                raise ValueError("vector field must return a tensor with the state shape")
            if not torch.isfinite(second).all():
                raise FloatingPointError(f"non-finite vector field at interval {index + 1}")
            state = state + step * (first + second) / 2
        if not torch.isfinite(state).all():
            raise FloatingPointError(f"non-finite ODE state at interval {index + 1}")
        if (index + 1) % save_every == 0 or index + 1 == intervals.numel():
            saved_times.append(time_grid[index + 1])
            saved_states.append(state)

    return ODESolution(
        times=torch.stack(saved_times),
        states=torch.stack(saved_states),
        nfe=nfe,
        method=method,
    )
