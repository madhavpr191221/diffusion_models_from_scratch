# Project Constitution

## Purpose

Teach the mechanics of an unconditional image DDPM through a small, inspectable implementation using EuroSAT RGB.

## Principles

1. **Mathematics stays visible.** Important functions identify the equation, input/output shapes, and corresponding variables.
2. **Unconditional means unconditional.** Do not use dataset labels or add class, text, or other conditioning.
3. **Use small, direct components.** Use PyTorch for the DDPM core; do not hide the forward or reverse process behind a diffusion library.
4. **Make evidence reproducible.** Record configuration, random seed, metrics, checkpoints, and generated figures for each training run.
5. **Prefer explanation over engineering overhead.** Add abstractions, dependencies, and tests only when they serve a stated requirement or prevent a demonstrated failure.
6. **Keep data local.** Do not commit downloaded imagery, checkpoints, or generated run outputs.

## Development process

Follow specification-driven development within a small SDLC: state requirements and acceptance criteria, design the smallest implementation that meets them, implement one bounded milestone, verify it, record the evidence, and review before starting the next milestone. If requirements change, update the specification first. Do not skip a failed milestone; fix or explicitly revise its acceptance criteria before continuing.

## Review standard

Every milestone report should say what changed, what was checked, observed tensor shapes or outputs, known limitations, and what milestone comes next. Tests should be few and tied to mathematical invariants or public interfaces; visual checks may be more informative than additional test scaffolding for this learning project.
