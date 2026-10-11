# Milestone Tasks

## Stage 0 — Foundation

- [x] Record project constitution, specification, design, detailed plan, and staged tasks.
- [x] Establish Python package metadata and pin the verified PyTorch/torchvision pair.
- [x] Establish React/TypeScript/Vite project scaffold.
- [x] Ignore datasets, run outputs, Python environments, and frontend build/dependency directories.
- [x] Review Stage 0 and accept the foundation before implementing data/model code.

## Stage 1 — Dataset

- [x] Add the pinned Hugging Face dataset dependency and lockfile update.
- [x] Download/cache `blanchon/EuroSAT_RGB` at the pinned revision under ignored `data/huggingface/`.
- [x] Preserve source splits; expose image tensors only to the model-facing adapter.
- [x] Inspect actual split counts, dimensions, channels, dtype, ranges, and batch shape.
- [x] Save and inspect a small normalization/de-normalization example grid and JSON report.

## Stage 2 — Schedule and forward process

- [x] Implement schedule coefficients and explicit timestep indexing/broadcasting.
- [x] Implement independent batch timestep sampling and direct forward noising with optional supplied noise.
- [x] Check schedule invariants and equation behavior; save a multi-timestep figure and JSON evidence report.

Stage 2 evidence: `uv run --locked python scripts/visualize_forward_process.py` completed on CUDA. All 1,000 schedule values were finite, `0 < alpha < 1`, and cumulative alpha strictly decreased (from 0.999900 to 0.00004036). The supplied-noise equation check had maximum absolute error 0; the generated-noise path also matched its returned noise exactly. A 64-item timestep draw was within `[0,999]`. The figure and report are saved under ignored `outputs/`.

## Stage 2.5 — Interactive forward-noise viewer

- [x] Add deterministic stepwise Markov forward trajectory generation while preserving direct marginal noising.
- [x] Add a local FastAPI interface for validation previews and seeded trace creation/frame retrieval.
- [x] Replace the Vite starter with an accessible React/TypeScript image selector and trace controls for both modes.
- [x] Add the learner-facing project log for completed stages and current implementation status.
- [x] Verify both modes, metadata, seed reproducibility, API proxy, learning-log route, and production frontend assets through focused local HTTP checks.
- [ ] Complete a visual browser click-through of image selection, scrubbing, and play/pause; the Windows browser helper failed setup twice during this run.

Stage 2.5 implementation evidence: `npm run build --prefix web` succeeded. The local API listed all 5,400 validation items without labels, returned PNG frames with correct timestep/`alpha_bar` headers for both modes, reproduced each mode byte-for-byte for the same seed, and returned 404 for a replaced trace. The Vite `/api` proxy, FastAPI-served production page/assets, and learning-log endpoint all responded successfully. The focused Markov math check verified the first two transitions with zero absolute error and identical repeat runs. The visual browser click-through remains unverified because the Computer Use helper returned `helper_unknown_error: setup refresh had errors` on both initialization attempts.

## Stage 3 — Timestep representation and U-Net

- [ ] Implement sinusoidal embedding and learned transform.
- [ ] Implement small conditioned U-Net and matching-resolution skips.
- [ ] Record representative tensor shapes and check finite output shape.

## Stage 4 — Training

- [ ] Implement simple noise-prediction training, metrics, and checkpoint format.
- [ ] Run tiny-subset debug/overfit experiment and measure runtime/memory.
- [ ] Set the baseline budget from pilot measurements and run it.

## Stage 5 — Sampling

- [ ] Implement documented ancestral posterior-variance update.
- [ ] Check one reverse update and a reduced smoke sampling loop.
- [ ] Generate and inspect samples and reverse trajectory from a saved checkpoint.

## Stage 6 — Local dashboard

- [ ] Implement dataset, forward-noising, run metrics/checkpoint, and generation views.
- [ ] Connect the frontend to the local API without duplicating diffusion logic.
- [ ] Verify the production build and one local end-to-end browser workflow.
