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

- [x] Specify the model input/output contract, zero-based timestep indexing, embedding frequencies, residual-block conditioning, and encoder/decoder shape map.
- [x] Implement cosine-then-sine timestep embedding with the learned 128 -> 512 -> 128 transform.
- [x] Implement timestep-conditioned residual blocks and the 32/64/128-channel U-Net with an 8x8 bottleneck and matching skip concatenations.
- [x] Run a CPU synthetic-batch smoke check for embedding shape, intermediate feature shapes, finite predictions, and output shape.
- [x] Record the observed tensor shapes and parameter count before review.

Stage 3 evidence: A CPU forward pass with batch size 2 and timesteps `[0,999]` returned finite predictions shaped `[2,3,64,64]`; the time embedding was finite with shape `[2,128]`. Encoder feature maps were `[2,32,64,64]`, `[2,64,32,32]`, `[2,128,16,16]`, then `[2,128,8,8]`. Decoder concatenations were `[2,256,16,16]`, `[2,128,32,32]`, and `[2,64,64,64]`, matching the saved skips. The model has 2,309,539 parameters. No training or optimizer steps were run.

## Stage 4 — Training

- [x] Implement a CLI noise-prediction trainer that loads only the train split for updates and the validation split for diagnostics; do not load the test split.
- [x] Add a fixed 64-image / 200-update GPU pilot, per-update JSONL metrics, run status, atomic resumable checkpoints, milestone snapshots, and best-validation checkpoint tracking.
- [x] Add read-only run, metric, and checkpoint API routes and a Training Monitor view in the existing React app.
- [x] Document run commands, artifact structure, loss interpretation, and the held-out test boundary.
- [x] Run the CUDA pilot: 200 updates completed with finite, falling loss and peak allocated CUDA memory recorded.
- [x] Run the five-epoch baseline: all 10,125 updates and all five full validation passes completed; final train/validation MSE was 0.01785 / 0.01663.
- [x] Save four progress snapshots, best-validation and latest checkpoints under the ignored run directory; document actual runtime, metrics, and checkpoint paths.
- [x] Review the completed-run React monitor and checkpoint metadata in the user-provided browser screenshots.
- [ ] Explicitly smoke-check checkpoint resumption as a Stage 4 follow-up.

## Stage 5 — Sampling

- [x] Implement the epsilon-parameterized ancestral update with posterior variance and no added noise at index 0.
- [x] Verify the reverse equation with supplied noise and verify index 0 is independent of supplied noise.
- [x] Run a full 1,000-step single-image sampling smoke run from `best-val.pt`.
- [x] Generate eight seeded samples and retain all 1,001 states for sample 0.
- [x] Save and inspect raw tensors, sample grid, trajectory contact sheet, and generation metadata.
- [x] Record the sample-quality limitation; loss does not establish visual quality.

## Stage 6 — Local dashboard

- [ ] Implement dataset, forward-noising, run metrics/checkpoint, and generation views.
- [ ] Connect the frontend to the local API without duplicating diffusion logic.
- [ ] Verify the production build and one local end-to-end browser workflow.
