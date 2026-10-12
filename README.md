# EuroSAT RGB DDPM

A small, unconditional DDPM project for learning the diffusion equations by implementing them directly in PyTorch.

The project is being developed in reviewed milestones. See:

- [Project principles](docs/CONSTITUTION.md)
- [Detailed plan](docs/plans/eurosat-ddpm.md)
- [Project specification](docs/specs/001-eurosat-ddpm/spec.md)
- [Design notes](docs/specs/001-eurosat-ddpm/design.md)
- [Milestone tasks](docs/specs/001-eurosat-ddpm/tasks.md)
- [Learning log](docs/learning-log.md)
- [Bug reports](docs/bug-reports.md)

Downloaded data and generated runs are local artifacts and are excluded from version control.

## Local setup

On Windows, install the locked Python environment and frontend dependencies with:

`````powershell
uv sync --locked
npm ci --prefix web
```

The Python lock uses the official PyTorch CUDA 13.0 wheel index for the RTX 5060 environment. To confirm the runtime after setup:

`````powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The local forward-noise viewer is available before model training. In separate terminals, start the API and Vite frontend:

`````powershell
uv run --locked uvicorn ddpm_eurosat.api:app --host 127.0.0.1 --port 8000
npm run dev --prefix web
```

Open the local URL printed by Vite. The viewer lets you select validation images and inspect direct marginal or stepwise forward noising; no trained model is needed. The production frontend is served by the API after `npm run build --prefix web`.

## Generate EuroSAT-style patches

Open the **Generate** view in the React app. Select a completed training run and checkpoint, enter a seed and patch count, then choose **Generate patches**. The app starts from Gaussian noise, animates one patch during the complete 1,000-step reverse process, and shows the generated patch grid when complete. Keep the seed and count fixed while generating `epoch-00010.pt```through `epoch-00050.pt```and `best-val.pt`; the app accumulates a side-by-side comparison for that setting. Each result reports raw-range, per-channel, and pairwise-diversity summaries against the first 128 unlabeled validation images. Raw patches, diagnostics, and the full trajectory for the animated patch are saved under `rruns/<run-id>/generation/<generation-id>/`. Display PNGs clip values to `[-1, 1]`; raw tensors remain unchanged. This comparison uses validation only; the test split remains untouched.

For a saved CLI comparison artifact, use the same seed and count for each checkpoint and add `--compare-validation`:

`````powershell
uv run --locked python -m ddpm_eurosat.sampling --checkpoint rruns/<run-id>/checkpoints/best-val.pt --num-samples 8 --seed 42 --compare-validation --output-dir rruns/<run-id>/generation/study-seed-42/best-val
```

In development, keep the API and Vite commands running in separate terminals as shown above. For a single-server local session, build the frontend with `npm run build --prefix web`, start only the API, and open `http://127.0.0.1:8000`.

## Stage 4 training

Start with the small GPU pilot, then run the configured five-epoch baseline:

`````powershell
uv run --locked python -m ddpm_eurosat.train --pilot
uv run --locked python -m ddpm_eurosat.train
```

Training uses the train split only; validation is used for epoch diagnostics and the test split is not loaded. Open **Training monitor** in the existing React app to watch status, loss curves, runtime, GPU memory, and checkpoints. Runs are stored under ignored `rruns/<run-id>/`. To resume the latest saved update:

`````powershell
uv run --locked python -m ddpm_eurosat.train --resume rruns/<run-id>/checkpoints/latest.pt
```

## Longer training run

The dedicated 50-epoch configuration preserves the baseline model and optimizer while saving selectable checkpoints every 10 epochs:

`````powershell
uv run --locked python -m ddpm_eurosat.train --config configs/eurosat_50_epoch.toml
```

The run has 101,250 optimizer updates. During the first two epochs on the RTX 5060 it averaged about 34 updates/second, suggesting roughly 50 minutes before validation and checkpoint overhead; the estimate may shift as the run progresses. Validation runs once per epoch. The run writes its own immutable config and logs under `rruns/<run-id>/`; use the Generate view to compare epoch checkpoints with the same seed. The test split is not loaded.

For the controlled EMA comparison, run the separate matching configuration with EMA decay 0.999:

`````powershell
uv run --locked python -m ddpm_eurosat.train --config configs/eurosat_50_epoch_ema.toml --device cuda
```

The training monitor plots raw and EMA validation losses. The run saves `best-val.pt```and `best-val-ema.pt`; the Generate view defaults to the EMA checkpoint when available and can accumulate same-seed raw/EMA grids for comparison. Record generation failures and their evidence in the append-only bug report.

## Dataset inspection

The first data milestone uses the pinned Hugging Face EuroSAT RGB revision. To download/reuse its local cache and regenerate the inspection report and preview:

`````powershell
uv run --locked python scripts/inspect_eurosat.py
```

The report and preview are written to the ignored `outputs/```directory. The dataset card currently lists its license as unknown; this project keeps the data local and does not redistribute it.

## Original DDPM U-Net run

Use the practical 50-epoch EuroSAT run as the default. It keeps the original DDPM U-Net, 64x64 patches, and effective batch 128; 6,350 updates are 50 passes over the 16,200 training patches. Pilot throughput suggests roughly 3.5–4 hours, plus validation and checkpoint overhead.

```powershell
uv run --locked python -m ddpm_eurosat.train --config configs/eurosat_original_ddpm_50_epoch.toml --device cuda
```

The 800,000-update configuration in `configs/eurosat_original_ddpm.toml` is a paper-scale reference, not the recommended EuroSAT run; the GPU pilot projected roughly 18 days. The practical run saves resumable state every epoch in `runs/<run-id>/checkpoints/latest.pt`. It validates every five epochs, saves best raw and EMA checkpoints, and never loads the test split. Follow progress in the existing React Training monitor and use the EMA checkpoint in Generate after validation selects it.