# EuroSAT RGB DDPM

A small, unconditional DDPM project for learning the diffusion equations by implementing them directly in PyTorch.

The project is being developed in reviewed milestones. See:

- [Project principles](docs/CONSTITUTION.md)
- [Detailed plan](docs/plans/eurosat-ddpm.md)
- [Project specification](docs/specs/001-eurosat-ddpm/spec.md)
- [Design notes](docs/specs/001-eurosat-ddpm/design.md)
- [Milestone tasks](docs/specs/001-eurosat-ddpm/tasks.md)
- [Learning log](docs/learning-log.md)

Downloaded data and generated runs are local artifacts and are excluded from version control.

## Local setup

On Windows, install the locked Python environment and frontend dependencies with:

```powershell
uv sync --locked
npm ci --prefix web
```

The Python lock uses the official PyTorch CUDA 13.0 wheel index for the RTX 5060 environment. To confirm the runtime after setup:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The local forward-noise viewer is available before model training. In separate terminals, start the API and Vite frontend:

```powershell
uv run --locked uvicorn ddpm_eurosat.api:app --host 127.0.0.1 --port 8000
npm run dev --prefix web
```

Open the local URL printed by Vite. The viewer lets you select validation images and inspect direct marginal or stepwise forward noising; no trained model is needed. The production frontend is served by the API after `npm run build --prefix web`.

## Stage 4 training

Start with the small GPU pilot, then run the configured five-epoch baseline:

```powershell
uv run --locked python -m ddpm_eurosat.train --pilot
uv run --locked python -m ddpm_eurosat.train
```

Training uses the train split only; validation is used for epoch diagnostics and the test split is not loaded. Open **Training monitor** in the existing React app to watch status, loss curves, runtime, GPU memory, and checkpoints. Runs are stored under ignored `runs/<run-id>/`. To resume the latest saved update:

```powershell
uv run --locked python -m ddpm_eurosat.train --resume runs/<run-id>/checkpoints/latest.pt
```

## Dataset inspection

The first data milestone uses the pinned Hugging Face EuroSAT RGB revision. To download/reuse its local cache and regenerate the inspection report and preview:

```powershell
uv run --locked python scripts/inspect_eurosat.py
```

The report and preview are written to the ignored `outputs/` directory. The dataset card currently lists its license as unknown; this project keeps the data local and does not redistribute it.
