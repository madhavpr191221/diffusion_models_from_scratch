import { useEffect, useState } from 'react'

type Run = {
  run_id: string
  state: string
  mode: string
  can_generate: boolean
}
type Checkpoint = {
  name: string
  kind: string
  epoch: number
  global_step: number
  validation_loss?: number | null
}
type Job = { generation_id: string; run_id: string; state: string }
type ImageSummary = {
  num_images: number
  value_range: [number, number]
  fraction_outside_training_range: number
  mean_per_image_pixel_std: number
  pairwise_pixel_rmse: number | null
  channels: Record<string, { mean: number; std: number; q01: number; median: number; q99: number }>
}
type ReferenceSummary = { split: 'validation'; selection: string; statistics: ImageSummary }
type Comparison = {
  generation_id: string
  run_id: string
  checkpoint: string
  epoch: number
  seed: number
  num_samples: number
  sample_statistics?: ImageSummary
  validation_reference?: ReferenceSummary
}
type Progress = {
  state: 'queued' | 'running' | 'completed' | 'failed' | 'interrupted'
  completed_steps: number
  total_steps: number
  current_timestep: number
  latest_frame_index: number
  frame_stride: number
  num_samples: number
  seed: number
  checkpoint: string
  generation_id?: string
  metadata?: {
    sample_value_range?: [number, number]
    fraction_outside_training_range?: number
    checkpoint_epoch?: number
    sample_statistics?: ImageSummary
    validation_reference?: ReferenceSummary
  }
  error?: string
}
type DisplayFrame = { url: string; timestep: number }

async function responseError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string }
    return body.detail ?? `Request failed (${response.status})`
  } catch {
    return `Request failed (${response.status})`
  }
}

function GenerationStudio() {
  const [runs, setRuns] = useState<Run[]>([])
  const [runId, setRunId] = useState('')
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [checkpointName, setCheckpointName] = useState('best-val-ema.pt')
  const [loadingCheckpoints, setLoadingCheckpoints] = useState(false)
  const [numSamples, setNumSamples] = useState(1)
  const [manualSeed, setManualSeed] = useState('')
  const [lastSeed, setLastSeed] = useState<number | null>(null)
  const [job, setJob] = useState<Job | null>(null)
  const [progress, setProgress] = useState<Progress | null>(null)
  const [comparisons, setComparisons] = useState<Comparison[]>([])
  const [frameIndex, setFrameIndex] = useState(-1)
  const [frame, setFrame] = useState<DisplayFrame | null>(null)
  const [playing, setPlaying] = useState(false)
  const [loadingRuns, setLoadingRuns] = useState(true)
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    let timer = 0
    const loadRuns = async () => {
      try {
        const response = await fetch('/api/runs', { signal: controller.signal })
        if (!response.ok) throw new Error(await responseError(response))
        const { items } = (await response.json()) as { items: Run[] }
        const eligible = items.filter((run) => run.can_generate)
        setRuns(eligible)
        setRunId((current) => eligible.some((run) => run.run_id === current) ? current : eligible[0]?.run_id ?? '')
      } catch (reason) {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Could not load trained runs')
      } finally {
        if (!controller.signal.aborted) {
          setLoadingRuns(false)
          timer = window.setTimeout(loadRuns, 10000)
        }
      }
    }
    void loadRuns()
    return () => {
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [])

  useEffect(() => {
    if (!runId) {
      setCheckpoints([])
      return
    }
    const controller = new AbortController()
    setLoadingCheckpoints(true)
    fetch(`/api/runs/${runId}/checkpoints`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response))
        return (await response.json()) as { items: Checkpoint[] }
      })
      .then(({ items }) => {
        setCheckpoints(items)
        setCheckpointName((current) => items.some((item) => item.name === current)
          ? current
        : items.some((item) => item.name === 'best-val-ema.pt') ? 'best-val-ema.pt'
          : items.some((item) => item.name === 'best-val.pt') ? 'best-val.pt' : items[0]?.name ?? '')
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Could not load checkpoints')
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingCheckpoints(false)
      })
    return () => controller.abort()
  }, [runId])

  useEffect(() => {
    if (!job) return
    const controller = new AbortController()
    let timer = 0
    let stopped = false
    const poll = async () => {
      try {
        const response = await fetch(`/api/runs/${job.run_id}/generations/${job.generation_id}`, { signal: controller.signal })
        if (!response.ok) throw new Error(await responseError(response))
        const result = (await response.json()) as Progress
        if (stopped) return
        setProgress(result)
        setJob((current) => current ? { ...current, state: result.state } : null)
        if (result.state === 'completed' && result.generation_id) {
          setComparisons((current) => [
            ...current.filter((item) => item.checkpoint !== result.checkpoint),
            {
              generation_id: result.generation_id!,
              run_id: job.run_id,
              checkpoint: result.checkpoint,
              epoch: result.metadata?.checkpoint_epoch ?? 0,
              seed: result.seed,
              num_samples: result.num_samples,
              sample_statistics: result.metadata?.sample_statistics,
              validation_reference: result.metadata?.validation_reference,
            },
          ].sort((first, second) => first.epoch - second.epoch || first.checkpoint.localeCompare(second.checkpoint)))
        }
        if (result.state === 'failed' || result.state === 'interrupted') {
          setError(result.error ?? 'Generation did not complete')
          setPlaying(false)
        } else if (result.state === 'queued' || result.state === 'running') {
          timer = window.setTimeout(poll, 100)
        }
      } catch (reason) {
        if (!controller.signal.aborted && !stopped) {
          setError(reason instanceof Error ? reason.message : 'Could not read generation progress')
          setPlaying(false)
        }
      }
    }
    void poll()
    return () => {
      stopped = true
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [job?.generation_id, job?.run_id])

  useEffect(() => {
    if (!job || frameIndex < 0) return
    const controller = new AbortController()
    let objectUrl: string | null = null
    fetch(`/api/runs/${job.run_id}/generations/${job.generation_id}/frames/${frameIndex}`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response))
        const timestep = Number(response.headers.get('X-Timestep'))
        objectUrl = URL.createObjectURL(await response.blob())
        setFrame({ url: objectUrl, timestep })
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Could not load this denoising frame')
      })
    return () => {
      controller.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [job?.generation_id, job?.run_id, frameIndex])

  useEffect(() => {
    if (!job || !progress || !playing) return
    if (frameIndex < 0 && progress.latest_frame_index >= 0) {
      setFrameIndex(0)
      return
    }
    if (frameIndex >= progress.latest_frame_index) {
      if (progress.state === 'completed') setPlaying(false)
      return
    }
    const timer = window.setTimeout(() => setFrameIndex((current) => current + 1), 50)
    return () => window.clearTimeout(timer)
  }, [job, progress, frameIndex, playing])

  function freshSeed(): number {
    return window.crypto.getRandomValues(new Uint32Array(1))[0]
  }

  async function startGeneration(seedOverride?: number, useManualSeed = false) {
    if (!runId) return
    const requestedSeed = seedOverride ?? (useManualSeed && manualSeed.trim() ? Number(manualSeed) : freshSeed())
    if (!Number.isInteger(requestedSeed) || requestedSeed < 0 || requestedSeed > 4294967295) {
      setError('Seed must be an integer from 0 to 4,294,967,295.')
      return
    }
    setStarting(true)
    setError(null)
    setJob(null)
    setProgress(null)
    setLastSeed(requestedSeed)
    setComparisons((current) => current.every((item) => item.seed === requestedSeed && item.num_samples === numSamples) ? current : [])
    setFrame(null)
    setFrameIndex(-1)
    setPlaying(true)
    try {
      const response = await fetch(`/api/runs/${runId}/generations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_samples: numSamples, seed: requestedSeed, checkpoint: checkpointName }),
      })
      if (!response.ok) throw new Error(await responseError(response))
      const created = (await response.json()) as Job
      setJob(created)
    } catch (reason) {
      setPlaying(false)
      setError(reason instanceof Error ? reason.message : 'Could not start generation')
    } finally {
      setStarting(false)
    }
  }

  const isRunning = job?.state === 'queued' || job?.state === 'running'
  const latestFrameIndex = progress?.latest_frame_index ?? -1
  const maxFrameIndex = Math.max(0, latestFrameIndex)
  const sampleGridUrl = job && progress?.state === 'completed'
    ? `/api/runs/${job.run_id}/generations/${job.generation_id}/samples.png`
    : null
  const runBusy = loadingRuns || loadingCheckpoints || starting || isRunning
  const chosenCheckpoint = checkpoints.find((item) => item.name === checkpointName)
  const clippedPercent = progress?.metadata?.fraction_outside_training_range === undefined
    ? null
    : 100 * progress.metadata.fraction_outside_training_range

  return (
    <section className="generation-view">
      <header className="generation-heading">
        <div>
          <div className="eyebrow">LIVE UNCONDITIONAL GENERATION</div>
          <h1>Watch noise <em>become a patch.</em></h1>
          <p>Start from fresh Gaussian noise and watch all 1,000 reverse steps create a new 64 × 64 RGB patch.</p>
        </div>
      </header>

      <section className="generation-controls panel" aria-label="Generation settings">
        <div className="panel-heading">
          <div><span className="section-index">01</span><h2>Generate a patch live</h2></div>
          <span className="local-tag">{chosenCheckpoint?.name ?? 'BEST EMA CHECKPOINT'}</span>
        </div>
        {loadingRuns ? <div className="empty-state">Loading trained checkpoints…</div> : runs.length === 0 ? (
          <div className="generation-empty">
            <p>No completed checkpoint is available for generation yet.</p>
          </div>
        ) : (
          <>
          <div className="live-generation-actions">
            <button className="primary-button generation-submit live-generate-button" onClick={() => void startGeneration()} disabled={!runId || runBusy || loadingCheckpoints || checkpoints.length === 0}>
              {isRunning || starting ? 'Generating live…' : sampleGridUrl ? 'Generate another patch' : 'Generate live patch'}
            </button>
            {lastSeed !== null && <div className="live-seed-status" role="status">
              <span>Noise seed <b>{lastSeed}</b></span>
              <button className="quiet-button" onClick={() => void startGeneration(lastSeed)} disabled={runBusy} aria-label={`Replay generation with seed ${lastSeed}`}>Replay same noise</button>
            </div>}
          </div>
          <details className="generation-advanced">
            <summary>Advanced settings</summary>
            <div className="generation-form">
              <label className="generation-run" htmlFor="generation-run">
                <span>Completed run</span>
                <select id="generation-run" value={runId} disabled={runBusy} onChange={(event) => {
                  setRunId(event.target.value)
                  setComparisons([])
                  setJob(null)
                  setProgress(null)
                  setFrame(null)
                  setFrameIndex(-1)
                }}>
                  {runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id}</option>)}
                </select>
              </label>
              <label className="generation-run generation-checkpoint" htmlFor="generation-checkpoint">
                <span>Checkpoint</span>
                <select id="generation-checkpoint" value={checkpointName} disabled={runBusy || checkpoints.length === 0} onChange={(event) => setCheckpointName(event.target.value)}>
                  {checkpoints.map((item) => <option key={item.name} value={item.name}>
                    {item.name} · epoch {item.epoch} · step {item.global_step.toLocaleString()}{item.validation_loss == null ? '' : ` · val ${item.validation_loss.toFixed(5)}`}
                  </option>)}
                </select>
                <small>{loadingCheckpoints ? 'Loading saved checkpoints…' : chosenCheckpoint ? `${chosenCheckpoint.kind} snapshot` : 'No checkpoints available'}</small>
              </label>
              <label className="generation-number" htmlFor="generation-count">
                <span>Batch size (1–8 patches)</span>
                <input id="generation-count" type="number" min="1" max="8" step="1" value={numSamples} disabled={runBusy} onChange={(event) => setNumSamples(Math.max(1, Math.min(8, Number(event.target.value) || 1)))} />
                <small>The demo animates patch 1.</small>
              </label>
              <label className="generation-number" htmlFor="generation-seed">
                <span>Manual seed (optional)</span>
                <input id="generation-seed" type="number" min="0" max="4294967295" step="1" placeholder="Fresh random seed" value={manualSeed} disabled={runBusy} onChange={(event) => setManualSeed(event.target.value)} />
                <small>Fresh noise is used by the main button. Use this seed to reproduce a trajectory.</small>
                <button className="quiet-button manual-seed-button" disabled={runBusy || !manualSeed.trim()} onClick={() => void startGeneration(undefined, true)}>Generate with this seed</button>
              </label>
            </div>
          </details>
          </>
        )}
        <div className="learning-note"><b>What you’re seeing:</b> one random noise image is refined by the trained denoiser. It is a new unconditional patch, not a reconstruction of a dataset image.</div>
        {error && <div className="error-banner" role="alert">{error}</div>}
      </section>

      <section className="generation-trace panel" aria-label="Live reverse sampling">
        <div className="panel-heading">
          <div><span className="section-index">02</span><h2>Follow the reverse process</h2></div>
          <span className="step-count">{progress ? `${progress.state.toUpperCase()} · ${progress.checkpoint}` : 'READY WHEN YOU ARE'}</span>
        </div>
        <div className="generation-image-stage">
          <div className="generation-image-display">
            {frame ? <img src={frame.url} alt={frame.timestep === 0 ? 'Generated EuroSAT-style patch' : `Current denoising state x at step ${frame.timestep}`} /> : <div className="image-placeholder">{isRunning ? 'Waiting for the first frame…' : 'Generate patches to begin from random noise'}</div>}
            <span className="image-caption">{frame?.timestep === 0 ? 'GENERATED PATCH' : 'CURRENT STATE'} <b>{frame ? (frame.timestep === 0 ? 'x₀' : `t = ${frame.timestep}`) : 'xₜ'}</b></span>
          </div>
          <div className="generation-progress-copy">
            <strong>{progress ? `${progress.completed_steps.toLocaleString()} / ${progress.total_steps.toLocaleString()}` : '1,000 reverse steps'}</strong>
            <span>{progress?.state === 'completed' ? 'Sampling complete' : isRunning ? 'Predicting and removing noise' : 'The model starts from Gaussian noise'}</span>
          </div>
        </div>
        <div className="progress-track generation-progress-track"><span style={{ width: `${progress ? 100 * progress.completed_steps / Math.max(progress.total_steps, 1) : 0}%` }} /></div>
        <div className="generation-frame-controls">
          <input
            aria-label="Denoising frame"
            type="range"
            min="0"
            max={maxFrameIndex}
            value={Math.max(0, Math.min(frameIndex, maxFrameIndex))}
            disabled={latestFrameIndex < 0}
            onChange={(event) => { setPlaying(false); setFrameIndex(Number(event.target.value)) }}
          />
          <div className="playback-controls">
            <button className="quiet-button" disabled={latestFrameIndex < 0} onClick={() => { setPlaying(false); setFrameIndex(0) }}>↺ Reset</button>
            <button className="play-button" disabled={latestFrameIndex < 0} onClick={() => {
              if (frameIndex >= latestFrameIndex && progress?.state === 'completed') setFrameIndex(0)
              setPlaying((current) => !current)
            }}>{playing ? 'Ⅱ  Pause animation' : '▶  Play animation'}</button>
            <span className="play-hint">101 snapshots · every 10 reverse steps</span>
          </div>
        </div>
      </section>

      {sampleGridUrl && (
        <section className="generation-results panel" aria-label="Generated patch grid">
          <div className="panel-heading">
            <div><span className="section-index">03</span><h2>Generated patches</h2></div>
            <span className="split-label">SEED {progress?.seed ?? lastSeed ?? '—'} · {progress?.num_samples ?? numSamples} {(progress?.num_samples ?? numSamples) === 1 ? 'PATCH' : 'PATCHES'}</span>
          </div>
          <img className="generated-grid-image" src={sampleGridUrl} alt={`${progress?.num_samples ?? numSamples} unconditional EuroSAT-style image patches generated from noise`} />
          <p className="panel-copy">The full batch is shown above. The animation follows patch 1 from its initial noise to its final sample. Checkpoint: <code>{progress?.checkpoint ?? checkpointName}</code>; seed: {progress?.seed ?? lastSeed ?? '—'}.</p>
          {clippedPercent !== null && <p className="generation-diagnostic" role="status">
            Raw values range from {progress?.metadata?.sample_value_range?.[0].toFixed(2)} to {progress?.metadata?.sample_value_range?.[1].toFixed(2)}. Display conversion clips {clippedPercent.toFixed(1)}% of tensor values to the trained image range [-1, 1]. Raw samples are preserved in the run artifacts.
          </p>}
        </section>
      )}

      {comparisons.length > 1 && (
        <section className="generation-comparison panel" aria-label="Checkpoint sample comparison">
          <div className="panel-heading">
            <div><span className="section-index">04</span><h2>Compare checkpoints</h2></div>
            <span className="split-label">SAME SEED {comparisons[0].seed} · {comparisons.length} CHECKPOINTS</span>
          </div>
          <div className="checkpoint-comparison-grid">
            {comparisons.map((item) => <figure key={item.checkpoint}>
              <figcaption>{item.checkpoint}</figcaption>
              <img src={`/api/runs/${item.run_id}/generations/${item.generation_id}/samples.png`} alt={`${item.num_samples} generated patches from ${item.checkpoint} with seed ${item.seed}`} />
              <small>{item.num_samples} patches · seed {item.seed}</small>
              {item.sample_statistics && <small>
                Outside range: {(100 * item.sample_statistics.fraction_outside_training_range).toFixed(1)}% ·
                diversity RMSE: {item.sample_statistics.pairwise_pixel_rmse?.toFixed(3) ?? 'n/a'}
              </small>}
              {item.sample_statistics && <small>
                RGB means: {(['red', 'green', 'blue'] as const).map((channel) => item.sample_statistics!.channels[channel].mean.toFixed(2)).join(' / ')} ·
                RGB SDs: {(['red', 'green', 'blue'] as const).map((channel) => item.sample_statistics!.channels[channel].std.toFixed(2)).join(' / ')}
              </small>}
            </figure>)}
          </div>
          {comparisons[0].validation_reference && <div className="comparison-reference">
            <b>Validation reference</b> · {comparisons[0].validation_reference.statistics.num_images} fixed images ({comparisons[0].validation_reference.selection}).
            Mean per-image pixel SD: {comparisons[0].validation_reference.statistics.mean_per_image_pixel_std.toFixed(3)} ·
            pairwise pixel RMSE: {comparisons[0].validation_reference.statistics.pairwise_pixel_rmse?.toFixed(3) ?? 'n/a'}.
            <div className="channel-summary">
              {(['red', 'green', 'blue'] as const).map((channel) => {
                const values = comparisons[0].validation_reference!.statistics.channels[channel]
                return <span key={channel}>{channel}: mean {values.mean.toFixed(3)}, SD {values.std.toFixed(3)}, q01/q50/q99 {values.q01.toFixed(2)}/{values.median.toFixed(2)}/{values.q99.toFixed(2)}</span>
              })}
            </div>
          </div>}
        </section>
      )}

      <footer className="page-footer generation-footer">
        <span>UNCONDITIONAL DDPM · EUROSAT RGB</span>
        <span>SELECTED CHECKPOINT · NO LABELS</span>
        <span>64 × 64 PATCHES</span>
      </footer>
    </section>
  )
}

export default GenerationStudio
