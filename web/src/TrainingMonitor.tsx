import { useEffect, useMemo, useRef, useState } from 'react'
import { getModelProfile, MODEL_PROFILES, resolveTrainingRun, type ModelFamily, type ModelRun } from './modelChoices'

type RunSummary = ModelRun & {
  global_step: number
  epoch: number
  total_epochs: number
}
type MetricEvent = {
  event_id: number
  kind: 'train' | 'validation' | 'validation_full'
  epoch: number
  global_step: number
  loss: number
  ema_loss?: number | null
  train_epoch_loss?: number
  elapsed_seconds: number
}
type RunStatus = {
  state?: string
  mode?: string
  device?: string
  epoch?: number
  total_epochs?: number
  global_step?: number
  planned_steps?: number
  elapsed_seconds?: number
  steps_per_second?: number
  peak_gpu_memory_mb?: number | null
  error?: string | null
}
type Checkpoint = {
  name: string
  kind: string
  epoch: number
  global_step: number
  validation_loss: number | null
  bytes: number
  saved_at: string
}
type ResolvedConfig = {
  process?: string
  seed?: number
  model?: { prediction?: 'epsilon' | 'score'; architecture?: string }
  training?: { batch_size?: number; effective_batch_size?: number; micro_batch_size?: number; epochs?: number; learning_rate?: number; optimizer?: string; validation_every_updates?: number; monitor_interval_updates?: number }
  diffusion?: { timesteps?: number }
  vp_sde?: { beta_min?: number; beta_max?: number; training_time_min?: number }
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: 'no-store' })
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`)
  return (await response.json()) as T
}

function LossChart({ events, windowUpdates, label }: { events: MetricEvent[]; windowUpdates: number; label: string }) {
  const latestByStep = new Map<string, MetricEvent>()
  for (const event of events) latestByStep.set(`${event.kind}:${event.global_step}`, event)
  const currentEvents = [...latestByStep.values()].sort((first, second) => first.global_step - second.global_step)
  const train = currentEvents.filter((event) => event.kind === 'train')
  const validation = currentEvents.filter((event) => event.kind === 'validation' || event.kind === 'validation_full')
  const smoothed = train.map((event, index) => {
    const window = train.slice(Math.max(0, index - 99), index + 1)
    return { step: event.global_step, loss: window.reduce((sum, item) => sum + item.loss, 0) / window.length }
  })
  const emaValidation = validation.filter((event) => event.ema_loss != null)
  const values = [...train.map((event) => event.loss), ...validation.map((event) => event.loss), ...emaValidation.map((event) => event.ema_loss as number)]
  if (!values.length) return <div className="chart-empty">Loss values will appear after the first update.</div>

  const width = 900
  const height = 260
  const left = 58
  const right = 18
  const top = 20
  const bottom = 34
  const minStep = Math.min(...currentEvents.map((event) => event.global_step))
  const maxStep = Math.max(1, ...currentEvents.map((event) => event.global_step))
  const minLoss = Math.min(...values)
  const maxLoss = Math.max(...values)
  const lossSpan = Math.max(maxLoss - minLoss, Math.abs(maxLoss) * 0.08, 1e-5)
  const x = (step: number) => left + ((step - minStep) / Math.max(maxStep - minStep, 1)) * (width - left - right)
  const y = (loss: number) => top + (1 - (loss - minLoss) / lossSpan) * (height - top - bottom)
  const rawPoints = train.map((event) => `${x(event.global_step).toFixed(1)},${y(event.loss).toFixed(1)}`).join(' ')
  const smoothPoints = smoothed.map((event) => `${x(event.step).toFixed(1)},${y(event.loss).toFixed(1)}`).join(' ')

  return (
    <div className="loss-chart-wrap">
      <svg className="loss-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${label} and validation loss by optimizer update`}>
        <line x1={left} y1={top} x2={left} y2={height - bottom} className="chart-axis" />
        <line x1={left} y1={height - bottom} x2={width - right} y2={height - bottom} className="chart-axis" />
        <text x="8" y={top + 5} className="chart-label">MSE</text>
        <text x={left} y={height - 8} className="chart-label">step {minStep.toLocaleString()}</text>
        <text x={width - right} y={height - 8} textAnchor="end" className="chart-label">step {maxStep.toLocaleString()}</text>
        {rawPoints && <polyline points={rawPoints} className="chart-raw" />}
        {smoothPoints && <polyline points={smoothPoints} className="chart-smooth" />}
        {validation.map((event) => <circle key={event.event_id} cx={x(event.global_step)} cy={y(event.loss)} r="4.5" className="chart-validation" />)}
        {emaValidation.map((event) => <rect key={`ema-${event.event_id}`} x={x(event.global_step) - 4} y={y(event.ema_loss as number) - 4} width="8" height="8" className="chart-validation-ema" />)}
      </svg>
      <div className="chart-legend"><span><i className="legend-raw" />Up to {windowUpdates}-update mean loss</span><span><i className="legend-smooth" />Smoothed monitor curve</span><span><i className="legend-validation" />Raw validation loss</span><span><i className="legend-validation-ema" />EMA validation loss</span></div>
      <div className="chart-range"><span>{minLoss.toFixed(4)}</span><span>{maxLoss.toFixed(4)}</span></div>
    </div>
  )
}

function formatDuration(seconds?: number) {
  if (seconds === undefined || !Number.isFinite(seconds)) return '—'
  if (seconds < 60) return `${seconds.toFixed(0)} sec`
  return `${Math.floor(seconds / 60)} min ${Math.floor(seconds % 60)} sec`
}

export default function TrainingMonitor() {
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [modelFamily, setModelFamily] = useState<ModelFamily>('ddpm')
  const [status, setStatus] = useState<RunStatus | null>(null)
  const [events, setEvents] = useState<MetricEvent[]>([])
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [resolvedConfig, setResolvedConfig] = useState<ResolvedConfig | null>(null)
  const [error, setError] = useState<string | null>(null)
  const cursor = useRef(0)
  const selectedRun = resolveTrainingRun(runs, modelFamily)
  const selectedRunId = selectedRun?.run_id ?? ''
  const modelProfile = getModelProfile(modelFamily)

  useEffect(() => {
    let stopped = false
    let busy = false
    const refresh = async () => {
      if (busy) return
      busy = true
      try {
        const result = await getJson<{ items: RunSummary[] }>('/api/runs')
        if (!stopped) {
          setRuns(result.items)
          setError(null)
        }
      } catch (reason) {
        if (!stopped) setError(reason instanceof Error ? reason.message : 'Could not load runs')
      } finally {
        busy = false
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), 2500)
    return () => { stopped = true; window.clearInterval(timer) }
  }, [])

  useEffect(() => {
    setEvents([])
    setStatus(null)
    setCheckpoints([])
    cursor.current = 0
    if (!selectedRunId) return
    let stopped = false
    let busy = false
    const refresh = async () => {
      if (busy) return
      busy = true
      try {
        const [detail, metricResult, checkpointResult] = await Promise.all([
          getJson<{ status: RunStatus; run: { config?: ResolvedConfig } }>(`/api/runs/${selectedRunId}`),
          getJson<{ items: MetricEvent[] }>(`/api/runs/${selectedRunId}/metrics?after_event=${cursor.current}`),
          getJson<{ items: Checkpoint[] }>(`/api/runs/${selectedRunId}/checkpoints`),
        ])
        if (stopped) return
        setStatus(detail.status)
        setResolvedConfig(detail.run.config ?? null)
        if (metricResult.items.length) {
          cursor.current = Math.max(cursor.current, ...metricResult.items.map((item) => item.event_id))
          setEvents((current) => [...current, ...metricResult.items])
        }
        setCheckpoints(checkpointResult.items)
        setError(null)
      } catch (reason) {
        if (!stopped) setError(reason instanceof Error ? reason.message : 'Could not load run details')
      } finally {
        busy = false
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), 2000)
    return () => { stopped = true; window.clearInterval(timer) }
  }, [selectedRunId])

  const progress = useMemo(() => {
    if (!status?.planned_steps) return 0
    return Math.min(100, (100 * (status.global_step ?? 0)) / status.planned_steps)
  }, [status])
  const selectedCheckpoint = checkpoints.find((checkpoint) => checkpoint.name === modelProfile.checkpoint)
  const isVpSde = resolvedConfig?.process === 'vp_sde'
  const isDirectScore = isVpSde && resolvedConfig?.model?.prediction === 'score'
  const lossTitle = isVpSde ? isDirectScore ? 'Weighted direct-score DSM' : 'Weighted DSM / epsilon MSE' : 'Noise-prediction loss'
  const monitorWindow = resolvedConfig?.training?.monitor_interval_updates ?? 100
  const validationInterval = resolvedConfig?.training?.validation_every_updates
  const validationCadence = validationInterval
    ? `Validation uses a fixed corruption set every ${validationInterval.toLocaleString()} updates.`
    : 'Validation cadence is recorded in the run config.'
  const lossDescription = isVpSde
    ? isDirectScore
      ? `The U-Net predicts the score directly and is trained against -epsilon/sigma(t), with sigma(t)^2 weighting. This is algebraically equal to epsilon MSE, but the model output is a score vector. ${validationCadence}`
      : `The U-Net predicts sampled epsilon. Its epsilon MSE is exactly the sigma-squared-weighted conditional DSM objective; sampling converts it by score = -epsilon/sigma(t). ${validationCadence}`
    : `Training minimizes mean squared error between the sampled noise and the U-Net prediction. ${validationCadence}`

  return (
    <section className="training-view" aria-label="Training monitor">
      <div className="monitor-heading">
        <div>
          <div className="eyebrow">STAGE 4 · LOCAL RUN MONITOR</div>
          <h1>Watch the model <em>learn noise.</em></h1>
          <p>Start training in PowerShell; this page reads the run artifacts and refreshes while it runs.</p>
        </div>
        <label className="run-picker">Model<select value={modelFamily} onChange={(event) => setModelFamily(event.target.value as ModelFamily)}>
          {MODEL_PROFILES.map((profile) => (
            <option key={profile.family} value={profile.family} disabled={!resolveTrainingRun(runs, profile.family)}>
              {profile.label}
            </option>
          ))}
        </select></label>
      </div>

      {error && <div className="error-banner" role="alert">{error}</div>}
      {!selectedRun ? (
        <div className="monitor-empty panel">
          <h2>{runs.length ? `${modelProfile.label} is not available yet` : 'No training run yet'}</h2>
          <p>{runs.length
            ? 'This model family has no active run or completed training run to monitor yet.'
            : 'Start a DDPM or direct-score VP-SDE training run from the repository root. The test split is not loaded by training.'}</p>
          {!runs.length && <code>uv run --locked python -m ddpm_eurosat.train --config configs/eurosat_original_ddpm.toml</code>}
        </div>
      ) : (
        <>
          <section className="panel run-summary">
            <div className="run-state-line"><div><span className="section-index">MODEL</span><h2>{modelProfile.alias}</h2><small>{selectedRun.run_id}</small></div><span className={`state-badge state-${status?.state ?? selectedRun.state ?? 'unknown'}`}>{status?.state ?? selectedRun.state ?? 'loading'}</span></div>
            <div className="progress-meta"><span>Epoch {status?.epoch ?? selectedRun.epoch} / {status?.total_epochs ?? selectedRun.total_epochs}</span><span>Update {(status?.global_step ?? selectedRun.global_step).toLocaleString()} / {(status?.planned_steps ?? 0).toLocaleString()}</span></div>
            <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
            <div className="run-stats">
              <div><small>DEVICE</small><b>{status?.device ?? '—'}</b></div>
              <div><small>ELAPSED</small><b>{formatDuration(status?.elapsed_seconds)}</b></div>
              <div><small>UPDATES / SEC</small><b>{status?.steps_per_second?.toFixed(2) ?? '—'}</b></div>
              <div><small>PEAK GPU MEMORY</small><b>{status?.peak_gpu_memory_mb ? `${status.peak_gpu_memory_mb.toFixed(0)} MB` : '—'}</b></div>
            </div>
            {resolvedConfig && <div className="resolved-config"><span>PROCESS <b>{isVpSde ? `VP-SDE · ${resolvedConfig.model?.prediction ?? 'epsilon'} output` : 'DDPM'}</b></span><span>SEED <b>{resolvedConfig.seed ?? '—'}</b></span><span>OPTIMIZER <b>{resolvedConfig.training?.optimizer ?? '—'}</b></span><span>LEARNING RATE <b>{resolvedConfig.training?.learning_rate ?? '—'}</b></span><span>EFFECTIVE BATCH <b>{resolvedConfig.training?.effective_batch_size ?? resolvedConfig.training?.batch_size ?? '—'}</b></span>{isVpSde ? <><span>BETA MIN / MAX <b>{resolvedConfig.vp_sde?.beta_min ?? '—'} / {resolvedConfig.vp_sde?.beta_max ?? '—'}</b></span><span>MIN TRAINING t <b>{resolvedConfig.vp_sde?.training_time_min ?? '—'}</b></span></> : <span>DIFFUSION STEPS <b>{resolvedConfig.diffusion?.timesteps ?? '—'}</b></span>}</div>}
            {status?.error && <div className="error-banner">{status.error}</div>}
          </section>

          <section className="panel chart-panel">
            <div className="panel-heading"><div><span className="section-index">01</span><h2>{lossTitle}</h2></div><span className="split-label">TRAIN UPDATES + VALIDATION CHECKPOINTS</span></div>
            <p className="panel-copy">{lossDescription} The curve shows a mean over up to {monitorWindow} updates.</p>
            <LossChart events={events} windowUpdates={monitorWindow} label={lossTitle} />
          <div className="learning-note"><b>How to read this:</b> A downward loss trend means the model is improving at its {isDirectScore ? 'score-estimation' : 'noise-prediction'} task. It does not by itself tell us whether generated images look good.</div>
          </section>

          <section className="panel checkpoint-panel">
            <div className="panel-heading"><div><span className="section-index">02</span><h2>Checkpoints</h2></div><span className="split-label">LOCAL .PT FILES</span></div>
            {selectedCheckpoint ? <div className="checkpoint-list"><div className="checkpoint-row" key={selectedCheckpoint.name}><div><b>{modelProfile.alias}</b><small>{selectedCheckpoint.kind} · source file {selectedCheckpoint.name} · step {selectedCheckpoint.global_step.toLocaleString()} · epoch {selectedCheckpoint.epoch}</small></div><span>{selectedCheckpoint.validation_loss === null ? '—' : `val ${selectedCheckpoint.validation_loss.toFixed(5)}`}</span><span>{(selectedCheckpoint.bytes / (1024 * 1024)).toFixed(1)} MB</span></div></div> : <div className="chart-empty">The validation-selected checkpoint will appear here after validation saves it.</div>}
            <p className="panel-copy checkpoint-path">Saved under <code>runs/{selectedRunId}/checkpoints/</code>. The underlying checkpoint remains available for resuming or inspecting the run.</p>
          </section>
        </>
      )}
    </section>
  )
}
