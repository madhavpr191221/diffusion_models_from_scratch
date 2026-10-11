import { useEffect, useMemo, useRef, useState } from 'react'

type RunSummary = {
  run_id: string
  state: string
  mode: string
  global_step: number
  epoch: number
  total_epochs: number
  created_at: string | null
}
type MetricEvent = {
  event_id: number
  kind: 'train' | 'validation'
  epoch: number
  global_step: number
  loss: number
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
  seed?: number
  training?: { batch_size?: number; epochs?: number; learning_rate?: number; optimizer?: string }
  diffusion?: { timesteps?: number }
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { cache: 'no-store' })
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`)
  return (await response.json()) as T
}

function LossChart({ events }: { events: MetricEvent[] }) {
  const train = events.filter((event) => event.kind === 'train')
  const validation = events.filter((event) => event.kind === 'validation')
  const smoothed = train.map((event, index) => {
    const window = train.slice(Math.max(0, index - 99), index + 1)
    return { step: event.global_step, loss: window.reduce((sum, item) => sum + item.loss, 0) / window.length }
  })
  const values = [...train.map((event) => event.loss), ...validation.map((event) => event.loss)]
  if (!values.length) return <div className="chart-empty">Loss values will appear after the first update.</div>

  const width = 900
  const height = 260
  const left = 58
  const right = 18
  const top = 20
  const bottom = 34
  const minStep = Math.min(...events.map((event) => event.global_step))
  const maxStep = Math.max(1, ...events.map((event) => event.global_step))
  const minLoss = Math.min(...values)
  const maxLoss = Math.max(...values)
  const lossSpan = Math.max(maxLoss - minLoss, Math.abs(maxLoss) * 0.08, 1e-5)
  const x = (step: number) => left + ((step - minStep) / Math.max(maxStep - minStep, 1)) * (width - left - right)
  const y = (loss: number) => top + (1 - (loss - minLoss) / lossSpan) * (height - top - bottom)
  const rawPoints = train.map((event) => `${x(event.global_step).toFixed(1)},${y(event.loss).toFixed(1)}`).join(' ')
  const smoothPoints = smoothed.map((event) => `${x(event.step).toFixed(1)},${y(event.loss).toFixed(1)}`).join(' ')

  return (
    <div className="loss-chart-wrap">
      <svg className="loss-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Training noise-prediction loss and validation loss by optimizer update">
        <line x1={left} y1={top} x2={left} y2={height - bottom} className="chart-axis" />
        <line x1={left} y1={height - bottom} x2={width - right} y2={height - bottom} className="chart-axis" />
        <text x="8" y={top + 5} className="chart-label">MSE</text>
        <text x={left} y={height - 8} className="chart-label">step {minStep.toLocaleString()}</text>
        <text x={width - right} y={height - 8} textAnchor="end" className="chart-label">step {maxStep.toLocaleString()}</text>
        {rawPoints && <polyline points={rawPoints} className="chart-raw" />}
        {smoothPoints && <polyline points={smoothPoints} className="chart-smooth" />}
        {validation.map((event) => <circle key={event.event_id} cx={x(event.global_step)} cy={y(event.loss)} r="4.5" className="chart-validation" />)}
      </svg>
      <div className="chart-legend"><span><i className="legend-raw" />Per-update training loss</span><span><i className="legend-smooth" />100-update moving average</span><span><i className="legend-validation" />Validation loss by epoch</span></div>
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
  const [selectedRunId, setSelectedRunId] = useState('')
  const [status, setStatus] = useState<RunStatus | null>(null)
  const [events, setEvents] = useState<MetricEvent[]>([])
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([])
  const [resolvedConfig, setResolvedConfig] = useState<ResolvedConfig | null>(null)
  const [error, setError] = useState<string | null>(null)
  const cursor = useRef(0)

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
          setSelectedRunId((current) => current && result.items.some((run) => run.run_id === current) ? current : result.items[0]?.run_id ?? '')
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
  const selected = runs.find((run) => run.run_id === selectedRunId)

  return (
    <section className="training-view" aria-label="Training monitor">
      <div className="monitor-heading">
        <div>
          <div className="eyebrow">STAGE 4 · LOCAL RUN MONITOR</div>
          <h1>Watch the model <em>learn noise.</em></h1>
          <p>Start training in PowerShell; this page reads the run artifacts and refreshes while it runs.</p>
        </div>
        <label className="run-picker">Run<select value={selectedRunId} onChange={(event) => setSelectedRunId(event.target.value)} disabled={!runs.length}>
          {!runs.length && <option value="">No runs found</option>}
          {runs.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id} · {run.state}</option>)}
        </select></label>
      </div>

      {error && <div className="error-banner" role="alert">{error}</div>}
      {!runs.length ? (
        <div className="monitor-empty panel">
          <h2>No training run yet</h2>
          <p>From the repository root, first run the small GPU pilot:</p>
          <code>uv run --locked python -m ddpm_eurosat.train --pilot</code>
          <p>Then run the baseline:</p>
          <code>uv run --locked python -m ddpm_eurosat.train</code>
          <p>The test split is not loaded by either training command.</p>
        </div>
      ) : (
        <>
          <section className="panel run-summary">
            <div className="run-state-line"><div><span className="section-index">RUN</span><h2>{selected?.run_id}</h2></div><span className={`state-badge state-${status?.state ?? selected?.state ?? 'unknown'}`}>{status?.state ?? selected?.state ?? 'loading'}</span></div>
            <div className="progress-meta"><span>Epoch {status?.epoch ?? 0} / {status?.total_epochs ?? selected?.total_epochs ?? 0}</span><span>Update {(status?.global_step ?? 0).toLocaleString()} / {(status?.planned_steps ?? 0).toLocaleString()}</span></div>
            <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
            <div className="run-stats">
              <div><small>DEVICE</small><b>{status?.device ?? '—'}</b></div>
              <div><small>ELAPSED</small><b>{formatDuration(status?.elapsed_seconds)}</b></div>
              <div><small>UPDATES / SEC</small><b>{status?.steps_per_second?.toFixed(2) ?? '—'}</b></div>
              <div><small>PEAK GPU MEMORY</small><b>{status?.peak_gpu_memory_mb ? `${status.peak_gpu_memory_mb.toFixed(0)} MB` : '—'}</b></div>
            </div>
            {resolvedConfig && <div className="resolved-config"><span>SEED <b>{resolvedConfig.seed ?? '—'}</b></span><span>OPTIMIZER <b>{resolvedConfig.training?.optimizer ?? '—'}</b></span><span>LEARNING RATE <b>{resolvedConfig.training?.learning_rate ?? '—'}</b></span><span>BATCH SIZE <b>{resolvedConfig.training?.batch_size ?? '—'}</b></span><span>DIFFUSION STEPS <b>{resolvedConfig.diffusion?.timesteps ?? '—'}</b></span></div>}
            {status?.error && <div className="error-banner">{status.error}</div>}
          </section>

          <section className="panel chart-panel">
            <div className="panel-heading"><div><span className="section-index">01</span><h2>Noise-prediction loss</h2></div><span className="split-label">TRAIN UPDATES + VALIDATION EPOCHS</span></div>
            <p className="panel-copy">Training minimizes mean squared error between the sampled noise and the U-Net prediction. The moving average makes the noisy per-batch curve easier to read.</p>
            <LossChart events={events} />
            <div className="learning-note"><b>How to read this:</b> A downward loss trend means the model is improving at this noise-prediction task. It does not by itself tell us whether generated images look good.</div>
          </section>

          <section className="panel checkpoint-panel">
            <div className="panel-heading"><div><span className="section-index">02</span><h2>Checkpoints</h2></div><span className="split-label">LOCAL .PT FILES</span></div>
            {checkpoints.length ? <div className="checkpoint-list">{checkpoints.map((checkpoint) => <div className="checkpoint-row" key={checkpoint.name}><div><b>{checkpoint.name}</b><small>{checkpoint.kind} · step {checkpoint.global_step.toLocaleString()} · epoch {checkpoint.epoch}</small></div><span>{checkpoint.validation_loss === null ? '—' : `val ${checkpoint.validation_loss.toFixed(5)}`}</span><span>{(checkpoint.bytes / (1024 * 1024)).toFixed(1)} MB</span></div>)}</div> : <div className="chart-empty">Checkpoint details will appear as training progresses.</div>}
            <p className="panel-copy checkpoint-path">Saved under <code>runs/{selectedRunId}/checkpoints/</code>. The latest checkpoint can resume from its saved update and batch position.</p>
          </section>
        </>
      )}
    </section>
  )
}
