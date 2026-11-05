import { useEffect, useMemo, useRef, useState } from 'react'

type FlowRun = {
  run_id: string
  state: string
  epoch: number
  total_epochs: number
  global_step: number
  planned_steps: number
  device?: string
  elapsed_seconds?: number
  steps_per_second?: number
  peak_gpu_memory_mb?: number
  train_loss?: number
  validation_loss?: number
  ema_validation_loss?: number
  error?: string
  available_checkpoints: string[]
}
type Metric = {
  event_id: number
  kind: 'train' | 'validation'
  global_step: number
  epoch: number
  loss: number
  ema_loss?: number
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { cache: 'no-store', signal })
  if (!response.ok) throw new Error(`Could not load ${url} (${response.status})`)
  return (await response.json()) as T
}

function duration(seconds?: number) {
  if (seconds === undefined || !Number.isFinite(seconds)) return '—'
  if (seconds < 60) return `${seconds.toFixed(0)} sec`
  return `${Math.floor(seconds / 60)} min ${Math.floor(seconds % 60)} sec`
}

function FlowLossChart({ events }: { events: Metric[] }) {
  const train = events.filter((event) => event.kind === 'train')
  const validation = events.filter((event) => event.kind === 'validation')
  const ema = validation.filter((event) => event.ema_loss !== undefined)
  const values = [...train.map((item) => item.loss), ...validation.map((item) => item.loss), ...ema.map((item) => item.ema_loss as number)]
  if (!values.length) return <div className="chart-empty">Training loss will appear after the first update.</div>
  const width = 900; const height = 230; const left = 54; const right = 16; const top = 16; const bottom = 30
  const minStep = Math.min(...events.map((item) => item.global_step))
  const maxStep = Math.max(1, ...events.map((item) => item.global_step))
  const minLoss = Math.min(...values); const maxLoss = Math.max(...values)
  const span = Math.max(maxLoss - minLoss, Math.abs(maxLoss) * 0.08, 1e-5)
  const x = (step: number) => left + ((step - minStep) / Math.max(1, maxStep - minStep)) * (width - left - right)
  const y = (loss: number) => top + (1 - (loss - minLoss) / span) * (height - top - bottom)
  const smooth = train.map((event, index) => {
    const window = train.slice(Math.max(0, index - 99), index + 1)
    return { step: event.global_step, loss: window.reduce((sum, item) => sum + item.loss, 0) / window.length }
  })
  const points = (items: { global_step?: number; step?: number; loss: number }[]) => items
    .map((item) => `${x(item.global_step ?? item.step ?? 0).toFixed(1)},${y(item.loss).toFixed(1)}`).join(' ')
  return <div className="loss-chart-wrap">
    <svg className="loss-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Flow Matching training and validation loss">
      <line x1={left} y1={top} x2={left} y2={height - bottom} className="chart-axis" />
      <line x1={left} y1={height - bottom} x2={width - right} y2={height - bottom} className="chart-axis" />
      <text x="8" y={top + 5} className="chart-label">MSE</text>
      <text x={left} y={height - 7} className="chart-label">update {minStep.toLocaleString()}</text>
      <text x={width - right} y={height - 7} textAnchor="end" className="chart-label">update {maxStep.toLocaleString()}</text>
      {train.length > 0 && <polyline points={points(train)} className="chart-raw" />}
      {smooth.length > 0 && <polyline points={points(smooth)} className="chart-smooth" />}
      {validation.map((event) => <circle key={event.event_id} cx={x(event.global_step)} cy={y(event.loss)} r="4" className="chart-validation" />)}
      {ema.map((event) => <rect key={`ema-${event.event_id}`} x={x(event.global_step) - 3} y={y(event.ema_loss as number) - 3} width="6" height="6" className="chart-validation-ema" />)}
    </svg>
    <div className="chart-legend"><span><i className="legend-raw" />Per-update training loss</span><span><i className="legend-smooth" />100-update mean</span><span><i className="legend-validation" />Raw validation</span><span><i className="legend-validation-ema" />EMA validation</span></div>
  </div>
}

export default function FlowTrainingMonitor() {
  const [runs, setRuns] = useState<FlowRun[]>([])
  const [selectedRunId, setSelectedRunId] = useState('')
  const [status, setStatus] = useState<FlowRun | null>(null)
  const [events, setEvents] = useState<Metric[]>([])
  const [error, setError] = useState<string | null>(null)
  const cursor = useRef(0)
  const selectedRun = runs.find((run) => run.run_id === selectedRunId)
    ?? runs.find((run) => run.state === 'running') ?? runs[0]

  useEffect(() => {
    let stopped = false
    let timer = 0
    const refresh = async () => {
      try {
        const result = await getJson<{ items: FlowRun[] }>('/api/flows/runs')
        if (stopped) return
        setRuns(result.items)
        setSelectedRunId(result.items.find((run) => run.state === 'running')?.run_id || result.items[0]?.run_id || '')
        setError(null)
      } catch (reason) {
        if (!stopped) setError(reason instanceof Error ? reason.message : 'Could not load Flow Matching runs')
      } finally {
        if (!stopped) timer = window.setTimeout(refresh, 2500)
      }
    }
    void refresh()
    return () => { stopped = true; window.clearTimeout(timer) }
  }, [])

  useEffect(() => {
    setEvents([]); setStatus(null); cursor.current = 0
    if (!selectedRunId) return
    let stopped = false
    let timer = 0
    const refresh = async () => {
      try {
        const [detail, metricResult] = await Promise.all([
          getJson<{ status: FlowRun }>(`/api/runs/${selectedRunId}`),
          getJson<{ items: Metric[] }>(`/api/runs/${selectedRunId}/metrics?after_event=${cursor.current}`),
        ])
        if (stopped) return
        setStatus(detail.status)
        if (metricResult.items.length) {
          cursor.current = Math.max(cursor.current, ...metricResult.items.map((item) => item.event_id))
          setEvents((current) => [...current, ...metricResult.items])
        }
        setError(null)
      } catch (reason) {
        if (!stopped) setError(reason instanceof Error ? reason.message : 'Could not load Flow Matching progress')
      } finally {
        if (!stopped) timer = window.setTimeout(refresh, 2000)
      }
    }
    void refresh()
    return () => { stopped = true; window.clearTimeout(timer) }
  }, [selectedRunId])

  const progress = useMemo(() => status?.planned_steps ? Math.min(100, 100 * (status.global_step / status.planned_steps)) : 0, [status])
  return <section className="panel flow-training-monitor">
    <div className="panel-heading"><div><span className="section-index">TRAINING</span><h2>Flow Matching run monitor</h2></div>
      {selectedRun && <span className={`state-badge state-${status?.state ?? selectedRun.state}`}>{status?.state ?? selectedRun.state}</span>}
    </div>
    {error && <div className="flow-error" role="alert">{error}</div>}
    {!selectedRun ? <div className="flow-monitor-empty"><b>No Flow Matching training run yet.</b><span>The trainer command is shown after the run configuration is frozen.</span></div> : <>
      <div className="flow-monitor-run"><b>{selectedRun.run_id}</b><span>Unconditional conditional-flow-matching objective · train split only</span></div>
      <div className="progress-meta"><span>Epoch {status?.epoch ?? selectedRun.epoch} / {status?.total_epochs ?? selectedRun.total_epochs}</span><span>Update {(status?.global_step ?? selectedRun.global_step).toLocaleString()} / {(status?.planned_steps ?? selectedRun.planned_steps).toLocaleString()}</span></div>
      <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
      <div className="run-stats">
        <div><small>DEVICE</small><b>{status?.device ?? selectedRun.device ?? '—'}</b></div>
        <div><small>ELAPSED</small><b>{duration(status?.elapsed_seconds ?? selectedRun.elapsed_seconds)}</b></div>
        <div><small>UPDATES / SEC</small><b>{status?.steps_per_second?.toFixed(2) ?? selectedRun.steps_per_second?.toFixed(2) ?? '—'}</b></div>
        <div><small>PEAK GPU MEMORY</small><b>{status?.peak_gpu_memory_mb ? `${status.peak_gpu_memory_mb.toFixed(0)} MB` : selectedRun.peak_gpu_memory_mb ? `${selectedRun.peak_gpu_memory_mb.toFixed(0)} MB` : '—'}</b></div>
      </div>
      <div className="flow-latest-losses"><span>Latest batch loss <b>{status?.train_loss?.toFixed(5) ?? selectedRun.train_loss?.toFixed(5) ?? '—'}</b></span><span>Validation <b>{status?.validation_loss?.toFixed(5) ?? selectedRun.validation_loss?.toFixed(5) ?? '—'}</b></span><span>EMA validation <b>{status?.ema_validation_loss?.toFixed(5) ?? selectedRun.ema_validation_loss?.toFixed(5) ?? '—'}</b></span></div>
      <FlowLossChart events={events} />
      <div className="learning-note"><b>Interpretation:</b> Flow Matching loss measures velocity-target regression. A lower loss is not by itself evidence that generated patches resemble EuroSAT.</div>
      {status?.error && <div className="flow-error" role="alert">{status.error}</div>}
    </>}
  </section>
}
