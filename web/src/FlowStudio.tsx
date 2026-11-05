import { useEffect, useState } from 'react'
import FlowTrainingMonitor from './FlowTrainingMonitor'

const FLOW_MATCHING_CHECKPOINT = 'best-eurosat-flow-matching.pt'
const FLOW_MATCHING_EMA_CHECKPOINT = 'best-eurosat-flow-matching-ema.pt'

type FlowRun = {
  run_id: string
  state: string
  epoch: number
  total_epochs: number
  global_step: number
  can_generate: boolean
  available_checkpoints?: string[]
}
type FlowStudy = {
  run_id: string
  study_id: string
  generation_state: 'queued' | 'running' | 'completed' | 'failed' | 'interrupted'
  generation_completed: number
  generation_total: number
  judgement_state: 'not_started' | 'queued' | 'running' | 'completed' | 'failed'
  judgement_completed: number
  judgement_total: number
  judgement_completed_batches?: number
  judgement_total_batches?: number
  num_samples: number
  seed: number
  solver: 'euler' | 'heun'
  steps: number
  can_judge: boolean
  nfe_per_sample?: number
  batch_model_calls?: number
  checkpoint?: string
  latest_frame_index?: number
  sample_statistics?: {
    value_range: [number, number]
    fraction_outside_training_range: number
    mean_per_image_pixel_std: number
    pairwise_pixel_rmse: number | null
  }
  validation_reference_selection?: string
  error?: string
  judgement_error?: string
}
type Judgment = {
  patch_id: string
  sample_index: number
  resemblance_score: number
  resemblance: 'yes' | 'partial' | 'no'
  gap_categories: string[]
  visual_evidence: string
}
type Report = {
  judged_images?: number
  mean_resemblance_score?: number | null
  resemblance_yes?: number
  resemblance_partial?: number
  resemblance_no?: number
  gap_counts?: Record<string, number>
  interpretation_note?: string
}
type StudyResult = { items: Judgment[]; report: Report }
type Frame = { url: string; time: number }

async function responseError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string }
    return body.detail ?? `Request failed (${response.status})`
  } catch {
    return `Request failed (${response.status})`
  }
}

function FlowStudio() {
  const [runs, setRuns] = useState<FlowRun[]>([])
  const [selectedRunId, setSelectedRunId] = useState('')
  const [seedText, setSeedText] = useState('42')
  const [solver, setSolver] = useState<'euler' | 'heun'>('heun')
  const [steps, setSteps] = useState(100)
  const [checkpoint, setCheckpoint] = useState(FLOW_MATCHING_EMA_CHECKPOINT)
  const [study, setStudy] = useState<FlowStudy | null>(null)
  const [studyResult, setStudyResult] = useState<StudyResult | null>(null)
  const [frameIndex, setFrameIndex] = useState(0)
  const [frame, setFrame] = useState<Frame | null>(null)
  const [playing, setPlaying] = useState(false)
  const [loadingRuns, setLoadingRuns] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const completedRuns = runs.filter((run) => run.can_generate)
  const selectedRun = completedRuns.find((run) => run.run_id === selectedRunId) ?? completedRuns[0]
  const sampleCount = study?.num_samples ?? 0
  const frameCount = (study?.latest_frame_index ?? -1) + 1

  useEffect(() => {
    const controller = new AbortController()
    let timer = 0
    const refresh = async () => {
      try {
        const response = await fetch('/api/flows/runs', { signal: controller.signal, cache: 'no-store' })
        if (!response.ok) throw new Error(await responseError(response))
        const { items } = await response.json() as { items: FlowRun[] }
        if (controller.signal.aborted) return
        setRuns(items)
        setSelectedRunId((current) => current || items.find((run) => run.can_generate)?.run_id || '')
        setLoadingRuns(false)
      } catch (reason) {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : 'Could not load Flow Matching runs')
          setLoadingRuns(false)
        }
      } finally {
        if (!controller.signal.aborted) timer = window.setTimeout(refresh, 2500)
      }
    }
    void refresh()
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [])

  useEffect(() => {
    if (!selectedRun?.available_checkpoints?.length || selectedRun.available_checkpoints.includes(checkpoint)) return
    setCheckpoint(selectedRun.available_checkpoints.includes(FLOW_MATCHING_EMA_CHECKPOINT)
      ? FLOW_MATCHING_EMA_CHECKPOINT : FLOW_MATCHING_CHECKPOINT)
  }, [selectedRun?.run_id, selectedRun?.available_checkpoints, checkpoint])

  useEffect(() => {
    if (!study || (study.generation_state !== 'queued' && study.generation_state !== 'running'
      && study.judgement_state !== 'queued' && study.judgement_state !== 'running')) return
    const controller = new AbortController()
    let timer = 0
    const poll = async () => {
      try {
        const response = await fetch(`/api/flows/runs/${study.run_id}/studies/${study.study_id}`, { signal: controller.signal })
        if (!response.ok) throw new Error(await responseError(response))
        const updated = (await response.json()) as FlowStudy
        if (controller.signal.aborted) return
        setStudy(updated)
        if (updated.generation_state === 'failed' || updated.generation_state === 'interrupted') {
          setError(updated.error ?? 'Flow generation failed')
        } else if (updated.judgement_state === 'failed') {
          setError(updated.judgement_error ?? 'VLM review failed')
        }
        if (updated.generation_state === 'queued' || updated.generation_state === 'running'
          || updated.judgement_state === 'queued' || updated.judgement_state === 'running') {
          timer = window.setTimeout(poll, 500)
        }
      } catch (reason) {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Could not read Flow Matching progress')
      }
    }
    void poll()
    return () => {
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [study?.study_id, study?.run_id, study?.generation_state, study?.judgement_state])

  useEffect(() => {
    if (!study || study.generation_state !== 'completed') return
    const controller = new AbortController()
    let objectUrl: string | null = null
    setFrame(null)
    fetch(`/api/flows/runs/${study.run_id}/studies/${study.study_id}/frames/${frameIndex}`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response))
        const time = Number(response.headers.get('X-Flow-Time'))
        objectUrl = URL.createObjectURL(await response.blob())
        setFrame({ url: objectUrl, time })
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted && frameCount > 0) setError(reason instanceof Error ? reason.message : 'Could not load trajectory frame')
      })
    return () => {
      controller.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [study?.study_id, study?.run_id, study?.generation_state, frameIndex, frameCount])

  useEffect(() => {
    if (!study || study.judgement_state !== 'completed') return
    const controller = new AbortController()
    fetch(`/api/flows/runs/${study.run_id}/studies/${study.study_id}/judgements`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response))
        return (await response.json()) as StudyResult
      })
      .then((result) => { if (!controller.signal.aborted) setStudyResult(result) })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : 'Could not load the VLM report')
      })
    return () => controller.abort()
  }, [study?.study_id, study?.run_id, study?.judgement_state])

  useEffect(() => {
    if (!playing || frameCount < 2) return
    if (frameIndex >= frameCount - 1) {
      setPlaying(false)
      return
    }
    const timer = window.setTimeout(() => setFrameIndex((index) => index + 1), 70)
    return () => window.clearTimeout(timer)
  }, [playing, frameIndex, frameCount])

  async function createStudy(count: 1 | 100) {
    if (!selectedRun) return
    const seed = Number(seedText)
    if (!Number.isInteger(seed) || seed < 0 || seed > 4_294_967_295) {
      setError('Seed must be an integer from 0 to 4,294,967,295.')
      return
    }
    setBusy(true)
    setError(null)
    setStudy(null)
    setStudyResult(null)
    setFrame(null)
    setFrameIndex(0)
    setPlaying(false)
    try {
      const response = await fetch(`/api/flows/runs/${selectedRun.run_id}/studies`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_samples: count, seed, solver, steps, checkpoint }),
      })
      if (!response.ok) throw new Error(await responseError(response))
      const created = (await response.json()) as { run_id: string; study_id: string }
      setStudy({
        run_id: created.run_id, study_id: created.study_id,
        generation_state: 'queued', generation_completed: 0, generation_total: count,
        judgement_state: 'not_started', judgement_completed: 0, judgement_total: 100,
        num_samples: count, seed, solver, steps, checkpoint, can_judge: false,
      })
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not start Flow Matching generation')
    } finally {
      setBusy(false)
    }
  }

  async function startJudgement() {
    if (!study || !study.can_judge) return
    setError(null)
    try {
      const response = await fetch(`/api/flows/runs/${study.run_id}/studies/${study.study_id}/judgements`, { method: 'POST' })
      if (!response.ok) throw new Error(await responseError(response))
      setStudy((current) => current ? { ...current, judgement_state: 'queued' } : current)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not start VLM review')
    }
  }

  const isGenerating = busy || study?.generation_state === 'queued' || study?.generation_state === 'running'
  const isJudging = study?.judgement_state === 'queued' || study?.judgement_state === 'running'

  return (
    <section className="flow-view">
      <header className="flow-heading">
        <div className="eyebrow">CONTINUOUS FLOW MATCHING · EUROSAT RGB</div>
        <h1>Watch noise <em>become a patch.</em></h1>
        <p>Flow time runs from Gaussian noise at <code>t = 0</code> to a new 64 × 64 EuroSAT-style patch at <code>t = 1</code>.</p>
      </header>

      <FlowTrainingMonitor />

      <section className="panel flow-controls">
        <div className="panel-heading">
          <div><span className="section-index">01</span><h2>Set up generation</h2></div>
          <span className="local-tag">FLOW MATCHING · VELOCITY MODEL</span>
        </div>
        {loadingRuns ? <p className="panel-copy">Loading trained Flow Matching models…</p> : completedRuns.length === 0 ? (
          <div className="learning-note flow-empty">
            <b>No trained Flow Matching checkpoint is available yet.</b>
            <span>The Flow Matching trainer uses the EuroSAT train split. First measure one real-image batch without updating weights:</span>
            <code>uv run --locked python -m ddpm_eurosat.flow_matching_train --preflight</code>
            <span>Review its VRAM and runtime estimate before approving the 50-epoch run.</span>
          </div>
        ) : (
          <div className="flow-form">
            <label>Training run
              <select value={selectedRun?.run_id ?? ''} disabled={isGenerating} onChange={(event) => setSelectedRunId(event.target.value)}>
                {completedRuns.map((run) => <option key={run.run_id} value={run.run_id}>{run.run_id} · epoch {run.epoch}/{run.total_epochs}</option>)}
              </select>
            </label>
            <label>Solver
              <select value={solver} disabled={isGenerating} onChange={(event) => setSolver(event.target.value as 'euler' | 'heun')}>
                <option value="heun">Heun (2 evaluations per step)</option>
                <option value="euler">Euler (1 evaluation per step)</option>
              </select>
            </label>
            <label>Best validation checkpoint
              <select value={checkpoint} disabled={isGenerating || isJudging} onChange={(event) => setCheckpoint(event.target.value)}>
                {selectedRun?.available_checkpoints?.includes(FLOW_MATCHING_EMA_CHECKPOINT) && <option value={FLOW_MATCHING_EMA_CHECKPOINT}>Best EMA · {FLOW_MATCHING_EMA_CHECKPOINT}</option>}
                {selectedRun?.available_checkpoints?.includes(FLOW_MATCHING_CHECKPOINT) && <option value={FLOW_MATCHING_CHECKPOINT}>Best raw · {FLOW_MATCHING_CHECKPOINT}</option>}
              </select>
            </label>
            <label>ODE steps
              <input type="number" min={2} max={500} value={steps} disabled={isGenerating} onChange={(event) => setSteps(Number(event.target.value))} />
            </label>
            <label>Noise seed
              <input type="number" min={0} max={4294967295} value={seedText} disabled={isGenerating} onChange={(event) => setSeedText(event.target.value)} />
            </label>
            <div className="flow-actions">
              <button className="primary-button" disabled={!selectedRun || isGenerating || isJudging} onClick={() => void createStudy(1)}>
                {isGenerating && sampleCount === 1 ? 'Generating…' : 'Generate one patch'}
              </button>
              <button className="quiet-button" disabled={!selectedRun || isGenerating || isJudging} onClick={() => void createStudy(100)}>
                Generate 100-patch VLM cohort
              </button>
            </div>
          </div>
        )}
        <div className="learning-note"><b>What this does:</b> the model predicts a velocity field. Euler or Heun numerically integrates that field from a standard Gaussian tensor to a new sample; this is not reconstruction of a selected EuroSAT image.</div>
      </section>

      {error && <div className="flow-error" role="alert">{error}</div>}

      {study && (
        <>
          <section className="panel flow-trajectory">
            <div className="panel-heading">
              <div><span className="section-index">02</span><h2>Follow the learned transport</h2></div>
              <span className="split-label">{study.generation_state.toUpperCase()} · {study.solver.toUpperCase()} · {study.steps} STEPS</span>
            </div>
            {study.generation_state === 'queued' || study.generation_state === 'running' ? (
              <div className="flow-progress">
                <progress max={study.generation_total} value={study.generation_completed} />
                <span>{study.generation_completed} / {study.generation_total} patches generated</span>
              </div>
            ) : study.generation_state === 'completed' && frame ? (
              <>
                <div className="flow-current">
                  <figure className="flow-frame">
                    <img src={frame.url} alt={`Flow trajectory state at t=${frame.time.toFixed(3)}`} />
                    <figcaption>First sample · t = {frame.time.toFixed(3)}</figcaption>
                  </figure>
                  <div className="flow-readout">
                    <strong>{study.nfe_per_sample ?? (study.solver === 'heun' ? 2 * study.steps : study.steps)}</strong>
                    <span>velocity-network evaluations for this sample</span>
                    <p>Initial noise at <code>t=0</code> is transported toward data at <code>t=1</code>.</p>
                  </div>
                </div>
                <input className="flow-slider" type="range" min={0} max={Math.max(frameCount - 1, 0)} value={Math.min(frameIndex, Math.max(frameCount - 1, 0))} onChange={(event) => { setPlaying(false); setFrameIndex(Number(event.target.value)) }} aria-label="Flow trajectory time" />
                <div className="flow-time-labels"><span>t = 0 · Gaussian noise</span><span>t = 1 · generated patch</span></div>
                <div className="flow-playback">
                  <button className="quiet-button" onClick={() => { setPlaying(false); setFrameIndex(0) }}>Reset</button>
                  <button className="quiet-button" disabled={frameCount < 2} onClick={() => setPlaying((value) => !value)}>{playing ? 'Pause' : 'Play trajectory'}</button>
                  <span>{study.batch_model_calls ?? '—'} batched field calls for this cohort · {study.checkpoint ?? checkpoint}</span>
                </div>
              </>
            ) : <p className="panel-copy">{study.error ?? 'Waiting for the sampler…'}</p>}
          </section>

          {study.generation_state === 'completed' && (
            <section className="panel flow-results">
              <div className="panel-heading">
                <div><span className="section-index">03</span><h2>{sampleCount === 100 ? 'Generated cohort' : 'Generated patch'}</h2></div>
                <span className="split-label">SEED {study.seed} · {sampleCount} PATCH{sampleCount === 1 ? '' : 'ES'}</span>
              </div>
              {study.sample_statistics && <p className="flow-diagnostics">
                Raw range {study.sample_statistics.value_range[0].toFixed(2)} to {study.sample_statistics.value_range[1].toFixed(2)} ·
                outside [-1,1]: {(study.sample_statistics.fraction_outside_training_range * 100).toFixed(1)}% ·
                mean per-patch SD: {study.sample_statistics.mean_per_image_pixel_std.toFixed(3)} ·
                diversity RMSE: {study.sample_statistics.pairwise_pixel_rmse?.toFixed(3) ?? 'n/a'}
              </p>}
              {sampleCount === 1 ? (
                <img className="flow-single-sample" src={`/api/flows/runs/${study.run_id}/studies/${study.study_id}/samples/0.png`} alt="Generated EuroSAT RGB patch" />
              ) : (
                <div className="flow-sample-grid">
                  {Array.from({ length: 100 }, (_, index) => {
                    const judgment = studyResult?.items.find((item) => item.sample_index === index)
                    return <figure key={index}>
                      <img src={`/api/flows/runs/${study.run_id}/studies/${study.study_id}/samples/${index}.png`} alt={`Generated patch ${index + 1}`} loading="lazy" />
                      {judgment && <figcaption>{judgment.resemblance_score}/5</figcaption>}
                    </figure>
                  })}
                </div>
              )}
              {sampleCount === 100 && <div className="flow-vlm-action">
                <p className="panel-copy">A 100-patch cohort is generated locally. VLM review sends ten images per request and uses the same fixed, unlabeled validation reference panel.</p>
                <button className="primary-button" disabled={!study.can_judge || isJudging || study.judgement_state === 'completed'} onClick={() => void startJudgement()}>
                  {isJudging ? `Judging ${study.judgement_completed}/100…` : study.judgement_state === 'completed' ? 'VLM review complete' : 'Judge 100 patches with VLM'}
                </button>
                {!study.can_judge && study.judgement_state !== 'completed' && <small>VLM review needs OPENAI_API_KEY in the API process.</small>}
              </div>}
              {studyResult?.report && <div className="flow-report">
                <div className="panel-heading"><div><span className="section-index">04</span><h2>Subjective VLM report</h2></div><span className="split-label">NOT A VALIDATED QUALITY METRIC</span></div>
                <div className="study-status-grid report-summary">
                  <div><span>Judged patches</span><b>{studyResult.report.judged_images ?? 0}</b></div>
                  <div><span>Mean resemblance</span><b>{studyResult.report.mean_resemblance_score?.toFixed(2) ?? '—'} / 5</b></div>
                  <div><span>Yes / partial / no</span><b>{studyResult.report.resemblance_yes ?? 0} / {studyResult.report.resemblance_partial ?? 0} / {studyResult.report.resemblance_no ?? 0}</b></div>
                </div>
                <div className="gap-summary">
                  <b>Most common visible gaps</b>
                  {Object.entries(studyResult.report.gap_counts ?? {}).filter(([, count]) => count > 0).sort((a, b) => b[1] - a[1]).map(([gap, count]) => <span key={gap}>{gap.replaceAll('_', ' ')} · {count}</span>)}
                </div>
                <div className="flow-judgment-list">
                  {studyResult.items.map((item) => <article key={item.patch_id}>
                    <img src={`/api/flows/runs/${study.run_id}/studies/${study.study_id}/samples/${item.sample_index}.png`} alt={`Judged patch ${item.sample_index + 1}`} loading="lazy" />
                    <div><b>Patch {String(item.sample_index + 1).padStart(3, '0')} · {item.resemblance_score}/5</b><span>{item.gap_categories.map((gap) => gap.replaceAll('_', ' ')).join(', ') || 'No visible gap category noted'}</span><p>{item.visual_evidence}</p></div>
                  </article>)}
                </div>
              </div>}
            </section>
          )}
        </>
      )}
      <footer className="page-footer generation-footer"><span>UNCONDITIONAL FLOW MATCHING · EUROSAT RGB</span><span>FLOW CLOCK · NOISE TO DATA</span><span>64 × 64 PATCHES</span></footer>
    </section>
  )
}

export default FlowStudio
