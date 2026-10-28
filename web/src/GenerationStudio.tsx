import { useEffect, useState } from 'react'
import { getModelProfile, MODEL_PROFILES, resolveGenerationRun, type ModelFamily, type ModelRun } from './modelChoices'

type Run = ModelRun & { process?: 'ddpm' | 'vp_sde'; prediction?: 'epsilon' | 'score'; can_generate: boolean }
type Job = { generation_id: string; run_id: string; state: string }
type Study = {
  study_id: string
  run_id: string
  generation_state: 'queued' | 'running' | 'completed' | 'failed' | 'interrupted'
  generation_completed: number
  generation_total: number
  judgement_state: 'not_started' | 'queued' | 'running' | 'completed' | 'failed'
  judgement_completed: number
  judgement_total: number
  judgement_completed_batches?: number
  judgement_total_batches?: number
  judgement_current_batch?: number
  judgement_batch_size?: number
  generation_chunk_size?: number
  seed: number
  num_samples: number
  checkpoint: string
  can_judge: boolean
  error?: string
  judgement_error?: string
  sample_statistics?: ImageSummary
  sample_value_range?: [number, number]
  fraction_outside_training_range?: number
}
type Judgement = {
  patch_id: string
  resemblance_score: number
  resemblance: 'yes' | 'partial' | 'no'
  gap_categories: string[]
  visual_evidence: string
}
type JudgementReport = {
  judged_images?: number
  mean_resemblance_score?: number | null
  resemblance_yes?: number
  resemblance_partial?: number
  resemblance_no?: number
  score_counts?: Record<string, number>
  gap_counts?: Record<string, number>
  lowest_scoring_examples?: Judgement[]
  highest_scoring_examples?: Judgement[]
}
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
  model_alias: string
  epoch: number
  seed: number
  num_samples: number
  sample_statistics?: ImageSummary
  validation_reference?: ReferenceSummary
}
type Progress = {
  state: 'queued' | 'running' | 'completed' | 'failed' | 'interrupted'
  process?: 'ddpm' | 'vp_sde'
  sampler?: 'reverse_sde' | 'annealed_langevin'
  completed_steps: number
  total_steps: number
  current_timestep: number
  current_t?: number
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
type DisplayFrame = { url: string; process: 'ddpm' | 'vp_sde'; timestep: number; time: number | null }

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
  const [modelFamily, setModelFamily] = useState<ModelFamily>('ddpm')
  const [sampler, setSampler] = useState<'reverse_sde' | 'annealed_langevin'>('reverse_sde')
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
  const [studyCount, setStudyCount] = useState(100)
  const [study, setStudy] = useState<Study | null>(null)
  const [studyStarting, setStudyStarting] = useState(false)
  const [studyError, setStudyError] = useState<string | null>(null)
  const [judgements, setJudgements] = useState<Judgement[]>([])
  const [judgementReport, setJudgementReport] = useState<JudgementReport | null>(null)
  const modelProfile = getModelProfile(modelFamily)
  const selectedRun = resolveGenerationRun(runs, modelFamily)
  const runId = selectedRun?.run_id ?? ''
  const checkpointName = modelProfile.checkpoint

  useEffect(() => {
    const controller = new AbortController()
    let timer = 0
    const loadRuns = async () => {
      try {
        const response = await fetch('/api/runs', { signal: controller.signal })
        if (!response.ok) throw new Error(await responseError(response))
        const { items } = (await response.json()) as { items: Run[] }
        setRuns(items)
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
            ...current.filter((item) => item.model_alias !== modelProfile.alias),
            {
              generation_id: result.generation_id!,
              run_id: job.run_id,
              checkpoint: result.checkpoint,
              model_alias: modelProfile.alias,
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
  }, [job?.generation_id, job?.run_id, modelProfile.alias])

  useEffect(() => {
    if (!runId) {
      setStudy(null)
      setJudgements([])
      setJudgementReport(null)
      return
    }
    const controller = new AbortController()
    fetch(`/api/runs/${runId}/generation-studies`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response))
        return (await response.json()) as { items: Study[] }
      })
      .then(({ items }) => {
        const latest = items[0]
        setStudy(latest ?? null)
        setJudgements([])
        setJudgementReport(null)
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setStudyError(reason instanceof Error ? reason.message : 'Could not load saved cohorts')
      })
    return () => controller.abort()
  }, [runId])

  useEffect(() => {
    if (!study) return
    const controller = new AbortController()
    let timer = 0
    const poll = async () => {
      try {
        const response = await fetch(`/api/runs/${study.run_id}/generation-studies/${study.study_id}`, { signal: controller.signal })
        if (!response.ok) throw new Error(await responseError(response))
        const latest = (await response.json()) as Study
        if (controller.signal.aborted) return
        setStudy((current) => current?.study_id === latest.study_id ? latest : current)
        const active = latest.generation_state === 'queued' || latest.generation_state === 'running'
          || latest.judgement_state === 'queued' || latest.judgement_state === 'running'
        if (active) timer = window.setTimeout(poll, 800)
      } catch (reason) {
        if (!controller.signal.aborted) setStudyError(reason instanceof Error ? reason.message : 'Could not read cohort progress')
      }
    }
    void poll()
    return () => {
      controller.abort()
      window.clearTimeout(timer)
    }
  }, [study?.study_id, study?.run_id, study?.generation_state, study?.judgement_state])

  useEffect(() => {
    if (!study || study.judgement_state !== 'completed') return
    const controller = new AbortController()
    fetch(`/api/runs/${study.run_id}/generation-studies/${study.study_id}/judgements`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response))
        return (await response.json()) as { items: Judgement[]; report: JudgementReport }
      })
      .then(({ items, report }) => {
        if (!controller.signal.aborted) {
          setJudgements(items)
          setJudgementReport(report)
        }
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setStudyError(reason instanceof Error ? reason.message : 'Could not load the judgement report')
      })
    return () => controller.abort()
  }, [study?.study_id, study?.run_id, study?.judgement_state])

  useEffect(() => {
    if (!job || frameIndex < 0) return
    const controller = new AbortController()
    let objectUrl: string | null = null
    fetch(`/api/runs/${job.run_id}/generations/${job.generation_id}/frames/${frameIndex}`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response))
        const process = response.headers.get('X-Process') === 'vp_sde' ? 'vp_sde' : 'ddpm'
        const timestep = Number(response.headers.get('X-Timestep') ?? 0)
        const timeHeader = response.headers.get('X-Time')
        objectUrl = URL.createObjectURL(await response.blob())
        setFrame({ url: objectUrl, process, timestep, time: timeHeader === null ? null : Number(timeHeader) })
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
        body: JSON.stringify({ num_samples: numSamples, seed: requestedSeed, checkpoint: checkpointName, sampler }),
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

  async function startCohortGeneration() {
    if (!runId || !checkpointName) return
    const requestedSeed = manualSeed.trim() ? Number(manualSeed) : freshSeed()
    if (!Number.isInteger(requestedSeed) || requestedSeed < 0 || requestedSeed > 4294967295) {
      setStudyError('Seed must be an integer from 0 to 4,294,967,295.')
      return
    }
    if (!Number.isInteger(studyCount) || studyCount < 1 || studyCount > 1000) {
      setStudyError('Choose between 1 and 1,000 patches.')
      return
    }
    setStudyStarting(true)
    setStudyError(null)
    setJudgements([])
    setJudgementReport(null)
    try {
      const response = await fetch(`/api/runs/${runId}/generation-studies`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_samples: studyCount, seed: requestedSeed, checkpoint: checkpointName }),
      })
      if (!response.ok) throw new Error(await responseError(response))
      const created = (await response.json()) as { study_id: string; run_id: string }
      setStudy({
        study_id: created.study_id,
        run_id: created.run_id,
        generation_state: 'queued',
        generation_completed: 0,
        generation_total: studyCount,
        judgement_state: 'not_started',
        judgement_completed: 0,
        judgement_total: studyCount,
        seed: requestedSeed,
        num_samples: studyCount,
        checkpoint: checkpointName,
        can_judge: true,
      })
    } catch (reason) {
      setStudyError(reason instanceof Error ? reason.message : 'Could not start cohort generation')
    } finally {
      setStudyStarting(false)
    }
  }

  async function startCohortJudgement() {
    if (!study || !study.can_judge) return
    setStudyError(null)
    try {
      const response = await fetch(`/api/runs/${study.run_id}/generation-studies/${study.study_id}/judgements`, { method: 'POST' })
      if (!response.ok) throw new Error(await responseError(response))
      setStudy((current) => current ? { ...current, judgement_state: 'queued', judgement_error: undefined } : current)
    } catch (reason) {
      setStudyError(reason instanceof Error ? reason.message : 'Could not start VLM judging')
    }
  }

  const isRunning = job?.state === 'queued' || job?.state === 'running'
  const latestFrameIndex = progress?.latest_frame_index ?? -1
  const maxFrameIndex = Math.max(0, latestFrameIndex)
  const sampleGridUrl = job && progress?.state === 'completed'
    ? `/api/runs/${job.run_id}/generations/${job.generation_id}/samples.png`
    : null
  const studySampling = studyStarting || study?.generation_state === 'queued' || study?.generation_state === 'running'
  const studyJudging = study?.judgement_state === 'queued' || study?.judgement_state === 'running'
  const runBusy = loadingRuns || starting || isRunning || studySampling
  const clippedPercent = progress?.metadata?.fraction_outside_training_range === undefined
    ? null
    : 100 * progress.metadata.fraction_outside_training_range
  const isVpSde = modelFamily === 'vp_sde_score' || progress?.process === 'vp_sde'
  const selectedPrediction = modelFamily === 'vp_sde_score' ? 'score' : selectedRun?.prediction ?? 'epsilon'
  const frameIsFinal = frame?.process === 'vp_sde'
    ? progress?.sampler === 'annealed_langevin'
      ? progress.state === 'completed' && frameIndex >= latestFrameIndex
      : frame.time === 0
      : frame?.timestep === 0

  function selectModelFamily(family: ModelFamily) {
    if (family === modelFamily) return
    setModelFamily(family)
    setSampler('reverse_sde')
    setComparisons([])
    setStudy(null)
    setJudgements([])
    setJudgementReport(null)
    setStudyError(null)
    setJob(null)
    setProgress(null)
    setFrame(null)
    setFrameIndex(-1)
    setPlaying(false)
    setError(null)
  }

  return (
    <section className="generation-view">
      <header className="generation-heading">
        <div>
          <div className="eyebrow">LIVE UNCONDITIONAL GENERATION</div>
          <h1>Watch noise <em>become a patch.</em></h1>
          <p>{isVpSde ? sampler === 'annealed_langevin' ? 'Start from Gaussian noise and follow 100 descending VP noise levels, with 10 Langevin updates at each level.' : 'Start from standard Gaussian noise and follow 1,000 stochastic reverse-SDE steps from t = 1 to t = 0.' : 'Start from fresh Gaussian noise and watch all 1,000 reverse steps create a new 64-by-64 RGB patch.'}</p>
        </div>
      </header>

      <section className="generation-controls panel" aria-label="Generation settings">
        <div className="panel-heading">
          <div><span className="section-index">01</span><h2>Generate a patch live</h2></div>
          <span className="local-tag">{modelProfile.alias}</span>
        </div>
        {loadingRuns ? <div className="empty-state">Loading trained models…</div> : (
          <>
          <label className="generation-run model-family-picker" htmlFor="generation-model">
            <span>Model</span>
            <select id="generation-model" value={modelFamily} disabled={runBusy} onChange={(event) => selectModelFamily(event.target.value as ModelFamily)}>
              {MODEL_PROFILES.map((profile) => (
                <option key={profile.family} value={profile.family} disabled={!resolveGenerationRun(runs, profile.family)}>
                  {profile.label}
                </option>
              ))}
            </select>
            <small>{selectedRun
              ? `${modelProfile.alias} · ${selectedRun.run_id} · ${modelProfile.checkpoint}`
              : `No completed ${modelProfile.label} run with its validation-selected checkpoint is available.`}</small>
          </label>
          {runs.length === 0 && <div className="generation-empty"><p>No completed trained model is available for generation yet.</p></div>}
          <div className="live-generation-actions">
            <button className="primary-button generation-submit live-generate-button" onClick={() => void startGeneration()} disabled={!selectedRun || runBusy}>
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
              {isVpSde && <label className="generation-run" htmlFor="generation-sampler">
                <span>Sampling method</span>
                <select id="generation-sampler" value={sampler} disabled={runBusy} onChange={(event) => setSampler(event.target.value as 'reverse_sde' | 'annealed_langevin')}>
                  <option value="reverse_sde">Reverse VP-SDE</option>
                  <option value="annealed_langevin">Annealed Langevin dynamics</option>
                </select>
                <small>{sampler === 'annealed_langevin' ? '100 noise levels × 10 score-ascent updates; 1,000 network evaluations.' : 'Euler–Maruyama reverse SDE; 1,000 network evaluations.'}</small>
              </label>}
              <label className="generation-number" htmlFor="generation-count">
                <span>Batch size (1–8 patches)</span>
                <input id="generation-count" type="number" min="1" max="8" step="1" value={numSamples} disabled={runBusy} onChange={(event) => setNumSamples(Math.max(1, Math.min(8, Number(event.target.value) || 1)))} />
                <small>The demo animates patch 1.</small>
              </label>
              <label className="generation-number" htmlFor="generation-seed">
                <span>Manual seed (optional)</span>
                <input id="generation-seed" type="number" min="0" max="4294967295" step="1" placeholder="Fresh random seed" value={manualSeed} disabled={runBusy} onChange={(event) => setManualSeed(event.target.value)} />
                <small>The Advanced Generate button uses this seed, or fresh noise if left blank.</small>
              </label>
            </div>
            <div className="generation-advanced-actions">
              <button
                className="primary-button generation-submit generation-advanced-button"
                onClick={() => void startGeneration(undefined, true)}
                disabled={!selectedRun || runBusy}
                aria-label="Generate using advanced settings"
              >
                {isRunning || starting ? 'Generating…' : '▶ Generate'}
              </button>
            </div>
          </details>
          </>
        )}
        <div className="learning-note"><b>What you’re seeing:</b> one random noise image is refined by the trained {isVpSde && selectedPrediction === 'score' ? 'direct-score model' : 'denoiser'}. {isVpSde ? `This checkpoint predicts ${selectedPrediction === 'score' ? 'the score directly' : 'epsilon, converted to a score using -epsilon/sigma(t)'}.` : ''} The output is a new unconditional patch, not a reconstruction of a dataset image.</div>
        {error && <div className="error-banner" role="alert">{error}</div>}
      </section>

      <section className="generation-trace panel" aria-label="Live reverse sampling">
        <div className="panel-heading">
          <div><span className="section-index">02</span><h2>Follow the reverse process</h2></div>
          <span className="step-count">{progress ? `${progress.state.toUpperCase()} · ${progress.sampler === 'annealed_langevin' ? 'ANNEALED LANGEVIN' : progress.process === 'vp_sde' ? 'REVERSE VP-SDE' : 'DDPM'} · ${progress.checkpoint}` : 'READY WHEN YOU ARE'}</span>
        </div>
        <div className="generation-image-stage">
          <div className="generation-image-display">
            {frame ? <img src={frame.url} alt={frameIsFinal ? 'Generated EuroSAT-style patch' : frame.process === 'vp_sde' ? `${progress?.sampler === 'annealed_langevin' ? 'Annealed Langevin state' : 'Reverse SDE state'} at t = ${frame.time?.toFixed(progress?.sampler === 'annealed_langevin' ? 5 : 3)}` : `Current denoising state at step ${frame.timestep}`} /> : <div className="image-placeholder">{isRunning ? 'Waiting for the first frame?' : 'Generate patches to begin from random noise'}</div>}
            <span className="image-caption">{frameIsFinal ? 'GENERATED PATCH' : 'CURRENT STATE'} <b>{frame ? (frame.process === 'vp_sde' ? `t = ${frame.time?.toFixed(progress?.sampler === 'annealed_langevin' ? 5 : 3)}` : frame.timestep === 0 ? 'x_0' : `k = ${frame.timestep}`) : 'x_t'}</b></span>
          </div>
          <div className="generation-progress-copy">
            <strong>{progress ? `${progress.completed_steps.toLocaleString()} / ${progress.total_steps.toLocaleString()}` : '1,000 reverse steps'}</strong>
            <span>{progress?.state === 'completed' ? 'Sampling complete' : isRunning ? (progress?.sampler === 'annealed_langevin' ? `Langevin update | VP noise level t = ${progress?.current_t?.toFixed(3) ?? '...'}` : isVpSde ? `Reverse SDE | t = ${progress?.current_t?.toFixed(3) ?? '...'}` : 'Predicting and removing noise') : isVpSde ? 'The model starts from Gaussian noise at t = 1' : 'The model starts from Gaussian noise'}</span>
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
            <span className="play-hint">101 snapshots · every 10 {progress?.sampler === 'annealed_langevin' ? 'Langevin updates' : 'reverse steps'}</span>
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
          <p className="panel-copy">The full batch is shown above. The animation follows patch 1 from its initial noise to its final sample. Model: <code>{modelProfile.alias}</code> · source checkpoint: {progress?.checkpoint ?? checkpointName} · run: {job?.run_id ?? '—'} · seed: {progress?.seed ?? lastSeed ?? '—'}.</p>
          {clippedPercent !== null && <p className="generation-diagnostic" role="status">
            Raw values range from {progress?.metadata?.sample_value_range?.[0].toFixed(2)} to {progress?.metadata?.sample_value_range?.[1].toFixed(2)}. Display conversion clips {clippedPercent.toFixed(1)}% of tensor values to the trained image range [-1, 1]. Raw samples are preserved in the run artifacts.
          </p>}
        </section>
      )}

      {!isVpSde && <section className="generation-study panel" aria-label="Generate and judge a patch cohort">
        <div className="panel-heading">
          <div><span className="section-index">04</span><h2>Generate and judge a cohort</h2></div>
          <span className="split-label">ONE RESULT PER PATCH</span>
        </div>
        <p className="panel-copy study-explainer">
          Choose a total number of patches. Generation runs locally in chunks of up to 8; after you review the result, judging sends 10 patches per OpenAI request.
        </p>
        <div className="study-controls">
          <label className="generation-number" htmlFor="study-count">
            <span>Total patches</span>
            <input
              id="study-count"
              type="number"
              min="1"
              max="1000"
              step="1"
              value={studyCount}
              disabled={studySampling || studyJudging}
              onChange={(event) => setStudyCount(Math.max(1, Math.min(1000, Number(event.target.value) || 1)))}
            />
            <small>From 1 to 1,000. This is the total cohort size.</small>
          </label>
          <button
            className="primary-button generation-submit"
            onClick={() => void startCohortGeneration()}
            disabled={!runId || runBusy || studyJudging || studyStarting}
          >
            {studySampling ? 'Generating cohort…' : 'Generate cohort'}
          </button>
          {study && study.generation_state === 'completed' && (
            <button
              className="primary-button generation-submit"
              onClick={() => void startCohortJudgement()}
              disabled={!study.can_judge || studyJudging || study.generation_state !== 'completed' || study.judgement_state === 'completed'}
            >
              {studyJudging ? 'Judging patches…' : study.judgement_state === 'failed' ? 'Resume judging' : 'Judge cohort'}
            </button>
          )}
        </div>
        {!study?.can_judge && study?.generation_state === 'completed' && study.judgement_state !== 'completed' && (
          <p className="study-key-note">Judging needs <code>OPENAI_API_KEY</code> available to the API process. Generation remains local.</p>
        )}
        {studyError && <div className="error-banner" role="alert">{studyError}</div>}
        {study?.error && <div className="error-banner" role="alert">{study.error}</div>}
        {study && (
          <>
            <div className="study-status-grid" role="status">
              <div><span>Study</span><b>{study.study_id}</b></div>
              <div><span>Generation</span><b>{study.generation_state} · {study.generation_completed}/{study.generation_total}</b></div>
              <div><span>Judgment</span><b>{study.judgement_state} · {study.judgement_completed}/{study.judgement_total}</b></div>
              <div><span>Seed / checkpoint</span><b>{study.seed} · {study.checkpoint}</b></div>
            </div>
            {study.generation_state === 'running' && (
              <div className="progress-track study-progress"><span style={{ width: `${100 * study.generation_completed / Math.max(study.generation_total, 1)}%` }} /></div>
            )}
            {study.judgement_state === 'running' && (
              <div className="study-progress-copy">
                Judged {study.judgement_completed} of {study.judgement_total} patches
                · request {study.judgement_current_batch ?? study.judgement_completed_batches ?? 0} of {study.judgement_total_batches ?? Math.ceil(study.judgement_total / 10)}
              </div>
            )}
            {study.judgement_error && <div className="error-banner" role="alert">{study.judgement_error}</div>}
            {study.generation_state === 'completed' && (
              <>
                <img
                  className="generated-grid-image study-grid-image"
                  src={`/api/runs/${study.run_id}/generation-studies/${study.study_id}/samples.png`}
                  alt={`${study.num_samples} generated 64 by 64 EuroSAT-style patches in cohort order`}
                />
                <p className="panel-copy study-grid-note">
                  Review this grid before judging. Each result ID maps to its position in the grid. The fixed comparison sheet contains 16 unlabeled validation patches; test images are not used.
                </p>
                {study.sample_statistics && <p className="generation-diagnostic">
                  Cohort raw range: {study.sample_value_range?.[0].toFixed(2)} to {study.sample_value_range?.[1].toFixed(2)} ·
                  outside [-1,1]: {((study.fraction_outside_training_range ?? 0) * 100).toFixed(1)}% ·
                  mean pixel SD: {study.sample_statistics.mean_per_image_pixel_std.toFixed(3)} ·
                  diversity RMSE: {study.sample_statistics.pairwise_pixel_rmse?.toFixed(3) ?? 'n/a'}
                </p>}
              </>
            )}
          </>
        )}
        {judgementReport && (
          <div className="judgement-report">
            <div className="panel-heading">
              <div><span className="section-index">05</span><h2>Visual judgment report</h2></div>
              <span className="split-label">SUBJECTIVE VLM ASSESSMENT</span>
            </div>
            <div className="study-status-grid report-summary">
              <div><span>Judged patches</span><b>{judgementReport.judged_images}</b></div>
              <div><span>Mean resemblance</span><b>{judgementReport.mean_resemblance_score?.toFixed(2) ?? '—'} / 5</b></div>
              <div><span>Yes / partial / no</span><b>{judgementReport.resemblance_yes} / {judgementReport.resemblance_partial} / {judgementReport.resemblance_no}</b></div>
            </div>
            <div className="gap-summary">
              <b>Most common visible gaps</b>
              {Object.entries(judgementReport.gap_counts ?? {}).filter(([, count]) => count > 0).sort((a, b) => b[1] - a[1]).map(([gap, count]) => (
                <span key={gap}>{gap.replaceAll('_', ' ')} · {count}</span>
              ))}
            </div>
            <p className="panel-copy study-explainer">Ratings describe visual resemblance, not a validated quality metric. Read them alongside the raw-range and diversity diagnostics shown above.</p>
            <div className="judgement-list">
              {judgements.map((item) => {
                const sampleIndex = Number(item.patch_id.replace('patch-', '')) - 1
                return <article className="judgement-card" key={item.patch_id}>
                  <img
                    src={`/api/runs/${study?.run_id}/generation-studies/${study?.study_id}/samples/${sampleIndex}.png`}
                    alt={`Generated patch ${item.patch_id}`}
                  />
                  <div className="judgement-card-copy">
                    <b>{item.patch_id} · {item.resemblance.toUpperCase()} · {item.resemblance_score}/5</b>
                    <span>{item.gap_categories.length ? item.gap_categories.map((gap) => gap.replaceAll('_', ' ')).join(', ') : 'No major visual gap noted'}</span>
                    <p>{item.visual_evidence}</p>
                  </div>
                </article>
              })}
            </div>
          </div>
        )}
      </section>}

      {comparisons.length > 1 && (
        <section className="generation-comparison panel" aria-label="Checkpoint sample comparison">
          <div className="panel-heading">
            <div><span className="section-index">04</span><h2>Compare models</h2></div>
            <span className="split-label">SAME SEED {comparisons[0].seed} · {comparisons.length} MODELS</span>
          </div>
          <div className="checkpoint-comparison-grid">
            {comparisons.map((item) => <figure key={item.model_alias}>
              <figcaption>{item.model_alias}</figcaption>
              <img src={`/api/runs/${item.run_id}/generations/${item.generation_id}/samples.png`} alt={`${item.num_samples} generated patches from ${item.model_alias} with seed ${item.seed}`} />
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
        <span>UNCONDITIONAL {isVpSde ? 'VP-SDE' : 'DDPM'} &middot; EUROSAT RGB</span>
        <span>SELECTED CHECKPOINT · NO LABELS</span>
        <span>64 × 64 PATCHES</span>
      </footer>
    </section>
  )
}

export default GenerationStudio
