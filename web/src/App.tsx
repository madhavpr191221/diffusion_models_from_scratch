import { useEffect, useState } from 'react'
import TrainingMonitor from './TrainingMonitor'
import './App.css'

type ImageItem = { index: number; url: string }
type ImagePage = { total: number; offset: number; limit: number; items: ImageItem[] }
type Trace = { trace_id: string; mode: 'direct' | 'markov'; steps: number; image_index: number; seed: number }
type Frame = { url: string; timestep: number; alphaBar: number }

const PAGE_SIZE = 24
const FRAME_DELAY_MS = 50

async function responseError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: string }
    return body.detail ?? `Request failed (${response.status})`
  } catch {
    return `Request failed (${response.status})`
  }
}

function App() {
  const [view, setView] = useState<'forward' | 'training'>('forward')
  const [pageOffset, setPageOffset] = useState(0)
  const [page, setPage] = useState<ImagePage | null>(null)
  const [selectedImage, setSelectedImage] = useState<ImageItem | null>(null)
  const [mode, setMode] = useState<'direct' | 'markov'>('markov')
  const [seed, setSeed] = useState(42)
  const [trace, setTrace] = useState<Trace | null>(null)
  const [stepIndex, setStepIndex] = useState(0)
  const [frame, setFrame] = useState<Frame | null>(null)
  const [playing, setPlaying] = useState(false)
  const [loadingImages, setLoadingImages] = useState(true)
  const [creatingTrace, setCreatingTrace] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    setLoadingImages(true)
    setError(null)
    fetch(`/api/validation/images?offset=${pageOffset}&limit=${PAGE_SIZE}`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response))
        return (await response.json()) as ImagePage
      })
      .then((result) => {
        setPage(result)
        setTrace(null)
        setFrame(null)
        setPlaying(false)
        setSelectedImage((current) => {
          if (current && result.items.some((item) => item.index === current.index)) return current
          return result.items[0] ?? null
        })
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : 'Could not load validation images')
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoadingImages(false)
      })
    return () => controller.abort()
  }, [pageOffset])

  useEffect(() => {
    if (!trace) return
    const controller = new AbortController()
    let objectUrl: string | null = null
    setFrame(null)
    fetch(`/api/forward-traces/${trace.trace_id}/frames/${stepIndex}`, {
      signal: controller.signal,
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(await responseError(response))
        const timestep = Number(response.headers.get('X-Timestep'))
        const alphaBar = Number(response.headers.get('X-Alpha-Bar'))
        const blob = await response.blob()
        objectUrl = URL.createObjectURL(blob)
        setFrame({ url: objectUrl, timestep, alphaBar })
      })
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : 'Could not load this frame')
        }
      })
    return () => {
      controller.abort()
      if (objectUrl) URL.revokeObjectURL(objectUrl)
    }
  }, [trace, stepIndex])

  useEffect(() => {
    if (!playing || !trace) return
    if (stepIndex >= trace.steps - 1) {
      setPlaying(false)
      return
    }
    const timer = window.setTimeout(() => setStepIndex((current) => current + 1), FRAME_DELAY_MS)
    return () => window.clearTimeout(timer)
  }, [playing, stepIndex, trace])

  async function createTrace() {
    if (!selectedImage) return
    setCreatingTrace(true)
    setError(null)
    setPlaying(false)
    setFrame(null)
    setTrace(null)
    try {
      const response = await fetch('/api/forward-traces', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_index: selectedImage.index, mode, seed }),
      })
      if (!response.ok) throw new Error(await responseError(response))
      const nextTrace = (await response.json()) as Trace
      setTrace(nextTrace)
      setStepIndex(0)
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not create the noise trace')
    } finally {
      setCreatingTrace(false)
    }
  }

  function updateImage(item: ImageItem) {
    setSelectedImage(item)
    setTrace(null)
    setFrame(null)
    setPlaying(false)
  }

  function updateMode(value: 'direct' | 'markov') {
    setMode(value)
    setTrace(null)
    setFrame(null)
    setPlaying(false)
  }

  function updateSeed(value: number) {
    setSeed(value)
    setTrace(null)
    setFrame(null)
    setPlaying(false)
  }

  const pageNumber = page ? Math.floor(page.offset / page.limit) + 1 : 1
  const pageCount = page ? Math.ceil(page.total / page.limit) : 1

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="EuroSAT DDPM home">
          <span className="brand-mark">ε</span>
          <span>FIELDNOTES <b>/ DDPM</b></span>
        </a>
        <div className="topbar-right">
          <span className="status-pill"><span className="status-dot" /> {view === 'forward' ? 'FORWARD PROCESS · NO TRAINED MODEL REQUIRED' : 'LOCAL TRAINING MONITOR'}</span>
          <a className="log-link" href="/api/learning-log" target="_blank" rel="noreferrer">Learning log ↗</a>
        </div>
      </header>

      <nav className="view-tabs" aria-label="Project views">
        <button className={view === 'forward' ? 'active' : ''} onClick={() => setView('forward')}>Forward process</button>
        <button className={view === 'training' ? 'active' : ''} onClick={() => setView('training')}>Training monitor</button>
      </nav>

      {view === 'training' ? <TrainingMonitor /> : <>
      <section className="intro" id="top">
        <div className="eyebrow">AN INTERACTIVE DDPM NOTEBOOK · STAGE 2.5</div>
        <h1>Watch an image <em>become noise.</em></h1>
        <p>Choose a EuroSAT image, set a seed, and follow the forward process. No trained model is needed for this experiment.</p>
      </section>

      <section className="workspace" aria-label="Forward diffusion explorer">
        <aside className="image-panel panel">
          <div className="panel-heading">
            <div><span className="section-index">01</span><h2>Choose an image</h2></div>
            <span className="split-label">VALIDATION · {page?.total.toLocaleString() ?? '…'}</span>
          </div>
          <p className="panel-copy">Unlabeled RGB patches, 64 × 64 pixels.</p>
          {loadingImages && <div className="empty-state">Loading the local dataset…</div>}
          {!loadingImages && page && (
            <>
              <div className="image-grid" role="list" aria-label="Validation image choices">
                {page.items.map((item) => (
                  <button
                    className={`image-choice${selectedImage?.index === item.index ? ' selected' : ''}`}
                    key={item.index}
                    onClick={() => updateImage(item)}
                    disabled={creatingTrace}
                    aria-label={`Select validation image ${item.index + 1}`}
                    aria-pressed={selectedImage?.index === item.index}
                    role="listitem"
                  >
                    <img src={item.url} alt="" loading="lazy" />
                    <span>{String(item.index + 1).padStart(4, '0')}</span>
                  </button>
                ))}
              </div>
              <div className="pagination">
                <button className="quiet-button" disabled={pageOffset === 0 || creatingTrace} onClick={() => setPageOffset(Math.max(0, pageOffset - PAGE_SIZE))}>← Previous</button>
                <span>{pageNumber} / {pageCount}</span>
                <button className="quiet-button" disabled={!page || pageOffset + PAGE_SIZE >= page.total || creatingTrace} onClick={() => setPageOffset(pageOffset + PAGE_SIZE)}>Next →</button>
              </div>
            </>
          )}
          <div className="privacy-note"><span>↳</span> Class labels are not loaded into this viewer.</div>
        </aside>

        <section className="experiment-panel panel" aria-label="Noise experiment">
          <div className="panel-heading">
            <div><span className="section-index">02</span><h2>Set up the experiment</h2></div>
            <span className="local-tag">LOCAL</span>
          </div>
          <div className="controls-grid">
            <fieldset className="mode-field">
              <legend>Forward-process view</legend>
              <label className={`mode-option${mode === 'markov' ? ' active' : ''}`}>
                <input type="radio" name="mode" value="markov" checked={mode === 'markov'} disabled={creatingTrace} onChange={() => updateMode('markov')} />
                <span><b>Stepwise path</b><small>Fresh noise at each transition</small></span>
                <span className="radio-dot" />
              </label>
              <label className={`mode-option${mode === 'direct' ? ' active' : ''}`}>
                <input type="radio" name="mode" value="direct" checked={mode === 'direct'} disabled={creatingTrace} onChange={() => updateMode('direct')} />
                <span><b>Direct marginal</b><small>Jump from the clean image to t</small></span>
                <span className="radio-dot" />
              </label>
            </fieldset>
            <label className="seed-field" htmlFor="seed-input">
              <span>Random seed</span>
              <input id="seed-input" type="number" min="0" max="4294967295" step="1" value={seed} disabled={creatingTrace} onChange={(event) => updateSeed(Math.max(0, Math.min(4294967295, Number(event.target.value) || 0)))} />
              <small>Same image + mode + seed = same trace</small>
            </label>
          </div>
          <div className="create-row">
            <span>{selectedImage ? `Image ${selectedImage.index + 1} · ${mode === 'markov' ? 'one realized chain' : 'closed-form marginals'}` : 'Choose an image to begin'}</span>
            <button className="primary-button" onClick={createTrace} disabled={!selectedImage || creatingTrace}>
              {creatingTrace ? 'Building trace…' : trace ? '↻ Create trace' : 'Create trace'}
            </button>
          </div>
          {error && <div className="error-banner" role="alert">{error}</div>}
        </section>

        <section className="trace-panel panel" aria-label="Noisy image trace">
          <div className="panel-heading">
            <div><span className="section-index">03</span><h2>Trace the noise</h2></div>
            <span className="step-count">{trace ? `${trace.steps.toLocaleString()} STEPS` : 'READY WHEN YOU ARE'}</span>
          </div>
          <div className="image-stage">
            <div className="image-display original-display">
              {selectedImage ? <img src={selectedImage.url} alt={`Original validation image ${selectedImage.index + 1}`} /> : <div className="image-placeholder">Select an image</div>}
              <span className="image-caption">CLEAN IMAGE <b>x₀</b></span>
            </div>
            <div className="image-display noisy-display">
              {frame ? <img src={frame.url} alt={`Noisy image at timestep ${frame.timestep}`} /> : <div className="image-placeholder">{trace ? 'Loading frame…' : 'Create a trace to begin'}</div>}
              <span className="image-caption">NOISY IMAGE <b>{frame ? `x${frame.timestep}` : 'xₜ'}</b></span>
            </div>
          </div>

          <div className="timestep-controls">
            <div className="timestep-readout">
              <div><span>DIFFUSION STEP</span><strong>{frame ? String(frame.timestep).padStart(4, '0') : '— — — —'}<small> / {trace?.steps ?? 1000}</small></strong></div>
              <div className="alpha-readout"><span>RETAINED SIGNAL · ᾱₜ</span><strong>{frame ? frame.alphaBar.toFixed(5) : '—'}</strong></div>
            </div>
            <input
              className="timestep-slider"
              aria-label="Diffusion timestep"
              type="range"
              min="0"
              max={(trace?.steps ?? 1000) - 1}
              value={stepIndex}
              disabled={!trace}
              onChange={(event) => { setPlaying(false); setStepIndex(Number(event.target.value)) }}
            />
            <div className="slider-labels"><span>t = 1 · mostly original</span><span>t = 1000 · mostly noise</span></div>
            <div className="playback-controls">
              <button className="quiet-button" disabled={!trace} onClick={() => { setPlaying(false); setStepIndex(0) }}>↺ Reset</button>
              <button className="play-button" disabled={!trace} onClick={() => {
                if (stepIndex >= (trace?.steps ?? 1) - 1) setStepIndex(0)
                setPlaying((current) => !current)
              }}>{playing ? 'Ⅱ  Pause' : '▶  Play'}</button>
              <span className="play-hint">20 frames / second</span>
            </div>
          </div>
          <p className="mode-explanation">
            {mode === 'markov'
              ? 'Each frame continues from the previous one: a single stochastic forward path.'
              : 'Each frame is sampled directly from x₀ with the same ε. Marginals are correct; frames are not one Markov path.'}
          </p>
        </section>
      </section>

      <footer className="page-footer">
        <span>UNCONDITIONAL DDPM STUDY · EUROSAT RGB</span>
        <a href="/api/learning-log" target="_blank" rel="noreferrer">Read what we’ve built so far ↗</a>
        <span>NO TRAINED MODEL YET</span>
      </footer>
      </>}
    </main>
  )
}

export default App
