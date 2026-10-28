export type ModelFamily = 'ddpm' | 'vp_sde_score'

export type ModelRun = {
  run_id: string
  state: string
  mode: string
  process?: string
  prediction?: string
  created_at?: string | null
  can_generate?: boolean
}

export type ModelProfile = {
  family: ModelFamily
  label: string
  alias: string
  checkpoint: string
  process: 'ddpm' | 'vp_sde'
  prediction?: 'score'
}

export const MODEL_PROFILES: ModelProfile[] = [
  {
    family: 'ddpm',
    label: 'EuroSAT DDPM',
    alias: 'best-eurosat-ddpm.pt',
    checkpoint: 'best-val.pt',
    process: 'ddpm',
  },
  {
    family: 'vp_sde_score',
    label: 'EuroSAT VP-SDE · direct score',
    alias: 'best-eurosat-vp-sde.pt',
    checkpoint: 'best-val-ema.pt',
    process: 'vp_sde',
    prediction: 'score',
  },
]

export function getModelProfile(family: ModelFamily): ModelProfile {
  return MODEL_PROFILES.find((profile) => profile.family === family) ?? MODEL_PROFILES[0]
}

function matchesFamily(run: ModelRun, profile: ModelProfile): boolean {
  if (profile.process === 'ddpm') return run.process === 'ddpm'
  return run.process === 'vp_sde' && run.prediction === 'score'
}

function newestFirst<T extends ModelRun>(runs: T[]): T[] {
  return [...runs].sort((first, second) => {
    const byCreatedAt = Date.parse(second.created_at ?? '') - Date.parse(first.created_at ?? '')
    return Number.isNaN(byCreatedAt) || byCreatedAt === 0
      ? second.run_id.localeCompare(first.run_id)
      : byCreatedAt
  })
}

export function resolveGenerationRun<T extends ModelRun>(runs: T[], family: ModelFamily): T | undefined {
  const profile = getModelProfile(family)
  return newestFirst(runs.filter((run) =>
    run.mode === 'train'
    && run.state === 'completed'
    && run.can_generate === true
    && matchesFamily(run, profile),
  ))[0]
}

export function resolveTrainingRun<T extends ModelRun>(runs: T[], family: ModelFamily): T | undefined {
  const profile = getModelProfile(family)
  const matchingRuns = runs.filter((run) => run.mode === 'train' && matchesFamily(run, profile))
  const active = newestFirst(matchingRuns.filter((run) => run.state === 'running'))[0]
  return active ?? newestFirst(matchingRuns.filter((run) => run.state === 'completed'))[0]
}
