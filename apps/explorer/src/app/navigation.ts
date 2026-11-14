export const organizationTabs = [
  ['', 'Overview'],
  ['explore', 'Visual exploration'],
  ['sequences', 'Sequences & linkage'],
  ['evaluation', 'Evaluation'],
] as const
export const legacyOrganizationRoutes: Record<string, string> = {
  dataset: '/organization',
  embeddings: '/organization/explore?view=embeddings',
  similarity: '/organization/explore?view=similarity',
  reduction: '/organization/explore?view=reduction',
  clustering: '/organization/explore?view=clustering',
  groups: '/organization/sequences',
  splits: '/organization/evaluation?view=splits',
  detector: '/organization/evaluation?view=detector',
}
export const nav = [
  ['/', 'Pipeline overview'],
  ['/preprocessing', 'Preprocessing'],
  ['/organization', 'Representation & organization'],
  ['/segmentation', 'Panoptic segmentation'],
  ['/experiments', 'Experiments'],
  ['/evaluation', 'Global evaluation'],
] as const
