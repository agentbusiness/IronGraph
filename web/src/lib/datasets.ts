export const BUNDLED_DATASETS = [
  'fraud',
  'flights',
  'trust',
  'epinions',
  'email',
  'social',
  'library',
  'dblp',
  'citations',
  'overflow',
] as const;

const BUNDLED_DATASET_NAMES = new Set<string>(BUNDLED_DATASETS);

export function isBundledDataset(name: string): boolean {
  return BUNDLED_DATASET_NAMES.has(name);
}

export function requiredBundledDatasets(markdown: string): string[] {
  const required = new Set<string>();
  for (const match of markdown.matchAll(/^\s*USE\s+`?([\w-]+)`?/gim)) {
    const name = match[1];
    if (name && isBundledDataset(name)) required.add(name);
  }
  return Array.from(required).sort();
}
