import { describe, expect, it } from 'vitest';
import { isBundledDataset, requiredBundledDatasets } from './datasets';

describe('documentation dataset requirements', () => {
  it('recognizes every bundled project used by a page and ignores scratch projects', () => {
    expect(requiredBundledDatasets(`USE trust\nRETURN 1\n\nUSE fraud\nRETURN 2\n\nUSE worked_example\nRETURN 3`))
      .toEqual(['fraud', 'trust']);
    expect(isBundledDataset('flights')).toBe(true);
    expect(isBundledDataset('worked_example')).toBe(false);
  });
});
