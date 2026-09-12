import { describe, expect, it } from 'vitest';
import { DATASETS, TRAINING_LESSONS } from './trainingData';

describe('training field guide', () => {
  it('offers ten distinct runnable investigations backed by canonical dataset slugs', () => {
    expect(TRAINING_LESSONS).toHaveLength(10);
    expect(new Set(TRAINING_LESSONS.map((lesson) => lesson.id)).size).toBe(10);
    const slugs = new Set(DATASETS.map((dataset) => dataset.slug));
    TRAINING_LESSONS.forEach((lesson) => {
      expect(slugs.has(lesson.dataset as typeof DATASETS[number]['slug'])).toBe(true);
      expect(lesson.query).toMatch(new RegExp(`^USE ${lesson.dataset}\\b`));
      expect(lesson.explanation.length).toBeGreaterThanOrEqual(3);
    });
  });
});
