import { describe, expect, it } from 'vitest';
import { perspectiveFitDistance } from './graphGeometry';

describe('graph viewport geometry', () => {
  it('fits wide layouts using both the camera aspect and graph depth', () => {
    const wideViewport = perspectiveFitDistance(400, 80, 20, 2, 48);
    const narrowViewport = perspectiveFitDistance(400, 80, 20, 0.5, 48);
    const deepLayout = perspectiveFitDistance(400, 80, 200, 2, 48);

    expect(wideViewport).toBeGreaterThan(0);
    expect(narrowViewport).toBeGreaterThan(wideViewport);
    expect(deepLayout).toBeGreaterThan(wideViewport);
  });
});
