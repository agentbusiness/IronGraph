import { describe, expect, it } from 'vitest';
import { dayLabel, elapsedMicroseconds, formatElapsed, whenLabel } from './format';

describe('when something is', () => {
  const now = new Date(2026, 7, 22, 12, 0, 0, 0).getTime();

  it('leaves the year off a date in the current year', () => {
    const label = whenLabel(new Date(2026, 7, 28, 14, 0, 0, 0).getTime(), now);
    expect(label).not.toMatch(/2026/);
    expect(label).toMatch(/28/);
  });

  it('states the year when the date is not in this one', () => {
    const label = whenLabel(new Date(2027, 2, 30, 13, 0, 0, 0).getTime(), now);
    expect(label).toMatch(/2027/);
  });

  it('states the year for a past year too', () => {
    expect(whenLabel(new Date(2025, 11, 24, 9, 0, 0, 0).getTime(), now)).toMatch(/2025/);
  });

  it('applies the same rule to a bare day', () => {
    expect(dayLabel(new Date(2026, 7, 28).getTime(), now)).not.toMatch(/2026/);
    expect(dayLabel(new Date(2027, 2, 30).getTime(), now)).toMatch(/2027/);
  });
});

describe('how long the engine took', () => {
  it('names microseconds below a millisecond, where this engine usually finishes', () => {
    expect(formatElapsed(420)).toBe('420 µs');
    expect(formatElapsed(999)).toBe('999 µs');
  });

  it('never reports a bare zero, which reads as broken rather than as fast', () => {
    expect(formatElapsed(0)).toBe('<1 µs');
  });

  it('changes unit with magnitude, keeping a glance-readable precision', () => {
    expect(formatElapsed(1_000)).toBe('1.0 ms');
    expect(formatElapsed(4_200)).toBe('4.2 ms');
    expect(formatElapsed(12_400)).toBe('12 ms');
    expect(formatElapsed(786_000)).toBe('786 ms');
    expect(formatElapsed(1_500_000)).toBe('1.5 s');
    expect(formatElapsed(45_000_000)).toBe('45 s');
  });

  it('reports nothing at all when nothing was measured', () => {
    expect(formatElapsed(undefined)).toBeUndefined();
    expect(formatElapsed(Number.NaN)).toBeUndefined();
    expect(formatElapsed(-1)).toBeUndefined();
  });
});

describe('which field the summary measured with', () => {
  it('prefers the microsecond field the current server sends', () => {
    expect(elapsedMicroseconds({ elapsed_us: 640, elapsed_ms: 0 })).toBe(640);
  });

  it('falls back to whole milliseconds from a server that only has those', () => {
    expect(elapsedMicroseconds({ elapsed_ms: 17 })).toBe(17_000);
  });

  it('treats an unmeasured zero as no measurement rather than as sub-microsecond', () => {
    expect(elapsedMicroseconds({ elapsed_ms: 0 })).toBeUndefined();
    expect(elapsedMicroseconds(undefined)).toBeUndefined();
  });
});
