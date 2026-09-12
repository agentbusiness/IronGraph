import { useEffect, useState } from 'react';

export interface CanvasPalette {
  background: string;
  foreground: string;
  muted: string;
  edge: string;
  accent: string;
  selected: string;
  /** What is changing right now — a traversal in flight, not a resting selection. */
  live: string;
}

/**
 * The canvas reads the same ink as the page.
 *
 * These used to name variables of their own (`--graph-selected`, `--canvas-bg`) that the approved
 * stylesheet never defines, so every one of them fell through to a literal and the plot was drawn
 * in a palette belonging to no design — an amber selection on a page whose accent is the rubric.
 * The design's own tokens are the fallback now, and the literals are the last resort for a document
 * that has not applied the stylesheet at all.
 */
function readPalette(): CanvasPalette {
  const style = getComputedStyle(document.documentElement);
  const read = (...names: [...string[], string]) => {
    for (const name of names.slice(0, -1)) {
      const value = style.getPropertyValue(name).trim();
      if (value) return value;
    }
    return names[names.length - 1] as string;
  };
  return {
    background: read('--canvas-bg', '--ground', '#14171A'),
    foreground: read('--text', '--ink', '#E9EBEC'),
    muted: read('--muted', '--ink-3', '#828A8F'),
    // Relationships are drawn at half alpha over the canvas, so the hairline rule colour — which is
    // already the faintest thing on the page — disappears against the light ground. The muted ink is
    // the quietest colour that still reads as a line on both.
    edge: read('--graph-edge', '--ink-3', '#828A8F'),
    accent: read('--accent', '--counter', '#6E9BB5'),
    selected: read('--graph-selected', '--rubric', '#E2603F'),
    live: read('--graph-live', '--rubric-lift', '#F59A5B'),
  };
}

export function useCanvasPalette(): CanvasPalette {
  const [palette, setPalette] = useState(readPalette);
  useEffect(() => {
    const observer = new MutationObserver(() => setPalette(readPalette()));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return () => observer.disconnect();
  }, []);
  return palette;
}
