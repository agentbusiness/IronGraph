import { useEffect, useState } from 'react';
import type { ThemePreference } from '../types';

const STORAGE_KEY = 'irongraph-theme';

/**
 * The reader's chosen ground, readable without a hook.
 *
 * The gate, the picker and the error frame all draw before — or instead of — the shell, and each
 * has to open on the same ground the reader chose. Exported so none of them has to guess.
 */
export function storedTheme(): ThemePreference {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    // The design is drawn on the plate. Nothing stored means nothing chosen, so it opens the
    // way the design opens rather than the way the machine happens to be set.
    return value === 'light' || value === 'dark' || value === 'system' ? value : 'dark';
  } catch {
    return 'dark';
  }
}

function resolvedTheme(preference: ThemePreference, media: MediaQueryList): 'light' | 'dark' {
  return preference === 'system' ? (media.matches ? 'dark' : 'light') : preference;
}

function applyTheme(preference: ThemePreference): void {
  const resolved = resolvedTheme(preference, window.matchMedia('(prefers-color-scheme: dark)'));
  document.documentElement.dataset.theme = resolved;
  document.documentElement.style.colorScheme = resolved;
}

/**
 * Put the stored ground on the document before React renders anything.
 *
 * The theme used to reach the document only from this hook's effect, which is to say only once the
 * shell had mounted. Everything that happens before that — the first paint, and any failure that
 * replaces the app instead of mounting it — was drawn on whatever the stylesheet defaulted to.
 * Setting it at startup costs one attribute and removes that whole class of mismatch.
 */
export function applyStoredTheme(): void {
  applyTheme(storedTheme());
}

export function useTheme(): [ThemePreference, (theme: ThemePreference) => void] {
  const [preference, setPreference] = useState<ThemePreference>(storedTheme);

  useEffect(() => {
    const media = window.matchMedia('(prefers-color-scheme: dark)');
    const apply = () => applyTheme(preference);
    apply();
    media.addEventListener('change', apply);
    try { localStorage.setItem(STORAGE_KEY, preference); } catch { /* The active theme still works for this page lifetime. */ }
    return () => media.removeEventListener('change', apply);
  }, [preference]);

  return [preference, setPreference];
}
