import { useEffect, useState, type ReactNode } from 'react';
import { Cross } from '../design/parts';
import { useTheme } from '../hooks/useTheme';

type Phase = 'downloading' | 'loading' | 'warming' | 'ready' | 'failed';
interface Progress {
  phase: Phase;
  downloaded_bytes: number;
  total_bytes: number;
  error?: string;
}

const LABEL: Record<Phase, string> = {
  downloading: 'Downloading the embedding model',
  loading: 'Loading the embedding model',
  warming: 'Warming the embedding device',
  ready: 'Ready',
  failed: 'Model startup failed',
};

function parseProgress(value: unknown): Progress {
  if (!value || typeof value !== 'object') throw new Error('Startup status is invalid.');
  const data = value as Record<string, unknown>;
  if (typeof data.phase !== 'string' || !Object.hasOwn(LABEL, data.phase)
    || typeof data.downloaded_bytes !== 'number' || !Number.isSafeInteger(data.downloaded_bytes) || data.downloaded_bytes < 0
    || typeof data.total_bytes !== 'number' || !Number.isSafeInteger(data.total_bytes) || data.total_bytes <= 0
    || data.downloaded_bytes > data.total_bytes
    || (data.error != null && typeof data.error !== 'string')) throw new Error('Startup status is invalid.');
  return { phase: data.phase as Phase, downloaded_bytes: data.downloaded_bytes, total_bytes: data.total_bytes, error: data.error as string | undefined };
}

export function StartupGate({ children }: { children: ReactNode }) {
  const [progress, setProgress] = useState<Progress>();
  const [error, setError] = useState<string>();
  const [theme] = useTheme();

  useEffect(() => {
    let disposed = false;
    let ready = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;
    const poll = async () => {
      if (disposed || ready || document.hidden) return;
      const active = new AbortController();
      controller = active;
      try {
        const response = await fetch('/system/startup', { cache: 'no-store', credentials: 'same-origin', signal: active.signal });
        if (!response.ok) throw new Error(`Startup status did not answer (${response.status}).`);
        const next = parseProgress(await response.json());
        if (disposed || active.signal.aborted) return;
        ready = next.phase === 'ready';
        setProgress(next);
        setError(undefined);
      } catch (cause) {
        if (!disposed && !active.signal.aborted) setError(cause instanceof Error ? cause.message : 'Startup status did not answer.');
      } finally {
        if (!disposed && !active.signal.aborted && !ready && !document.hidden) timer = setTimeout(() => { void poll(); }, 1000);
      }
    };
    const visibility = () => {
      clearTimeout(timer);
      controller?.abort();
      if (!document.hidden) void poll();
    };
    document.addEventListener('visibilitychange', visibility);
    void poll();
    return () => { disposed = true; clearTimeout(timer); controller?.abort(); document.removeEventListener('visibilitychange', visibility); };
  }, []);

  if (progress?.phase === 'ready') return children;
  const failed = progress?.phase === 'failed';
  const percentage = progress ? Math.floor(progress.downloaded_bytes / progress.total_bytes * 100) : undefined;
  const bytes = (value: number) => `${(value / 1024 ** 3).toFixed(2)} GiB`;
  return (
    <main id="ig" data-view={theme === 'dark' ? 'plate' : theme === 'light' ? 'page' : 'auto'} className="drawn">
      <header className="band"><div className="mark"><b>IronGraph</b><Cross /><i>starting</i></div></header>
      <section className="console-leaf startup-leaf" aria-label="Database startup">
        <aside className="col-i"><span className="ap">Local text embedding</span><p>The model downloads once and stays on this machine.</p></aside>
        <div className="rail" aria-hidden="true"><span className="cap">Start</span><Cross /></div>
        <div className="col-main startup-body">
          <span className="ap faint">Preparing the console</span>
          <h1>{progress ? LABEL[progress.phase] : 'Connecting to the database'}</h1>
          <p className="empty-line" role="status">{progress?.phase === 'downloading'
            ? 'Downloading and verifying the model files. This page updates automatically.'
            : failed ? 'Resolve the reported problem, then restart IronGraph.'
              : 'The console opens automatically when the embedding model is ready.'}</p>
          {progress && !failed && <>
            <progress className="startup-progress" aria-label="Model download" max={progress.total_bytes} value={progress.downloaded_bytes} />
            <p className="ap">{percentage}% · {bytes(progress.downloaded_bytes)} of {bytes(progress.total_bytes)}</p>
          </>}
          {(error || progress?.error) && <div className="notice contradiction" role="alert"><p>{error ?? progress?.error}</p></div>}
        </div>
        <div className="marg-rule" aria-hidden="true" />
        <aside className="marg"><span className="ap">Startup</span><p>Download → load → warm up</p><p>Keep this page open to see progress.</p></aside>
      </section>
    </main>
  );
}
