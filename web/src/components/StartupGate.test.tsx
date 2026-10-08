import { act, cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { StartupGate } from './StartupGate';

vi.mock('../hooks/useTheme', () => ({ useTheme: () => ['dark', vi.fn()] }));
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); vi.restoreAllMocks(); });

const status = (phase: string, downloaded_bytes = 20, error?: string) =>
  ({ ok: true, json: () => Promise.resolve({ phase, downloaded_bytes, total_bytes: 100, error }) });

describe('startup progress', () => {
  it('shows measured bytes and all phases before mounting the console once ready', async () => {
    vi.useFakeTimers();
    const fetch = vi.fn().mockResolvedValueOnce(status('downloading')).mockResolvedValueOnce(status('loading', 100))
      .mockResolvedValueOnce(status('warming', 100)).mockResolvedValue(status('ready', 100));
    vi.stubGlobal('fetch', fetch);
    await act(async () => { render(<StartupGate><h1>Query console</h1></StartupGate>); await Promise.resolve(); });
    expect(screen.getByRole('heading', { name: 'Downloading the embedding model' })).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toHaveAttribute('value', '20');
    expect(screen.queryByRole('heading', { name: 'Query console' })).not.toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(screen.getByRole('heading', { name: 'Loading the embedding model' })).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(screen.getByRole('heading', { name: 'Warming the embedding device' })).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(1000); });
    expect(screen.getByRole('heading', { name: 'Query console' })).toBeInTheDocument();
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(fetch).toHaveBeenCalledTimes(4);
  });

  it('keeps a startup failure visible with the actual cause', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(status('failed', 20, 'Model checksum verification failed')));
    await act(async () => { render(<StartupGate><p>Console</p></StartupGate>); await Promise.resolve(); });
    expect(screen.getByRole('alert')).toHaveTextContent('Model checksum verification failed');
    expect(screen.queryByText('Console')).not.toBeInTheDocument();
  });

  it('stops polling and cancels in-flight status requests while hidden', async () => {
    vi.useFakeTimers();
    const hidden = vi.spyOn(document, 'hidden', 'get').mockReturnValue(false);
    const fetch = vi.fn().mockResolvedValue(status('downloading'));
    vi.stubGlobal('fetch', fetch);
    await act(async () => { render(<StartupGate><p>Console</p></StartupGate>); await Promise.resolve(); });
    hidden.mockReturnValue(true);
    await act(async () => { document.dispatchEvent(new Event('visibilitychange')); await vi.advanceTimersByTimeAsync(5000); });
    expect(fetch).toHaveBeenCalledTimes(1);
    expect(fetch.mock.calls[0]![1].signal.aborted).toBe(true);
    hidden.mockReturnValue(false);
    await act(async () => { document.dispatchEvent(new Event('visibilitychange')); await Promise.resolve(); });
    expect(fetch).toHaveBeenCalledTimes(2);
  });

  it('reports invalid progress instead of inventing a percentage', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(status('downloading', 101)));
    await act(async () => { render(<StartupGate><p>Console</p></StartupGate>); await Promise.resolve(); });
    expect(screen.getByRole('alert')).toHaveTextContent('Startup status is invalid');
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
  });
});
