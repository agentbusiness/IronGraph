import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

vi.mock('./hooks/useProjects', () => ({
  useProjects: () => ({ projects: [], loading: false, select: vi.fn(), refresh: vi.fn(() => Promise.resolve()), create: vi.fn() }),
}));
vi.mock('./hooks/useTheme', () => ({ useTheme: () => ['dark', vi.fn()] }));
vi.mock('./components/settings/SettingsPage', () => ({ SettingsPage: () => <h1>Settings integration manager</h1> }));
vi.mock('./components/graph/GraphPage', () => ({ GraphPage: () => <h1>Query screen</h1> }));

beforeEach(() => window.history.replaceState({}, '', '/web/'));
afterEach(cleanup);

describe('Settings navigation', () => {
  it('places Settings in the top-level console navigation', () => {
    render(<App />);
    expect(screen.getByRole('button', { name: 'Settings' })).toBeInTheDocument();
  });

  it('opens Settings and writes its route', async () => {
    render(<App />);
    fireEvent.click(screen.getByRole('button', { name: 'Settings' }));
    expect(await screen.findByRole('heading', { name: 'Settings integration manager' })).toBeInTheDocument();
    expect(window.location.pathname).toBe('/web/settings');
  });

  it('opens Settings directly from its URL', async () => {
    window.history.replaceState({}, '', '/web/settings');
    render(<App />);
    expect(await screen.findByRole('heading', { name: 'Settings integration manager' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Settings' })).toHaveAttribute('aria-pressed', 'true');
  });
});
