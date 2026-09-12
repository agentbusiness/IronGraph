import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { SettingsPage } from './SettingsPage';
import { changeLocalAiIntegration, fetchLocalAiIntegrations } from '../../lib/api';
import type { LocalAiIntegration } from '../../types';

vi.mock('../../lib/api', () => ({
  fetchLocalAiIntegrations: vi.fn(),
  changeLocalAiIntegration: vi.fn(),
}));

const integrations: LocalAiIntegration[] = [
  { host: 'cursor', display_name: 'Cursor', integration: 'Agent Plugin', detected: false, state: 'not-installed', available_version: '0.1.0', activation_required: false, install_command: '', update_command: '' },
  { host: 'hermes', display_name: 'Hermes Agent', integration: 'Agent Plugin', detected: true, state: 'not-installed', available_version: '0.1.0', activation_required: true, install_command: '', update_command: '' },
  { host: 'pi', display_name: 'Pi', integration: 'MCP extension + Agent Skill', detected: true, state: 'repair-required', installed_version: '0.1.0', available_version: '0.1.0', activation_required: true, install_command: '', update_command: '' },
  { host: 'openclaw', display_name: 'OpenClaw', integration: 'Agent Plugin', detected: true, state: 'update-available', installed_version: '0.0.9', available_version: '0.1.0', activation_required: true, install_command: '', update_command: '' },
  { host: 'unsloth', display_name: 'Unsloth Studio', integration: 'Streamable HTTP MCP import', detected: true, state: 'not-installed', available_version: '0.1.0', activation_required: true, activation_instruction: 'Import the generated config in Unsloth Studio.', install_command: '', update_command: '' },
];

beforeEach(() => {
  vi.mocked(fetchLocalAiIntegrations).mockResolvedValue(integrations);
  vi.mocked(changeLocalAiIntegration).mockResolvedValue(integrations);
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('SettingsPage', () => {
  it('detects hosts and presents install, update, and repair actions', async () => {
    render(<SettingsPage />);
    expect(await screen.findByRole('heading', { name: 'Use IronGraph from your assistants' })).toBeInTheDocument();
    expect(screen.getAllByRole('button', { name: 'Install' })).toHaveLength(2);
    expect(screen.getByRole('button', { name: 'Update' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Repair' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Not found' })).toBeDisabled();
  });

  it('offers the native Unsloth Studio MCP import integration', async () => {
    render(<SettingsPage />);
    const name = await screen.findByText('Unsloth Studio');
    expect(name).toBeInTheDocument();
    expect(screen.getByText('Streamable HTTP MCP import')).toBeInTheDocument();
    const row = name.closest('article');
    expect(row).not.toBeNull();
    fireEvent.click(within(row!).getByRole('button', { name: 'Install' }));
    expect(await screen.findByRole('status')).toHaveTextContent('Import the generated config in Unsloth Studio.');
  });

  it('runs repair through the local integration API', async () => {
    render(<SettingsPage />);
    fireEvent.click(await screen.findByRole('button', { name: 'Repair' }));
    await waitFor(() => expect(changeLocalAiIntegration).toHaveBeenCalledWith('pi', 'repair'));
    expect(await screen.findByRole('status')).toHaveTextContent('Pi: installation repaired');
  });

  it('re-detects installed applications on demand', async () => {
    render(<SettingsPage />);
    fireEvent.click(await screen.findByRole('button', { name: 'Detect again' }));
    await waitFor(() => expect(fetchLocalAiIntegrations).toHaveBeenCalledTimes(2));
  });

  it('keeps model-runtime explanations out of the settings interface', async () => {
    render(<SettingsPage />);
    await screen.findByRole('heading', { name: 'Use IronGraph from your assistants' });
    expect(screen.queryByText(/Ollama and Unsloth/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/Install IronGraph into the detected harness/i)).not.toBeInTheDocument();
  });
});
