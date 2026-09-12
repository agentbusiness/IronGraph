import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchLocalAiIntegrations } from './api';

afterEach(() => vi.unstubAllGlobals());

describe('local AI integration responses', () => {
  it('preserves host-specific activation instructions', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify([{
      host: 'unsloth',
      display_name: 'Unsloth Studio',
      integration: 'Streamable HTTP MCP import',
      detected: true,
      state: 'not-installed',
      available_version: '0.1.0',
      activation_required: true,
      activation_instruction: 'Import the generated config in Unsloth Studio.',
      install_command: '',
      update_command: '',
    }]), { status: 200, headers: { 'content-type': 'application/json' } })));

    const integrations = await fetchLocalAiIntegrations();

    expect(integrations[0]?.activation_instruction).toBe('Import the generated config in Unsloth Studio.');
  });
});
