import { afterEach, describe, expect, it, vi } from 'vitest';
import { fetchLocalAiIntegrations, runQuery } from './api';
import { QueryResultCollector } from './queryResult';

afterEach(() => vi.unstubAllGlobals());

describe('relationship evidence paths', () => {
  it('decodes bare protocol path entities and makes their real endpoints available to the plot', async () => {
    const columns = [{ name: 'evidence', values: [{ type: 'path', value: {
      nodes: [
        { id: '1', labels: ['Document'], properties: { body: { type: 'string', value: 'E104 recovery' } } },
        { id: '2', labels: ['Person'], properties: { name: { type: 'string', value: 'Ada' } } },
      ],
      relationships: [{ id: '1', source: '1', target: '2', relationship_type: 'OWNED_BY',
        properties: { description: { type: 'string', value: 'handles recovery' } } }],
    } }] }];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(
      `${JSON.stringify({ type: 'batch', row_count: 1, columns })}\n`,
      { status: 200, headers: { 'content-type': 'application/x-ndjson' } },
    )));
    const collector = new QueryResultCollector();
    for await (const event of runQuery({ query: 'RETURN evidence' })) collector.append(event);
    const result = collector.finish();
    expect(result.nodes.map((node) => node.id)).toEqual(['1', '2']);
    expect(result.nodes[0]?.properties.body).toBe('E104 recovery');
    expect(result.edges).toEqual([{ id: '1', source: '1', target: '2',
      relationshipType: 'OWNED_BY', properties: { description: 'handles recovery' } }]);
    expect(result.rows).toHaveLength(1);
  });
});

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
