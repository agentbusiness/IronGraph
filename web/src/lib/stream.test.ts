import { describe, expect, it } from 'vitest';
import { decodeResponse } from './stream';

function splitResponse(parts: string[], status = 200): Response {
  const encoder = new TextEncoder();
  return new Response(new ReadableStream({
    start(controller) {
      parts.forEach((part) => controller.enqueue(encoder.encode(part)));
      controller.close();
    },
  }), { status, headers: { 'content-type': 'application/x-ndjson' } });
}

describe('NDJSON decoder', () => {
  it('preserves events split across arbitrary transport chunks', async () => {
    const response = splitResponse(['{"type":"sche', 'ma","columns":[]}\n{"type":"sum', 'mary","truncated":false}\n']);
    const events: unknown[] = [];
    for await (const envelope of decodeResponse(response)) events.push(envelope.data);
    expect(events).toEqual([
      { type: 'schema', columns: [] },
      { type: 'summary', truncated: false },
    ]);
  });

  it('surfaces HTTP errors before attempting stream parsing', async () => {
    const response = new Response(JSON.stringify({ message: 'Project not found' }), {
      status: 404,
      headers: { 'content-type': 'application/json' },
    });
    const consume = async () => {
      for await (const envelope of decodeResponse(response)) void envelope;
    };
    await expect(consume()).rejects.toThrow('Project not found');
  });
});
