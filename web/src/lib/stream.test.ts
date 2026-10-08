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
  it('preserves a large dirty value across UTF-8 byte boundaries and a final event without newline', async () => {
    const body = '東京 🛫 complete owner content\n'.repeat(16384);
    const bytes = new TextEncoder().encode(JSON.stringify({ type: 'batch', body }) + '\r\n{"type":"summary"}');
    let offset = 0;
    const response = new Response(new ReadableStream({
      pull(controller) {
        if (offset === bytes.length) { controller.close(); return; }
        const end = Math.min(offset + 257, bytes.length);
        controller.enqueue(bytes.subarray(offset, end));
        offset = end;
      },
    }), { headers: { 'content-type': 'application/x-ndjson' } });
    const events: unknown[] = [];
    for await (const event of decodeResponse(response)) events.push(event.data);
    expect(events).toEqual([{ type: 'batch', body }, { type: 'summary' }]);
  });

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
