export interface StreamEnvelope {
  event?: string;
  data: unknown;
}

function parseJson(value: string): unknown {
  try {
    return JSON.parse(value) as unknown;
  } catch {
    throw new Error('The server returned a malformed streaming event.');
  }
}

function decodeBlock(block: string): StreamEnvelope[] {
  const trimmed = block.trim();
  if (!trimmed) return [];

  if (trimmed.startsWith('{') || trimmed.startsWith('[')) return [{ data: parseJson(trimmed) }];

  let event: string | undefined;
  const data: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith(':')) continue;
    const separator = line.indexOf(':');
    const field = separator < 0 ? line : line.slice(0, separator);
    const value = separator < 0 ? '' : line.slice(separator + 1).replace(/^ /, '');
    if (field === 'event') event = value;
    if (field === 'data') data.push(value);
  }
  if (data.length === 0) return [];
  return [{ event, data: parseJson(data.join('\n')) }];
}

export async function* decodeResponse(response: Response): AsyncGenerator<StreamEnvelope> {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`.trim();
    try {
      const payload = (await response.json()) as { message?: string; error?: { message?: string } };
      detail = payload.message ?? payload.error?.message ?? detail;
    } catch {
      // The stable HTTP status remains useful when the body is not JSON.
    }
    throw new Error(detail || 'Request failed.');
  }

  const contentType = response.headers.get('content-type')?.toLowerCase() ?? '';
  if (!response.body) {
    const payload = (await response.json()) as unknown;
    yield { data: payload };
    return;
  }

  if (contentType.includes('application/json') && !contentType.includes('ndjson')) {
    yield { data: (await response.json()) as unknown };
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let fragments: string[] = [];
  const isSse = contentType.includes('text/event-stream');

  try {
    while (true) {
      const { done, value } = await reader.read();
      const chunk = decoder.decode(value, { stream: !done });

      if (isSse) {
        buffer += chunk;
        let boundary = buffer.search(/\r?\n\r?\n/);
        while (boundary >= 0) {
          const block = buffer.slice(0, boundary);
          const match = /\r?\n\r?\n/.exec(buffer.slice(boundary));
          const boundaryLength = match?.[0].length ?? 2;
          buffer = buffer.slice(boundary + boundaryLength);
          for (const envelope of decodeBlock(block)) yield envelope;
          boundary = buffer.search(/\r?\n\r?\n/);
        }
      } else {
        let start = 0;
        for (let newline = chunk.indexOf('\n'); newline !== -1; newline = chunk.indexOf('\n', start)) {
          fragments.push(chunk.slice(start, newline));
          const line = (fragments.length === 1 ? fragments[0]! : fragments.join('')).trim();
          fragments = [];
          start = newline + 1;
          if (line) yield { data: parseJson(line) };
        }
        if (start < chunk.length) fragments.push(chunk.slice(start));
      }

      if (done) break;
    }
    if (!isSse) buffer = fragments.join('');
    if (buffer.trim()) {
      for (const envelope of decodeBlock(buffer)) yield envelope;
    }
  } finally {
    reader.releaseLock();
  }
}

export function envelopeType(envelope: StreamEnvelope): string | undefined {
  if (envelope.event) return envelope.event;
  if (typeof envelope.data === 'object' && envelope.data !== null && 'type' in envelope.data) {
    const type = (envelope.data as { type?: unknown }).type;
    return typeof type === 'string' ? type : undefined;
  }
  return undefined;
}
