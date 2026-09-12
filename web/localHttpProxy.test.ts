import { describe, expect, it } from 'vitest';
import type { IncomingMessage } from 'node:http';
import { createServer as createHttpServer } from 'node:http';
import type { AddressInfo } from 'node:net';
import { createServer, preview } from 'vite';
import { localHttpProxyGuard, localProxyRequestAllowed } from './localHttpProxy';

function request(host: string, origin?: string, url = '/system/local-ai-integrations/cline/install'): IncomingMessage {
  return {
    method: 'POST', url, headers: { host, ...(origin === undefined ? {} : { origin }) },
    rawHeaders: ['Host', host, ...(origin === undefined ? [] : ['Origin', origin])],
    socket: { localPort: 18489 },
  } as IncomingMessage;
}

describe('local development proxy authority', () => {
  it('preserves same-origin bodyless Settings and native Query requests', () => {
    expect(localProxyRequestAllowed(request('127.0.0.1:18489', 'http://127.0.0.1:18489'))).toBe(true);
    expect(localProxyRequestAllowed(request('localhost:18489', 'http://localhost:18489'))).toBe(true);
    expect(localProxyRequestAllowed(request('127.0.0.1:18489', undefined, '/api/query'))).toBe(true);
  });
  it('rejects foreign, missing, null, wrong-port and malformed origins before rewriting', () => {
    for (const origin of [undefined, 'null', 'http://foreign.example', 'http://127.0.0.1:18484', 'https://127.0.0.1:18489', 'http://127.0.0.1:18489/path']) {
      expect(localProxyRequestAllowed(request('127.0.0.1:18489', origin))).toBe(false);
    }
    for (const host of ['foreign.example:18489', 'localhost.foreign.example:18489', 'localhost:18484', 'user@localhost:18489']) {
      expect(localProxyRequestAllowed(request(host, 'http://' + host))).toBe(false);
    }
  });
  it('rejects ambiguous duplicate headers', () => {
    for (const header of ['Host', 'Origin']) {
      const req = request('localhost:18489', 'http://localhost:18489');
      req.rawHeaders.push(header, header === 'Host' ? 'localhost:18489' : 'http://localhost:18489');
      expect(localProxyRequestAllowed(req)).toBe(false);
    }
  });
  for (const mode of ['development', 'preview'] as const) {
    it(`${mode} rejects requests before the proxy can rewrite their origin`, async () => {
      let calls = 0;
      const upstream = createHttpServer((_req, res) => { calls++; res.statusCode = 204; res.end(); });
      await new Promise<void>((resolve) => upstream.listen(0, '127.0.0.1', resolve));
      const target = `http://127.0.0.1:${(upstream.address() as AddressInfo).port}`;
      const proxy = { '/system/local-ai-integrations': { target, changeOrigin: true, headers: { origin: target } } };
      const config = { configFile: false as const, plugins: [localHttpProxyGuard()], server: { host: '127.0.0.1', port: 0, proxy }, preview: { host: '127.0.0.1', port: 0, proxy } };
      const server = mode === 'development' ? await createServer(config) : await preview(config);
      try {
        if ('listen' in server) await server.listen();
        const origin = `http://127.0.0.1:${(server.httpServer!.address() as AddressInfo).port}`;
        const endpoint = origin + '/system/local-ai-integrations/cline/install';
        for (const value of [undefined, 'http://foreign.example', 'null']) {
          const headers: Record<string, string> = value === undefined ? {} : { Origin: value };
          expect((await fetch(endpoint, { method: 'POST', headers })).status).toBe(403);
        }
        expect(calls).toBe(0);
        expect((await fetch(endpoint, { method: 'POST', headers: { Origin: origin } })).status).toBe(204);
        expect(calls).toBe(1);
      } finally {
        await server.close();
        await new Promise<void>((resolve, reject) => upstream.close(error => error ? reject(error) : resolve()));
      }
    });
  }
});
