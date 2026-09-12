import type { IncomingMessage } from 'node:http';
import type { Connect, Plugin } from 'vite';

/** Check the browser's original authority before the proxy rewrites it for the backend. */
export function localProxyRequestAllowed(req: IncomingMessage): boolean {
  const host = req.headers.host;
  if (!host || !/^(localhost|127\.0\.0\.1):[0-9]+$/i.test(host)) return false;
  if (req.rawHeaders.filter((_, i) => i % 2 === 0 && req.rawHeaders[i].toLowerCase() === 'host').length !== 1) return false;
  const expected = new URL('http://' + host);
  if (Number(expected.port || 80) !== req.socket.localPort) return false;
  const origin = req.headers.origin;
  if (origin === undefined) {
    return !(req.method === 'POST' && req.url?.startsWith('/system/local-ai-integrations/'));
  }
  if (req.rawHeaders.filter((_, i) => i % 2 === 0 && req.rawHeaders[i].toLowerCase() === 'origin').length !== 1) return false;
  return origin.toLowerCase() === expected.origin;
}

export function localHttpProxyGuard(): Plugin {
  const guard: Connect.NextHandleFunction = (req, res, next) => {
    if (!req.url?.startsWith('/api/') && !req.url?.startsWith('/system/local-ai-integrations')) return next();
    let allowed = false;
    try { allowed = localProxyRequestAllowed(req); } catch { /* Malformed authority. */ }
    if (!allowed) {
      res.statusCode = 403;
      res.end('Local HTTP authority or origin is not allowed');
      return;
    }
    next();
  };
  return {
    name: 'local-http-proxy-guard',
    configureServer(server) {
      server.middlewares.use(guard);
    },
    configurePreviewServer(server) {
      server.middlewares.use(guard);
    },
  };
}
