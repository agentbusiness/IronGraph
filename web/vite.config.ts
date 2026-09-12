import react from '@vitejs/plugin-react';
import type { ProxyOptions } from 'vite';
import { defineConfig } from 'vitest/config';
import { localHttpProxyGuard } from './localHttpProxy';

/** The graph-data endpoint plus the loopback-only Settings control plane. */
const API_PREFIXES = ['/api', '/system/local-ai-integrations'];

/**
 * In production the server embeds and serves `dist`, so the client speaks in relative paths. The
 * dev server keeps that contract by forwarding those same paths to a locally running server —
 * except browser navigations (Accept: text/html), which are the SPA's own pages.
 *
 * `IRONGRAPH_SERVER` points the proxy at a differently-addressed server, so a second dev stack can run
 * beside the usual one without either touching the other's ports.
 */
const target = process.env.IRONGRAPH_SERVER ?? 'http://127.0.0.1:18484';

const apiProxy: ProxyOptions = {
  target,
  // The server accepts only hosts it was bound to; keep the upstream's own Host header.
  changeOrigin: true,
  // Its cross-origin guard would also refuse the dev origin, and rightly — so the proxied request
  // carries the server's own origin, the same claim a same-origin browser request makes.
  headers: { origin: target },
  bypass: (req) => {
    const accept = req.headers.accept;
    const wantsHtml = typeof accept === 'string' && accept.includes('text/html');
    return wantsHtml ? '/index.html' : undefined;
  },
};

export default defineConfig({
  base: '/web/',
  plugins: [localHttpProxyGuard(), react()],
  server: {
    host: '127.0.0.1',
    port: 18489,
    strictPort: true,
    proxy: Object.fromEntries(API_PREFIXES.map((prefix) => [prefix, apiProxy])),
    // The Docs screen compiles the public Markdown one directory above the web package.
    fs: { allow: ['..'] },
  },
  build: {
    target: 'es2022',
    sourcemap: false,
    cssCodeSplit: true,
    reportCompressedSize: true,
    chunkSizeWarningLimit: 800,
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    coverage: { reporter: ['text', 'json', 'html'] },
  },
});
