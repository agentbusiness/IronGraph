#!/usr/bin/env node

import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const mode = process.argv[2];

function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, encoding: 'utf8', stdio: 'inherit' });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

if (mode === 'libraries') {
  run('cargo', ['test', '-p', 'irongraph-graph']);
  run('cargo', ['test', '-p', 'irongraph-cypher']);
  console.log('first-run adjacency library verification passed');
} else if (mode === 'integration') {
  run('cargo', ['check', '-p', 'irongraph-server', '--features', 'accelerator']);
  run('node', ['scripts/perf/verify-adaptive-executor.mjs', 'cypher-suite']);
  console.log('first-run adjacency integration verification passed');
} else {
  console.error('usage: verify-first-run-adjacency.mjs libraries|integration');
  process.exit(2);
}
