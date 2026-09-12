#!/usr/bin/env node

import { spawnSync } from "node:child_process";

const checks = [
  ["cargo", ["test", "-p", "irongraph-embedding"]],
  ["cargo", ["test", "-p", "irongraph-graph"]],
  ["cargo", ["test", "-p", "irongraph-execution"]],
  ["cargo", ["test", "-p", "irongraph-storage"]],
  ["cargo", ["test", "-p", "irongraph-cypher"]],
  ["cargo", ["test", "-p", "irongraph-gpu"]],
  ["cargo", ["test", "-p", "irongraph-server"]],
  ["npm", ["--prefix", "web", "run", "build"]],
  ["npm", ["--prefix", "web", "test"]],
];

for (const [program, args] of checks) {
  const label = [program, ...args].join(" ");
  process.stdout.write(`\n[regression] ${label}\n`);
  const result = spawnSync(program, args, {
    cwd: process.cwd(),
    env: process.env,
    stdio: "inherit",
  });
  if (result.error) {
    throw new Error(`${label}: ${result.error.message}`);
  }
  if (result.status !== 0) {
    process.stderr.write(`[regression] failed: ${label}\n`);
    process.exit(result.status ?? 1);
  }
}

process.stdout.write("\nintegrated regression verification passed\n");
