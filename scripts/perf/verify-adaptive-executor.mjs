#!/usr/bin/env node

import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const mode = process.argv[2];

function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, encoding: 'utf8', stdio: 'inherit' });
  if (result.status !== 0) process.exit(result.status ?? 1);
}

function source(relative) {
  return fs.readFileSync(path.join(root, relative), 'utf8');
}

function requireCondition(condition, message) {
  if (!condition) throw new Error(message);
}

switch (mode) {
  case 'routing': {
    run('cargo', [
      'test',
      '-p',
      'irongraph-cypher',
      'adaptive_execution_policy',
      '--',
      '--nocapture',
    ]);
    const executor = source('crates/cypher/src/executor.rs');
    requireCondition(executor.includes('"graph.degree" => nodes >= 10_000'), 'degree crossover missing');
    requireCondition(executor.includes('nodes >= 100_000'), '100k graph crossover missing');
    requireCondition(
      executor.includes('_ => false'),
      'unmeasured sizes must retain the last measured CPU winner',
    );
    console.log('adaptive routing verification passed');
    break;
  }
  case 'cpu-only': {
    run('cargo', [
      'test',
      '--test',
      'gpu',
      '--features',
      'accelerator',
      'cpu_sparse_algorithm_contract_covers_parallel_self_loop_disconnected_and_layers',
      '--',
      '--nocapture',
    ]);
    const executor = source('crates/cypher/src/executor.rs');
    requireCondition(
      executor.includes('backend.kind() == crate::execution::BackendKind::Metal'),
      'adaptive override is not restricted to an active Metal backend',
    );
    requireCondition(!executor.includes('CpuBackend::new'), 'query routing constructs a duplicate CPU resident backend');
    console.log('cpu-only fallback verification passed');
    break;
  }
  case 'semantics': {
    run('cargo', [
      'test',
      '-p',
      'irongraph-cypher',
      'algorithm_adjacency_dense_mapping_tracks_rows_not_dirty_property_bytes',
      '--',
      '--nocapture',
    ]);
    run('cargo', [
      'test',
      '--test',
      'gpu',
      '--features',
      'accelerator',
      'real_metal_all_graph_algorithms_match_cpu_on_adversarial_sparse_graph',
      '--',
      '--ignored',
      '--nocapture',
    ]);
    const executor = source('crates/cypher/src/executor.rs');
    const view = source('crates/cypher/src/view.rs');
    requireCondition(!executor.includes('CpuBackend::new'), 'adaptive path duplicates the resident graph');
    requireCondition(view.includes('let mut ordinal_by_dense = vec![u32::MAX; node_capacity]'), 'dense O(N) ordinal map missing');
    requireCondition(!view.includes('collect::<Result<BTreeMap<_, _>>>()?'), 'tree ordinal rebuild remains');
    console.log('adaptive semantics verification passed');
    break;
  }
  case 'integration': {
    run('cargo', ['check', '-p', 'irongraph-graph']);
    run('cargo', ['check', '-p', 'irongraph-cypher']);
    run('cargo', ['check', '--bench', 'performance_matrix', '--features', 'accelerator']);
    run('cargo', ['test', '-p', 'irongraph-cypher', 'plan_cache', '--', '--nocapture']);
    run('cargo', [
      'test',
      '--test',
      'gpu',
      '--features',
      'accelerator',
      'active_accelerator_dispatches_every_graph_algorithm_without_host_fallback',
      '--',
      '--nocapture',
    ]);
    console.log('adaptive integration verification passed');
    break;
  }
  case 'cypher-suite': {
    const reportPath = path.join(root, 'performance-results/adaptive-routing/opencypher-tck-report.json');
    const timingPath = path.join(root, 'performance-results/adaptive-routing/opencypher-tck-timing.json');
    requireCondition(fs.existsSync(reportPath), 'full TCK report is missing');
    requireCondition(fs.existsSync(timingPath), 'full TCK timing is missing');
    const report = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
    const timing = JSON.parse(fs.readFileSync(timingPath, 'utf8'));
    for (const field of ['total', 'cpu_passed', 'metal_passed', 'matched', 'fully_conformant']) {
      requireCondition(report[field] === 3897, `${field}=${report[field]} instead of 3897`);
    }
    requireCondition(report.scenarios?.length === 3897, 'per-scenario report is incomplete');
    requireCondition(Number.isFinite(timing.elapsed_seconds) && timing.elapsed_seconds > 0, 'elapsed_seconds is invalid');
    requireCondition(timing.completed_unix_millis > 0, 'completion timestamp is invalid');
    console.log('full Cypher compatibility verification passed');
    break;
  }
  default:
    console.error('usage: verify-adaptive-executor.mjs routing|cpu-only|semantics|integration|cypher-suite');
    process.exit(2);
}
