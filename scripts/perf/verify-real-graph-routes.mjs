#!/usr/bin/env node

import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';

const [reportPath, mode] = process.argv.slice(2);
if (!reportPath || !mode) throw new Error('usage: verify-real-graph-routes.mjs REPORT MODE');
const report = JSON.parse(readFileSync(reportPath, 'utf8'));
const operations = report.operations ?? [];

function invariant(condition, message) {
  if (!condition) throw new Error(message);
}

function transportOperations(transport) {
  return operations.filter((operation) => operation.transport === transport);
}

function operationFamily(name) {
  if (name.startsWith('write_nodes_')) return 'write_nodes';
  if (name.startsWith('write_edges_')) return 'write_edges';
  return name;
}

function verifyTransport(transport) {
  const rows = transportOperations(transport);
  const families = new Set(rows.map((operation) => operationFamily(operation.operation)));
  for (const required of ['write_nodes', 'write_edges', 'point_read', 'traversal', 'aggregate', 'louvain']) {
    invariant(families.has(required), `${transport} is missing ${required}`);
  }
  invariant(rows.every((operation) => operation.wall_us > 0), `${transport} has an invalid timing`);
}

if (mode === 'icij-louvain') {
  invariant(report.icij_louvain?.assignments === 2_017_662, 'ICIJ assignment count is wrong');
  invariant(report.icij_louvain?.communities === 1_281_802, 'ICIJ community count is wrong');
  invariant(report.icij_louvain.samples.length === 3, 'ICIJ requires three samples');
  invariant(Math.max(...report.icij_louvain.samples) < 120_000_000, 'ICIJ exceeded deadline');
  invariant(report.metal_capable_dynamic_icij?.configured_backend === 'metal', 'Metal-capable route evidence is absent');
  invariant(report.metal_capable_dynamic_icij?.selected_executor === 'cpu', 'sparse ICIJ selected the wrong executor');
  invariant(report.metal_capable_dynamic_icij?.assignments === 2_017_662, 'dynamic ICIJ assignment count is wrong');
  console.log('ICIJ Louvain verification passed');
} else if (mode === 'query-api') {
  verifyTransport('http_query');
  invariant(report.validation?.node_count === 20_000 && report.validation?.edge_count === 19_998, 'route cardinality is wrong');
  console.log('query API real-graph verification passed');
} else if (mode === 'bolt') {
  invariant(report.neo4j_driver_version === '6.2.0', 'official Neo4j driver version is absent');
  verifyTransport('neo4j_driver_bolt');
  invariant(report.validation?.node_count === 20_000 && report.validation?.edge_count === 19_998, 'route cardinality is wrong');
  console.log('Bolt real-graph verification passed');
} else if (mode === 'semantics') {
  for (const args of [
    ['test', '-p', 'irongraph-graph', '--release', 'louvain', '--', '--nocapture'],
    ['test', '-p', 'irongraph-cypher', '--release', 'adaptive_execution_policy_keeps_only_measured_graph_procedure_wins_on_metal', '--', '--nocapture'],
    ['test', '--test', 'gpu', '--release', '--features', 'metal', 'real_metal_components_metrics_and_louvain_match_cpu_independently', '--', '--ignored', '--nocapture'],
  ]) {
    const result = spawnSync('cargo', args, { stdio: 'inherit' });
    invariant(result.status === 0, `cargo ${args.join(' ')} failed`);
  }
  console.log('Louvain semantics verification passed');
} else if (mode === 'tck') {
  const tck = JSON.parse(readFileSync('performance-results/real-graph-datasets/tck-final.json', 'utf8'));
  for (const field of ['total', 'cpu_passed', 'metal_passed', 'matched', 'fully_conformant']) {
    invariant(tck[field] === 3_897, `TCK ${field} is ${tck[field]}`);
  }
  invariant(tck.scenarios?.length === 3_897, 'TCK scenario evidence is incomplete');
  console.log('full Cypher compatibility verification passed');
} else {
  throw new Error(`unknown verification mode ${mode}`);
}
