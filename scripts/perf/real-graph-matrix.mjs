#!/usr/bin/env node

/**
 * Import licensed real-world graph data through IronGraph's public Cypher endpoint and record
 * correctness/performance measurements. The adapters only decode source formats; every durable
 * effect is an ordinary Cypher write, exercising the same path as a real client.
 *
 * Usage:
 *   node scripts/perf/real-graph-matrix.mjs \
 *     --dataset icij|openalex|osm \
 *     --source /absolute/path \
 *     --server http://127.0.0.1:18484 \
 *     --output /absolute/path/results.json
 */

import { createHash, randomUUID } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, stat, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { pipeline } from 'node:stream/promises';
import { createGunzip } from 'node:zlib';
import { DatabaseSync } from 'node:sqlite';
import { performance } from 'node:perf_hooks';

const DATASETS = new Set(['icij', 'openalex', 'osm']);
const DEFAULT_BATCH_SIZE = 1_000;

function parseArgs(argv) {
  const values = new Map();
  for (let index = 0; index < argv.length; index += 2) {
    const flag = argv[index];
    const value = argv[index + 1];
    if (!flag?.startsWith('--') || value === undefined) {
      throw new Error(`expected --name value, got ${JSON.stringify(argv.slice(index))}`);
    }
    values.set(flag.slice(2), value);
  }
  const dataset = values.get('dataset');
  const source = values.get('source');
  const output = values.get('output');
  if (!DATASETS.has(dataset) || !source || !output) {
    throw new Error('--dataset icij|openalex|osm, --source, and --output are required');
  }
  const batchSize = Number(values.get('batch-size') ?? DEFAULT_BATCH_SIZE);
  if (!Number.isSafeInteger(batchSize) || batchSize < 1 || batchSize > 5_000) {
    throw new Error('--batch-size must be an integer in 1..5000');
  }
  return {
    dataset,
    source: resolve(source),
    output: resolve(output),
    server: new URL(values.get('server') ?? 'http://127.0.0.1:18484'),
    projectName: values.get('project') ?? `real-${dataset}-${Date.now()}`,
    batchSize,
    maxNodes: nonnegativeInteger(values.get('max-nodes') ?? '0', '--max-nodes'),
    maxEdges: nonnegativeInteger(values.get('max-edges') ?? '0', '--max-edges'),
  };
}

function nonnegativeInteger(value, flag) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0) throw new Error(`${flag} must be non-negative`);
  return parsed;
}

function quoteIdentifier(value) {
  return `\`${value.replaceAll('`', '``')}\``;
}

function typedValue(value) {
  if (!value || typeof value !== 'object' || typeof value.type !== 'string') return value;
  if (value.type === 'null') return null;
  if (!Object.hasOwn(value, 'value')) return value;
  if (value.type === 'integer') return Number(value.value);
  if (value.type === 'float') return Number(value.value);
  if (value.type === 'list') return Array.isArray(value.value) ? value.value.map(typedValue) : [];
  if (value.type === 'map' && value.value && typeof value.value === 'object') {
    return Object.fromEntries(Object.entries(value.value).map(([key, nested]) => [key, typedValue(nested)]));
  }
  return value.value;
}

function rowsFromEvents(events) {
  const rows = [];
  for (const event of events) {
    if (event.type !== 'batch' || !Array.isArray(event.columns)) continue;
    const count = Number(event.row_count ?? event.columns[0]?.values?.length ?? 0);
    for (let row = 0; row < count; row += 1) {
      rows.push(Object.fromEntries(event.columns.map((column) => [column.name, typedValue(column.values[row])])));
    }
  }
  return rows;
}

async function query(config, projectId, cypher, parameters = {}) {
  const started = performance.now();
  const response = await fetch(new URL('/api/query', config.server), {
    method: 'POST',
    headers: { Accept: 'application/x-ndjson', 'Content-Type': 'application/json' },
    body: JSON.stringify({
      request_id: randomUUID(),
      project_id: projectId,
      query: cypher,
      parameters,
      consistency: 'PUBLISHED',
      bookmark: null,
      limits: {},
    }),
  });
  const body = await response.text();
  if (!response.ok) throw new Error(`query HTTP ${response.status}: ${body.slice(0, 1_000)}`);
  const events = body
    .split('\n')
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  const failure = events.find((event) => event.type === 'error');
  if (failure) throw new Error(`${failure.code ?? 'QueryError'}: ${failure.message ?? 'query failed'}`);
  const summary = events.findLast((event) => event.type === 'summary');
  return {
    elapsed_ms: performance.now() - started,
    server_elapsed_us: Number(summary?.statistics?.elapsed_us ?? 0),
    statistics: summary?.statistics ?? {},
    rows: rowsFromEvents(events),
  };
}

async function createProject(config) {
  await query(config, null, `CREATE PROJECT ${quoteIdentifier(config.projectName)}`);
  const projects = await query(config, null, 'SHOW PROJECTS');
  const project = projects.rows.find((row) => row.display_name === config.projectName);
  if (!project?.project_id) throw new Error(`created project ${config.projectName} was not listed`);
  return String(project.project_id);
}

async function sha256(path) {
  const hash = createHash('sha256');
  await pipeline(createReadStream(path), hash);
  return hash.digest('hex');
}

async function sourceManifest(path) {
  const metadata = await stat(path);
  return { path, bytes: metadata.size, sha256: await sha256(path) };
}

async function writeReport(report, output) {
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
}

async function importBatches(config, report, projectId, stage, cypher, iterable) {
  const measurement = { stage, rows: 0, requests: 0, elapsed_ms: 0, rows_per_second: 0 };
  report.ingestion.push(measurement);
  const started = performance.now();
  let batch = [];
  for await (const row of iterable) {
    batch.push(row);
    if (batch.length < config.batchSize) continue;
    await query(config, projectId, cypher, { rows: batch });
    measurement.rows += batch.length;
    measurement.requests += 1;
    batch = [];
    if (measurement.requests % 50 === 0) await writeReport(report, config.output);
  }
  if (batch.length > 0) {
    await query(config, projectId, cypher, { rows: batch });
    measurement.rows += batch.length;
    measurement.requests += 1;
  }
  measurement.elapsed_ms = performance.now() - started;
  measurement.rows_per_second = measurement.rows / (measurement.elapsed_ms / 1_000);
  await writeReport(report, config.output);
  return measurement;
}

async function* csvRows(path) {
  const input = createReadStream(path, { encoding: 'utf8' });
  let row = [];
  let field = '';
  let quoted = false;
  let afterQuote = false;
  for await (const chunk of input) {
    for (const character of chunk) {
      if (quoted) {
        if (character === '"') {
          quoted = false;
          afterQuote = true;
        } else {
          field += character;
        }
        continue;
      }
      if (afterQuote) {
        if (character === '"') {
          field += '"';
          quoted = true;
          afterQuote = false;
          continue;
        }
        afterQuote = false;
      }
      if (character === '"' && field.length === 0) {
        quoted = true;
      } else if (character === ',') {
        row.push(field);
        field = '';
      } else if (character === '\n') {
        row.push(field);
        yield row;
        row = [];
        field = '';
      } else if (character !== '\r') {
        field += character;
      }
    }
  }
  if (quoted) throw new Error(`unterminated CSV quote in ${path}`);
  if (field.length > 0 || row.length > 0) {
    row.push(field);
    yield row;
  }
}

async function* csvObjects(path) {
  let header;
  for await (const row of csvRows(path)) {
    if (!header) {
      header = row;
      continue;
    }
    yield Object.fromEntries(header.map((name, index) => [name, row[index] ?? '']));
  }
}

async function* bounded(iterable, maximum) {
  let count = 0;
  for await (const value of iterable) {
    if (maximum > 0 && count >= maximum) break;
    yield value;
    count += 1;
  }
}

async function importIcij(config, report, projectId) {
  const files = [
    ['entity', 'nodes-entities.csv'],
    ['officer', 'nodes-officers.csv'],
    ['address', 'nodes-addresses.csv'],
    ['intermediary', 'nodes-intermediaries.csv'],
    ['other', 'nodes-others.csv'],
  ];
  let remaining = config.maxNodes;
  for (const [fileIndex, [kind, file]] of files.entries()) {
    if (config.maxNodes > 0 && remaining === 0) break;
    const path = join(config.source, file);
    report.sources.push(await sourceManifest(path));
    const limit = config.maxNodes === 0 ? 0 : remaining;
    const source = (async function* () {
      for await (const record of bounded(csvObjects(path), limit)) {
        const id = String(record.node_id ?? '').trim();
        if (!id) continue;
        yield {
          id,
          kind,
          role: kind,
          name: String(record.name || record.address || '').slice(0, 2_048),
          jurisdiction: String(record.jurisdiction_description || record.jurisdiction || '').slice(0, 256),
          countries: String(record.countries || record.country_codes || '').slice(0, 512),
          source: String(record.sourceID || '').slice(0, 128),
        };
      }
    })();
    const measurement = await importBatches(
      config,
      report,
      projectId,
      `icij-nodes-${kind}`,
      fileIndex === 0
        ? 'UNWIND $rows AS row CREATE (:IcijNode {dataset_id: row.id, kind: row.kind, roles: row.role, name: row.name, jurisdiction: row.jurisdiction, countries: row.countries, source: row.source})'
        : "UNWIND $rows AS row MERGE (n:IcijNode {dataset_id: row.id}) ON CREATE SET n.kind = row.kind, n.roles = row.role, n.name = row.name, n.jurisdiction = row.jurisdiction, n.countries = row.countries, n.source = row.source ON MATCH SET n.roles = n.roles + '|' + row.role",
      source,
    );
    // The first published node category declares the label without a technical seed node. Build
    // the equality index once, then let later categories MERGE duplicate published IDs onto that
    // real node while preserving every category in `roles`. ICIJ repeats 1,139 IDs across role
    // files; treating those rows as separate endpoints fans one relationship row into many edges.
    if (fileIndex === 0) {
      await query(config, projectId, 'CREATE INDEX icij_id FOR (n:IcijNode) ON (n.dataset_id)');
    }
    if (config.maxNodes > 0) remaining -= measurement.rows;
  }

  const relationships = join(config.source, 'relationships.csv');
  report.sources.push(await sourceManifest(relationships));
  let firstEdge;
  const edgeRows = (async function* () {
    for await (const record of bounded(csvObjects(relationships), config.maxEdges)) {
      const row = {
        start: String(record.node_id_start ?? '').trim(),
        end: String(record.node_id_end ?? '').trim(),
        kind: String(record.rel_type || record.link || 'related').slice(0, 128),
        source: String(record.sourceID || '').slice(0, 128),
      };
      if (!row.start || !row.end) continue;
      firstEdge ??= row;
      yield row;
    }
  })();
  await importBatches(
    config,
    report,
    projectId,
    'icij-relationships',
    'UNWIND $rows AS row MATCH (a:IcijNode {dataset_id: row.start}), (b:IcijNode {dataset_id: row.end}) CREATE (a)-[:ICIJ_LINK {relationship_kind: row.kind, provenance_source: row.source}]->(b)',
    edgeRows,
  );
  return { label: 'IcijNode', idProperty: 'dataset_id', firstEdge };
}

async function* jsonLinesGzip(path) {
  const input = createReadStream(path).pipe(createGunzip());
  const lines = createInterface({ input, crlfDelay: Infinity });
  for await (const line of lines) {
    if (line.trim()) yield JSON.parse(line);
  }
}

function openAlexId(value) {
  return String(value ?? '').replace('https://openalex.org/', '');
}

async function importOpenAlex(config, report, projectId) {
  report.sources.push(await sourceManifest(config.source));

  const authors = new Map();
  const topics = new Map();
  const references = new Set();
  const authorEdges = [];
  const topicEdges = [];
  const citationEdges = [];
  const maximumWorks = config.maxNodes || Number.MAX_SAFE_INTEGER;
  const maximumEdges = config.maxEdges || Number.MAX_SAFE_INTEGER;
  let works = 0;
  const workRows = (async function* () {
    for await (const record of jsonLinesGzip(config.source)) {
      if (works >= maximumWorks) break;
      const work = openAlexId(record.id);
      if (!work) continue;
      works += 1;
      for (const authorship of record.authorships ?? []) {
        const author = openAlexId(authorship.author?.id);
        if (!author) continue;
        authors.set(author, String(authorship.author?.display_name ?? '').slice(0, 1_024));
        if (authorEdges.length < maximumEdges) authorEdges.push({ author, work });
      }
      const topic = openAlexId(record.primary_topic?.id);
      if (topic) {
        topics.set(topic, String(record.primary_topic?.display_name ?? '').slice(0, 1_024));
        if (topicEdges.length < maximumEdges) topicEdges.push({ work, topic });
      }
      for (const citedValue of record.referenced_works ?? []) {
        if (citationEdges.length >= maximumEdges) break;
        const cited = openAlexId(citedValue);
        if (!cited) continue;
        references.add(cited);
        citationEdges.push({ work, cited });
      }
      yield {
        id: work,
        title: String(record.title ?? '').slice(0, 4_096),
        year: Number(record.publication_year ?? 0),
        kind: String(record.type ?? '').slice(0, 128),
        citedBy: Number(record.cited_by_count ?? 0),
      };
    }
  })();
  await importBatches(
    config,
    report,
    projectId,
    'openalex-works',
    'UNWIND $rows AS row CREATE (:OpenAlexWork {openalex_id: row.id, title: row.title, year: row.year, kind: row.kind, cited_by_count: row.citedBy})',
    workRows,
  );
  await importBatches(
    config,
    report,
    projectId,
    'openalex-authors',
    'UNWIND $rows AS row CREATE (:OpenAlexAuthor {openalex_id: row.id, name: row.name})',
    [...authors].map(([id, name]) => ({ id, name })),
  );
  await importBatches(
    config,
    report,
    projectId,
    'openalex-topics',
    'UNWIND $rows AS row CREATE (:OpenAlexTopic {openalex_id: row.id, name: row.name})',
    [...topics].map(([id, name]) => ({ id, name })),
  );
  const knownWorks = new Set();
  for await (const record of jsonLinesGzip(config.source)) {
    const id = openAlexId(record.id);
    if (id) knownWorks.add(id);
    if (knownWorks.size >= works) break;
  }
  const externalReferences = [...references].filter((id) => !knownWorks.has(id));
  await importBatches(
    config,
    report,
    projectId,
    'openalex-reference-nodes',
    'UNWIND $rows AS row CREATE (:OpenAlexReference {openalex_id: row.id})',
    externalReferences.map((id) => ({ id })),
  );
  for (const statement of [
    'CREATE INDEX openalex_work_id FOR (n:OpenAlexWork) ON (n.openalex_id)',
    'CREATE INDEX openalex_author_id FOR (n:OpenAlexAuthor) ON (n.openalex_id)',
    'CREATE INDEX openalex_topic_id FOR (n:OpenAlexTopic) ON (n.openalex_id)',
    'CREATE INDEX openalex_reference_id FOR (n:OpenAlexReference) ON (n.openalex_id)',
  ]) await query(config, projectId, statement);
  await importBatches(
    config,
    report,
    projectId,
    'openalex-authorships',
    'UNWIND $rows AS row MATCH (a:OpenAlexAuthor {openalex_id: row.author}), (w:OpenAlexWork {openalex_id: row.work}) CREATE (a)-[:AUTHORED]->(w)',
    authorEdges,
  );
  await importBatches(
    config,
    report,
    projectId,
    'openalex-topics-edges',
    'UNWIND $rows AS row MATCH (w:OpenAlexWork {openalex_id: row.work}), (t:OpenAlexTopic {openalex_id: row.topic}) CREATE (w)-[:HAS_TOPIC]->(t)',
    topicEdges,
  );
  await importBatches(
    config,
    report,
    projectId,
    'openalex-known-work-citations',
    'UNWIND $rows AS row MATCH (w:OpenAlexWork {openalex_id: row.work}), (cited:OpenAlexWork {openalex_id: row.cited}) CREATE (w)-[:CITES]->(cited)',
    citationEdges.filter((edge) => knownWorks.has(edge.cited)),
  );
  await importBatches(
    config,
    report,
    projectId,
    'openalex-external-citations',
    'UNWIND $rows AS row MATCH (w:OpenAlexWork {openalex_id: row.work}), (cited:OpenAlexReference {openalex_id: row.cited}) CREATE (w)-[:CITES]->(cited)',
    citationEdges.filter((edge) => !knownWorks.has(edge.cited)),
  );
  const first = authorEdges[0];
  return {
    label: 'OpenAlexWork',
    idProperty: 'openalex_id',
    firstEdge: first ? { startLabel: 'OpenAlexAuthor', start: first.author, endLabel: 'OpenAlexWork', end: first.work } : undefined,
  };
}

const ENVELOPE_BYTES = [0, 32, 48, 48, 64];

function geometryOffset(blob) {
  if (blob[0] !== 0x47 || blob[1] !== 0x50) throw new Error('GeoPackage geometry has no GP header');
  const envelope = (blob[3] >> 1) & 0x07;
  const bytes = ENVELOPE_BYTES[envelope];
  if (bytes === undefined) throw new Error(`unsupported GeoPackage envelope ${envelope}`);
  return 8 + bytes;
}

function readWkbGeometry(buffer, start) {
  let offset = start;
  const little = buffer[offset] === 1;
  offset += 1;
  const readU32 = () => {
    const value = little ? buffer.readUInt32LE(offset) : buffer.readUInt32BE(offset);
    offset += 4;
    return value;
  };
  const readF64 = () => {
    const value = little ? buffer.readDoubleLE(offset) : buffer.readDoubleBE(offset);
    offset += 8;
    return value;
  };
  let rawType = readU32();
  const hasSrid = (rawType & 0x2000_0000) !== 0;
  const ewkbZ = (rawType & 0x8000_0000) !== 0;
  const ewkbM = (rawType & 0x4000_0000) !== 0;
  rawType &= 0x0fff_ffff;
  if (hasSrid) readU32();
  const isoDimension = Math.floor(rawType / 1_000);
  const type = rawType % 1_000;
  const dimensions = isoDimension === 3 || (ewkbZ && ewkbM) ? 4 : isoDimension > 0 || ewkbZ || ewkbM ? 3 : 2;
  const point = () => {
    const coordinates = [readF64(), readF64()];
    for (let index = 2; index < dimensions; index += 1) readF64();
    return coordinates;
  };
  if (type === 2) {
    const count = readU32();
    let first;
    let last;
    for (let index = 0; index < count; index += 1) {
      const current = point();
      first ??= current;
      last = current;
    }
    return { offset, lines: first && last ? [[first, last]] : [] };
  }
  if (type === 5) {
    const count = readU32();
    const lines = [];
    for (let index = 0; index < count; index += 1) {
      const nested = readWkbGeometry(buffer, offset);
      offset = nested.offset;
      lines.push(...nested.lines);
    }
    return { offset, lines };
  }
  throw new Error(`unsupported road WKB type ${type}`);
}

function roadEndpoints(blob) {
  return readWkbGeometry(blob, geometryOffset(blob)).lines;
}

function junctionId([x, y]) {
  return `${x.toFixed(7)},${y.toFixed(7)}`;
}

async function importOsm(config, report, projectId) {
  report.sources.push(await sourceManifest(config.source));
  const database = new DatabaseSync(config.source, { readOnly: true });
  const junctions = new Map();
  const roads = [];
  let featureCount = 0;
  const maximumFeatures = config.maxEdges || Number.MAX_SAFE_INTEGER;
  const statement = database.prepare(
    'SELECT osm_id, fclass, name, ref, oneway, maxspeed, geom FROM gis_osm_roads_free ORDER BY fid',
  );
  for (const feature of statement.iterate()) {
    if (featureCount >= maximumFeatures) break;
    featureCount += 1;
    for (const [startPoint, endPoint] of roadEndpoints(Buffer.from(feature.geom))) {
      const start = junctionId(startPoint);
      const end = junctionId(endPoint);
      junctions.set(start, { id: start, longitude: startPoint[0], latitude: startPoint[1] });
      junctions.set(end, { id: end, longitude: endPoint[0], latitude: endPoint[1] });
      const road = {
        start,
        end,
        osmId: String(feature.osm_id ?? ''),
        kind: String(feature.fclass ?? ''),
        name: String(feature.name ?? '').slice(0, 1_024),
        ref: String(feature.ref ?? '').slice(0, 128),
        maxspeed: Number(feature.maxspeed ?? 0),
      };
      const direction = String(feature.oneway ?? '').toUpperCase();
      if (direction !== 'T') roads.push(road);
      if (direction !== 'F') roads.push({ ...road, start: end, end: start });
    }
  }
  database.close();
  await importBatches(
    config,
    report,
    projectId,
    'osm-road-junctions',
    'UNWIND $rows AS row CREATE (:RoadJunction {junction_id: row.id, longitude: row.longitude, latitude: row.latitude})',
    junctions.values(),
  );
  await query(config, projectId, 'CREATE INDEX osm_junction_id FOR (n:RoadJunction) ON (n.junction_id)');
  await importBatches(
    config,
    report,
    projectId,
    'osm-roads',
    'UNWIND $rows AS row MATCH (a:RoadJunction {junction_id: row.start}), (b:RoadJunction {junction_id: row.end}) CREATE (a)-[:ROAD {osm_id: row.osmId, kind: row.kind, name: row.name, ref: row.ref, maxspeed: row.maxspeed}]->(b)',
    config.maxEdges > 0 ? roads.slice(0, config.maxEdges) : roads,
  );
  const first = roads[0];
  return { label: 'RoadJunction', idProperty: 'junction_id', firstEdge: first };
}

async function measuredQuery(config, report, projectId, name, cypher, parameters = {}) {
  const result = await query(config, projectId, cypher, parameters);
  const measurement = { name, cypher, elapsed_ms: result.elapsed_ms, server_elapsed_us: result.server_elapsed_us, rows: result.rows };
  report.queries.push(measurement);
  await writeReport(report, config.output);
  return result;
}

async function runAnalytics(config, report, projectId, fixture) {
  await new Promise((resolvePromise) => setTimeout(resolvePromise, 8_000));
  const label = quoteIdentifier(fixture.label);
  const property = quoteIdentifier(fixture.idProperty);
  await measuredQuery(config, report, projectId, 'node-cardinality', 'MATCH (n) RETURN count(n) AS nodes');
  await measuredQuery(config, report, projectId, 'edge-cardinality', 'MATCH ()-[r]->() RETURN count(r) AS edges');
  if (fixture.firstEdge) {
    const startLabel = quoteIdentifier(fixture.firstEdge.startLabel ?? fixture.label);
    const endLabel = quoteIdentifier(fixture.firstEdge.endLabel ?? fixture.label);
    await measuredQuery(
      config,
      report,
      projectId,
      'exact-retrieval',
      `MATCH (n:${label} {${property}: $id}) RETURN n.${property} AS id`,
      { id: fixture.firstEdge.end },
    );
    await measuredQuery(
      config,
      report,
      projectId,
      'shortest-path',
      `MATCH (a:${startLabel} {${property}: $start}), (b:${endLabel} {${property}: $end}) CALL graph.shortestpath(a, b) YIELD cost RETURN cost`,
      { start: fixture.firstEdge.start, end: fixture.firstEdge.end },
    );
  }
  await measuredQuery(config, report, projectId, 'degree', 'CALL graph.degree() YIELD degree RETURN count(*) AS nodes, max(degree) AS maximum_degree');
  await measuredQuery(config, report, projectId, 'weak-components', 'CALL graph.wcc() YIELD component RETURN count(*) AS assignments, count(DISTINCT component) AS components');
  await measuredQuery(config, report, projectId, 'strong-components', 'CALL graph.scc() YIELD component RETURN count(*) AS assignments, count(DISTINCT component) AS components');
  await measuredQuery(config, report, projectId, 'pagerank', 'CALL graph.pagerank(0.85, 0.000001, 20) YIELD score RETURN count(*) AS ranked, sum(score) AS score_sum');
  await measuredQuery(config, report, projectId, 'louvain', 'CALL graph.louvain() YIELD community RETURN count(*) AS assignments, count(DISTINCT community) AS communities');
}

async function main() {
  const config = parseArgs(process.argv.slice(2));
  const report = {
    format: 1,
    dataset: config.dataset,
    project_name: config.projectName,
    server: config.server.toString(),
    started_at: new Date().toISOString(),
    configuration: { batch_size: config.batchSize, max_nodes: config.maxNodes, max_edges: config.maxEdges },
    sources: [],
    ingestion: [],
    queries: [],
  };
  await writeReport(report, config.output);
  try {
    const projectId = await createProject(config);
    report.project_id = projectId;
    await writeReport(report, config.output);
    const fixture = config.dataset === 'icij'
      ? await importIcij(config, report, projectId)
      : config.dataset === 'openalex'
        ? await importOpenAlex(config, report, projectId)
        : await importOsm(config, report, projectId);
    await runAnalytics(config, report, projectId, fixture);
    report.finished_at = new Date().toISOString();
    report.status = 'passed';
  } catch (error) {
    report.finished_at = new Date().toISOString();
    report.status = 'failed';
    report.error = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
    throw error;
  } finally {
    await writeReport(report, config.output);
  }
}

await main();
