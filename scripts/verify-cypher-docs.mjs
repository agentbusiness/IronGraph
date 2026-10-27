#!/usr/bin/env node
/** Source-derived Cypher reference coverage and executable-example verification. */
import assert from 'node:assert/strict';
import { readFile, readdir, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { randomUUID } from 'node:crypto';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const read = (name) => readFile(path.join(root, name), 'utf8');
async function walk(directory) {
  const entries = await readdir(path.join(root, directory), { withFileTypes: true });
  return (await Promise.all(entries.map(async (entry) => {
    const name = `${directory}/${entry.name}`;
    return entry.isDirectory() ? walk(name) : name.endsWith('.md') ? [name] : [];
  }))).flat();
}
const docs = new Map(await Promise.all((await walk('docs/cypher')).map(async (name) => [name, await read(name)])));
const binder = await read('crates/cypher/src/binder.rs');
const ast = await read('crates/cypher/src/ast.rs');
const procedures = await read('crates/cypher/src/procedure.rs');
const temporal = await read('crates/execution/src/temporal.rs');
const database = await read('crates/server/src/server/database.rs');

function strings(source) { return [...source.matchAll(/"([^"\n]+)"/g)].map((match) => match[1]); }
function constant(source, name) {
  const match = source.match(new RegExp(`const ${name}: [\\s\\S]*?= &\\[([\\s\\S]*?)\\];`));
  assert(match, `source catalogue ${name} not found`);
  return strings(match[1]);
}
function variants(source, name) {
  const match = source.match(new RegExp(`pub enum ${name} \\{([\\s\\S]*?)\\n\\}`));
  assert(match, `source enum ${name} not found`);
  return [...match[1].matchAll(/^    (\w+)(?:\s*\{|\(|,)/gm)].map((item) => item[1]);
}
const functions = [...constant(binder, 'BUILTIN_FUNCTIONS'), ...constant(binder, 'TEMPORAL_CLOCK_FUNCTIONS')];
assert.equal(new Set(functions).size, functions.length, 'duplicate function catalogue entries');
const discovery = database.match(/fn builtin_functions\(\) -> Vec<String> \{([\s\S]*?)\n}/)?.[1];
assert(discovery?.includes('crate::cypher::builtin_function_names()') && !discovery.includes('"'), 'server function discovery must collect the canonical binder catalogue');
const canonicalDiscovery = binder.match(/pub fn builtin_function_names\([\s\S]*?\n}/)?.[0] ?? '';
assert(canonicalDiscovery.includes('BUILTIN_FUNCTIONS') && canonicalDiscovery.includes('.chain(TEMPORAL_CLOCK_FUNCTIONS)'), 'function discovery must include scalar and clock-qualified names');
const procedureSignatures = [...procedures.matchAll(/ProcedureSignature \{\s*name: "([^"]+)",\s*accepted_arities: &\[([^\]]*)\],\s*outputs: &\[([^\]]*)\]/g)]
  .map((match) => ({ name: match[1], arities: match[2].split(',').map((n) => Number(n.trim())).filter(Number.isInteger), outputs: strings(match[3]) }));
assert(procedureSignatures.length > 0, 'procedure signatures not found');

const statements = {
  Query: 'clauses/core/match', CreateProject: 'statements/projects/create-project',
  ShowProjects: 'statements/projects/show-projects', ImportDataset: 'statements/import-dataset',
  ShowTopics: 'statements/brokers/show-topics', ShowQueues: 'statements/brokers/show-queues',
  ShowExchanges: 'statements/brokers/show-exchanges', ShowConsumerLag: 'statements/brokers/show-consumer-lag',
  CheckReadOnly: 'statements/check-read-only', CreateTopic: 'statements/brokers/create-topic',
  AlterTopicRetention: 'statements/brokers/alter-topic', DropTopic: 'statements/brokers/drop-topic',
  ClearTopic: 'statements/brokers/clear-topic', CreateQueue: 'statements/brokers/create-queue',
  AlterQueueRetention: 'statements/brokers/alter-queue', DropQueue: 'statements/brokers/drop-queue',
  PurgeQueue: 'statements/brokers/purge-queue', CreateExchange: 'statements/brokers/create-exchange',
  DropExchange: 'statements/brokers/drop-exchange', BindQueue: 'statements/brokers/bind-queue',
  UnbindQueue: 'statements/brokers/unbind-queue', AlterProjectRename: 'statements/projects/alter-project',
  DropProject: 'statements/projects/drop-project', CreateIndex: 'statements/indexes/create-index',
  ShowIndexes: 'statements/indexes/show-indexes', CreateConstraint: 'statements/constraints/create-constraint',
  ShowConstraints: 'statements/constraints/show-constraints', RebuildIndex: 'statements/indexes/rebuild-index',
  DropIndex: 'statements/indexes/drop-index', DropConstraint: 'statements/constraints/drop-constraint',
  DeclareTemporal: 'statements/temporal/alter-set-temporal', CreateRollup: 'statements/temporal/create-rollup',
  CreateEmbedding: 'clauses/search/create-embedding-index', AlterAutomaticSemantic: 'statements/indexes/alter-automatic-semantic',
};
const clauses = Object.fromEntries(['Match', 'Where', 'Unwind', 'For', 'Let', 'Filter', 'Create', 'Merge', 'Set', 'Remove', 'Delete', 'With', 'Return', 'Skip', 'Limit', 'Call', 'Finish']
  .map((name) => [name, `clauses/core/${name.toLowerCase()}`]));
Object.assign(clauses, { OrderBy: 'clauses/core/order-by', History: 'clauses/time/history', Window: 'clauses/time/window', Search: 'clauses/search/search' });
const expressionTerms = {
  Literal: 'Values, names and parameters', Parameter: '$name', Variable: 'identifiers', Property: 'entity.property',
  List: '[1, 2, null]', Map: "{name: 'Ada', active: true}", MapProjection: 'map{.name', Case: 'CASE',
  ListComprehension: '[x IN list WHERE predicate | projection]', Reduce: 'reduce(', ListPredicate: 'single',
  Function: 'functions', ExistentialSubquery: 'EXISTS { MATCH', Unary: 'Unary', Binary: 'Operators',
  IsNull: 'IS NULL', Index: 'value[index]', Slice: 'list[start..end]', Star: 'RETURN *',
};
const operators = {
  Not: 'NOT', Positive: 'Unary `+`', Negative: '`-`', Or: 'OR', Xor: 'XOR', And: 'AND', Equal: '`=`',
  NotEqual: '`<>`', Less: '`<`', LessOrEqual: '`<=`', Greater: '`>`', GreaterOrEqual: '`>=`',
  In: '`IN`', StartsWith: 'STARTS WITH', EndsWith: 'ENDS WITH', Contains: 'CONTAINS', RegexMatch: '`=~`',
  Concat: '\\|\\|', Add: '`+`', Subtract: '`-`', Multiply: '`*`', Divide: '`/`', Modulo: '`%`', Power: '`^`',
};
const selectors = { All: 'No prefix', Any: '`ANY`', AnyShortest: 'ANY SHORTEST', AllShortest: 'ALL SHORTEST', Shortest: 'GROUPS' };
const modes = { DifferentRelationships: 'DIFFERENT RELATIONSHIPS', RepeatableElements: 'REPEATABLE ELEMENTS', Acyclic: 'ACYCLIC' };

function checkCoverage(files) {
  const errors = [];
  const requirePage = (relative, item) => {
    const name = `docs/cypher/${relative}.md`;
    const text = files.get(name);
    if (!text) { errors.push(`${item}: missing ${name}`); return null; }
    for (const field of ['| Kind |', '| Signature |', '| Relationship to standard Cypher |']) {
      if (!text.includes(field)) errors.push(`${name}: missing ${field}`);
    }
    if (!text.includes('```cypher')) errors.push(`${name}: no executable Cypher example`);
    return text;
  };
  for (const name of functions) {
    const matches = [...files.entries()].filter(([file, text]) => file.includes('/functions/') && text.match(/^# `([^`]+)`/m)?.[1].toLowerCase() === name);
    if (matches.length !== 1) { errors.push(`${name}: expected one function reference page, found ${matches.length}`); continue; }
    const [file, text] = matches[0];
    requirePage(file.replace('docs/cypher/', '').replace(/\.md$/, ''), name);
    if (!/^\| Signature \|.*`[^`]*\(/m.test(text)) errors.push(`${file}: signature must state arguments`);
    const signatureName = text.match(/^\| Signature \|\s*`([^`(]+)\(/m)?.[1].toLowerCase();
    if (signatureName !== name) errors.push(`${file}: function signature names another function`);
    for (const heading of ['## What it does', '## How it behaves', '## Simple example', '## Advanced example']) {
      if (!text.includes(heading)) errors.push(`${file}: missing ${heading}`);
    }
    if ([...text.matchAll(/```cypher\n([\s\S]*?)```/g)].length < 2) errors.push(`${file}: needs two examples`);
  }
  for (const procedure of procedureSignatures) {
    const matches = [...files.entries()].filter(([file, text]) => file.includes('/procedures/') && text.match(/^# `([^`]+)`/m)?.[1].toLowerCase() === procedure.name);
    if (matches.length !== 1) { errors.push(`${procedure.name}: expected one procedure reference`); continue; }
    const [file, text] = matches[0];
    const signature = text.match(/^\| Signature \| (.+)$/m)?.[1] ?? '';
    requirePage(file.replace('docs/cypher/', '').replace(/\.md$/, ''), procedure.name);
    for (const output of procedure.outputs) if (!new RegExp(`\\b${output}\\b`).test(signature)) errors.push(`${file}: missing YIELD ${output}`);
    const argumentsText = signature.match(/\((.*?)\)/)?.[1] ?? '';
    const countArguments = (text) => text.replaceAll('[', '').replaceAll(']', '').split(',').map((value) => value.trim()).filter(Boolean).length;
    const arities = [...new Set([countArguments(argumentsText.replace(/\[[^\]]*\]/g, '')), countArguments(argumentsText)])].sort();
    if (JSON.stringify(arities) !== JSON.stringify([...procedure.arities].sort())) errors.push(`${file}: procedure argument counts disagree with the supported signature`);
    for (const heading of ['## What it does', '## How it behaves', '## Simple example', '## Advanced example', '## Limitations and trade-offs']) {
      if (!text.includes(heading)) errors.push(`${file}: missing ${heading}`);
    }
  }
  for (const [catalogue, mapping] of [['Statement', statements], ['Clause', clauses]]) {
    for (const variant of variants(ast, catalogue)) {
      if (!mapping[variant]) errors.push(`${catalogue}::${variant}: no documentation mapping`);
      else requirePage(mapping[variant], `${catalogue}::${variant}`);
    }
  }
  const expression = files.get('docs/cypher/expressions.md') ?? '';
  const allClauses = `${expression}\n${files.get('docs/cypher/clauses/core/return.md') ?? ''}`;
  for (const variant of variants(ast, 'Expression')) {
    if (!expressionTerms[variant] || !allClauses.includes(expressionTerms[variant])) errors.push(`Expression::${variant}: undocumented value form`);
  }
  for (const catalogue of ['UnaryOperator', 'BinaryOperator']) {
    for (const variant of variants(ast, catalogue)) if (!operators[variant] || !expression.includes(operators[variant])) errors.push(`${catalogue}::${variant}: undocumented operator`);
  }
  const paths = files.get('docs/cypher/clauses/core/paths.md') ?? '';
  for (const [catalogue, mapping] of [['PathSelector', selectors], ['PathMode', modes]]) {
    for (const variant of variants(ast, catalogue)) if (!mapping[variant] || !paths.includes(mapping[variant])) errors.push(`${catalogue}::${variant}: undocumented path syntax`);
  }
  const choices = [
    ['Direction', 'clauses/core/paths', { Outgoing: '-->', Incoming: '<--', Undirected: '`--`' }],
    ['WindowSyntax', 'clauses/time/window', { Tumbling: 'TUMBLING', Hopping: 'HOPPING' }],
    ['SearchMethod', 'clauses/search/search', { Default: 'without `USING`', Exact: 'EXACT', Prefix: 'PREFIX', Adaptive: 'ADAPTIVE', Quantized: 'QUANTIZED' }],
    ['SearchInput', 'clauses/search/search', { Text: 'TEXT', Vector: 'VECTOR', HybridText: 'HYBRID TEXT' }],
    ['IndexKind', 'statements/indexes/create-index', { Equality: 'Equality', Range: 'Range', Text: 'Text', Vector: 'Vector' }],
    ['TemporalTarget', 'statements/temporal/alter-set-temporal', { Node: 'NODE', Relationship: 'RELATIONSHIP' }],
    ['ListPredicateKind', 'expressions', { All: '`all`', Any: '`any`', None: '`none`', Single: '`single`' }],
    ['SetItem', 'clauses/core/set', { Property: '<property>', MergeMap: 'SET n += map', ReplaceMap: 'SET n = map', Labels: 'SET n:Label' }],
    ['RemoveItem', 'clauses/core/remove', { Property: '<property>', Labels: 'label' }],
    ['LabelName', 'expressions', { Static: 'literal', Dynamic: '$(expression)', DynamicAll: '$all(expression)' }],
    ['MapProjectionItem', 'expressions', { AllProperties: '.*', Property: '.name', Variable: 'map{variable}', Entry: 'computed: expression' }],
    ['CallArgumentMode', 'clauses/core/call', { Explicit: 'explicit parentheses', Implicit: 'query parameters' }],
    ['CallYieldMode', 'clauses/core/call', { Omitted: 'omitted YIELD', Explicit: 'YIELD and optional AS', All: 'YIELD *' }],
  ];
  for (const [catalogue, page, mapping] of choices) {
    const text = files.get(`docs/cypher/${page}.md`) ?? '';
    for (const variant of variants(ast, catalogue)) if (!mapping[variant] || !text.includes(mapping[variant])) errors.push(`${catalogue}::${variant}: undocumented syntax choice`);
  }
  const temporalNames = temporal.match(/pub fn from_property_name\([\s\S]*?\n    }/)?.[0] ?? '';
  assert(temporalNames, 'temporal field catalogue not found');
  for (const name of strings(temporalNames)) if (!expression.toLowerCase().includes(`\`${name}\``)) errors.push(`${name}: undocumented temporal accessor`);
  for (const [file, text] of files) {
    if (/\bcrates\/|\b\w+\.rs\b/.test(text)) errors.push(`${file}: private source reference in public documentation`);
    for (const link of text.matchAll(/\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g)) {
      const target = link[1].split('#')[0];
      if (!target || /^[a-z]+:/i.test(target) || target.startsWith('/')) continue;
      const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(file), decodeURIComponent(target)));
      if (resolved.startsWith('docs/cypher/') && !files.has(resolved)) errors.push(`${file}: broken link ${target}`);
    }
  }
  const reachable = new Set();
  const pending = ['docs/cypher/README.md'];
  while (pending.length) {
    const file = pending.pop();
    if (reachable.has(file)) continue;
    reachable.add(file);
    for (const link of (files.get(file) ?? '').matchAll(/\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g)) {
      const target = link[1].split('#')[0];
      if (!target || /^[a-z]+:/i.test(target) || target.startsWith('/')) continue;
      const resolved = path.posix.normalize(path.posix.join(path.posix.dirname(file), decodeURIComponent(target)));
      if (files.has(resolved) && !reachable.has(resolved)) pending.push(resolved);
    }
  }
  for (const file of files.keys()) if (!reachable.has(file)) errors.push(`${file}: unreachable from the Cypher reference navigation`);
  return errors;
}

const errors = checkCoverage(docs);
if (errors.length) { console.error(errors.join('\n')); process.exit(1); }
// A missing page and a mismatched procedure output must both fail this exact checker.
const missingPage = new Map(docs);
missingPage.delete('docs/cypher/functions/numeric/abs.md');
assert(checkCoverage(missingPage).some((message) => message.includes('abs: expected one function reference')), 'missing-page negative control failed');
const wrongOutput = new Map(docs);
const degreeFile = 'docs/cypher/procedures/centrality/graph-degree.md';
wrongOutput.set(degreeFile, docs.get(degreeFile).replace(/^\| Signature \|.*$/m, '| Signature | `graph.degree() YIELD node` |'));
assert(checkCoverage(wrongOutput).some((message) => message.includes('missing YIELD outDegree')), 'procedure-output negative control failed');

function exampleInventory() {
  const examples = [];
  const onlyIndex = process.argv.indexOf('--only');
  const only = onlyIndex === -1 ? null : process.argv[onlyIndex + 1];
  assert(onlyIndex === -1 || only, '--only requires a reference-relative prefix');
  for (const [file, text] of docs) {
    if (only && !file.replace('docs/cypher/', '').startsWith(only)) continue;
    // New self-contained references have no multi-million-row dataset prerequisite.
    if ((!process.argv.includes('--all-examples') && !/\/functions\/(numeric|string|list|entity|conversion|temporal)\/[^/]+\.md$|\/clauses\/core\/[^/]+\.md$|\/statements\/brokers\/[^/]+\.md$|\/statements\/(check-read-only|import-dataset)\.md$|\/statements\/projects\/alter-project\.md$|\/statements\/constraints\/show-constraints\.md$|\/statements\/indexes\/alter-automatic-semantic\.md$|\/expressions\.md$/.test(file)) || file.endsWith('/README.md')) continue;
    for (const match of text.matchAll(/```cypher\n([\s\S]*?)```/g)) {
      const before = text.slice(0, match.index);
      const parameterBlock = [...before.matchAll(/```json\n([\s\S]*?)```/g)].at(-1);
      const parameters = file.endsWith('/check-read-only.md') && parameterBlock ? JSON.parse(parameterBlock[1]) : {};
      examples.push({ file, query: match[1].trim(), parameters, expectError: /<!-- expect-error -->\s*$/.test(before) });
    }
  }
  return examples;
}
const examples = exampleInventory();
if (process.argv.includes('--examples-json')) {
  process.stdout.write(JSON.stringify(examples, null, 2));
  process.exit(0);
}

const executeIndex = process.argv.indexOf('--execute');
if (executeIndex !== -1) {
  const endpoint = process.argv[executeIndex + 1];
  assert(endpoint?.startsWith('http://127.0.0.1:'), '--execute requires an explicit disposable loopback Query API URL');
  const runId = randomUUID().replaceAll('-', '').slice(0, 12);
  async function run(query, parameters = {}, expectError = false) {
    if (process.argv.includes('--trace')) console.error(query);
    const response = await fetch(endpoint, { method: 'POST', signal: AbortSignal.timeout(120_000), headers: { 'content-type': 'application/json' }, body: JSON.stringify({ request_id: randomUUID(), project_id: null, query, parameters }) });
    const body = await response.text();
    assert(response.ok, `HTTP ${response.status}: ${body}`);
    const frames = body.trim().split('\n').filter(Boolean).map((line) => JSON.parse(line));
    const error = frames.find((frame) => frame.type === 'error');
    if (expectError) {
      assert(error, `expected example rejection but query succeeded: ${query}`);
      return frames;
    }
    assert(!error, `${error?.message}\n${query}`);
    assert(frames.some((frame) => frame.type === 'summary'), `missing terminal summary: ${query}`);
    return frames;
  }
  const results = [];
  const projectByFile = new Map();
  const fixtureSuffixByFile = new Map();
  const initializedProjects = new Set();
  for (const example of examples) {
    let project = projectByFile.get(example.file);
    if (!project) {
      project = example.file.includes('/brokers/') || example.file.endsWith('/alter-automatic-semantic.md')
        ? `cypher_docs_${runId}_${projectByFile.size}` : `cypher_docs_${runId}`;
      projectByFile.set(example.file, project);
      fixtureSuffixByFile.set(example.file, `${runId}_${projectByFile.size}`);
      if (!initializedProjects.has(project)) {
        await run(`CREATE PROJECT ${project}`);
        initializedProjects.add(project);
      }
      if (example.file.includes('/brokers/')) {
        const slug = path.basename(example.file, '.md');
        const setup = [
          ['create-topic', 'CREATE TOPIC reference_topic PARTITIONS 1'], ['create-topic', 'CREATE TOPIC reference_topic_partitioned PARTITIONS 3 RETENTION 7 DAYS'],
          ['create-queue', 'CREATE QUEUE reference_queue CLASSIC'], ['create-queue', 'CREATE QUEUE reference_stream STREAM RETENTION 7 DAYS'],
          ['create-exchange', 'CREATE EXCHANGE reference_exchange TYPE direct'], ['create-exchange', 'CREATE EXCHANGE reference_fanout TYPE fanout'],
        ];
        for (const [creationPage, query] of setup) if (slug !== creationPage) await run(`USE ${project} ${query}`);
        if (slug === 'unbind-queue') {
          await run(`USE ${project} BIND QUEUE reference_queue TO EXCHANGE reference_exchange KEY created`);
          await run(`USE ${project} BIND QUEUE reference_stream TO EXCHANGE reference_exchange KEY \`events.created\``);
        }
      }
      if (example.file.endsWith('/alter-project.md')) await run(`CREATE PROJECT rename_example_${runId}`);
    }
    const query = example.query.replace(/\bworked_example_other\b/g, `${project}_other`).replace(/\bworked_example\b/g, project).replaceAll('USE code', `USE ${project}`)
      .replaceAll('rename_example', `rename_example_${runId}`).replaceAll('renamed_example', `renamed_example_${runId}`).replaceAll('`Renamed example`', `\`Renamed example ${runId}\``)
      .replace(/\b(index_example|temporal_example|rollup_example|backfill_example|naming_example_renamed|naming_example|embedding_example|failed_index_example)\b/g, (name) => `${name}_${fixtureSuffixByFile.get(example.file)}`);
    const parameters = Object.fromEntries(Object.entries(example.parameters).map(([key, value]) => [key, typeof value === 'string' ? value.replaceAll('USE worked_example', `USE ${project}`) : value]));
    const frames = await run(query, parameters, example.expectError);
    results.push({ ...example, executedQuery: query, frames });
  }
  await mkdir(path.join(root, '.unlazy/semantic-scope'), { recursive: true });
  await writeFile(path.join(root, '.unlazy/semantic-scope/doc-results.json'), JSON.stringify(results, null, 2));
  console.log(`CYPHER_EXAMPLES_OK ${results.length} examples, ${initializedProjects.size} isolated fixture projects; results saved in ignored verification output`);
}
console.log(`CYPHER_DOCS_OK ${functions.length} functions, ${procedureSignatures.length} procedures, ${variants(ast, 'Statement').length} statement forms, ${variants(ast, 'Clause').length} clauses; missing-page and wrong-output controls rejected`);
