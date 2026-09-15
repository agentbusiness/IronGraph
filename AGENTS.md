## Output style

The reader has ADHD. Shape every response so it can be acted on:

1. Lead with the answer or next action: command, path, or snippet first.
2. Number multi-step work; one bounded action per step.
3. End with one next action doable in under two minutes.
4. Finish the current issue before raising a new one.
5. Restate progress each turn ("step 3 of 5 done").
6. Give time estimates in concrete units, never "a bit".
7. After a change, show what now works.
8. Errors: state location, cause, and fix. No drama.
9. Cap lists at 5 items.
10. No preamble, no recaps, no closers.

Exceptions: explain fully when asked to explain. Confirm before destructive actions. If the request is ambiguous, ask one short question.

Fix all issues encountered during the task immediately, including build, tooling, dependency, and
verification failures. Continue until each issue is resolved and the fix is verified. Failed attempts
require investigating the cause and changing the approach; never stop because an attempt count has
been reached or leave a fixable issue for the user to resolve. Ask for user input only when required
information or authorization cannot be obtained within the task's existing scope.

---

# IronGraph repository contract

IronGraph is a standalone, GPU-first, single-node graph database. This file is the only
authoritative repository-guidance document for future work. User-facing Markdown documentation,
including `README.md` and files below `docs/`, is allowed when it describes the current product.

## Product boundary

- Cypher is the only query and administration language.
- Local MCP is available over stdio for process-based hosts and over a loopback-only Streamable
  HTTP listener for URL-based local hosts. Both transports expose the same tool and resource surface
  and reach graph data only through the canonical Query API.
- The only browser-facing graph-data endpoint is `POST /api/query`. Loopback-only Settings controls
  use `/system/local-ai-integrations` to inspect and install local host packages; remote listeners
  never expose those controls. The web application is served below `/web/` and contains Query,
  Streams, Documents, Training, Docs, and Settings surfaces. Graph is the Plot result view inside
  Query rather than a separate route.
- Bolt remains supported. Remote Query, Bolt, Kafka-compatible Streams, and AMQP-compatible Queues
  require mutual TLS. Plain local listeners bind only to loopback.
- The Rust library exposes both in-process database access and remote API/Bolt access. Python and
  Node.js packages expose the same two modes; the browser and React package is remote-only. An embedded
  caller chooses the database directory, while the embedded node still uses the one canonical
  WAL/snapshot path, one selected device, and one-process single-node boundary.
- Projects and the OBSERVED, KNOWLEDGE, and WORKSPACE graph layers remain first-class database
  semantics. There is no implicit default project.
- Graph storage, indexes, temporal data, algorithms, transactions, WAL recovery, periodic snapshots,
  and asynchronous WAL durability remain database core.
- CPU is the reference execution backend. Metal is the primary local accelerator and CUDA is an
  optional build target. These are the complete execution-backend set.
- Every GPU-backed process selects one device and keeps each admitted project graph and its derived
  indexes resident. Admission failure is explicit; canonical graph rows are never silently paged or
  truncated.
- Full vector storage, vector indexes, text embedding, and vector search remain. The verified local
  embedding model installs, loads, binds to the selected device, and warms automatically at startup.
  IronGraph does not install, host, or invoke generative language models.
- Documents are ordinary Cypher-native graph records, for example `(:Document {body: ...})`. They
  persist through the normal WAL/snapshot path and participate in automatic embedding and vector
  search through declared graph indexes. No document-specific REST protocol or second store exists.
- Kafka-compatible topic operations and AMQP-compatible queue/exchange/binding operations remain.
  Administration, clearing/purging/deleting, and monitoring use Cypher statements such as `SHOW
  TOPICS`, `SHOW QUEUES`, `SHOW EXCHANGES`, and `SHOW CONSUMER LAG`.

## Web console design system

The console has one visual language, established by `web/src/reference.css` and composed by
`web/src/console.css`. Every surface — the band, Query, Streams, Documents, Training, Docs, Settings, and any
screen added later — is written in it. A new surface that introduces its own palette, type, or
control shapes is wrong even if it looks good.

- Colour comes only from the tokens: `--ground`, `--ink`/`--ink-2`/`--ink-3`, `--unwritten`,
  `--rule`/`--rule-2`, `--rubric`/`--rubric-deep`/`--rubric-lift`, `--counter`, `--rubric-wash`.
  Never write a literal colour into a component or stylesheet rule. Every surface must read
  correctly on both grounds (`data-view="plate"`, `"page"`, and `"auto"` on `#ig`).
- The rubric is selection, emphasis, and trouble. The counter is schema, provenance, and code.
  Neither is ever a decorative or categorical colour.
- Type: Source Serif 4 for prose and names; Archivo for apparatus (labels, controls, tabular
  numbers); Archivo/monospace for statements and identifiers. Both families are bundled through
  `@fontsource` imports in `web/src/main.tsx`; the CSS keeps system fallbacks.
- Idioms, not new controls: `.ap` labels, `.detent` actions, `.pick`/`Menu` for stated choices,
  `.seg-set` for one-of sets, `.field`/`.fld` inputs, `.bar` toolbars, `.lst` indexes, `.notice`
  (with `.contradiction`) for trouble and inline confirmation, `.sheet` for the rare interruption,
  `.stmt` for a statement a control is about to run. Corners are square; rules are 1px; marks are
  squares; the only drawn glyphs are the register cross and the caret in `web/src/design/parts.tsx`.
- Screens live on the manuscript grid (`.console-leaf`): index column, rail, reading column,
  margin rule, margin. The margin is where a selection is read and operated on. On small screens
  the margin follows the reading column; it is never simply hidden.
- The graph canvas is part of the design: entity nodes are squares, merged communities diamonds;
  categorical colours come from `paletteColor` in `web/src/lib/graphScene.ts` (muted inks, no
  reds — the rubric marks selection and the expanding node only); canvas labels are Archivo on a
  ground plate; relationships are hairlines that carry their arrowhead; the selected node takes a
  rubric ring and a ruled label plate.
- Administration controls state the exact Cypher statement they run before they run it. Document
  creation and editing use ordinary writing controls and do not expose their Cypher implementation.
  Destructive administration statements (`CLEAR`, `PURGE`, `DROP`) confirm inline with the
  consequence written out; document deletion confirms the consequence in plain language. Nothing
  destructive runs on a single press. No screen may require typing raw Cypher for an operation the
  screen exists to provide — the Query screen is the one raw surface.
- Waiting is always local: the acting control shows its working state and disables itself, never
  the screen; long reads are cancellable; polls stand down while the page is hidden; empty,
  loading, and error states are written in the design's voice (serif italic `.empty-line`,
  `.notice` for trouble).

## Closed scope

The product consists exactly of the database capabilities named above. New work must extend one of
those capabilities directly; it must not introduce a parallel application domain, second data store,
second query language, or bespoke data endpoint. The database is one process on one node, with one
selected execution device and one asynchronous WAL/snapshot durability path. Kafka and AMQP wire
fields exist solely for client compatibility.

Repository guidance is maintained only here. User-facing Markdown must not contain agent
instructions or establish a second repository contract. Do not add historical material, migration
narratives, compatibility aliases, deprecated modules, placeholder APIs, or descriptions of
functionality outside the current product boundary.

## Public documentation standard

- Public documentation describes supported behavior, public interfaces, operating requirements,
  and user-visible guarantees. It must not name or link to source files, private modules, crates,
  internal functions, traits, data structures, execution plans, unpublished algorithms, or other
  implementation details. Public SDK types and methods may be documented when they are part of the
  supported developer interface. Architecture documentation stays at the product-boundary level.
- Treat IronGraph as open-source software distributed under the Apache License 2.0. Document source
  builds and published packages without claiming that an unpublished artifact is available.
  State distribution prerequisites explicitly when availability depends on a package source.
- Organize documentation by developer intent: start, understand, install, embed, connect, operate,
  and troubleshoot. Lead each page with the outcome and intended audience, then use progressive
  disclosure from a minimal working path to production considerations.
- Write in direct, calm, technically precise English. Address the reader as "you" where it shortens
  instructions. Use active voice, present tense, and imperative verbs for procedures. Prefer short
  paragraphs and descriptive headings over slogans, conversational filler, or dense walls of text.
- Define IronGraph-specific concepts before using them. Keep the canonical terms `project`, `node`,
  `relationship`, `property`, `OBSERVED`, `KNOWLEDGE`, and `WORKSPACE` consistent. Format commands,
  values, and identifiers as code; write Cypher keywords in uppercase in examples.
- Reserve `node` for a graph entity. Use `instance`, `database process`, or `host` for deployment
  prose, except when naming the product's explicit single-node boundary.
- Every quickstart includes prerequisites, copyable commands, an expected result, and a next step.
  Every installation or configuration procedure identifies its deployment mode and distinguishes a
  safe local default from a production requirement.
- Make claims that can be verified from current product behavior. Qualify performance numbers with
  workload, data size, backend, and measurement method. Avoid unqualified superlatives, competitor
  comparisons, promises of future behavior, and broad compatibility claims that exceed tested
  surfaces.
- Use notes sparingly for prerequisites, security boundaries, destructive actions, and common
  mistakes. State limitations directly: IronGraph is single-node, has no implicit default project,
  selects one execution device per process, and rejects GPU admission when resident data does not
  fit.

## Graph invariants

- Every canonical node and relationship represents real user or domain data. Embedding rows,
  passages, scores, cache entries, prompts, and processing intermediates are derived state and never
  graph entities.
- A derived structure has a cold build and a bounded incremental path. Delta work must scale with
  changed rows, not unrelated graph size. Add shape tests using dirty fixtures with realistically
  large values whenever a derived structure changes.
- Source text, including document bodies, remains complete on its owning node. Encoder bounds are
  handled by invisible rebuildable spans that point back to that owner.
- Credentials remain outside graph properties, query results, WAL records, and snapshots.

## Editing and verification

- Multiple sessions may share this worktree. Preserve unrelated changes and never edit or delete by
  computed line range. Match unique content and fail when an anchor is absent or ambiguous.
- Announce before changing a struct field, trait method, or function signature. Check for an existing
  implementation before adding one.
- Use `apply_patch` for source edits. Use `rg` for repository searches. Do not use destructive Git
  commands.
- A substantial change starts with observable acceptance gates. Every gate needs a runnable check and
  an expected result; a claim without current command output is not complete.
- Verify Rust with formatting, workspace checks for default and no-default features, tests, and
  Clippy. Verify the web application with its test suite, production build, and an actual rendered
  browser pass. Browser verification uses ego-browser, never Puppeteer.
- Before completion, prove there is exactly one authoritative repository-guidance Markdown file,
  exactly one browser data endpoint, only the declared web surfaces and execution backends, and no
  out-of-scope terminology or untracked generated material.
