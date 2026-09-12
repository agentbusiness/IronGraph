import { useCallback, useEffect, useMemo, useState } from 'react';
import type { Project } from '../../types';
import { execute, query } from '../../lib/cypher';
import {
  bindQueue,
  alterQueueRetention,
  alterTopicRetention,
  clearTopic,
  createExchange,
  createQueue,
  createTopic,
  dropExchange,
  dropQueue,
  dropTopic,
  exchangeSummaries,
  formatBytes,
  lagRows as readLagRows,
  PARTITION_LIMIT,
  purgeQueue,
  queueSummaries,
  SHOW_CONSUMER_LAG,
  SHOW_EXCHANGES,
  SHOW_QUEUES,
  SHOW_TOPICS,
  topicSummaries,
  unbindQueue,
  validName,
  type ExchangeKind,
  type ExchangeSummary,
  type LagRow,
  type QueueKind,
  type QueueSummary,
  type TopicSummary,
} from '../../lib/broker';
import { errorMessage, formatTime } from '../../lib/format';
import { Cross, Rail, Screen } from '../../design/parts';

type Register = 'topics' | 'queues' | 'exchanges' | 'lag';

/** What one click on the broker tables is about, read and operated on in the margin. */
type Selection =
  | { kind: 'topic'; name: string }
  | { kind: 'queue'; name: string }
  | { kind: 'exchange'; name: string }
  | { kind: 'lag'; group: string; topic: string; partition: number };

type MarginMode = { at: 'inspect' } | { at: 'create'; register: 'topics' | 'queues' | 'exchanges' };

/** A destructive statement waiting for its second press, with the consequence written out. */
interface Confirm {
  statement: string;
  label: string;
  consequence: string;
  verb: string;
  /** Selecting nothing afterwards is right when the thing itself is gone. */
  clearsSelection?: boolean;
}

interface BrokerSample {
  topics: TopicSummary[];
  queues: QueueSummary[];
  exchanges: ExchangeSummary[];
  lag: LagRow[];
}

const EMPTY_SAMPLE: BrokerSample = { topics: [], queues: [], exchanges: [], lag: [] };
const POLL_MS = 2_000;

const REGISTERS: { id: Register; key: string; label: string }[] = [
  { id: 'topics', key: 'T', label: 'Topics' },
  { id: 'queues', key: 'Q', label: 'Queues' },
  { id: 'exchanges', key: 'X', label: 'Exchanges' },
  { id: 'lag', key: 'L', label: 'Consumer lag' },
];

/** The statement a control is about to run, stated in the open. */
function Stmt({ statement }: { statement: string }) {
  return (
    <code className="stmt">
      <b>{statement}</b>
    </code>
  );
}

function Section({ label, count }: { label: string; count?: number }) {
  return (
    <div className="insp-sec">
      <span className="ap">{label}</span>
      <span className="insp-sec-rule" />
      {count !== undefined && <span className="ap faint">{count.toLocaleString()}</span>}
    </div>
  );
}

export function StreamConsole({ project }: { project?: Project }) {
  const [register, setRegister] = useState<Register>('topics');
  const [sample, setSample] = useState<BrokerSample>(EMPTY_SAMPLE);
  const [sampledAt, setSampledAt] = useState<Date>();
  const [pollError, setPollError] = useState<string>();
  const [selection, setSelection] = useState<Selection>();
  const [margin, setMargin] = useState<MarginMode>({ at: 'inspect' });
  const [confirm, setConfirm] = useState<Confirm>();
  const [running, setRunning] = useState<string>();
  const [actionError, setActionError] = useState<string>();

  /**
   * One sample reads all four registers, not just the open one: the index states every count,
   * and the margin's routing forms offer queues and exchanges whichever table is open. Each is
   * a metric read the broker answers from memory, so four every two seconds is cheap; the poll
   * still stands down whenever the page is hidden, because nobody is reading the answer.
   */
  const sampler = useCallback(async (projectId: string, signal?: AbortSignal) => {
    try {
      const [topicRows, queueRows, exchangeRows, lagRowsRaw] = await Promise.all([
        query(projectId, SHOW_TOPICS, {}, signal),
        query(projectId, SHOW_QUEUES, {}, signal),
        query(projectId, SHOW_EXCHANGES, {}, signal),
        query(projectId, SHOW_CONSUMER_LAG, {}, signal),
      ]);
      setSample({
        topics: topicSummaries(topicRows),
        queues: queueSummaries(queueRows),
        exchanges: exchangeSummaries(exchangeRows),
        lag: readLagRows(lagRowsRaw),
      });
      setSampledAt(new Date());
      setPollError(undefined);
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError')) {
        setPollError(errorMessage(cause));
      }
    }
  }, []);

  useEffect(() => {
    setSample(EMPTY_SAMPLE);
    setSampledAt(undefined);
    setPollError(undefined);
    setSelection(undefined);
    setMargin({ at: 'inspect' });
    setConfirm(undefined);
    setActionError(undefined);
    if (!project) return;
    const controller = new AbortController();
    let interval: number | undefined;
    const start = () => {
      if (interval !== undefined) return;
      void sampler(project.id, controller.signal);
      interval = window.setInterval(() => void sampler(project.id, controller.signal), POLL_MS);
    };
    const stop = () => {
      if (interval === undefined) return;
      window.clearInterval(interval);
      interval = undefined;
    };
    // A hidden page reads nothing, so it asks for nothing; coming back asks at once.
    const onVisibility = () => (document.hidden ? stop() : start());
    onVisibility();
    document.addEventListener('visibilitychange', onVisibility);
    return () => {
      document.removeEventListener('visibilitychange', onVisibility);
      stop();
      controller.abort();
    };
  }, [project, sampler]);

  /** Runs one administration statement, then reads the registers again without waiting for the poll. */
  const run = useCallback(
    async (statement: string, after?: () => void) => {
      if (!project || running) return;
      setRunning(statement);
      setActionError(undefined);
      try {
        await execute(project.id, statement);
        after?.();
        await sampler(project.id);
      } catch (cause) {
        setActionError(errorMessage(cause));
      } finally {
        setRunning(undefined);
      }
    },
    [project, running, sampler],
  );

  const openRegister = (next: Register) => {
    setRegister(next);
    setSelection(undefined);
    setMargin({ at: 'inspect' });
    setConfirm(undefined);
    setActionError(undefined);
  };

  const select = (next: Selection) => {
    setSelection(next);
    setMargin({ at: 'inspect' });
    setConfirm(undefined);
    setActionError(undefined);
  };

  const worstLag = sample.lag[0]?.lag ?? 0;
  const lagGroups = useMemo(() => new Set(sample.lag.map((row) => row.group)).size, [sample.lag]);

  const registerCounts: Record<Register, string> = {
    topics:
      sample.topics.length === 0
        ? 'none declared'
        : `${sample.topics.length.toLocaleString()} · ${sample.topics
            .reduce((total, topic) => total + topic.partitions.length, 0)
            .toLocaleString()} partitions`,
    queues: sample.queues.length === 0 ? 'none declared' : sample.queues.length.toLocaleString(),
    exchanges: sample.exchanges.length === 0 ? 'none declared' : sample.exchanges.length.toLocaleString(),
    lag:
      sample.lag.length === 0
        ? 'no groups'
        : `${lagGroups.toLocaleString()} group${lagGroups === 1 ? '' : 's'}${worstLag > 0 ? ` · behind by ${worstLag.toLocaleString()}` : ' · caught up'}`,
  };

  const open = REGISTERS.find((entry) => entry.id === register) ?? REGISTERS[0]!;

  return (
    <Screen name="streams">
      <div className="col-i">
        <div className="bar">
          <span className="ap">Broker</span>
          <div className="grow"></div>
          <span className={project && !pollError ? 'pip' : 'pip hollow'} aria-hidden></span>
          <span className="ap faint">{project ? (pollError ? 'not answering' : '2 s poll') : 'idle'}</span>
        </div>
        <ul className="lst regs">
          {REGISTERS.map((entry) => (
            <li key={entry.id} className={entry.id === register ? 'on' : undefined}>
              <button type="button" onClick={() => openRegister(entry.id)} aria-pressed={entry.id === register}>
                <span className="t">{entry.label}</span>
                <span className="r">
                  <span className={entry.id === 'lag' && worstLag > 0 ? 'ap hot' : 'ap faint'}>
                    {registerCounts[entry.id]}
                  </span>
                  <span className="grow"></span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>

      <Rail
        screen="streams"
        cap="Broker"
        keys={REGISTERS.map((entry) => ({
          t: entry.key,
          title: entry.label,
          on: entry.id === register,
          ticks: [entry.id === register],
          onSelect: () => openRegister(entry.id),
        }))}
        foot={sampledAt ? formatTime(sampledAt.getTime()) : '—'}
      />

      <div className="col-ii" style={{ display: 'flex', flexDirection: 'column', paddingRight: '22px' }}>
        {pollError && (
          <div className="notice contradiction" role="alert">
            <span className="kindmark"></span>
            <span className="ap lbl">Trouble</span>
            <p>The broker did not answer: {pollError}</p>
          </div>
        )}

        {!project ? (
          <p className="empty-line">Create or select a project at the top; its broker state reads here.</p>
        ) : (
          <>
            <div className="bar" style={{ paddingLeft: 0, paddingRight: 0 }}>
              <span className="ap">{open.label}</span>
              <span className="ap faint">{registerCounts[register]}</span>
              <div className="grow"></div>
              {register !== 'lag' && (
                <button
                  className="detent"
                  type="button"
                  aria-pressed={margin.at === 'create' && margin.register === register}
                  onClick={() => {
                    setMargin(
                      margin.at === 'create' && margin.register === register
                        ? { at: 'inspect' }
                        : { at: 'create', register },
                    );
                    setConfirm(undefined);
                    setActionError(undefined);
                  }}
                >
                  {register === 'topics' ? 'New topic' : register === 'queues' ? 'New queue' : 'New exchange'}
                </button>
              )}
            </div>

            {register === 'topics' && (
              <TopicsTable topics={sample.topics} selection={selection} onSelect={select} sampled={sampledAt !== undefined} />
            )}
            {register === 'queues' && (
              <QueuesTable queues={sample.queues} selection={selection} onSelect={select} sampled={sampledAt !== undefined} />
            )}
            {register === 'exchanges' && (
              <ExchangesTable exchanges={sample.exchanges} selection={selection} onSelect={select} sampled={sampledAt !== undefined} />
            )}
            {register === 'lag' && (
              <LagTable rows={sample.lag} selection={selection} onSelect={select} sampled={sampledAt !== undefined} />
            )}
          </>
        )}
      </div>

      <StreamMargin
        project={project}
        sample={sample}
        selection={selection}
        margin={margin}
        confirm={confirm}
        running={running}
        actionError={actionError}
        onCloseCreate={() => setMargin({ at: 'inspect' })}
        onClearSelection={() => setSelection(undefined)}
        onConfirm={setConfirm}
        onCancelConfirm={() => setConfirm(undefined)}
        onRun={(statement, after) => void run(statement, after)}
      />
    </Screen>
  );
}

/* ── The four registers, as the design's table ────────────────────────────── */

function EmptyRegister({ sampled, children }: { sampled: boolean; children: React.ReactNode }) {
  return <p className="empty-line">{sampled ? children : 'Sampling the broker…'}</p>;
}

interface TableProps {
  selection?: Selection;
  onSelect: (selection: Selection) => void;
  sampled: boolean;
}

function rowProps(selected: boolean, onSelect: () => void) {
  return {
    tabIndex: 0,
    'aria-selected': selected,
    onClick: onSelect,
    onKeyDown: (event: React.KeyboardEvent) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        onSelect();
      }
    },
  };
}

function TopicsTable({ topics, selection, onSelect, sampled }: TableProps & { topics: TopicSummary[] }) {
  if (topics.length === 0) {
    return (
      <EmptyRegister sampled={sampled}>
        No topics. A topic is a partitioned, replayable log of records; declare one with New topic.
      </EmptyRegister>
    );
  }
  return (
    <div className="bt-frame">
      <table className="bt">
        <thead>
          <tr>
            <th scope="col">Name</th>
            <th scope="col" className="n">Partitions</th>
            <th scope="col" className="n">Records</th>
            <th scope="col" className="n">Retained</th>
          </tr>
        </thead>
        <tbody>
          {topics.map((topic) => (
            <tr
              key={topic.name}
              {...rowProps(selection?.kind === 'topic' && selection.name === topic.name, () =>
                onSelect({ kind: 'topic', name: topic.name }),
              )}
            >
              <td className="name">{topic.name}</td>
              <td className="n">{topic.partitions.length.toLocaleString()}</td>
              <td className="n">{topic.records.toLocaleString()}</td>
              <td className="n">{formatBytes(topic.retainedBytes)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function QueuesTable({ queues, selection, onSelect, sampled }: TableProps & { queues: QueueSummary[] }) {
  if (queues.length === 0) {
    return (
      <EmptyRegister sampled={sampled}>
        No queues. A queue delivers each message to one consumer; declare one with New queue.
      </EmptyRegister>
    );
  }
  return (
    <div className="bt-frame">
      <table className="bt">
        <thead>
          <tr>
            <th scope="col">Name</th>
            <th scope="col">Kind</th>
            <th scope="col" className="n">Messages</th>
            <th scope="col" className="n">Ready</th>
            <th scope="col" className="n">Retained</th>
          </tr>
        </thead>
        <tbody>
          {queues.map((queue) => (
            <tr
              key={queue.name}
              {...rowProps(selection?.kind === 'queue' && selection.name === queue.name, () =>
                onSelect({ kind: 'queue', name: queue.name }),
              )}
            >
              <td className="name">{queue.name}</td>
              <td><span className="kindword">{queue.kind}</span></td>
              <td className="n">{queue.messages.toLocaleString()}</td>
              <td className="n">{queue.available.toLocaleString()}</td>
              <td className="n">{formatBytes(queue.retainedBytes)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ExchangesTable({ exchanges, selection, onSelect, sampled }: TableProps & { exchanges: ExchangeSummary[] }) {
  if (exchanges.length === 0) {
    return (
      <EmptyRegister sampled={sampled}>
        No exchanges. An exchange routes published messages into bound queues; declare one with New exchange.
      </EmptyRegister>
    );
  }
  return (
    <div className="bt-frame">
      <table className="bt">
        <thead>
          <tr>
            <th scope="col">Name</th>
            <th scope="col">Kind</th>
            <th scope="col">Durable</th>
            <th scope="col" className="n">Bindings</th>
          </tr>
        </thead>
        <tbody>
          {exchanges.map((exchange) => (
            <tr
              key={exchange.name}
              {...rowProps(selection?.kind === 'exchange' && selection.name === exchange.name, () =>
                onSelect({ kind: 'exchange', name: exchange.name }),
              )}
            >
              <td className="name">{exchange.name}</td>
              <td><span className="kindword">{exchange.kind}</span></td>
              <td>{exchange.durable ? 'Yes' : 'No'}</td>
              <td className="n">{exchange.bindings.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function LagTable({ rows, selection, onSelect, sampled }: TableProps & { rows: LagRow[] }) {
  if (rows.length === 0) {
    return (
      <EmptyRegister sampled={sampled}>
        No consumer groups have committed offsets against this project&rsquo;s topics yet.
      </EmptyRegister>
    );
  }
  return (
    <div className="bt-frame">
      <table className="bt">
        <thead>
          <tr>
            <th scope="col">Group</th>
            <th scope="col">Topic</th>
            <th scope="col" className="n">Partition</th>
            <th scope="col" className="n">Committed</th>
            <th scope="col" className="n">Next</th>
            <th scope="col" className="n">Lag</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const key = `${row.group}·${row.topic}·${row.partition}`;
            const selected =
              selection?.kind === 'lag' &&
              selection.group === row.group &&
              selection.topic === row.topic &&
              selection.partition === row.partition;
            return (
              <tr
                key={key}
                {...rowProps(selected, () =>
                  onSelect({ kind: 'lag', group: row.group, topic: row.topic, partition: row.partition }),
                )}
              >
                <td className="name">{row.group}</td>
                <td>{row.topic}</td>
                <td className="n">{row.partition.toLocaleString()}</td>
                <td className="n">{row.committed.toLocaleString()}</td>
                <td className="n">{row.next.toLocaleString()}</td>
                <td className={row.lag > 0 ? 'n hot' : 'n'}>{row.lag.toLocaleString()}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── The margin: reading one row, and every operation that can be run on it ── */

interface MarginProps {
  project?: Project;
  sample: BrokerSample;
  selection?: Selection;
  margin: MarginMode;
  confirm?: Confirm;
  running?: string;
  actionError?: string;
  onCloseCreate: () => void;
  onClearSelection: () => void;
  onConfirm: (confirm: Confirm) => void;
  onCancelConfirm: () => void;
  onRun: (statement: string, after?: () => void) => void;
}

function StreamMargin({
  project,
  sample,
  selection,
  margin,
  confirm,
  running,
  actionError,
  onCloseCreate,
  onClearSelection,
  onConfirm,
  onCancelConfirm,
  onRun,
}: MarginProps) {
  const heading =
    margin.at === 'create'
      ? margin.register === 'topics'
        ? 'New topic'
        : margin.register === 'queues'
          ? 'New queue'
          : 'New exchange'
      : selection
        ? 'Selected'
        : 'The broker';

  const census = (
    <div className="insp-census">
      <span className="ap origin">In this project</span>
      <div className="insp-census-row">
        <span className="insp-tally"><b>{sample.topics.length.toLocaleString()}</b><span className="ap faint">topics</span></span>
        <span className="insp-tally"><b>{sample.queues.length.toLocaleString()}</b><span className="ap faint">queues</span></span>
        <span className="insp-tally"><b>{sample.exchanges.length.toLocaleString()}</b><span className="ap faint">exchanges</span></span>
      </div>
      <p>
        Sampled every 2 seconds while this screen is open. Topics speak the Kafka protocol; queues and
        exchanges speak AMQP. One process, this machine.
      </p>
    </div>
  );

  return (
    <>
      <div className="vrule marg-rule"></div>
      <div className="marg insp">
        <div className="marg-h">
          <Cross />
          <span className="ap">{heading}</span>
          <span className="grow"></span>
          {margin.at === 'create' ? (
            <button className="insp-clear ap faint" type="button" onClick={onCloseCreate}>Close</button>
          ) : (
            selection && (
              <button className="insp-clear ap faint" type="button" onClick={onClearSelection}>Clear</button>
            )
          )}
        </div>

        {actionError && (
          <p className="insp-trouble" role="alert">{actionError}</p>
        )}

        {margin.at === 'create' && project ? (
          <>
            {margin.register === 'topics' && <CreateTopic running={running} onRun={onRun} onDone={onCloseCreate} />}
            {margin.register === 'queues' && <CreateQueue running={running} onRun={onRun} onDone={onCloseCreate} />}
            {margin.register === 'exchanges' && <CreateExchange running={running} onRun={onRun} onDone={onCloseCreate} />}
          </>
        ) : !selection ? (
          <>
            <p className="insp-lede">
              Select a row to read it here, with every operation that can be run against it. Each
              control states the one Cypher statement it runs — the query screen accepts the same
              statements typed by hand.
            </p>
            {census}
          </>
        ) : (
          <>
            <SelectedDetail
              selection={selection}
              sample={sample}
              confirm={confirm}
              running={running}
              onConfirm={onConfirm}
              onCancelConfirm={onCancelConfirm}
              onRun={onRun}
              onClearSelection={onClearSelection}
            />
            {census}
          </>
        )}
      </div>
    </>
  );
}

/** The pair every destructive control resolves into: the consequence, then the two detents. */
function ConfirmNotice({
  confirm,
  running,
  onRun,
  onCancel,
  onDone,
}: {
  confirm: Confirm;
  running?: string;
  onRun: (statement: string, after?: () => void) => void;
  onCancel: () => void;
  onDone?: () => void;
}) {
  return (
    <div className="notice contradiction" role="alertdialog" aria-label={confirm.label}>
      <span className="kindmark"></span>
      <span className="ap lbl">{confirm.label}</span>
      <p>{confirm.consequence}</p>
      <Stmt statement={confirm.statement} />
      <div className="acts">
        <button
          className="detent warn"
          type="button"
          disabled={running !== undefined}
          onClick={() => onRun(confirm.statement, onDone)}
        >
          {running === confirm.statement ? 'Working…' : confirm.verb}
        </button>
        <button className="detent" type="button" disabled={running !== undefined} onClick={onCancel}>
          Keep it
        </button>
      </div>
    </div>
  );
}

function SelectedDetail({
  selection,
  sample,
  confirm,
  running,
  onConfirm,
  onCancelConfirm,
  onRun,
  onClearSelection,
}: {
  selection: Selection;
  sample: BrokerSample;
  confirm?: Confirm;
  running?: string;
  onConfirm: (confirm: Confirm) => void;
  onCancelConfirm: () => void;
  onRun: (statement: string, after?: () => void) => void;
  onClearSelection: () => void;
}) {
  if (selection.kind === 'topic') {
    const topic = sample.topics.find((candidate) => candidate.name === selection.name);
    if (!topic) return <Gone onClearSelection={onClearSelection} />;
    return (
      <TopicDetail
        topic={topic}
        confirm={confirm}
        running={running}
        onConfirm={onConfirm}
        onCancelConfirm={onCancelConfirm}
        onRun={onRun}
        onClearSelection={onClearSelection}
      />
    );
  }
  if (selection.kind === 'queue') {
    const queue = sample.queues.find((candidate) => candidate.name === selection.name);
    if (!queue) return <Gone onClearSelection={onClearSelection} />;
    return (
      <QueueDetail
        queue={queue}
        exchanges={sample.exchanges}
        confirm={confirm}
        running={running}
        onConfirm={onConfirm}
        onCancelConfirm={onCancelConfirm}
        onRun={onRun}
        onClearSelection={onClearSelection}
      />
    );
  }
  if (selection.kind === 'exchange') {
    const exchange = sample.exchanges.find((candidate) => candidate.name === selection.name);
    if (!exchange) return <Gone onClearSelection={onClearSelection} />;
    return (
      <ExchangeDetail
        exchange={exchange}
        queues={sample.queues}
        confirm={confirm}
        running={running}
        onConfirm={onConfirm}
        onCancelConfirm={onCancelConfirm}
        onRun={onRun}
        onClearSelection={onClearSelection}
      />
    );
  }
  const row = sample.lag.find(
    (candidate) =>
      candidate.group === selection.group &&
      candidate.topic === selection.topic &&
      candidate.partition === selection.partition,
  );
  if (!row) return <Gone onClearSelection={onClearSelection} />;
  return <LagDetail row={row} />;
}

function Gone({ onClearSelection }: { onClearSelection: () => void }) {
  return (
    <>
      <p className="insp-lede">What was selected is no longer in the broker&rsquo;s answer.</p>
      <div className="insp-acts">
        <button className="detent" type="button" onClick={onClearSelection}>Clear selection</button>
      </div>
    </>
  );
}

function TopicDetail({
  topic,
  confirm,
  running,
  onConfirm,
  onCancelConfirm,
  onRun,
  onClearSelection,
}: {
  topic: TopicSummary;
  confirm?: Confirm;
  running?: string;
  onConfirm: (confirm: Confirm) => void;
  onCancelConfirm: () => void;
  onRun: (statement: string, after?: () => void) => void;
  onClearSelection: () => void;
}) {
  const clear = clearTopic(topic.name);
  const drop = dropTopic(topic.name);
  return (
    <>
      <div className="insp-head">
        <span className="ap">Topic</span>
        <span className="insp-id">{topic.partitions.length.toLocaleString()} partition{topic.partitions.length === 1 ? '' : 's'}</span>
      </div>
      <h3 className="insp-title">{topic.name}</h3>
      <dl className="insp-props">
        <div>
          <dt>records</dt>
          <dd><span className="insp-value">{topic.records.toLocaleString()}</span></dd>
        </div>
        <div>
          <dt>retained</dt>
          <dd><span className="insp-value">{formatBytes(topic.retainedBytes)}</span></dd>
        </div>
      </dl>

      <Section label="Partitions" count={topic.partitions.length} />
      <ul className="insp-rels">
        {topic.partitions.map((partition) => (
          <li key={partition.partition}>
            <span className="insp-rel-name">#{partition.partition}</span>
            <span className="insp-rel-count">{partition.records.toLocaleString()} records</span>
            <span className="insp-rel-count">offsets {partition.baseOffset.toLocaleString()}-{partition.nextOffset.toLocaleString()}</span>
          </li>
        ))}
      </ul>

      <RetentionEditor
        kind="topic"
        name={topic.name}
        currentDays={topic.retentionDays}
        running={running}
        onRun={onRun}
      />

      <Section label="Operations" />
      {confirm ? (
        <ConfirmNotice
          confirm={confirm}
          running={running}
          onRun={onRun}
          onCancel={onCancelConfirm}
          onDone={confirm.clearsSelection ? onClearSelection : undefined}
        />
      ) : (
        <div className="insp-acts">
          <button
            className="detent"
            type="button"
            disabled={running !== undefined}
            onClick={() =>
              onConfirm({
                statement: clear,
                label: 'Erases records',
                consequence: `Every record in all ${topic.partitions.length.toLocaleString()} partition${topic.partitions.length === 1 ? '' : 's'} of ${topic.name} is erased. The topic itself stays declared.`,
                verb: 'Clear it',
              })
            }
          >
            Clear records
          </button>
          <button
            className="detent warn"
            type="button"
            disabled={running !== undefined}
            onClick={() =>
              onConfirm({
                statement: drop,
                label: 'Removes the topic',
                consequence: `${topic.name} and its ${topic.records.toLocaleString()} record${topic.records === 1 ? '' : 's'} are removed. Producers and consumers using it will find nothing.`,
                verb: 'Drop it',
                clearsSelection: true,
              })
            }
          >
            Drop topic
          </button>
        </div>
      )}
    </>
  );
}

function QueueDetail({
  queue,
  exchanges,
  confirm,
  running,
  onConfirm,
  onCancelConfirm,
  onRun,
  onClearSelection,
}: {
  queue: QueueSummary;
  exchanges: ExchangeSummary[];
  confirm?: Confirm;
  running?: string;
  onConfirm: (confirm: Confirm) => void;
  onCancelConfirm: () => void;
  onRun: (statement: string, after?: () => void) => void;
  onClearSelection: () => void;
}) {
  const purge = purgeQueue(queue.name);
  const drop = dropQueue(queue.name);
  return (
    <>
      <div className="insp-head">
        <span className="ap">Queue</span>
      </div>
      <h3 className="insp-title">{queue.name}</h3>
      <div className="insp-labels">
        <span className="insp-lb">{queue.kind}</span>
      </div>
      <dl className="insp-props">
        <div>
          <dt>messages</dt>
          <dd><span className="insp-value">{queue.messages.toLocaleString()} · {queue.available.toLocaleString()} ready</span></dd>
        </div>
        <div>
          <dt>retained</dt>
          <dd><span className="insp-value">{formatBytes(queue.retainedBytes)}</span></dd>
        </div>
      </dl>

      <RetentionEditor
        kind="queue"
        name={queue.name}
        currentDays={queue.retentionDays}
        running={running}
        onRun={onRun}
      />

      <Section label="Operations" />
      {confirm ? (
        <ConfirmNotice
          confirm={confirm}
          running={running}
          onRun={onRun}
          onCancel={onCancelConfirm}
          onDone={confirm.clearsSelection ? onClearSelection : undefined}
        />
      ) : (
        <div className="insp-acts">
          <button
            className="detent"
            type="button"
            disabled={running !== undefined}
            onClick={() =>
              onConfirm({
                statement: purge,
                label: 'Erases messages',
                consequence: `Every message waiting in ${queue.name} is erased (${queue.messages.toLocaleString()} right now). The queue and its bindings stay.`,
                verb: 'Purge it',
              })
            }
          >
            Purge messages
          </button>
          <button
            className="detent warn"
            type="button"
            disabled={running !== undefined}
            onClick={() =>
              onConfirm({
                statement: drop,
                label: 'Removes the queue',
                consequence: `${queue.name} is removed with everything in it, and every binding pointing at it stops delivering.`,
                verb: 'Drop it',
                clearsSelection: true,
              })
            }
          >
            Drop queue
          </button>
        </div>
      )}

      <RoutingForm
        fixed={{ side: 'queue', name: queue.name }}
        counterparts={exchanges.map((exchange) => exchange.name)}
        running={running}
        onRun={onRun}
      />
    </>
  );
}

function RetentionEditor({
  kind,
  name,
  currentDays,
  running,
  onRun,
}: {
  kind: 'topic' | 'queue';
  name: string;
  currentDays?: number;
  running?: string;
  onRun: (statement: string) => void;
}) {
  const [days, setDays] = useState(currentDays === undefined ? '' : String(currentDays));
  const count = Number.parseInt(days, 10);
  const ready = Number.isFinite(count) && count >= 1;
  const statement = ready
    ? kind === 'topic'
      ? alterTopicRetention(name, count)
      : alterQueueRetention(name, count)
    : undefined;
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        if (statement && running === undefined) onRun(statement);
      }}
    >
      <Section label="Retention" />
      <label className="fld">
        <span className="ap faint">Length in days</span>
        <input type="number" min={1} step={1} value={days} onChange={(event) => setDays(event.target.value)} />
        <span className="fld-note">
          {currentDays === undefined ? 'Currently unlimited.' : `Currently ${currentDays.toLocaleString()} days.`}
        </span>
      </label>
      {statement && <Stmt statement={statement} />}
      <div className="insp-acts">
        <button className="detent" type="submit" disabled={!ready || running !== undefined}>
          {statement !== undefined && running === statement ? 'Working…' : 'Set retention'}
        </button>
      </div>
    </form>
  );
}

function ExchangeDetail({
  exchange,
  queues,
  confirm,
  running,
  onConfirm,
  onCancelConfirm,
  onRun,
  onClearSelection,
}: {
  exchange: ExchangeSummary;
  queues: QueueSummary[];
  confirm?: Confirm;
  running?: string;
  onConfirm: (confirm: Confirm) => void;
  onCancelConfirm: () => void;
  onRun: (statement: string, after?: () => void) => void;
  onClearSelection: () => void;
}) {
  const drop = dropExchange(exchange.name);
  return (
    <>
      <div className="insp-head">
        <span className="ap">Exchange</span>
      </div>
      <h3 className="insp-title">{exchange.name}</h3>
      <div className="insp-labels">
        <span className="insp-lb">{exchange.kind}</span>
        {exchange.durable && <span className="insp-lb">DURABLE</span>}
      </div>
      <dl className="insp-props">
        <div>
          <dt>bindings</dt>
          <dd>
            <span className="insp-value">
              {exchange.bindings.toLocaleString()}
            </span>
          </dd>
        </div>
      </dl>
      <p className="insp-note">
        The broker counts bindings; it does not list them. A binding is named by its queue, this
        exchange and its key — the three the routing form below takes.
      </p>

      <Section label="Operations" />
      {confirm ? (
        <ConfirmNotice
          confirm={confirm}
          running={running}
          onRun={onRun}
          onCancel={onCancelConfirm}
          onDone={confirm.clearsSelection ? onClearSelection : undefined}
        />
      ) : (
        <div className="insp-acts">
          <button
            className="detent warn"
            type="button"
            disabled={running !== undefined}
            onClick={() =>
              onConfirm({
                statement: drop,
                label: 'Removes the exchange',
                consequence: `${exchange.name} is removed, and everything published to it stops being routed anywhere.`,
                verb: 'Drop it',
                clearsSelection: true,
              })
            }
          >
            Drop exchange
          </button>
        </div>
      )}

      <RoutingForm
        fixed={{ side: 'exchange', name: exchange.name }}
        counterparts={queues.map((queue) => queue.name)}
        running={running}
        onRun={onRun}
      />
    </>
  );
}

function LagDetail({ row }: { row: LagRow }) {
  return (
    <>
      <div className="insp-head">
        <span className="ap">Consumer group</span>
        <span className="insp-id">partition #{row.partition}</span>
      </div>
      <h3 className="insp-title">{row.group}</h3>
      <div className="insp-labels">
        <span className="insp-lb">{row.topic}</span>
      </div>
      <dl className="insp-props">
        <div>
          <dt>committed offset</dt>
          <dd><span className="insp-value">{row.committed.toLocaleString()}</span></dd>
        </div>
        <div>
          <dt>next offset</dt>
          <dd><span className="insp-value">{row.next.toLocaleString()}</span></dd>
        </div>
        <div>
          <dt>lag</dt>
          <dd>
            <span className="insp-value">
              {row.lag.toLocaleString()}
              {row.lag === 0 ? ' · caught up' : ' records behind'}
            </span>
          </dd>
        </div>
      </dl>
      <p className="insp-note">
        Offsets are committed by the consumers themselves; there is nothing to operate on here.
        A group that stops consuming keeps its committed offset and its lag grows with the topic.
      </p>
    </>
  );
}

/* ── The routing form: one binding, named in full, bound or unbound ────────── */

function RoutingForm({
  fixed,
  counterparts,
  running,
  onRun,
}: {
  /** The side the margin is already about; the form asks only for the other side and the key. */
  fixed: { side: 'queue' | 'exchange'; name: string };
  counterparts: string[];
  running?: string;
  onRun: (statement: string, after?: () => void) => void;
}) {
  const [verb, setVerb] = useState<'BIND' | 'UNBIND'>('BIND');
  const [counterpart, setCounterpart] = useState('');
  const [key, setKey] = useState('');
  const chosen = counterpart || counterparts[0] || '';

  const queue = fixed.side === 'queue' ? fixed.name : chosen;
  const exchange = fixed.side === 'exchange' ? fixed.name : chosen;
  const ready = validName(queue) && validName(exchange) && validName(key);
  const statement = ready
    ? verb === 'BIND'
      ? bindQueue(queue, exchange, key)
      : unbindQueue(queue, exchange, key)
    : undefined;

  if (counterparts.length === 0) {
    return (
      <>
        <Section label="Routing" />
        <p className="insp-none">
          {fixed.side === 'queue'
            ? 'No exchanges to bind to. Declare one under Exchanges first.'
            : 'No queues to bind. Declare one under Queues first.'}
        </p>
      </>
    );
  }

  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        if (statement && running === undefined) onRun(statement, () => setKey(''));
      }}
    >
      <Section label="Routing" />
      <div className="seg-set" role="group" aria-label="Bind or unbind">
        <button type="button" aria-pressed={verb === 'BIND'} onClick={() => setVerb('BIND')}>Bind</button>
        <button type="button" aria-pressed={verb === 'UNBIND'} onClick={() => setVerb('UNBIND')}>Unbind</button>
      </div>
      <label className="fld">
        <span className="ap faint">{fixed.side === 'queue' ? 'To exchange' : 'Queue'}</span>
        <select value={chosen} onChange={(event) => setCounterpart(event.target.value)}>
          {counterparts.map((name) => (
            <option key={name} value={name}>{name}</option>
          ))}
        </select>
      </label>
      <label className="fld">
        <span className="ap faint">Routing key</span>
        <input
          type="text"
          value={key}
          onChange={(event) => setKey(event.target.value)}
          spellCheck={false}
          autoComplete="off"
          placeholder={fixed.side === 'queue' ? 'orders' : 'orders.created'}
        />
        <span className="fld-note">
          A DIRECT exchange routes on the exact key; a TOPIC exchange on dotted patterns; FANOUT
          ignores the key but the binding still names one.
        </span>
      </label>
      {statement && <Stmt statement={statement} />}
      <div className="insp-acts">
        <button className="detent" type="submit" disabled={!ready || running !== undefined}>
          {statement !== undefined && running === statement ? 'Working…' : verb === 'BIND' ? 'Bind' : 'Unbind'}
        </button>
      </div>
    </form>
  );
}

/* ── The three declarations ───────────────────────────────────────────────── */

function CreateTopic({
  running,
  onRun,
  onDone,
}: {
  running?: string;
  onRun: (statement: string, after?: () => void) => void;
  onDone: () => void;
}) {
  const [name, setName] = useState('');
  const [partitions, setPartitions] = useState('1');
  const [retentionDays, setRetentionDays] = useState('7');
  const count = Number.parseInt(partitions, 10);
  const retention = Number.parseInt(retentionDays, 10);
  const ready = validName(name) && Number.isFinite(count) && count >= 1 && count <= PARTITION_LIMIT
    && Number.isFinite(retention) && retention >= 1;
  const statement = ready ? createTopic(name.trim(), count, retention) : undefined;
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        if (statement && running === undefined) onRun(statement, onDone);
      }}
    >
      <p className="insp-lede">
        A topic is a partitioned, replayable log. Producers append records; consumer groups read
        them in order within each partition, at their own pace.
      </p>
      <label className="fld">
        <span className="ap faint">Topic name</span>
        <input type="text" value={name} onChange={(event) => setName(event.target.value)} autoFocus spellCheck={false} autoComplete="off" />
      </label>
      <label className="fld">
        <span className="ap faint">Retention days</span>
        <input type="number" min={1} step={1} value={retentionDays} onChange={(event) => setRetentionDays(event.target.value)} />
        <span className="fld-note">Records older than this are removed by broker retention.</span>
      </label>
      <label className="fld">
        <span className="ap faint">Partitions</span>
        <input
          type="number"
          min={1}
          max={PARTITION_LIMIT}
          value={partitions}
          onChange={(event) => setPartitions(event.target.value)}
        />
        <span className="fld-note">
          The topic&rsquo;s parallelism: consumers in one group split partitions between them.
          Records in one partition stay ordered. This cannot be changed later.
        </span>
      </label>
      {statement && <Stmt statement={statement} />}
      <div className="insp-acts">
        <button className="detent" type="submit" disabled={!ready || running !== undefined}>
          {statement !== undefined && running === statement ? 'Working…' : 'Create topic'}
        </button>
      </div>
    </form>
  );
}

function CreateQueue({
  running,
  onRun,
  onDone,
}: {
  running?: string;
  onRun: (statement: string, after?: () => void) => void;
  onDone: () => void;
}) {
  const [name, setName] = useState('');
  const [kind, setKind] = useState<QueueKind>('CLASSIC');
  const [retentionDays, setRetentionDays] = useState('7');
  const retention = Number.parseInt(retentionDays, 10);
  const ready = validName(name) && Number.isFinite(retention) && retention >= 1;
  const statement = ready ? createQueue(name.trim(), kind, retention) : undefined;
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        if (statement && running === undefined) onRun(statement, onDone);
      }}
    >
      <p className="insp-lede">
        A queue delivers each message to one consumer. Publish into it directly, or bind it to an
        exchange and let the exchange route.
      </p>
      <label className="fld">
        <span className="ap faint">Queue name</span>
        <input type="text" value={name} onChange={(event) => setName(event.target.value)} autoFocus spellCheck={false} autoComplete="off" />
      </label>
      <div className="seg-set" role="group" aria-label="Queue kind">
        <button type="button" aria-pressed={kind === 'CLASSIC'} onClick={() => setKind('CLASSIC')}>Classic</button>
        <button type="button" aria-pressed={kind === 'STREAM'} onClick={() => setKind('STREAM')}>Stream</button>
      </div>
      <p className="insp-note">
        {kind === 'CLASSIC'
          ? 'A classic queue deletes each message once a consumer acknowledges it.'
          : 'A stream queue keeps a replayable log; consumers read from any offset without consuming it away.'}
      </p>
      <label className="fld">
        <span className="ap faint">Retention days</span>
        <input type="number" min={1} step={1} value={retentionDays} onChange={(event) => setRetentionDays(event.target.value)} />
        <span className="fld-note">Messages older than this are removed by broker retention.</span>
      </label>
      {statement && <Stmt statement={statement} />}
      <div className="insp-acts">
        <button className="detent" type="submit" disabled={!ready || running !== undefined}>
          {statement !== undefined && running === statement ? 'Working…' : 'Create queue'}
        </button>
      </div>
    </form>
  );
}

function CreateExchange({
  running,
  onRun,
  onDone,
}: {
  running?: string;
  onRun: (statement: string, after?: () => void) => void;
  onDone: () => void;
}) {
  const [name, setName] = useState('');
  const [kind, setKind] = useState<ExchangeKind>('DIRECT');
  const ready = validName(name);
  const statement = ready ? createExchange(name.trim(), kind) : undefined;
  const notes: Record<ExchangeKind, string> = {
    DIRECT: 'Routes a message to the queues whose binding key equals the message key exactly.',
    FANOUT: 'Routes every message to every bound queue, keys ignored.',
    TOPIC: 'Routes on dotted patterns: a binding key like orders.* matches orders.created.',
  };
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault();
        if (statement && running === undefined) onRun(statement, onDone);
      }}
    >
      <p className="insp-lede">
        An exchange receives published messages and routes them into whichever queues are bound to
        it. What a binding key means depends on the exchange&rsquo;s kind.
      </p>
      <label className="fld">
        <span className="ap faint">Exchange name</span>
        <input type="text" value={name} onChange={(event) => setName(event.target.value)} autoFocus spellCheck={false} autoComplete="off" />
      </label>
      <div className="seg-set" role="group" aria-label="Exchange kind">
        <button type="button" aria-pressed={kind === 'DIRECT'} onClick={() => setKind('DIRECT')}>Direct</button>
        <button type="button" aria-pressed={kind === 'FANOUT'} onClick={() => setKind('FANOUT')}>Fanout</button>
        <button type="button" aria-pressed={kind === 'TOPIC'} onClick={() => setKind('TOPIC')}>Topic</button>
      </div>
      <p className="insp-note">{notes[kind]}</p>
      {statement && <Stmt statement={statement} />}
      <div className="insp-acts">
        <button className="detent" type="submit" disabled={!ready || running !== undefined}>
          {statement !== undefined && running === statement ? 'Working…' : 'Create exchange'}
        </button>
      </div>
    </form>
  );
}
