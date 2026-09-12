import { useMemo, useRef, useState } from 'react';
import type { Project, QueryResult } from '../../types';
import { errorMessage } from '../../lib/format';
import { executeQuery } from '../../lib/simpleQuery';
import { Apparatus, Rail, Screen } from '../../design/parts';
import { ResultTable } from '../common/ResultTable';
import { DATASETS, TRAINING_LESSONS } from './trainingData';

interface Props {
  projects: Project[];
  onProjectsChanged: () => Promise<void>;
}

export function TrainingPage({ projects, onProjectsChanged }: Props) {
  const [selected, setSelected] = useState(TRAINING_LESSONS[0]!.id);
  const [result, setResult] = useState<QueryResult>();
  const [running, setRunning] = useState(false);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState<string>();
  const abortRef = useRef<AbortController | undefined>(undefined);
  const lesson = TRAINING_LESSONS.find((entry) => entry.id === selected) ?? TRAINING_LESSONS[0]!;
  const dataset = DATASETS.find((entry) => entry.slug === lesson.dataset)!;
  const project = projects.find((entry) => entry.name === lesson.dataset);
  const sameDataset = useMemo(() => TRAINING_LESSONS.filter((entry) => entry.dataset === lesson.dataset), [lesson.dataset]);

  const choose = (id: string) => {
    setSelected(id);
    setResult(undefined);
    setError(undefined);
    abortRef.current?.abort();
    setRunning(false);
  };

  const importDataset = async () => {
    if (importing || project) return;
    setImporting(true);
    setError(undefined);
    try {
      await executeQuery(`IMPORT DATASET ${dataset.slug}`);
      await onProjectsChanged();
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setImporting(false);
    }
  };

  const run = async () => {
    if (!project || running) return;
    const controller = new AbortController();
    abortRef.current = controller;
    setRunning(true);
    setError(undefined);
    setResult(undefined);
    try {
      setResult(await executeQuery(lesson.query, project.id, {}, controller.signal));
    } catch (cause) {
      if (!(cause instanceof DOMException && cause.name === 'AbortError')) setError(errorMessage(cause));
    } finally {
      if (!controller.signal.aborted) setRunning(false);
    }
  };

  return (
    <Screen name="training">
      <div className="col-i training-index">
        <div className="bar"><span className="ap">Field guide</span><span className="grow"></span><span className="ap faint">10 investigations</span></div>
        <ul className="lst lesson-list">{TRAINING_LESSONS.map((entry) => (
          <li key={entry.id} className={entry.id === lesson.id ? 'on' : undefined}>
            <button type="button" onClick={() => choose(entry.id)} aria-current={entry.id === lesson.id ? 'step' : undefined}>
              <span className="ap origin">{entry.eyebrow}</span><span className="t">{entry.title}</span>
              <span className="r"><span className="ap faint">{entry.dataset}</span><span className="grow"></span><span className={`pip${projects.some((item) => item.name === entry.dataset) ? '' : ' hollow'}`}></span></span>
            </button>
          </li>
        ))}</ul>
      </div>
      <Rail screen="training" cap="Lessons" keys={TRAINING_LESSONS.map((entry, index) => ({
        t: String(index + 1).padStart(2, '0'), on: entry.id === lesson.id, title: entry.title, onSelect: () => choose(entry.id),
      }))} foot={`${sameDataset.length} on ${lesson.dataset}`} />
      <article className="col-ii training-work">
        <svg className="rd-over" aria-hidden></svg>
        <p className="ap origin">{lesson.eyebrow} · {dataset.slug}</p>
        <h1 className="pagekey">{lesson.title}</h1>
        <p className="lesson-question" data-mark="question">{lesson.question}</p>
        {!project && (
          <section className="import-gate" data-mark="import">
            <div className="import-state"><span className="pip hollow"></span><span className="ap live">Dataset not loaded</span></div>
            <h2>Import into <code>{dataset.slug}</code></h2>
            <p>IronGraph includes this dataset and imports it into its own project. If that project already exists, it is left untouched.</p>
            <div className="bar bare">
              <button className="detent" type="button" aria-pressed="true" onClick={() => void importDataset()} disabled={importing}>{importing ? 'Importing…' : 'Import dataset'}</button>
            </div>
            <p className="ap faint">{dataset.scale}</p>
          </section>
        )}
        {project && <div className="dataset-ready" role="status"><span className="pip"></span><span><b>{dataset.slug}</b> is loaded. Re-running the loader will leave it untouched.</span></div>}
        <section className="lesson-query" data-mark="query">
          <div className="bar bare"><span className="ap">Query</span><span className="grow"></span>{running && <span className="ap live">Running…</span>}<button className="detent" type="button" aria-pressed={project ? true : undefined} onClick={() => void run()} disabled={!project || running}>{running ? 'Working…' : 'Run query'}</button></div>
          <pre className="stmt"><code>{lesson.query}</code></pre>
        </section>
        {error && <div className="notice contradiction" role="alert"><span className="kindmark"></span><span className="ap lbl">Operation failed</span><p>{error}</p></div>}
        {result && <section className="lesson-answer" data-mark="result"><div className="bar bare"><span className="ap origin">Result</span><span className="grow"></span><span className="ap faint">{result.rows.length} row{result.rows.length === 1 ? '' : 's'}</span></div><ResultTable result={result} /></section>}
        <section className="lesson-explanation" data-mark="explanation"><p className="ap">How it works</p><ol>{lesson.explanation.map((paragraph) => <li key={paragraph}>{paragraph}</li>)}</ol></section>
      </article>
      <Apparatus heading="Lesson" notes={[
        { lbl: 'Question', p: lesson.question, anchor: 'question' },
        { lbl: project ? 'Ready' : 'Import first', p: project ? `${dataset.slug} is present as its own project.` : `Import ${dataset.slug}; IronGraph creates its project and loads the bundled data.`, anchor: project ? 'query' : 'import', origin: true },
        { lbl: 'Result', p: result ? `${result.rows.length} rows returned. Read them beside the query that produced them.` : 'Run the query to make the explanation concrete.', anchor: result ? 'result' : 'query' },
        { lbl: 'Method', p: 'Each clause is unpacked below the live result.', anchor: 'explanation' },
      ]} />
    </Screen>
  );
}
