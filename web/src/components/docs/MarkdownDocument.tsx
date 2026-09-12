import { Fragment, type ReactNode, useState } from 'react';
import type { Project, QueryResult } from '../../types';
import { executeQuery } from '../../lib/simpleQuery';
import { errorMessage } from '../../lib/format';
import { ResultTable } from '../common/ResultTable';
import { resolveDocLink } from './docsCatalog';
import { createProject } from '../../lib/api';
import { isBundledDataset } from '../../lib/datasets';

interface Props {
  body: string;
  docId: string;
  projects: Project[];
  onProjectsChanged: () => Promise<void>;
  onNavigate: (id: string) => void;
}

function inline(text: string, docId: string, onNavigate: (id: string) => void): ReactNode[] {
  const pieces: ReactNode[] = [];
  const pattern = /(`[^`]+`|\[[^\]]+\]\([^)]+\)|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let at = 0;
  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > at) pieces.push(text.slice(at, index));
    const token = match[0];
    if (token.startsWith('`')) pieces.push(<code key={index}>{token.slice(1, -1)}</code>);
    else if (token.startsWith('**')) pieces.push(<strong key={index}>{token.slice(2, -2)}</strong>);
    else if (token.startsWith('*')) pieces.push(<em key={index}>{token.slice(1, -1)}</em>);
    else {
      const link = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(token)!;
      const target = resolveDocLink(docId, link[2]!);
      const codeLabel = /^`([^`]+)`$/.exec(link[1]!);
      const label = codeLabel ? <code>{codeLabel[1]}</code> : link[1];
      pieces.push(target
        ? <button className="doc-link" type="button" key={index} onClick={() => onNavigate(target)}>{label}</button>
        : <a key={index} href={link[2]} target={link[2]!.startsWith('http') ? '_blank' : undefined} rel="noreferrer">{label}</a>);
    }
    at = index + token.length;
  }
  if (at < text.length) pieces.push(text.slice(at));
  return pieces;
}

function CodeBlock({ code, language, projects, onProjectsChanged }: { code: string; language: string; projects: Project[]; onProjectsChanged: () => Promise<void> }) {
  const [result, setResult] = useState<QueryResult>();
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string>();
  const [copied, setCopied] = useState(false);
  const [provisioning, setProvisioning] = useState(false);
  const cypher = language === 'cypher';
  const projectName = /^\s*USE\s+`?([\w-]+)`?/im.exec(code)?.[1];
  const project = projectName ? projects.find((entry) => entry.name === projectName) : undefined;
  const runnable = cypher;
  const missingProject = Boolean(projectName && !project);
  const importsDataset = Boolean(projectName && isBundledDataset(projectName));
  const lines = code.trim().split('\n');
  const divider = lines.findIndex((line) => /^[-+|\s]+$/.test(line) && line.includes('-'));
  const tableRows = divider === 1 && lines[0]?.includes('|')
    ? lines.filter((line, index) => index !== divider && line.includes('|')).map((line) => line.split('|').map((cell) => cell.trim()))
    : [];
  const rowNote = tableRows.length ? lines.find((line) => /^\d+\s+rows?$/.test(line.trim()))?.trim() : undefined;

  const run = async () => {
    if (!runnable || running || missingProject) return;
    setRunning(true);
    setResult(undefined);
    setError(undefined);
    try { setResult(await executeQuery(code, project?.id)); }
    catch (cause) { setError(errorMessage(cause)); }
    finally { setRunning(false); }
  };

  const provisionProject = async () => {
    if (!projectName || provisioning) return;
    setProvisioning(true);
    setResult(undefined);
    setError(undefined);
    try {
      if (importsDataset) await executeQuery(`IMPORT DATASET ${projectName}`);
      else await createProject(projectName);
      await onProjectsChanged();
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setProvisioning(false);
    }
  };

  return (
    <div className="doc-code">
      <div className="bar bare"><span className="ap origin">{language || 'text'}</span><span className="grow"></span>{cypher && missingProject && <span className="ap faint">Requires {projectName}</span>}{cypher && missingProject && <button className="detent" type="button" onClick={() => void provisionProject()} disabled={provisioning}>{provisioning ? (importsDataset ? 'Importing…' : 'Creating…') : `${importsDataset ? 'Import' : 'Create'} ${projectName}`}</button>}<button className="detent" type="button" onClick={() => void navigator.clipboard.writeText(code).then(() => setCopied(true))}>{copied ? 'Copied' : 'Copy'}</button>{cypher && <button className="detent" type="button" aria-pressed="true" onClick={() => void run()} disabled={running || missingProject}>{running ? 'Running…' : 'Run'}</button>}</div>
      {tableRows.length ? (
        <div className="doc-result-table">
          <div className="bt-frame"><table className="bt"><thead><tr>{tableRows[0]!.map((cell, index) => <th key={index}>{cell}</th>)}</tr></thead><tbody>{tableRows.slice(1).map((row, rowIndex) => <tr key={rowIndex}>{row.map((cell, cellIndex) => <td key={cellIndex}>{cell}</td>)}</tr>)}</tbody></table></div>
          {rowNote && <p className="ap faint">{rowNote}</p>}
        </div>
      ) : <pre className="stmt"><code>{code}</code></pre>}
      {error && <div className="notice contradiction" role="alert"><span className="kindmark"></span><span className="ap lbl">Query failed</span><p>{error}</p></div>}
      {result && <ResultTable result={result} />}
    </div>
  );
}

/** Small, safe renderer for the Markdown forms used by the bundled public documentation. */
export function MarkdownDocument({ body, docId, projects, onProjectsChanged, onNavigate }: Props) {
  const lines = body.split('\n');
  const blocks: ReactNode[] = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index] ?? '';
    if (!line.trim()) { index += 1; continue; }
    if (line.startsWith('```')) {
      const language = line.slice(3).trim();
      const code: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index]!.startsWith('```')) code.push(lines[index++]!);
      index += 1;
      blocks.push(<CodeBlock key={`code-${index}`} code={code.join('\n')} language={language} projects={projects} onProjectsChanged={onProjectsChanged} />);
      continue;
    }
    const heading = /^(#{1,4})\s+(.+)$/.exec(line);
    if (heading) {
      const level = heading[1]!.length;
      const content = inline(heading[2]!, docId, onNavigate);
      if (level === 1) blocks.push(<h1 className="pagekey" key={index}>{content}</h1>);
      else if (level === 2) blocks.push(<h2 key={index}>{content}</h2>);
      else blocks.push(<h3 key={index}>{content}</h3>);
      index += 1; continue;
    }
    if (/^[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^[-*]\s+/.test(lines[index]!)) items.push(lines[index++]!.replace(/^[-*]\s+/, ''));
      blocks.push(<ul key={`list-${index}`}>{items.map((item, itemIndex) => <li key={itemIndex}>{inline(item, docId, onNavigate)}</li>)}</ul>);
      continue;
    }
    if (/^\d+\.\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && /^\d+\.\s+/.test(lines[index]!)) items.push(lines[index++]!.replace(/^\d+\.\s+/, ''));
      blocks.push(<ol key={`order-${index}`}>{items.map((item, itemIndex) => <li key={itemIndex}>{inline(item, docId, onNavigate)}</li>)}</ol>);
      continue;
    }
    if (line.startsWith('|') && lines[index + 1]?.match(/^\|?\s*:?-+/)) {
      const rows: string[][] = [];
      rows.push(line.split('|').slice(1, -1).map((cell) => cell.trim()));
      index += 2;
      while (index < lines.length && lines[index]!.startsWith('|')) rows.push(lines[index++]!.split('|').slice(1, -1).map((cell) => cell.trim()));
      blocks.push(<div className="doc-table" key={`table-${index}`}><table className="bt"><thead><tr>{rows[0]!.map((cell, cellIndex) => <th key={cellIndex}>{inline(cell, docId, onNavigate)}</th>)}</tr></thead><tbody>{rows.slice(1).map((row, rowIndex) => <tr key={rowIndex}>{row.map((cell, cellIndex) => <td key={cellIndex}>{inline(cell, docId, onNavigate)}</td>)}</tr>)}</tbody></table></div>);
      continue;
    }
    if (line === '---') { blocks.push(<hr key={index} />); index += 1; continue; }
    const paragraph = [line];
    index += 1;
    while (index < lines.length && lines[index]!.trim() && !/^(#{1,4})\s|^```|^[-*]\s+|^\d+\.\s+|^\|/.test(lines[index]!)) paragraph.push(lines[index++]!);
    blocks.push(<p key={`p-${index}`}>{inline(paragraph.join(' '), docId, onNavigate)}</p>);
  }
  return <Fragment>{blocks}</Fragment>;
}
