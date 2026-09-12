const sources: Record<string, string> = import.meta.glob('../../../../docs/**/*.md', {
  query: '?raw',
  import: 'default',
  eager: true,
});

export interface DocPage {
  id: string;
  title: string;
  section: string;
  body: string;
}

function titleOf(body: string, fallback: string): string {
  return /^#\s+(.+)$/m.exec(body)?.[1]?.replaceAll('`', '') ?? fallback;
}

function sectionOf(id: string): string {
  if (!id.includes('/')) return 'Start';
  if (id === 'cypher/datasets') return 'Cypher';
  if (id.startsWith('cypher/')) {
    const part = id.split('/')[1];
    return part ? `Cypher · ${part.replaceAll('-', ' ')}` : 'Cypher';
  }
  return id.split('/')[0]!.replaceAll('-', ' ');
}

const SECTION_ORDER = ['Start', 'Cypher', 'Cypher · clauses', 'Cypher · statements', 'Cypher · functions', 'Cypher · procedures'];
const START_ORDER = ['getting-started', 'installation', 'README', 'features', 'embedding', 'architecture'];

function orderOf(value: string, order: string[]): number {
  const index = order.indexOf(value);
  return index === -1 ? order.length : index;
}

export const DOC_PAGES: DocPage[] = Object.entries(sources).map(([path, body]) => {
  const id = path.split('/docs/')[1]!.replace(/\.md$/, '').replace(/\/README$/, '');
  const section = id === 'cypher' ? 'Cypher' : sectionOf(id);
  return { id, title: titleOf(body, id), section, body };
}).sort((a, b) => {
  const section = orderOf(a.section, SECTION_ORDER) - orderOf(b.section, SECTION_ORDER);
  if (section) return section;
  if (a.section === 'Start') {
    const start = orderOf(a.id, START_ORDER) - orderOf(b.id, START_ORDER);
    if (start) return start;
  }
  return a.title.localeCompare(b.title);
});

export function resolveDocLink(currentId: string, href: string): string | undefined {
  if (/^(?:https?:|mailto:|#)/.test(href)) return undefined;
  const clean = href.split('#')[0]?.replace(/\.md$/, '').replace(/\/README$/, '');
  if (!clean) return currentId;
  const resolveFrom = (base: string) => {
    const resolved: string[] = [];
    `${base}${clean}`.split('/').forEach((part) => {
      if (part === '..') resolved.pop();
      else if (part !== '.' && part) resolved.push(part);
    });
    return resolved.join('/');
  };
  const parent = currentId.includes('/') ? currentId.slice(0, currentId.lastIndexOf('/') + 1) : '';
  const candidates = [resolveFrom(parent), resolveFrom(currentId ? `${currentId}/` : '')];
  return candidates.find((id) => DOC_PAGES.some((page) => page.id === id));
}
