import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { CompletionSchema } from '../../types';
import type { SchemaCensus as Census } from '../../lib/api';
import { SchemaCensus } from './SchemaCensus';

afterEach(cleanup);

const schema: CompletionSchema = {
  labels: ['Event', 'Audit', 'Endpoint'],
  relationshipTypes: ['SENT_TO', 'HAS_TURN'],
  properties: [],
  functions: [],
};

const census: Census = {
  nodes: 929,
  edges: 455,
  labels: new Map([['Event', 73], ['Endpoint', 62], ['Audit', 7]]),
  relationshipTypes: new Map([['SENT_TO', 77], ['HAS_TURN', 4]]),
  layersOfLabel: new Map([['Event', ['OBSERVED']], ['Endpoint', ['KNOWLEDGE']], ['Audit', ['WORKSPACE']]]),
  layersOfRelationshipType: new Map([['SENT_TO', ['OBSERVED']], ['HAS_TURN', ['WORKSPACE']]]),
  byLayer: [
    { layer: 'OBSERVED', nodes: 554, edges: 300, labels: new Map(), relationshipTypes: new Map() },
    { layer: 'KNOWLEDGE', nodes: 278, edges: 143, labels: new Map(), relationshipTypes: new Map() },
    { layer: 'WORKSPACE', nodes: 97, edges: 12, labels: new Map(), relationshipTypes: new Map() },
  ],
};

function panel(props: Partial<Parameters<typeof SchemaCensus>[0]> = {}) {
  return (
    <SchemaCensus
      schema={schema}
      census={census}
      loading={false}
      open
      onOpenChange={vi.fn()}
      onPick={vi.fn()}
      {...props}
    />
  );
}

describe('the schema census', () => {
  it('states the totals without being opened, and lists nothing until it is', () => {
    const { container, rerender } = render(panel({ open: false }));

    expect(container.querySelector('.census-bar')?.textContent)
      .toContain('929 nodes · 455 edges in 3 layers');
    expect(screen.queryByText(':Event')).not.toBeInTheDocument();

    rerender(panel({ open: true }));
    expect(screen.getByText(':Event')).toBeInTheDocument();
  });

  it('orders both lists by how much of each thing there is', () => {
    render(panel());

    const labels = screen.getAllByText(/^:/).map((node) => node.textContent);
    expect(labels).toEqual([':Event', ':Endpoint', ':Audit']);
  });

  it('marks what a plain statement cannot see, and only that', () => {
    const { container } = render(panel());

    // Two rows live in WORKSPACE — one label, one relationship type — and only those are marked.
    const marks = [...container.querySelectorAll('.census-layer')].map((node) => node.textContent);
    expect(marks).toEqual(['workspace', 'workspace']);
    expect(container.querySelector('.census-layer')?.closest('button')?.textContent).toContain(':Audit');
  });

  it('writes a workspace row with its layer, and a default-visible row without one', () => {
    const onPick = vi.fn();
    render(panel({ onPick }));

    fireEvent.click(screen.getByRole('button', { name: /:Audit/ }));
    expect(onPick).toHaveBeenCalledWith('USE LAYER WORKSPACE\nMATCH (n:Audit)\nRETURN n\nLIMIT 100');

    fireEvent.click(screen.getByRole('button', { name: /:Event/ }));
    expect(onPick).toHaveBeenLastCalledWith('MATCH (n:Event)\nRETURN n\nLIMIT 100');
  });

  it('asks a relationship type for both of its endpoints, so the result is a graph', () => {
    const onPick = vi.fn();
    render(panel({ onPick }));

    fireEvent.click(screen.getByRole('button', { name: /SENT_TO/ }));
    expect(onPick).toHaveBeenCalledWith(
      'MATCH (source)-[relationship:SENT_TO]->(target)\nRETURN source, relationship, target\nLIMIT 100',
    );
  });

  it('accounts for every layer, so the user’s own workspace is not left out of the total', () => {
    render(panel());

    expect(screen.getByText('554')).toBeInTheDocument();
    expect(screen.getByText('278')).toBeInTheDocument();
    expect(screen.getByText('97')).toBeInTheDocument();
    expect(screen.getByText(/reads observed and knowledge/)).toBeInTheDocument();
  });

  it('says the catalogue is still being read rather than claiming it is empty', () => {
    render(panel({ schema: { labels: [], relationshipTypes: [], properties: [], functions: [] }, census: undefined, loading: true }));

    expect(screen.getByText('reading the catalogue…')).toBeInTheDocument();
  });
});
