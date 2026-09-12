import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { GraphEdge, GraphNode } from '../../types';
import { DEFAULT_VIEW_STATE } from '../../lib/graphView';
import { GraphInspector } from './GraphInspector';

afterEach(cleanup);

const nodes: GraphNode[] = [
  {
    id: '11',
    labels: ['Person', 'Sender'],
    properties: { name: 'Ada', posts: 37, retired: null, __irongraph_layer: 'OBSERVED' },
  },
  { id: '12', labels: ['Thread'], properties: { subject: 'Back to school' } },
];

const edges: GraphEdge[] = [
  { id: '90', source: '11', target: '12', relationshipType: 'IN_THREAD', properties: { at: 1755 } },
];

function inspector(props: Partial<Parameters<typeof GraphInspector>[0]> = {}) {
  return (
    <GraphInspector
      nodes={nodes}
      edges={edges}
      rowCount={5}
      view={DEFAULT_VIEW_STATE}
      onViewChange={vi.fn()}
      onSelect={vi.fn()}
      onExpand={vi.fn()}
      {...props}
    />
  );
}

describe('reading a selected node in the margin', () => {
  it('names the node, its kinds, what it is attached to here, and every readable property', () => {
    render(inspector({ selection: { kind: 'node', id: '11' } }));

    expect(screen.getByRole('heading', { name: 'Ada' })).toBeInTheDocument();
    expect(screen.getByText(':Person')).toBeInTheDocument();
    expect(screen.getByText(':Sender')).toBeInTheDocument();
    expect(screen.getByText('#11')).toBeInTheDocument();
    expect(screen.getByText('IN_THREAD')).toBeInTheDocument();
    expect(screen.getByText('posts')).toBeInTheDocument();
    expect(screen.getByText('37')).toBeInTheDocument();
    expect(screen.queryByText('retired')).not.toBeInTheDocument();
    expect(screen.queryByText('__irongraph_layer')).not.toBeInTheDocument();
  });

  it('runs the traversal for the selected node and says so while it is running', () => {
    const onExpand = vi.fn();
    const { rerender } = render(inspector({ selection: { kind: 'node', id: '11' }, onExpand }));

    fireEvent.click(screen.getByRole('button', { name: 'Expand' }));
    expect(onExpand).toHaveBeenCalledWith('11');

    rerender(inspector({ selection: { kind: 'node', id: '11' }, onExpand, expandingNodeId: '11' }));
    expect(screen.getByRole('button', { name: 'Traversing…' })).toBeDisabled();
  });

  it('reports what the last traversal brought back, and what went wrong when it failed', () => {
    const { rerender } = render(inspector({
      selection: { kind: 'node', id: '11' },
      expandNote: 'Pulled in 4 nodes and 6 relationships.',
    }));
    expect(screen.getByText('Pulled in 4 nodes and 6 relationships.')).toBeInTheDocument();

    rerender(inspector({ selection: { kind: 'node', id: '11' }, expandError: 'GPU_ADMISSION_FAILURE: refused' }));
    expect(screen.getByRole('alert')).toHaveTextContent('GPU_ADMISSION_FAILURE: refused');
  });
});

describe('reading a selected relationship in the margin', () => {
  it('shows both endpoints and its properties, and selects an endpoint when it is clicked', () => {
    const onSelect = vi.fn();
    render(inspector({ selection: { kind: 'edge', id: '90' }, onSelect }));

    expect(screen.getByText('IN_THREAD')).toBeInTheDocument();
    expect(screen.getByText('at')).toBeInTheDocument();
    expect(screen.getByText('1755')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: /Ada/ }));
    expect(onSelect).toHaveBeenCalledWith({ kind: 'node', id: '11' });
  });
});

describe('reading a merged community in the margin', () => {
  it('says how big it is and offers to break it open', () => {
    const onViewChange = vi.fn();
    render(inspector({
      onViewChange,
      selection: {
        kind: 'cluster',
        id: 'cluster:2',
        community: 2,
        primaryLabel: 'Item',
        memberCount: 2,
        internalEdges: 9,
        members: ['11', '12'],
      },
    }));

    expect(screen.getByRole('heading', { name: '2 nodes merged' })).toBeInTheDocument();
    expect(screen.getByText('9')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Break open' }));
    expect(onViewChange).toHaveBeenCalledWith({ clusterMode: 'off', colorBy: 'community' });
  });
});

describe('with nothing selected', () => {
  it('says how to select something and what is on the canvas', () => {
    render(inspector());

    expect(screen.getByText(/Double-click a node/)).toBeInTheDocument();
    expect(screen.getByText('nodes').previousSibling).toHaveTextContent('2');
    expect(screen.getByText('edges').previousSibling).toHaveTextContent('1');
    expect(screen.getByText('rows').previousSibling).toHaveTextContent('5');
  });
});
