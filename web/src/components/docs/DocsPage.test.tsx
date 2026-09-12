import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { waitFor } from '@testing-library/react';
import { DocsPage } from './DocsPage';
import { executeQuery } from '../../lib/simpleQuery';

vi.mock('../../lib/simpleQuery', () => ({ executeQuery: vi.fn(() => Promise.resolve({})) }));

beforeAll(() => {
  HTMLElement.prototype.scrollTo = vi.fn();
});

afterEach(cleanup);

describe('DocsPage index', () => {
  it('opens with the beginner section expanded and lets categories collapse', () => {
    render(<DocsPage projects={[]} onProjectsChanged={vi.fn(() => Promise.resolve())} />);

    const start = screen.getByRole('button', { name: /^start\d+$/i });
    expect(start).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('button', { name: /^Start here/ })).toBeInTheDocument();

    fireEvent.click(start);
    expect(start).toHaveAttribute('aria-expanded', 'false');
    expect(screen.queryByRole('button', { name: /^Start here/ })).not.toBeInTheDocument();
  });

  it('reveals matching pages inside collapsed categories while searching', () => {
    render(<DocsPage projects={[]} onProjectsChanged={vi.fn(() => Promise.resolve())} />);

    fireEvent.change(screen.getByRole('searchbox', { name: 'Search documentation' }), { target: { value: 'pagerank' } });
    expect(screen.getByRole('button', { name: /cypher · procedures/i })).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('button', { name: /pagerank/i })).toBeInTheDocument();
  });

  it('offers the Training-style importer on a page whose dataset project is missing', async () => {
    const onProjectsChanged = vi.fn(() => Promise.resolve());
    render(<DocsPage projects={[]} onProjectsChanged={onProjectsChanged} />);

    fireEvent.click(screen.getByRole('button', { name: /^Features$/ }));
    expect(screen.getByText('Dataset not loaded')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: /Import into fraud/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Import dataset' }));
    await waitFor(() => expect(executeQuery).toHaveBeenCalledWith('IMPORT DATASET fraud'));
    await waitFor(() => expect(onProjectsChanged).toHaveBeenCalled());
  });
});
