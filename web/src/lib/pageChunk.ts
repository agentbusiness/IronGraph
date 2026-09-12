/**
 * Pages arrive as their own script, and a page's script can go missing.
 *
 * Every page is code-split, so opening one is a fetch. The build stamps a content hash into each
 * file name and the server hands those out `immutable` for a year, which is right: the file at a
 * hashed name never changes. What does change is which names exist. A rebuild replaces the whole
 * set, and a tab that was opened before it is holding the old index — so the first page it opens
 * after the rebuild asks for a file the server no longer has, and the import rejects.
 *
 * That rejection is not recoverable in place. A failed module URL is recorded as failed in the
 * document's module map, and React records the rejection on the lazy component, so re-trying the
 * same import in the same document returns the same failure however many times it is called. The
 * only cure is a new document, which is also the only way to pick up the new index.
 *
 * So the failure is read rather than retried: ask the server what it is serving now, and if the
 * entry script has a different name than the one this document is running, the build moved and a
 * reload is the fix. If it has the same name, the build did not move — the server is down, or the
 * network refused — and reloading would replace a drawn interface with a browser error page. That
 * case is reported to the page's boundary instead, which keeps the shell up and says so.
 */
/** A page's script could not be fetched, and reloading is not known to cure it. */
export class PageChunkError extends Error {
  constructor(cause: unknown) {
    super('This page could not be loaded from the server.', { cause });
    this.name = 'PageChunkError';
  }
}

/**
 * Wraps a page's module loader with the one recovery a code-split page can actually perform.
 *
 * Sits between `React.lazy` and the `import()`: `lazy(pageLoader(() => import('./Page')))`.
 */
export function pageLoader<T>(load: () => Promise<T>): () => Promise<T> {
  return async () => {
    try {
      return await load();
    } catch (error) {
      throw new PageChunkError(error);
    }
  };
}
