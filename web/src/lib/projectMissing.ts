/**
 * The one signal a page has that the project it is asking about is gone.
 *
 * Every project-scoped query answers `PROJECT_NOT_FOUND` when the request names a project this
 * database does not have — a page left open across a database rebuild, or a project dropped from
 * another window. The page cannot notice that on its own: it holds the project list it loaded when
 * it mounted, so its selection still looks valid and its polls keep asking, once a second and a
 * half, for a project that will never answer. Reporting it here lets the project list re-read once
 * and move the app onto a project that exists.
 */

const EVENT = 'irongraph:project-missing';

/** The server's error code for a project this database does not have. */
export const PROJECT_NOT_FOUND = 'PROJECT_NOT_FOUND';

/** Say that `projectId` was refused as missing. Safe to call on every failed poll. */
export function reportProjectMissing(projectId?: string): void {
  window.dispatchEvent(new CustomEvent(EVENT, { detail: projectId }));
}

/**
 * Report a failed request if what came back was that refusal.
 *
 * `payload` is the parsed error envelope, `body` the request it answers — the project the caller
 * named is read from there, so every client wrapper reports the same thing in one line.
 */
export function reportIfProjectMissing(payload: unknown, body: unknown): void {
  if (!payload || typeof payload !== 'object') return;
  if ((payload as { code?: unknown }).code !== PROJECT_NOT_FOUND) return;
  const named = body && typeof body === 'object' ? (body as { project_id?: unknown }).project_id : undefined;
  reportProjectMissing(typeof named === 'string' ? named : undefined);
}

/** Listen for those reports. Returns the unsubscribe. */
export function onProjectMissing(listener: (projectId?: string) => void): () => void {
  const handler = (event: Event): void => {
    listener((event as CustomEvent<string | undefined>).detail);
  };
  window.addEventListener(EVENT, handler);
  return () => window.removeEventListener(EVENT, handler);
}
