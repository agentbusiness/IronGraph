import { Fragment, useCallback, useEffect, useRef, useState } from 'react';
import type { ThemePreference } from '../types';

const NS = 'http://www.w3.org/2000/svg';

/** The plate is the dark ground, the page the light one; auto follows the machine. */
const VIEW_LABEL: Record<ThemePreference, string> = { system: 'Auto', dark: 'Plate', light: 'Page' };
const VIEW_ATTR: Record<ThemePreference, string> = { system: 'auto', dark: 'plate', light: 'page' };

/** The register mark. One glyph, drawn wherever the design asks for a crossing. */
export function Cross({ className = 'cross' }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 9 9" aria-hidden>
      <path d="M4.5 0v9M0 4.5h9" stroke="currentColor" strokeWidth="1.1" />
    </svg>
  );
}

export interface RailKey {
  /** What the key says. */
  t: string;
  on?: boolean;
  /**
   * One tick per thing under the key; a true tick is hot. The design draws the
   * count rather than printing it, so an empty array is a key with nothing under it.
   */
  ticks?: boolean[];
  onSelect?: () => void;
  title?: string;
}

/**
 * One grammar for a keyed rail: a cap, keys with square detents, a cap. Every
 * screen's centre rail is this, so a new one costs a list rather than a layout.
 */
/**
 * The caret the design draws, in the four directions a control needs it.
 *
 * The design has no icon set — apparatus is set in type and marks are rules and squares — but it
 * does draw one glyph itself: the chevron on a `.pick`, `M1 3l3 3 3-3` on an 8×8 field. That path
 * is the design's, so every disclosure and every step-through in the product uses it rather than
 * an imported icon that happens to look similar. Rotation, not a second path: down is the drawn
 * one, the rest are quarter turns of it.
 */
const CARET_TURN = { down: 0, up: 180, right: -90, left: 90 };

export function Caret({ dir = 'down', className }: { dir?: keyof typeof CARET_TURN; className?: string }) {
  return (
    <svg viewBox="0 0 8 8" aria-hidden className={className} style={{ transform: `rotate(${CARET_TURN[dir]}deg)` }}>
      <path d="M1 3l3 3 3-3" stroke="currentColor" fill="none" strokeWidth="1.1" />
    </svg>
  );
}

export function Rail({ screen, cap, keys, foot }: { screen: string; cap: string; keys: RailKey[]; foot: string }) {
  return (
    <div className="rail" data-rail={screen}>
      <span className="cap">{cap}</span>
      {keys.map((key, index) => (
        <Fragment key={`${key.t}-${index}`}>
          <button
            className={`rk${key.on ? ' on' : ''}`}
            type="button"
            data-k={key.t}
            title={key.title}
            onClick={key.onSelect}
            disabled={!key.onSelect}
          >
            <span className="sq"></span>
            {key.t}
          </button>
          {/*
            * The ramp descends across everything it marks, not across four and then stops.
            *
            * The design gives four steps — lift, rubric, deep, deep — and the mockup spends them
            * on the first four ticks, which on a static page with four ticks is the whole ramp.
            * Against a real day of arrivals it left a short burst of colour above a column of
            * grey, so the ranking read as "four things matter" rather than as a descent.
            *
            * The steps are spread over however many lit ticks there are instead, so the top of a
            * day is always `--rubric-lift` and the bottom always `--rubric-deep`, whether the day
            * holds four arrivals or forty.
            */}
          {(key.ticks ?? []).map((hot, tick, all) => {
            const lit = all.filter(Boolean).length;
            const rank = all.slice(0, tick + 1).filter(Boolean).length;
            const step = lit <= 1 ? 1 : Math.min(4, Math.ceil((rank / lit) * 4) || 1);
            return <span className={`tick${hot ? ` hot r${step}` : ''}`} key={tick}></span>;
          })}
        </Fragment>
      ))}
      <span className="spacer"></span>
      <span className="cap">{foot}</span>
    </div>
  );
}

export interface Note {
  lbl: string;
  /** The note itself. Plain text — the design sets it, the note does not mark itself up. */
  p: React.ReactNode;
  /** Which `data-mark` in the reading column this note brackets back into. */
  anchor?: string;
  /** A note about provenance is drawn in the counter colour rather than the rubric. */
  origin?: boolean;
  go?: string;
  onGo?: () => void;
}

/**
 * The signature of the design: a note in the margin brackets back into the exact
 * line it annotates. Hovering draws the bracket, clicking sticks it.
 *
 * The drawing is imperative because it measures: it needs the laid-out position of
 * the anchor inside a scrolled column, which React does not know and should not.
 */
export function Apparatus({ notes, heading = 'Apparatus' }: { notes: Note[]; heading?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [stuck, setStuck] = useState<string | null>(null);

  const draw = useCallback((anchor: string | null) => {
    const screen = ref.current?.closest('.screen');
    if (!screen) return;
    const over = screen.querySelector('.rd-over');
    if (over) while (over.firstChild) over.removeChild(over.firstChild);
    screen.querySelectorAll('.mark-anchor.hot').forEach((n) => n.classList.remove('hot', 'cite'));
    if (!anchor) return;
    const col = screen.querySelector('.col-ii');
    const target = screen.querySelector(`[data-mark="${CSS.escape(anchor)}"]`);
    if (!col || !over || !target) return;
    const origin = notes.find((note) => note.anchor === anchor)?.origin ?? false;
    target.classList.add('hot');
    target.classList.toggle('cite', origin);
    const marg = screen.querySelector('.marg');
    if (marg && getComputedStyle(marg).display === 'none') return;
    const cb = col.getBoundingClientRect();
    const tb = target.getBoundingClientRect();
    const height = Math.max(col.scrollHeight, cb.height);
    over.setAttribute('viewBox', `0 0 ${cb.width} ${height}`);
    over.setAttribute('width', String(cb.width));
    over.setAttribute('height', String(height));
    (over as SVGElement & { style: CSSStyleDeclaration }).style.width = `${cb.width}px`;
    (over as SVGElement & { style: CSSStyleDeclaration }).style.height = `${height}px`;
    const top = tb.top - cb.top + col.scrollTop;
    const bottom = tb.bottom - cb.top + col.scrollTop;
    const x = cb.width - 10;
    const path = document.createElementNS(NS, 'path');
    path.setAttribute(
      'd',
      `M${x - 8} ${top.toFixed(1)} H${x} V${bottom.toFixed(1)} H${x - 8} M${x} ${((top + bottom) / 2).toFixed(1)} H${cb.width.toFixed(1)}`,
    );
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', origin ? 'var(--counter)' : 'var(--rubric)');
    path.setAttribute('stroke-width', '1');
    path.setAttribute('shape-rendering', 'crispEdges');
    over.appendChild(path);
  }, [notes]);

  // The bracket is measured, so it is redrawn whenever the measurement could have changed.
  useEffect(() => {
    draw(stuck);
    const onResize = () => draw(stuck);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [draw, stuck]);

  return (
    <>
      <div className="vrule marg-rule"></div>
      <div className="marg" ref={ref}>
        <div className="marg-h">
          <Cross />
          <span className="ap">{heading}</span>
        </div>
        {notes.map((note, index) => (
            <button
              className={`nota${note.origin ? ' origin' : ''}${stuck && stuck === note.anchor ? ' hot' : ''}`}
              type="button"
              key={`${note.lbl}-${index}`}
              data-anchor={note.anchor}
              onMouseEnter={() => { if (!stuck && note.anchor) draw(note.anchor); }}
              onFocus={() => { if (!stuck && note.anchor) draw(note.anchor); }}
              onMouseLeave={() => { if (!stuck) draw(null); }}
              onBlur={() => { if (!stuck) draw(null); }}
              onClick={(event) => {
                if ((event.target as HTMLElement).closest('.go')) {
                  note.onGo?.();
                  return;
                }
                if (note.anchor) setStuck((current) => (current === note.anchor ? null : note.anchor!));
              }}
            >
            <span className="ap lbl">{note.lbl}</span>
            <p>{note.p}</p>
            {note.go && <span className="go">{note.go} →</span>}
          </button>
        ))}
      </div>
    </>
  );
}

/**
 * A screen is the work area between the index and the edge of the plate. `on`
 * is always set: the app renders one screen, where the mockup kept eight and hid seven.
 */
export function Screen({ name, children }: { name: string; children: React.ReactNode }) {
  return (
    <section className="screen on" data-screen={name}>
      {children}
    </section>
  );
}

/**
 * The plate, before the workspace exists.
 *
 * The startup gate and the model picker run before the shell mounts, and they used to render
 * outside `#ig` entirely — which meant none of the design reached them: system sans-serif on a
 * white ground, in a product whose whole argument is one ground and one measure. They are the
 * first thing anyone sees, so they are the last place to be off-design.
 *
 * `data-view="plate"` rather than the reader's preference: nothing has loaded yet, including the
 * stored choice, and a flash from one ground to the other is worse than opening on the design's.
 */
export function Frame({
  children,
  theme = 'dark',
  onThemeChange,
}: {
  children: React.ReactNode;
  theme?: ThemePreference;
  onThemeChange?: (theme: ThemePreference) => void;
}) {
  const next: Record<ThemePreference, ThemePreference> = { system: 'light', light: 'dark', dark: 'system' };
  return (
    <div id="ig" data-view={VIEW_ATTR[theme]} className="drawn frame">
      <svg className="reg tl" viewBox="0 0 11 11" aria-hidden><path d="M5.5 0v11M0 5.5h11" stroke="currentColor" strokeWidth="1" /></svg>
      <svg className="reg tr" viewBox="0 0 11 11" aria-hidden><path d="M5.5 0v11M0 5.5h11" stroke="currentColor" strokeWidth="1" /></svg>
      <svg className="reg bl" viewBox="0 0 11 11" aria-hidden><path d="M5.5 0v11M0 5.5h11" stroke="currentColor" strokeWidth="1" /></svg>
      <svg className="reg br" viewBox="0 0 11 11" aria-hidden><path d="M5.5 0v11M0 5.5h11" stroke="currentColor" strokeWidth="1" /></svg>
      <div className="band">
        <div className="mark">
          <b>IronGraph</b>
          <Cross />
          <i>IronGraph</i>
        </div>
        <div className="grow"></div>
        {/* See the note in AppShell: the absolute claim went when rule 15 admitted the code tool. */}
        <span className="ap faint">Local embedding · no cloud, no account</span>
        {/* The ground is selected directly on the console frame. */}
        {onThemeChange && (
          <button
            className="detent"
            type="button"
            onClick={() => onThemeChange(next[theme])}
            aria-label={`View: ${VIEW_LABEL[theme]}. Change view`}
          >
            {VIEW_LABEL[theme]}
          </button>
        )}
      </div>
      {children}
    </div>
  );
}
