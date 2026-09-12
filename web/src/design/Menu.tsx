import { useEffect, useRef, useState } from 'react';

/** Compact control menu used by the graph result toolbar. */

export interface MenuItem {
  key: string;
  label: string;
  /** Shown as a checked state — for menus that pick one of a set rather than run an action. */
  selected?: boolean;
  disabled?: boolean;
  /** Why it is disabled, so a greyed row is never unexplained. */
  hint?: string;
  /**
   * What this choice means, shown on the row whether or not it can be picked.
   *
   * Separate from `hint`: a hint explains an absence, a note explains a choice. The design draws
   * both as `.mh`, but a menu of four ways to colour a graph needs the second one on every row.
   */
  note?: string;
  onSelect: () => void;
}

interface Props {
  label: string;
  items: MenuItem[];
  /**
   * `pick` states a current value and opens to change it; `detent` is a control you
   * press. The design draws them differently because they mean different things.
   */
  variant?: 'pick' | 'detent';
  /**
   * Words before the value, for a pick that names what it is picking.
   *
   * "Colour by **Kind**" rather than a bare "Kind" — the design puts the noun outside `.v` so only
   * the value carries the value's ink.
   */
  prefix?: string;
  disabled?: boolean;
  /** A count carried on a `detent` trigger, drawn as the design's `.k`. Zero shows nothing. */
  count?: number;
  /**
   * Choices that stack rather than replace one another.
   *
   * A radio menu answers a question and closes; a checkbox menu is a set being assembled, and
   * closing after each toggle would make picking three things three trips. The mark is the same
   * square either way — what differs is the role announced and whether the list stays open.
   */
  multi?: boolean;
  /**
   * What the control announces, when the label alone does not say what pressing it does.
   * A pick that states a project needs to say it is a project and that it switches.
   */
  ariaLabel?: string;
  title?: string;
}

/**
 * A control that opens a short list of choices.
 *
 * One component for both the Write menu and the filters, because the difference between "do this"
 * and "show this" is a mark against the current row, not a different control. Extending either is
 * adding an entry to an array, so graph controls share one interaction.
 *
 * The list is positioned rather than nested: the design gives it a hard border and a fixed
 * position so it sits over the page's rules instead of stretching the bar it hangs from.
 */
export function Menu({ label, items, variant = 'pick', prefix, disabled, count, multi = false, ariaLabel, title }: Props) {
  const [open, setOpen] = useState(false);
  const [at, setAt] = useState<{ left: number; top: number }>();
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    const onPointer = (event: MouseEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onPointer);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onPointer);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  /**
   * Measured when it opens, not after it has opened.
   *
   * This used to run in a layout effect keyed on `open`, which meant the menu rendered once with
   * no position and then again with one — a placed element arriving in two passes, which is both a
   * cascading render and a visible jump on a slow frame. The button's box is already known at the
   * moment of the press, so the measurement belongs there: one render, in the right place.
   */
  const toggle = () => {
    if (open) {
      setOpen(false);
      return;
    }
    const box = trigger.current?.getBoundingClientRect();
    if (box) setAt({ left: box.left, top: box.bottom + 4 });
    setOpen(true);
  };

  return (
    <div className="menu-anchor" ref={root} style={{ position: 'relative', display: 'inline-flex' }}>
      <button
        ref={trigger}
        type="button"
        className={variant}
        disabled={disabled}
        aria-label={ariaLabel}
        title={title}
        onClick={toggle}
        aria-haspopup="menu"
        aria-expanded={open}
      >
        {variant === 'pick' ? (
          <>
            {prefix ? `${prefix} ` : null}
            <span className="v">{label}</span>
          </>
        ) : (
          <>
            {label}
            {count ? <span className="k">{count}</span> : null}
          </>
        )}
        {variant === 'pick' && (
          <svg viewBox="0 0 8 8" aria-hidden>
            <path d="M1 3l3 3 3-3" stroke="currentColor" fill="none" strokeWidth="1.1" />
          </svg>
        )}
      </button>
      {open && at && (
        <div className="menu open" role="menu" aria-label={ariaLabel ?? (prefix ? `${prefix} ${label}` : label)} style={{ left: `${at.left}px`, top: `${at.top}px` }}>
          {items.map((item) => (
            <button
              key={item.key}
              type="button"
              role={multi ? 'menuitemcheckbox' : item.selected === undefined ? 'menuitem' : 'menuitemradio'}
              aria-selected={item.selected}
              aria-checked={item.selected}
              disabled={item.disabled}
              title={item.hint}
              onClick={() => {
                item.onSelect();
                if (!multi) setOpen(false);
              }}
            >
              <span className="mk" aria-hidden></span>
              <span className="ml">{item.label}</span>
              {(item.note ?? (item.disabled ? item.hint : undefined)) && (
                <span className="mh">{item.note ?? item.hint}</span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
