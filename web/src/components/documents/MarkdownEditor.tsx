import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { EditorState, StateField, type Range } from '@codemirror/state';
import { Decoration, EditorView, keymap, placeholder as editorPlaceholder, type DecorationSet } from '@codemirror/view';
import { useCallback, useEffect, useRef } from 'react';

interface Props {
  value: string;
  placeholder?: string;
  onChange: (value: string) => void;
  onSave?: () => void;
}

const hidden = Decoration.replace({});
const syntax = Decoration.mark({ class: 'lm-syntax' });
const strong = Decoration.mark({ class: 'lm-strong' });
const emphasis = Decoration.mark({ class: 'lm-em' });
const strike = Decoration.mark({ class: 'lm-strike' });
const code = Decoration.mark({ class: 'lm-codespan' });

function touchesSelection(state: EditorState, from: number, to: number) {
  return state.selection.ranges.some((range) => range.from <= to && range.to >= from);
}

function decoratePair(
  state: EditorState,
  ranges: Range<Decoration>[],
  from: number,
  to: number,
  markerWidth: number,
  mark: Decoration,
) {
  if (to - from <= markerWidth * 2) return;
  const marker = touchesSelection(state, from, to) ? syntax : hidden;
  ranges.push(marker.range(from, from + markerWidth));
  ranges.push(mark.range(from + markerWidth, to - markerWidth));
  ranges.push(marker.range(to - markerWidth, to));
}

function buildDecorations(state: EditorState): DecorationSet {
  const ranges: Range<Decoration>[] = [];
  for (let number = 1; number <= state.doc.lines; number += 1) {
    const line = state.doc.line(number);
    const heading = /^(#{1,6})\s/.exec(line.text);
    if (heading) {
      const width = (heading[0] ?? '').length;
      const level = (heading[1] ?? '').length;
      ranges.push(Decoration.line({ class: `lm-heading lm-h${level}` }).range(line.from));
      ranges.push((touchesSelection(state, line.from, line.to) ? syntax : hidden).range(line.from, line.from + width));
    }

    const patterns: Array<[RegExp, number, Decoration]> = [
      [/\*\*[^*\n]+\*\*/g, 2, strong],
      [/~~[^~\n]+~~/g, 2, strike],
      [/`[^`\n]+`/g, 1, code],
      [/(?<!\*)\*[^*\n]+\*(?!\*)/g, 1, emphasis],
    ];
    for (const [pattern, width, mark] of patterns) {
      for (const match of line.text.matchAll(pattern)) {
        const from = line.from + (match.index ?? 0);
        decoratePair(state, ranges, from, from + match[0].length, width, mark);
      }
    }
  }
  return Decoration.set(ranges, true);
}

const liveMarkdown = StateField.define<DecorationSet>({
  create: buildDecorations,
  update(value, transaction) {
    return transaction.docChanged || transaction.selection ? buildDecorations(transaction.state) : value;
  },
  provide: (field) => EditorView.decorations.from(field),
});

function toggleWrap(view: EditorView, marker: string) {
  const { from, to } = view.state.selection.main;
  const selected = view.state.sliceDoc(from, to);
  const wrapped = selected.startsWith(marker) && selected.endsWith(marker) && selected.length >= marker.length * 2;
  const replacement = wrapped ? selected.slice(marker.length, -marker.length) : `${marker}${selected || 'text'}${marker}`;
  view.dispatch({
    changes: { from, to, insert: replacement },
    selection: wrapped
      ? { anchor: from, head: from + replacement.length }
      : { anchor: from + marker.length, head: from + replacement.length - marker.length },
  });
  view.focus();
}

function toggleLine(view: EditorView, prefix: string) {
  const { from, to } = view.state.selection.main;
  const first = view.state.doc.lineAt(from);
  const last = view.state.doc.lineAt(to);
  const lines = Array.from({ length: last.number - first.number + 1 }, (_, index) =>
    view.state.doc.line(first.number + index).text);
  const remove = lines.every((line) => line.startsWith(prefix));
  view.dispatch({
    changes: {
      from: first.from,
      to: last.to,
      insert: lines.map((line) => remove ? line.slice(prefix.length) : `${prefix}${line}`).join('\n'),
    },
  });
  view.focus();
}

function insertLink(view: EditorView) {
  const { from, to } = view.state.selection.main;
  const label = view.state.sliceDoc(from, to) || 'link text';
  const insert = `[${label}](url)`;
  view.dispatch({
    changes: { from, to, insert },
    selection: { anchor: from + label.length + 3, head: from + insert.length - 1 },
  });
  view.focus();
}

export function MarkdownEditor({ value, placeholder, onChange, onSave }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const editor = useRef<EditorView>(null);
  const onChangeRef = useRef(onChange);
  const onSaveRef = useRef(onSave);
  onChangeRef.current = onChange;
  onSaveRef.current = onSave;

  useEffect(() => {
    if (!host.current) return;
    const instance = new EditorView({
      parent: host.current,
      state: EditorState.create({
        doc: value,
        extensions: [
          history(),
          keymap.of([
            { key: 'Mod-s', preventDefault: true, run: () => { onSaveRef.current?.(); return true; } },
            { key: 'Mod-b', preventDefault: true, run: (view) => { toggleWrap(view, '**'); return true; } },
            { key: 'Mod-i', preventDefault: true, run: (view) => { toggleWrap(view, '*'); return true; } },
            ...historyKeymap,
            ...defaultKeymap,
          ]),
          liveMarkdown,
          EditorView.lineWrapping,
          EditorView.contentAttributes.of({ 'aria-label': 'Document body' }),
          editorPlaceholder(placeholder ?? 'Begin the document…'),
          EditorView.updateListener.of((update) => {
            if (update.docChanged) onChangeRef.current(update.state.doc.toString());
          }),
        ],
      }),
    });
    editor.current = instance;
    return () => { instance.destroy(); editor.current = null; };
    // The controlled value is reconciled below so typing never rebuilds the editor.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const instance = editor.current;
    if (!instance || instance.state.doc.toString() === value) return;
    instance.dispatch({ changes: { from: 0, to: instance.state.doc.length, insert: value } });
  }, [value]);

  const command = useCallback((action: (view: EditorView) => void) => {
    if (editor.current) action(editor.current);
  }, []);

  return (
    <div className="markdown-editor">
      <div className="markdown-tools" role="toolbar" aria-label="Formatting">
        <button type="button" aria-label="Heading" title="Heading" onClick={() => command((view) => toggleLine(view, '## '))}>H2</button>
        <button type="button" aria-label="Bold" title="Bold (⌘B)" onClick={() => command((view) => toggleWrap(view, '**'))}><b>B</b></button>
        <button type="button" aria-label="Italic" title="Italic (⌘I)" onClick={() => command((view) => toggleWrap(view, '*'))}><i>I</i></button>
        <button type="button" aria-label="Strikethrough" title="Strikethrough" onClick={() => command((view) => toggleWrap(view, '~~'))}><s>S</s></button>
        <button type="button" aria-label="Inline code" title="Inline code" onClick={() => command((view) => toggleWrap(view, '`'))}>Code</button>
        <span className="markdown-tool-divider" aria-hidden />
        <button type="button" aria-label="List" title="List" onClick={() => command((view) => toggleLine(view, '- '))}>List</button>
        <button type="button" aria-label="Quote" title="Quote" onClick={() => command((view) => toggleLine(view, '> '))}>Quote</button>
        <button type="button" aria-label="Link" title="Link" onClick={() => command(insertLink)}>Link</button>
      </div>
      <div className="markdown-live" ref={host} />
    </div>
  );
}
