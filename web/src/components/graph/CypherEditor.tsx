import { autocompletion, type Completion, type CompletionContext } from '@codemirror/autocomplete';
import { defaultKeymap, history, historyKeymap, indentWithTab } from '@codemirror/commands';
import { HighlightStyle, StreamLanguage, syntaxHighlighting, type StreamParser } from '@codemirror/language';
import { tags } from '@lezer/highlight';
import { Compartment, EditorState } from '@codemirror/state';
import { EditorView, keymap, lineNumbers, placeholder as editorPlaceholder } from '@codemirror/view';
import { useEffect, useRef, useState } from 'react';
import type { CompletionSchema } from '../../types';

const KEYWORDS = new Set(`ALL ALTER AND ANY AS ASC ASCENDING AT BY CALL CASE CONSTRAINT CONTAINS CREATE DELETE DESC DESCENDING DETACH DISTINCT DROP ELSE END ENDS EVERY EXISTS EXPLAIN FALSE FOR FROM HISTORY IN INDEX IS LIMIT MATCH MERGE NODE NONE NOT NULL ON OPTIONAL OR ORDER PROJECT PROPERTY RANGE REBUILD RELATIONSHIP REMOVE RETURN ROLLUP SEARCH SET SHOW SKIP STARTS TEMPORAL THEN TO TRUE UNION UNIQUE UNWIND USE WHEN WHERE WINDOW WITH WRITE YIELD`.split(' '));
const BUILT_INS = new Set(`abs avg ceil collect count date datetime degrees duration e exp floor labels last localdatetime localtime log max min nodes percentileCont percentileDisc pi power radians range relationships reverse round sign size sqrt stDev stDevP sum time toBoolean toFloat toInteger toString type variance varianceP vector`.split(' '));

/**
 * Cypher in this design's colours.
 *
 * CodeMirror's stock theme paints keywords purple, strings green and numbers pink — three hues from
 * outside this world, in a design that spends exactly one accent and one steel. The query bar was
 * the brightest thing on any screen because of it.
 *
 * The mapping is the design's own: the language's verbs take `--rubric`, the accent; labels and the
 * functions that read them take `--counter`, the steel this design reserves for provenance and
 * schema — the same ink the schema list writes `:Label` in; and everything else is one of the three
 * inks. Colours are named as custom properties rather than resolved, so the editor follows the
 * plate/page switch without being told.
 */
const cypherHighlight = HighlightStyle.define([
  { tag: tags.keyword, color: 'var(--rubric)' },
  { tag: tags.typeName, color: 'var(--counter)' },
  { tag: tags.function(tags.variableName), color: 'var(--counter)' },
  { tag: tags.special(tags.variableName), color: 'var(--counter)' },
  { tag: tags.variableName, color: 'var(--ink)' },
  { tag: tags.string, color: 'var(--ink-2)' },
  { tag: tags.number, color: 'var(--ink-2)' },
  { tag: tags.operator, color: 'var(--ink-3)' },
  { tag: tags.comment, color: 'var(--ink-3)', fontStyle: 'italic' },
]);

/**
 * The editor's chrome, in the design's inks.
 *
 * CodeMirror ships a paper chrome of its own: a light gutter, a sky-blue marker behind the active
 * line's number, a black caret, a white completion panel. Every one is a literal from outside this
 * world, and on the plate they are the brightest things on the screen. Rewritten as the design's
 * variables, the editor follows the ground it sits on. The caret keeps the console's block shape —
 * that is inherited from the frame — and takes the rubric, the same ink every other caret in this
 * console writes with; the completion list is drawn as the design's menu.
 */
const cypherChrome = EditorView.theme({
  '&': { backgroundColor: 'transparent' },
  '.cm-content': { caretColor: 'var(--rubric)' },
  '.cm-cursor, .cm-dropCursor': { borderLeftColor: 'var(--rubric)' },
  '.cm-gutters': {
    backgroundColor: 'transparent',
    color: 'var(--unwritten)',
    border: 'none',
    paddingRight: '4px',
  },
  '.cm-activeLineGutter': { backgroundColor: 'transparent', color: 'var(--ink-2)' },
  '.cm-activeLine': { backgroundColor: 'transparent' },
  '.cm-placeholder': { color: 'var(--unwritten)' },
  '.cm-tooltip': {
    border: '1px solid var(--rubric)',
    backgroundColor: 'var(--ground)',
    color: 'var(--ink-2)',
  },
  '.cm-tooltip.cm-tooltip-autocomplete > ul > li': {
    fontFamily: 'Archivo, ui-monospace, monospace',
    fontSize: '11.5px',
  },
  '.cm-tooltip.cm-tooltip-autocomplete > ul > li[aria-selected]': {
    backgroundColor: 'var(--rubric-wash)',
    color: 'var(--ink)',
  },
  '.cm-completionDetail': { color: 'var(--ink-3)', fontStyle: 'italic' },
  '.cm-completionMatchedText': { textDecoration: 'none', color: 'var(--rubric)' },
  // The design has no icon set; a row's kind is written out in its detail text instead.
  '.cm-completionIcon': { display: 'none' },
});

interface TokenState { blockComment: boolean }

const cypherParser: StreamParser<TokenState> = {
  startState: () => ({ blockComment: false }),
  token(stream, state) {
    if (state.blockComment) {
      if (stream.skipTo('*/')) {
        stream.match('*/');
        state.blockComment = false;
      } else stream.skipToEnd();
      return 'comment';
    }
    if (stream.match('//')) { stream.skipToEnd(); return 'comment'; }
    if (stream.match('/*')) { state.blockComment = true; return 'comment'; }
    if (stream.match(/^(?:"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/)) return 'string';
    if (stream.match(/^`(?:[^`]|``)*`/)) return 'variableName.special';
    if (stream.match(/^\$[A-Za-z_][\w]*/)) return 'variableName';
    if (stream.match(/^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?/)) return 'number';
    if (stream.match(/^[:][A-Za-z_][\w]*/)) return 'typeName';
    if (stream.match(/^[A-Za-z_][\w]*/)) {
      const word = stream.current();
      if (KEYWORDS.has(word.toUpperCase())) return 'keyword';
      if (BUILT_INS.has(word)) return 'function(variableName)';
      return 'variableName';
    }
    if (stream.match(/^(?:<>|<=|>=|=~|[-+*/%=<>|&^~])/)) return 'operator';
    stream.next();
    return null;
  },
};

interface Props {
  value: string;
  schema: CompletionSchema;
  running: boolean;
  onChange: (value: string) => void;
  onRun: (query: string) => void;
  onCancel: () => void;
}

function options(schema: CompletionSchema): Completion[] {
  const keywords: Completion[] = [...KEYWORDS].map((label) => ({ label, type: 'keyword', boost: 20 }));
  const functions = [...new Set([...BUILT_INS, ...schema.functions])].map((label) => ({ label, type: 'function', apply: `${label}()`, boost: 15 }));
  return [
    ...keywords,
    ...functions,
    ...schema.labels.map((label) => ({ label, type: 'class', detail: 'node label', boost: 10 })),
    ...schema.relationshipTypes.map((label) => ({ label, type: 'type', detail: 'relationship', boost: 10 })),
    ...schema.properties.map((label) => ({ label, type: 'property', detail: 'property', boost: 8 })),
  ];
}

export function CypherEditor({ value, schema, running, onChange, onRun, onCancel }: Props) {
  const parentRef = useRef<HTMLDivElement>(null);
  const viewRef = useRef<EditorView | undefined>(undefined);
  const completionCompartment = useRef(new Compartment());
  const callbacks = useRef({ onChange, onRun });
  const [hasSelection, setHasSelection] = useState(false);
  callbacks.current = { onChange, onRun };

  useEffect(() => {
    if (!parentRef.current) return;
    const completionSource = (context: CompletionContext) => {
      const word = context.matchBefore(/[\w:]*/);
      if (!word || (!context.explicit && word.from === word.to)) return null;
      return { from: word.from, options: options(schema), validFor: /^[\w:]*$/ };
    };
    const state = EditorState.create({
      doc: value,
      extensions: [
        lineNumbers(),
        history(),
        cypherChrome,
        StreamLanguage.define(cypherParser),
        syntaxHighlighting(cypherHighlight, { fallback: true }),
        EditorView.lineWrapping,
        editorPlaceholder('USE project\nMATCH (n)\nRETURN n\nLIMIT 100'),
        keymap.of([
          ...defaultKeymap,
          ...historyKeymap,
          indentWithTab,
          {
            key: 'Mod-Enter',
            run(view) { callbacks.current.onRun(view.state.doc.toString()); return true; },
          },
          {
            key: 'Mod-Shift-Enter',
            run(view) {
              const selection = view.state.sliceDoc(view.state.selection.main.from, view.state.selection.main.to);
              if (selection.trim()) callbacks.current.onRun(selection);
              return true;
            },
          },
        ]),
        EditorView.updateListener.of((update) => {
          if (update.docChanged) callbacks.current.onChange(update.state.doc.toString());
          if (update.selectionSet) setHasSelection(!update.state.selection.main.empty);
        }),
        completionCompartment.current.of(autocompletion({ override: [completionSource], activateOnTyping: true })),
      ],
    });
    const view = new EditorView({ state, parent: parentRef.current });
    viewRef.current = view;
    return () => { view.destroy(); viewRef.current = undefined; };
    // Schema is reconfigured separately to avoid rebuilding editor state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const view = viewRef.current;
    if (!view || view.state.doc.toString() === value) return;
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: value } });
  }, [value]);

  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    const completionSource = (context: CompletionContext) => {
      const word = context.matchBefore(/[\w:]*/);
      if (!word || (!context.explicit && word.from === word.to)) return null;
      return { from: word.from, options: options(schema), validFor: /^[\w:]*$/ };
    };
    view.dispatch({ effects: completionCompartment.current.reconfigure(autocompletion({ override: [completionSource] })) });
  }, [schema]);

  const selection = () => {
    const view = viewRef.current;
    if (!view) return '';
    return view.state.sliceDoc(view.state.selection.main.from, view.state.selection.main.to);
  };

  return (
    <div className="qbar" aria-label="Cypher query editor">
      {/* CodeMirror draws its own gutter and lines inside; the design's frame is what changes. */}
      <div className="qbar-top">
        <div className="qedit" ref={parentRef} />
      </div>
      <div className="qbar-foot">
        <span className="ap faint">Cypher</span>
        <span className="ap origin">{running ? 'running' : 'ready'}</span>
        <div className="grow"></div>
        <span className="ap faint">Tab completes · ⌘↵ runs · ⌘⇧↵ runs the selection</span>
        <button className="detent" type="button" disabled={running || !hasSelection} onClick={() => onRun(selection())}>
          Run selection
        </button>
        {running ? (
          <button className="detent warn" type="button" onClick={onCancel}>Cancel</button>
        ) : (
          <button className="detent" type="button" aria-pressed={Boolean(value.trim())} disabled={!value.trim()} onClick={() => onRun(value)}>
            Run
          </button>
        )}
      </div>
    </div>
  );
}
