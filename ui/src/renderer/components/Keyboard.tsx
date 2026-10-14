// On-screen keyboard for instrument plugins: click the keys, or (while it has focus) play with the
// computer keyboard like a piano: A S D F G H J K are white keys, W E T Y U the black ones,
// Z / X shift the octave.
import { useEffect, useRef, useState } from 'react';
import { engine } from '../store';
import { t } from '../i18n';

const NAMES = ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'];
const BLACK = new Set([1, 3, 6, 8, 10]);
// Computer key -> semitone above the octave's C.
const KEYMAP: Record<string, number> = { a: 0, w: 1, s: 2, e: 3, d: 4, f: 5, t: 6, g: 7, y: 8, h: 9, u: 10, j: 11, k: 12 };
const VELOCITY = 100;

export function Keyboard({ nodeId, fxId }: { nodeId: string; fxId: string }) {
  const [octave, setOctave] = useState(4); // C4 = MIDI note 60
  const [held, setHeld] = useState<Set<number>>(new Set());
  const down = useRef(new Map<string, number>()); // computer key -> note (so octave changes can't strand a note)
  const send = (events: number[][]) => void engine.call('fx.midi', { nodeId, fxId, events }).catch(() => {});
  const on = (note: number) => {
    send([[0x90, note, VELOCITY]]);
    setHeld((h) => new Set(h).add(note));
  };
  const off = (note: number) => {
    send([[0x80, note, 0]]);
    setHeld((h) => {
      const n = new Set(h);
      n.delete(note);
      return n;
    });
  };
  // Never leave a note hanging when the keyboard closes.
  const heldRef = useRef(held);
  heldRef.current = held;
  useEffect(() => () => heldRef.current.forEach((n) => send([[0x80, n, 0]])), []);

  const base = (octave + 1) * 12;
  const notes = Array.from({ length: 25 }, (_, i) => base + i);

  const onKeyDown = (e: React.KeyboardEvent) => {
    const k = e.key.toLowerCase();
    if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
    if (k === 'z' || k === 'x') {
      setOctave((o) => Math.min(8, Math.max(0, o + (k === 'x' ? 1 : -1))));
      e.preventDefault();
      return;
    }
    if (k in KEYMAP && !down.current.has(k)) {
      const note = base + KEYMAP[k];
      down.current.set(k, note);
      on(note);
      e.preventDefault();
    }
  };
  const onKeyUp = (e: React.KeyboardEvent) => {
    const k = e.key.toLowerCase();
    const note = down.current.get(k);
    if (note !== undefined) {
      down.current.delete(k);
      off(note);
    }
  };

  return (
    <div className="keyboard" role="group" aria-label={`On-screen keyboard, octave ${octave}`} tabIndex={0} onKeyDown={onKeyDown} onKeyUp={onKeyUp}>
      <div className="keyboard-bar">
        <button className="btn small" onClick={() => setOctave((o) => Math.max(0, o - 1))} aria-label={t("Octave down")} title={t("Octave down (Z)")}>
          −
        </button>
        <span className="mono small">C{octave}–C{octave + 2}</span>
        <button className="btn small" onClick={() => setOctave((o) => Math.min(8, o + 1))} aria-label={t("Octave up")} title={t("Octave up (X)")}>
          +
        </button>
        <span className="hint">{t("Click here, then play with A–K (W E T Y U for sharps).")}</span>
      </div>
      <div className="keys">
        {notes.map((n) => {
          const black = BLACK.has(n % 12);
          return (
            <button
              key={n}
              className={`key ${black ? 'black' : 'white'} ${held.has(n) ? 'down' : ''}`}
              aria-label={`${NAMES[n % 12]}${Math.floor(n / 12) - 1}`}
              aria-pressed={held.has(n)}
              tabIndex={-1}
              onPointerDown={(e) => {
                (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
                on(n);
              }}
              onPointerUp={() => held.has(n) && off(n)}
              onPointerCancel={() => held.has(n) && off(n)}
            />
          );
        })}
      </div>
    </div>
  );
}
