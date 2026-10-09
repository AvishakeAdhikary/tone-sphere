import { useEffect, useRef, useState } from 'react';

const MIN_DB = -60;
const MAX_DB = 12;

/** Fader law: more resolution around 0 dB; position 0 = silence (−∞). */
export function dbToPos(db: number): number {
  if (db <= MIN_DB) return 0;
  return Math.pow((Math.min(db, MAX_DB) - MIN_DB) / (MAX_DB - MIN_DB), 2);
}
export function posToDb(pos: number): number {
  if (pos <= 0.0005) return -120;
  return Math.sqrt(pos) * (MAX_DB - MIN_DB) + MIN_DB;
}
export function formatDb(db: number): string {
  if (db <= -119) return '−∞';
  const r = Math.round(db * 10) / 10;
  return `${r > 0 ? '+' : r < 0 ? '−' : ''}${Math.abs(r).toFixed(1)}`;
}

/**
 * Gain slider. Sends updates while dragging (throttled by rAF) and on release.
 * Double-click resets to 0 dB. Arrow keys move 0.5 dB, PageUp/Down 3 dB.
 */
export function Fader({
  valueDb,
  onChange,
  label,
  orientation = 'horizontal',
  compact = false,
  disabled = false,
}: {
  valueDb: number;
  onChange: (db: number) => void;
  label: string;
  orientation?: 'horizontal' | 'vertical';
  compact?: boolean;
  disabled?: boolean;
}) {
  const [local, setLocal] = useState<number | null>(null);
  const pendingRef = useRef<number | null>(null);
  const frame = useRef<number | null>(null);
  const shown = local ?? valueDb;

  useEffect(() => () => {
    if (frame.current) cancelAnimationFrame(frame.current);
  }, []);

  const push = (db: number) => {
    setLocal(db);
    pendingRef.current = db;
    if (frame.current == null) {
      frame.current = requestAnimationFrame(() => {
        frame.current = null;
        if (pendingRef.current != null) onChange(pendingRef.current);
      });
    }
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    const step = e.key === 'PageUp' ? 3 : e.key === 'PageDown' ? -3 : e.key === 'ArrowUp' || e.key === 'ArrowRight' ? 0.5 : e.key === 'ArrowDown' || e.key === 'ArrowLeft' ? -0.5 : 0;
    if (!step) return;
    e.preventDefault();
    const base = shown <= -119 ? MIN_DB : shown;
    const next = Math.max(MIN_DB - 0.5, Math.min(MAX_DB, base + step));
    onChange(next < MIN_DB ? -120 : next);
  };

  return (
    <div className={`fader ${orientation} ${compact ? 'compact' : ''}`}>
      <input
        type="range"
        min={0}
        max={1000}
        step={1}
        value={Math.round(dbToPos(shown) * 1000)}
        disabled={disabled}
        aria-label={label}
        aria-valuetext={`${formatDb(shown)} dB`}
        onChange={(e) => push(posToDb(Number(e.target.value) / 1000))}
        onPointerUp={() => {
          if (pendingRef.current != null) onChange(pendingRef.current);
          setLocal(null);
        }}
        onBlur={() => setLocal(null)}
        onKeyDown={onKeyDown}
        onDoubleClick={() => {
          setLocal(null);
          onChange(0);
        }}
        style={{ '--pos': dbToPos(shown) } as React.CSSProperties}
      />
      {!compact && <span className="fader-value">{formatDb(shown)} dB</span>}
    </div>
  );
}
