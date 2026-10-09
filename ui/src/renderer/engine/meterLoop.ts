// One shared requestAnimationFrame loop drives every level meter in the UI.
// Meter elements register a (value index, element) pair; the loop writes CSS
// custom properties directly so React never re-renders for meter motion.
import { meters } from './client';

interface MeterBinding {
  el: HTMLElement;
  /** index of the peak value; rms is index + 1 */
  index: number;
  peakHold: number;
  holdUntil: number;
  level: number;
}

const bindings = new Set<MeterBinding>();
let running = false;
let currentLayout = -1;

/** Maps linear amplitude to a 0..1 meter position on a -60..+6 dB scale. */
export function meterPosition(lin: number): number {
  if (lin <= 0.001) return 0; // -60 dB
  const db = 20 * Math.log10(lin);
  return Math.min(1, Math.max(0, (db + 60) / 66));
}

function tick(now: number): void {
  const stale = now - meters.lastFrameAt > 500;
  for (const b of bindings) {
    const valid = !stale && meters.layoutVersion === currentLayout && b.index + 1 < meters.values.length;
    const peak = valid ? meters.values[b.index] : 0;
    const target = meterPosition(peak);
    // fast attack, ~20 dB/s release
    b.level = target > b.level ? target : Math.max(target, b.level - 0.012);
    if (b.level >= b.peakHold || now > b.holdUntil) {
      b.peakHold = b.level;
      b.holdUntil = now + 1200;
    }
    b.el.style.setProperty('--level', b.level.toFixed(4));
    b.el.style.setProperty('--hold', b.peakHold.toFixed(4));
    b.el.dataset.clip = peak >= 0.999 ? '1' : '0';
  }
  if (bindings.size > 0) requestAnimationFrame(tick);
  else running = false;
}

export function setMeterLayout(layout: number): void {
  currentLayout = layout;
}

export function bindMeter(el: HTMLElement, index: number): () => void {
  const b: MeterBinding = { el, index, peakHold: 0, holdUntil: 0, level: 0 };
  bindings.add(b);
  if (!running) {
    running = true;
    requestAnimationFrame(tick);
  }
  return () => {
    bindings.delete(b);
  };
}
