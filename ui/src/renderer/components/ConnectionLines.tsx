import { useEffect, useLayoutEffect, useState, type RefObject } from 'react';
import { useStore } from '../store';

interface Line {
  id: string;
  from: string;
  to: string;
  d: string;
  muted: boolean;
}

/** Draws the "audio flow" as curves from source ports to destination ports. */
export function ConnectionLines({ container }: { container: RefObject<HTMLDivElement | null> }) {
  const routes = useStore((s) => s.state?.routes);
  const hover = useStore((s) => s.hoverNode);
  const selection = useStore((s) => s.selection);
  const [lines, setLines] = useState<Line[]>([]);
  const [tick, setTick] = useState(0);

  // Re-measure when layout changes (resize, collapse/expand, cards appearing).
  useEffect(() => {
    const el = container.current;
    if (!el) return;
    let frame = 0;
    const bump = () => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => setTick((t) => t + 1));
    };
    const ro = new ResizeObserver(bump);
    ro.observe(el);
    const mo = new MutationObserver((records) => {
      // Ignore our own SVG updates to avoid a measure → render → measure loop.
      if (records.every((r) => (r.target as Element).closest?.('.connection-lines'))) return;
      bump();
    });
    mo.observe(el, { childList: true, subtree: true });
    window.addEventListener('resize', bump);
    return () => {
      cancelAnimationFrame(frame);
      ro.disconnect();
      mo.disconnect();
      window.removeEventListener('resize', bump);
    };
  }, [container]);

  useLayoutEffect(() => {
    const el = container.current;
    if (!el || !routes) return;
    const base = el.getBoundingClientRect();
    const port = (id: string, cls: string) =>
      el.querySelector<HTMLElement>(`[data-node-id="${CSS.escape(id)}"] .${cls}`)?.getBoundingClientRect();
    const out: Line[] = [];
    for (const r of routes) {
      const a = port(r.from, 'port-out');
      const b = port(r.to, 'port-in');
      if (!a || !b) continue;
      const x1 = a.left + a.width / 2 - base.left;
      const y1 = a.top + a.height / 2 - base.top;
      const x2 = b.left + b.width / 2 - base.left;
      const y2 = b.top + b.height / 2 - base.top;
      const dx = Math.max(40, Math.abs(x2 - x1) * 0.45);
      out.push({
        id: r.id,
        from: r.from,
        to: r.to,
        muted: r.muted || !r.enabled,
        d: `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`,
      });
    }
    setLines(out);
  }, [routes, tick, container]);

  const focus = hover ?? (selection?.type === 'node' ? selection.id : null);
  const selectedRoute = selection?.type === 'route' ? selection.id : null;

  return (
    <svg className="connection-lines" aria-hidden>
      {lines.map((l) => {
        const active = l.id === selectedRoute || (focus != null && (l.from === focus || l.to === focus));
        const dim = focus != null && !active;
        return (
          <g key={l.id} className={`conn ${active ? 'active' : ''} ${dim ? 'dim' : ''} ${l.muted ? 'muted' : ''}`}>
            <path d={l.d} className="conn-path" />
            <path
              d={l.d}
              className="conn-hit"
              onClick={() => useStore.getState().select({ type: 'route', id: l.id })}
            >
              <title>Click to edit this connection</title>
            </path>
          </g>
        );
      })}
    </svg>
  );
}
