import type { NodeInfo, RouteInfo } from '@shared/protocol';
import { act, engine } from '../store';
import { formatDb } from './Fader';

/** Channel-level crosspoint grid for one connection (ASIO Link Pro's "mapping window"). */
export function RouteMatrix({ route, from, to }: { route: RouteInfo; from: NodeInfo; to: NodeInfo }) {
  const cp = new Map(route.crosspoints.map((c) => [`${c.f}:${c.t}`, c.g]));
  const set = (f: number, t: number, g: number | null) =>
    void act(engine.call('route.setCrosspoint', { id: route.id, fromCh: f, toCh: t, gainDb: g }));

  const preset = (mode: 'one-to-one' | 'mono-sum' | 'spread' | 'clear') =>
    void act(engine.call('route.applyPreset', { id: route.id, mode }));

  const fromNames = from.channelNames.length ? from.channelNames : Array.from({ length: from.channels }, (_, i) => `${i + 1}`);
  const toNames = to.channelNames.length ? to.channelNames : Array.from({ length: to.channels }, (_, i) => `${i + 1}`);

  return (
    <div className="route-matrix">
      <div className="preset-row">
        <button className="btn small" onClick={() => preset('one-to-one')} title="Channel 1 → 1, 2 → 2, …">1 → 1</button>
        <button className="btn small" onClick={() => preset('mono-sum')} title="Every input channel into every output channel">Mix to all</button>
        <button className="btn small" onClick={() => preset('spread')} title="Mono to both sides / wrap extra channels">Spread</button>
        <button className="btn small" onClick={() => preset('clear')}>Clear</button>
      </div>
      <div className="xp-scroll">
        <table className="xp-grid">
          <thead>
            <tr>
              <th className="xp-corner">in ↓ out →</th>
              {toNames.map((n, t) => (
                <th key={t} title={n}>
                  {n}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {fromNames.map((n, f) => (
              <tr key={f}>
                <th title={n}>{n}</th>
                {toNames.map((_, t) => {
                  const g = cp.get(`${f}:${t}`);
                  const on = g !== undefined;
                  return (
                    <td key={t}>
                      <button
                        className={`xp ${on ? 'on' : ''}`}
                        aria-pressed={on}
                        aria-label={`Input ${fromNames[f]} to output ${toNames[t]}${on ? `, ${formatDb(g!)} dB` : ''}`}
                        title={on ? `${formatDb(g!)} dB — scroll to adjust, click to remove` : 'Click to connect'}
                        onClick={() => set(f, t, on ? null : 0)}
                        onWheel={(e) => {
                          if (!on) return;
                          const next = Math.max(-60, Math.min(12, g! + (e.deltaY < 0 ? 1 : -1)));
                          set(f, t, next);
                        }}
                      >
                        {on && <span style={{ opacity: Math.max(0.35, Math.min(1, (g! + 60) / 60)) }} />}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
