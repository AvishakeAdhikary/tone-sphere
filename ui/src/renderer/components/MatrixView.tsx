import { useMemo } from 'react';
import { isDestination, isSource, type Bin, type NodeInfo } from '@shared/protocol';
import { BIN_META, DEST_ORDER, SOURCE_ORDER, nodeIcon } from '../bins';
import { act, engine, useStore } from '../store';
import { formatDb } from './Fader';
import { t } from '../i18n';

const ROW_BINS: Bin[] = [...SOURCE_ORDER, 'bus'];
const COL_BINS: Bin[] = ['bus', ...DEST_ORDER];

/**
 * Advanced matrix: every source (rows) × every destination (columns).
 * Click toggles a connection, mouse wheel adjusts its gain, double-click opens it.
 */
export function MatrixView() {
  const state = useStore((s) => s.state)!;
  const hover = useStore((s) => s.hoverNode);

  const rows = useMemo(
    () => ROW_BINS.flatMap((b) => state.nodes.filter((n) => n.bin === b && isSource(n.bin) && !n.hidden)),
    [state.nodes],
  );
  const cols = useMemo(
    () => COL_BINS.flatMap((b) => state.nodes.filter((n) => n.bin === b && isDestination(n.bin) && !n.hidden)),
    [state.nodes],
  );
  const routeMap = useMemo(() => {
    const m = new Map<string, (typeof state.routes)[number]>();
    for (const r of state.routes) m.set(`${r.from}\u0000${r.to}`, r);
    return m;
  }, [state.routes]);

  const toggle = (from: NodeInfo, to: NodeInfo) => {
    if (from.id === to.id) return;
    const r = routeMap.get(`${from.id}\u0000${to.id}`);
    if (r) void act(engine.call('route.disconnect', { id: r.id }));
    else void act(engine.call('route.connect', { from: from.id, to: to.id, mode: 'auto' }));
  };

  const wheel = (e: React.WheelEvent, from: NodeInfo, to: NodeInfo) => {
    const r = routeMap.get(`${from.id}\u0000${to.id}`);
    if (!r) return;
    const step = e.deltaY < 0 ? 1 : -1;
    const g = Math.max(-60, Math.min(12, (r.gainDb <= -119 ? -60 : r.gainDb) + step));
    void act(engine.call('route.update', { id: r.id, gainDb: g }));
  };

  const groupStart = (list: NodeInfo[], i: number) => i === 0 || list[i - 1].bin !== list[i].bin;

  return (
    <div className="matrix-view">
      <div className="matrix-help muted small">
        {t('Rows send sound to columns. Click a cell to connect or disconnect, scroll on a connected cell to change its volume, double-click to edit individual channels.')}
      </div>
      <div className="matrix-scroll">
        <table className="matrix" role="grid" aria-label={t("Routing matrix")}>
          <thead>
            <tr>
              <th className="corner">
                <span>{t("From ↓ / To →")}</span>
              </th>
              {cols.map((c, i) => {
                const Icon = nodeIcon(c);
                return (
                  <th
                    key={c.id}
                    scope="col"
                    className={`col-head hue-${BIN_META[c.bin].hue} ${groupStart(cols, i) ? 'group-start' : ''} ${hover === c.id ? 'hl' : ''}`}
                    title={`${c.name} — ${BIN_META[c.bin].title}`}
                    onMouseEnter={() => useStore.getState().setHover(c.id)}
                    onMouseLeave={() => useStore.getState().setHover(null)}
                    onClick={() => useStore.getState().select({ type: 'node', id: c.id })}
                  >
                    <div className="col-label">
                      {c.icon ? <img src={c.icon} alt="" className="mini-icon" /> : <Icon size={13} />}
                      <span>{c.name}</span>
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, ri) => {
              const Icon = nodeIcon(r);
              return (
                <tr key={r.id} className={groupStart(rows, ri) ? 'group-start' : ''}>
                  <th
                    scope="row"
                    className={`row-head hue-${BIN_META[r.bin].hue} ${hover === r.id ? 'hl' : ''}`}
                    title={`${r.name} — ${BIN_META[r.bin].title}`}
                    onMouseEnter={() => useStore.getState().setHover(r.id)}
                    onMouseLeave={() => useStore.getState().setHover(null)}
                    onClick={() => useStore.getState().select({ type: 'node', id: r.id })}
                  >
                    {groupStart(rows, ri) && <div className="row-group">{BIN_META[r.bin].title}</div>}
                    <div className="row-label">
                      {r.icon ? <img src={r.icon} alt="" className="mini-icon" /> : <Icon size={13} />}
                      <span className="ellipsis">{r.name}</span>
                    </div>
                  </th>
                  {cols.map((c, ci) => {
                    const route = routeMap.get(`${r.id}\u0000${c.id}`);
                    const self = r.id === c.id;
                    return (
                      <td
                        key={c.id}
                        className={[
                          'cell',
                          route ? 'on' : '',
                          route?.muted ? 'muted' : '',
                          self ? 'self' : '',
                          groupStart(cols, ci) ? 'group-start' : '',
                          hover === r.id || hover === c.id ? 'hl' : '',
                        ].join(' ')}
                        role="gridcell"
                        aria-label={`${r.name} to ${c.name}: ${route ? `connected, ${formatDb(route.gainDb)} dB` : 'not connected'}`}
                        tabIndex={self ? -1 : 0}
                        onClick={() => toggle(r, c)}
                        onKeyDown={(e) => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault();
                            toggle(r, c);
                          }
                        }}
                        onDoubleClick={() => route && useStore.getState().select({ type: 'route', id: route.id })}
                        onWheel={(e) => wheel(e, r, c)}
                      >
                        {route && <span className="cell-gain">{formatDb(route.gainDb)}</span>}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
