import { useEffect, useState } from 'react';
import { ArrowRight, ExternalLink, Trash2, X } from 'lucide-react';
import type { NodeInfo, RouteInfo } from '@shared/protocol';
import { BIN_META, kindLabel, nodeIcon } from '../bins';
import { act, engine, nodeById, routesFrom, routesTo, useStore } from '../store';
import { Fader, formatDb } from './Fader';
import { Meter } from './Meter';
import { FxChain } from './FxChain';
import { RouteMatrix } from './RouteMatrix';

export function Inspector() {
  const selection = useStore((s) => s.selection);
  const state = useStore((s) => s.state)!;
  const close = () => useStore.getState().select(null);

  if (selection?.type === 'node') {
    const n = nodeById(state, selection.id);
    if (!n) return null;
    return (
      <aside className="inspector" aria-label={`${n.name} details`}>
        <NodeInspector node={n} onClose={close} />
      </aside>
    );
  }
  if (selection?.type === 'route') {
    const r = state.routes.find((x) => x.id === selection.id);
    if (!r) return null;
    return (
      <aside className="inspector" aria-label="Connection details">
        <RouteInspector route={r} onClose={close} />
      </aside>
    );
  }
  return null;
}

function NameField({ node }: { node: NodeInfo }) {
  const [v, setV] = useState(node.name);
  useEffect(() => setV(node.name), [node.name]);
  if (!node.caps.canRename) return <h2 className="insp-name">{node.name}</h2>;
  const commit = () => {
    const t = v.trim();
    if (t && t !== node.name) void act(engine.call('node.update', { id: node.id, name: t }));
    else setV(node.name);
  };
  return (
    <input
      className="insp-name editable"
      value={v}
      aria-label="Name"
      onChange={(e) => setV(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => e.key === 'Enter' && (e.target as HTMLInputElement).blur()}
    />
  );
}

function Toggle({ checked, onChange, label, hint }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string }) {
  return (
    <label className="toggle-row">
      <span className="grow">
        <span>{label}</span>
        {hint && <span className="hint">{hint}</span>}
      </span>
      <input type="checkbox" role="switch" checked={checked} onChange={(e) => onChange(e.target.checked)} className="switch" />
    </label>
  );
}

function NumberRow({
  label,
  value,
  unit,
  min,
  max,
  step,
  hint,
  onChange,
}: {
  label: string;
  value: number;
  unit: string;
  min: number;
  max: number;
  step: number;
  hint?: string;
  onChange: (v: number) => void;
}) {
  const [v, setV] = useState(String(value));
  useEffect(() => setV(String(value)), [value]);
  const commit = () => {
    const n = Number(v);
    if (Number.isFinite(n)) onChange(Math.min(max, Math.max(min, n)));
    else setV(String(value));
  };
  return (
    <label className="num-row">
      <span className="grow">
        <span>{label}</span>
        {hint && <span className="hint">{hint}</span>}
      </span>
      <input
        type="number"
        value={v}
        min={min}
        max={max}
        step={step}
        onChange={(e) => setV(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => e.key === 'Enter' && commit()}
      />
      <span className="unit">{unit}</span>
    </label>
  );
}

function RouteRow({ route, other, direction }: { route: RouteInfo; other: NodeInfo | undefined; direction: 'to' | 'from' }) {
  const Icon = other ? nodeIcon(other) : ArrowRight;
  return (
    <div className={`route-row ${route.muted ? 'muted' : ''}`}>
      <button className="route-target" onClick={() => other && useStore.getState().select({ type: 'node', id: other.id })} title={`Open ${other?.name ?? ''}`}>
        {other?.icon ? <img src={other.icon} alt="" className="mini-icon" /> : <Icon size={13} />}
        <span className="ellipsis">{other?.name ?? '(missing)'}</span>
      </button>
      <Fader
        compact
        valueDb={route.gainDb}
        label={`Level ${direction} ${other?.name ?? ''}`}
        onChange={(db) => void act(engine.call('route.update', { id: route.id, gainDb: db }))}
      />
      <span className="route-db">{formatDb(route.gainDb)}</span>
      <button className="icon-btn subtle" title="Edit channels" onClick={() => useStore.getState().select({ type: 'route', id: route.id })}>
        <ExternalLink size={13} />
      </button>
      <button className="icon-btn subtle danger" title="Disconnect" onClick={() => void act(engine.call('route.disconnect', { id: route.id }))}>
        <X size={13} />
      </button>
    </div>
  );
}

function NodeInspector({ node, onClose }: { node: NodeInfo; onClose: () => void }) {
  const state = useStore((s) => s.state)!;
  const Icon = nodeIcon(node);
  const meta = BIN_META[node.bin];
  const sends = routesFrom(state, node.id);
  const receives = routesTo(state, node.id);
  const upd = (p: Parameters<typeof engine.call<'node.update'>>[1]) => void act(engine.call('node.update', p));
  const set = (s: NodeInfo['settings']) => upd({ id: node.id, settings: s });
  const isVirtualish = node.kind === 'vad' || node.kind === 'compat' || node.kind === 'vban' || node.kind === 'app' || node.kind === 'asioclient';

  return (
    <div className="insp">
      <div className={`insp-header hue-${meta.hue}`}>
        <div className="node-icon large">{node.icon ? <img src={node.icon} alt="" /> : <Icon size={22} />}</div>
        <div className="grow">
          <NameField node={node} />
          <div className="muted small">
            {meta.title} · {node.subtitle ?? kindLabel(node)}
          </div>
        </div>
        <button className="icon-btn" onClick={onClose} aria-label="Close details">
          <X size={16} />
        </button>
      </div>

      {node.statusText && node.state !== 'active' && <div className={`notice s-${node.state}`}>{node.statusText}</div>}

      <div className="insp-section">
        <div className="volume-block">
          <Meter slot={node.meter} className="big" />
          <div className="volume-row">
            <button
              className={`btn small ${node.muted ? 'danger-on' : ''}`}
              aria-pressed={node.muted}
              onClick={() => upd({ id: node.id, muted: !node.muted })}
            >
              {node.muted ? 'Muted' : 'Mute'}
            </button>
            {node.bin === 'bus' && (
              <button className={`btn small ${node.solo ? 'accent-on' : ''}`} aria-pressed={node.solo} onClick={() => upd({ id: node.id, solo: !node.solo })}>
                Listen
              </button>
            )}
            <Fader valueDb={node.gainDb} label={`${node.name} volume`} onChange={(db) => upd({ id: node.id, gainDb: db })} />
          </div>
        </div>
      </div>

      {node.bin === 'app-out' && node.app && (
        <div className="insp-section">
          <h3>Capture mode</h3>
          <div className="choice-group" role="radiogroup" aria-label="Capture mode">
            {(
              [
                ['copy', 'Copy', 'Hear the app normally and also route a copy of its sound.'],
                ['isolate', 'Isolate', 'Take the app’s sound out of your speakers and route it only through NexTone.'],
                ['off', 'Off', 'Don’t capture this app.'],
              ] as const
            ).map(([mode, label, hint]) => (
              <label key={mode} className={`choice ${node.app!.mode === mode ? 'on' : ''}`}>
                <input
                  type="radio"
                  name="capmode"
                  checked={node.app!.mode === mode}
                  onChange={() => void act(engine.call('app.setMode', { exe: node.app!.exe, mode }))}
                />
                <span className="choice-label">{label}</span>
                <span className="hint">{hint}</span>
              </label>
            ))}
          </div>
          <Toggle
            label="Remember this app"
            hint="Keep it listed and restore its routing whenever it starts."
            checked={node.app.pinned}
            onChange={(v) => void act(engine.call('app.setRule', { exe: node.app!.exe, pinned: v }))}
          />
        </div>
      )}

      {node.bin === 'app-in' && node.app && (
        <div className="insp-section">
          <h3>Microphone for this app</h3>
          <p className="muted small">
            Anything you connect here becomes this app’s microphone. NexTone points the app at a private virtual
            microphone automatically.
          </p>
          <Toggle
            label="Remember this app"
            hint="Keep it listed and restore its routing whenever it starts."
            checked={node.app.pinned}
            onChange={(v) => void act(engine.call('app.setRule', { exe: node.app!.exe, pinned: v }))}
          />
        </div>
      )}

      {sends.length > 0 && (
        <div className="insp-section">
          <h3>Sends to</h3>
          {sends.map((r) => (
            <RouteRow key={r.id} route={r} other={nodeById(state, r.to)} direction="to" />
          ))}
        </div>
      )}
      {receives.length > 0 && (
        <div className="insp-section">
          <h3>Receives from</h3>
          {receives.map((r) => (
            <RouteRow key={r.id} route={r} other={nodeById(state, r.from)} direction="from" />
          ))}
        </div>
      )}

      <div className="insp-section">
        <h3>Effects</h3>
        <FxChain node={node} />
      </div>

      <details className="insp-section advanced" open={state.settings.advancedMode}>
        <summary>
          <h3>Advanced</h3>
        </summary>
        {node.caps.canDisable && (
          <Toggle
            label="Enabled"
            hint="Turn this device off to release it for other apps."
            checked={node.enabled}
            onChange={(v) => upd({ id: node.id, enabled: v })}
          />
        )}
        {node.caps.exclusiveMode && (
          <Toggle
            label="Exclusive mode (lowest latency)"
            hint="NexTone takes the device for itself. Other apps can’t use it directly."
            checked={!!node.settings.exclusive}
            onChange={(v) => set({ exclusive: v })}
          />
        )}
        {node.kind === 'wasapi' && (
          <NumberRow label="Device buffer" unit="ms" min={1} max={200} step={0.5} value={node.settings.bufferMs ?? 10} hint="Lower = less delay, higher = safer." onChange={(v) => set({ bufferMs: v })} />
        )}
        {isVirtualish && (
          <NumberRow
            label="Safety buffer"
            unit="ms"
            min={0}
            max={500}
            step={1}
            value={node.settings.safetyMs ?? 0}
            hint="Add delay if you hear crackles from this source."
            onChange={(v) => set({ safetyMs: v })}
          />
        )}
        <NumberRow label="Delay" unit="ms" min={0} max={2000} step={1} value={node.settings.delayMs ?? 0} hint="Align audio with video or other sources." onChange={(v) => set({ delayMs: v })} />
        {node.channels === 2 && (
          <NumberRow label="Balance" unit="L ↔ R" min={-1} max={1} step={0.05} value={node.settings.pan ?? 0} onChange={(v) => set({ pan: v })} />
        )}
        <Toggle label="Invert polarity" checked={!!node.settings.invert} onChange={(v) => set({ invert: v })} />
        {node.kind === 'vban' && (
          <>
            <label className="num-row">
              <span className="grow">Host</span>
              <input defaultValue={node.settings.host} onBlur={(e) => set({ host: e.target.value })} />
            </label>
            <NumberRow label="Port" unit="" min={1} max={65535} step={1} value={node.settings.port ?? 6980} onChange={(v) => set({ port: v })} />
            <label className="num-row">
              <span className="grow">Stream name</span>
              <input defaultValue={node.settings.streamName} maxLength={16} onBlur={(e) => set({ streamName: e.target.value })} />
            </label>
          </>
        )}
        {node.caps.hasControlPanel && (
          <button className="btn small" onClick={() => void act(engine.call('node.openControlPanel', { id: node.id }))}>
            Open driver control panel
          </button>
        )}
        <dl className="stats">
          {node.sampleRate ? (
            <>
              <dt>Sample rate</dt>
              <dd>{(node.sampleRate / 1000).toFixed(1)} kHz</dd>
            </>
          ) : null}
          {node.latencyMs != null && (
            <>
              <dt>Latency</dt>
              <dd>{node.latencyMs.toFixed(1)} ms</dd>
            </>
          )}
          {node.driftPpm != null && (
            <>
              <dt>Clock drift</dt>
              <dd>{node.driftPpm.toFixed(0)} ppm</dd>
            </>
          )}
          {node.xruns != null && (
            <>
              <dt>Dropouts</dt>
              <dd>{node.xruns}</dd>
            </>
          )}
          {node.cpuUs != null && (
            <>
              <dt>Processing</dt>
              <dd>{node.cpuUs.toFixed(0)} µs</dd>
            </>
          )}
          <dt>Channels</dt>
          <dd>{node.channelNames.join(', ') || node.channels}</dd>
          <dt>ID</dt>
          <dd className="mono small">{node.id}</dd>
        </dl>
      </details>

      {node.caps.canDelete && (
        <div className="insp-section">
          <button
            className="btn danger small"
            onClick={() => {
              const call = node.kind === 'vban' ? engine.call('vban.remove', { id: node.id }) : engine.call('bus.delete', { id: node.id });
              void act(call).then(onClose);
            }}
          >
            <Trash2 size={13} /> Delete {node.kind === 'bus' ? 'mix' : 'stream'}
          </button>
        </div>
      )}
    </div>
  );
}

function RouteInspector({ route, onClose }: { route: RouteInfo; onClose: () => void }) {
  const state = useStore((s) => s.state)!;
  const from = nodeById(state, route.from);
  const to = nodeById(state, route.to);
  if (!from || !to) return null;
  const FromIcon = nodeIcon(from);
  const ToIcon = nodeIcon(to);
  return (
    <div className="insp">
      <div className="insp-header">
        <div className="grow">
          <h2 className="insp-name route-title">
            <button className="link" onClick={() => useStore.getState().select({ type: 'node', id: from.id })}>
              <FromIcon size={15} /> {from.name}
            </button>
            <ArrowRight size={15} className="dim" />
            <button className="link" onClick={() => useStore.getState().select({ type: 'node', id: to.id })}>
              <ToIcon size={15} /> {to.name}
            </button>
          </h2>
          <div className="muted small">Connection</div>
        </div>
        <button className="icon-btn" onClick={onClose} aria-label="Close details">
          <X size={16} />
        </button>
      </div>
      <div className="insp-section">
        <div className="volume-row">
          <button
            className={`btn small ${route.muted ? 'danger-on' : ''}`}
            aria-pressed={route.muted}
            onClick={() => void act(engine.call('route.update', { id: route.id, muted: !route.muted }))}
          >
            {route.muted ? 'Muted' : 'Mute'}
          </button>
          <Fader valueDb={route.gainDb} label="Connection level" onChange={(db) => void act(engine.call('route.update', { id: route.id, gainDb: db }))} />
        </div>
      </div>
      <div className="insp-section">
        <h3>Channels</h3>
        <p className="muted small">Choose which input channel feeds which output channel. Scroll on a dot to change its level.</p>
        <RouteMatrix route={route} from={from} to={to} />
      </div>
      <div className="insp-section">
        <button className="btn danger small" onClick={() => void act(engine.call('route.disconnect', { id: route.id })).then(onClose)}>
          <Trash2 size={13} /> Disconnect
        </button>
      </div>
    </div>
  );
}
