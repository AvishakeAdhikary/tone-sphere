import { useRef, useState } from 'react';
import { ChevronDown, ChevronUp, ExternalLink, Plus, Power, RefreshCw, Trash2 } from 'lucide-react';
import { isSource, type FxInfo, type FxParamInfo, type FxType, type NodeInfo } from '@shared/protocol';
import { act, engine, useStore } from '../store';
import { Popover } from './Popover';

const BUILTINS: { type: FxType; name: string; hint: string }[] = [
  { type: 'denoise', name: 'Noise suppression', hint: 'Removes background noise from voice (AI, RNNoise).' },
  { type: 'gate', name: 'Noise gate', hint: 'Silences the signal when it falls below a threshold.' },
  { type: 'eq', name: 'Equalizer', hint: 'Shape the tone: bass, mids, treble.' },
  { type: 'compressor', name: 'Compressor', hint: 'Evens out loud and quiet parts.' },
  { type: 'limiter', name: 'Limiter', hint: 'Prevents clipping and sudden loud peaks.' },
  { type: 'ducker', name: 'Auto-duck', hint: 'Lowers this sound while another one (e.g. your mic) is active.' },
  { type: 'delay', name: 'Delay', hint: 'Delays the signal (sync audio with video).' },
];

function fmt(p: FxParamInfo): string {
  if (p.choices) return p.choices[Math.round(p.value)] ?? String(p.value);
  const v = Math.abs(p.value) >= 100 ? p.value.toFixed(0) : Math.abs(p.value) >= 10 ? p.value.toFixed(1) : p.value.toFixed(2);
  return `${v}${p.unit ? ` ${p.unit}` : ''}`;
}

function toSlider(p: FxParamInfo): number {
  if (p.scale === 'log' && p.min > 0) return Math.log(p.value / p.min) / Math.log(p.max / p.min);
  return (p.value - p.min) / (p.max - p.min);
}
function fromSlider(p: FxParamInfo, x: number): number {
  if (p.scale === 'log' && p.min > 0) return p.min * Math.pow(p.max / p.min, x);
  return p.min + x * (p.max - p.min);
}

function FxItem({ node, fx, index, count }: { node: NodeInfo; fx: FxInfo; index: number; count: number }) {
  const state = useStore((s) => s.state)!;
  const [open, setOpen] = useState(false);
  const update = (p: Partial<{ bypass: boolean; params: Record<string, number>; sidechain: string | null }>) =>
    void act(engine.call('fx.update', { nodeId: node.id, fxId: fx.id, ...p }));

  return (
    <div className={`fx-item ${fx.bypass ? 'bypassed' : ''}`}>
      <div className="fx-head">
        <button
          className={`icon-btn subtle power ${fx.bypass ? '' : 'on'}`}
          title={fx.bypass ? 'Turn on' : 'Bypass'}
          aria-pressed={!fx.bypass}
          onClick={() => update({ bypass: !fx.bypass })}
        >
          <Power size={13} />
        </button>
        <button className="fx-name" onClick={() => setOpen(!open)} aria-expanded={open}>
          {fx.name}
          {fx.reductionDb != null && fx.reductionDb < -0.5 && <span className="gr">{fx.reductionDb.toFixed(1)} dB</span>}
        </button>
        {fx.hasEditor && (
          <button className="icon-btn subtle" title="Open plugin window" onClick={() => void act(engine.call('fx.openEditor', { nodeId: node.id, fxId: fx.id }))}>
            <ExternalLink size={13} />
          </button>
        )}
        <button className="icon-btn subtle" disabled={index === 0} title="Move up" onClick={() => void act(engine.call('fx.move', { nodeId: node.id, fxId: fx.id, index: index - 1 }))}>
          <ChevronUp size={13} />
        </button>
        <button className="icon-btn subtle" disabled={index === count - 1} title="Move down" onClick={() => void act(engine.call('fx.move', { nodeId: node.id, fxId: fx.id, index: index + 1 }))}>
          <ChevronDown size={13} />
        </button>
        <button className="icon-btn subtle danger" title="Remove effect" onClick={() => void act(engine.call('fx.remove', { nodeId: node.id, fxId: fx.id }))}>
          <Trash2 size={13} />
        </button>
      </div>
      {fx.error && (
        <div className="fx-error" role="status">
          {fx.error}
        </div>
      )}
      {open && (
        <div className="fx-params">
          {fx.type === 'ducker' && (
            <label className="param">
              <span className="param-name">Duck when</span>
              <select value={fx.sidechain ?? ''} onChange={(e) => update({ sidechain: e.target.value || null })}>
                <option value="">— choose a source —</option>
                {state.nodes
                  .filter((n) => isSource(n.bin) && n.id !== node.id)
                  .map((n) => (
                    <option key={n.id} value={n.id}>
                      {n.name}
                    </option>
                  ))}
              </select>
            </label>
          )}
          {fx.params.map((p) =>
            p.choices ? (
              <label key={p.id} className="param">
                <span className="param-name">{p.name}</span>
                <select value={Math.round(p.value)} onChange={(e) => update({ params: { [p.id]: Number(e.target.value) } })}>
                  {p.choices.map((c, i) => (
                    <option key={i} value={i}>
                      {c}
                    </option>
                  ))}
                </select>
              </label>
            ) : (
              <label key={p.id} className="param">
                <span className="param-name">{p.name}</span>
                <input
                  type="range"
                  min={0}
                  max={1000}
                  value={Math.round(toSlider(p) * 1000)}
                  onChange={(e) => update({ params: { [p.id]: fromSlider(p, Number(e.target.value) / 1000) } })}
                  aria-valuetext={fmt(p)}
                />
                <span className="param-value">{fmt(p)}</span>
              </label>
            ),
          )}
          {fx.params.length === 0 && fx.type !== 'ducker' && <div className="muted small">No adjustable settings. Use the plugin window.</div>}
        </div>
      )}
    </div>
  );
}

export function FxChain({ node }: { node: NodeInfo }) {
  const state = useStore((s) => s.state)!;
  const [menu, setMenu] = useState(false);
  const btn = useRef<HTMLButtonElement>(null);

  const add = (type: FxType, pluginId?: string) => {
    setMenu(false);
    void act(
      engine.call('fx.add', { nodeId: node.id, type, pluginId }).then(async (fx) => {
        // A new plugin opens its window right away, as in a DAW.
        if (type === 'vst3' && fx.hasEditor !== false) await engine.call('fx.openEditor', { nodeId: node.id, fxId: fx.id });
      }),
    );
  };

  return (
    <div className="fx-chain">
      {node.fx.length === 0 && <div className="muted small">No effects. Add noise suppression, EQ, compression or your own VST3 plugins.</div>}
      {node.fx.map((fx, i) => (
        <FxItem key={fx.id} node={node} fx={fx} index={i} count={node.fx.length} />
      ))}
      <button ref={btn} className="btn small" onClick={() => setMenu(!menu)} aria-expanded={menu}>
        <Plus size={13} /> Add effect
      </button>
      {menu && (
        <Popover anchor={btn.current} onClose={() => setMenu(false)} label="Add effect" width={320}>
          <div className="popover-title">Built-in effects</div>
          {BUILTINS.map((b) => (
            <button key={b.type} className="menu-item two-line" onClick={() => add(b.type)}>
              <span>{b.name}</span>
              <span className="hint">{b.hint}</span>
            </button>
          ))}
          <div className="popover-title with-action">
            <span>VST3 plugins</span>
            <button
              className="icon-btn subtle"
              title={state.pluginsScanning ? 'Scanning…' : 'Scan for plugins'}
              disabled={state.pluginsScanning}
              onClick={() => void act(engine.call('plugins.scan'))}
            >
              <RefreshCw size={13} className={state.pluginsScanning ? 'spin' : ''} />
            </button>
          </div>
          {state.plugins.length === 0 && (
            <div className="muted small pad">
              {state.pluginsScanning ? 'Looking for plugins…' : 'No VST3 plugins found. Install plugins, or add your plugin folder in Settings › Plugins.'}
            </div>
          )}
          <div className="plugin-list">
            {state.plugins.map((p) => (
              <button key={p.id} className="menu-item two-line" onClick={() => add('vst3', p.id)}>
                <span>{p.name}</span>
                <span className="hint">
                  {p.vendor}
                  {p.category ? ` · ${p.category}` : ''}
                </span>
              </button>
            ))}
          </div>
        </Popover>
      )}
    </div>
  );
}
