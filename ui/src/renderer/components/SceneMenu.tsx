import { useEffect, useRef, useState } from 'react';
import { Bookmark, Check, ChevronDown, Download, Plus, Trash2, Upload } from 'lucide-react';
import { act, engine, useStore } from '../store';
import { bridge } from '../engine/bridge';

/** Scenes = saved routing snapshots (ASIO Link Pro's "Profiles", but unlimited). */
export function SceneMenu() {
  const state = useStore((s) => s.state)!;
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');
  const ref = useRef<HTMLDivElement>(null);
  const active = state.scenes.find((s) => s.id === state.activeScene);

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener('mousedown', close);
    return () => window.removeEventListener('mousedown', close);
  }, [open]);

  const save = async () => {
    const n = name.trim() || `Scene ${state.scenes.length + 1}`;
    await act(engine.call('scene.save', { name: n }));
    setName('');
  };

  const toast = useStore.getState().toast;
  const importPreset = async () => {
    try {
      const f = await bridge.openFile('Import a ToneSphere preset', ['json']);
      if (!f) return;
      let preset: unknown;
      try {
        preset = JSON.parse(f.text);
      } catch {
        throw new Error("This file isn't a ToneSphere preset (it isn't valid JSON).");
      }
      const r = await engine.call('preset.import', { preset });
      setOpen(false);
      toast('info', `Preset “${r.name}” is active: ${r.connections} connection${r.connections === 1 ? '' : 's'}.`);
      if (r.waiting.length)
        toast('info', `Waiting for: ${r.waiting.join(', ')}. They connect automatically as soon as they appear.`);
    } catch (e) {
      toast('error', (e as Error).message);
    }
  };
  const exportPreset = async () => {
    const name = active?.name ?? 'My ToneSphere setup';
    try {
      const p = await engine.call('preset.export', { name });
      const file = `${name.replace(/[^\w -]+/g, '').trim() || 'preset'}.tonesphere.json`;
      const path = await bridge.saveFile('Export preset', file, JSON.stringify(p, null, 2) + '\n', ['json']);
      if (path) toast('info', `Saved ${path}`);
    } catch (e) {
      toast('error', (e as Error).message);
    }
  };

  return (
    <div className="menu-anchor no-drag" ref={ref}>
      <button className="icon-btn" onClick={() => setOpen(!open)} aria-expanded={open} title="Scenes">
        <Bookmark size={15} />
        <span className="scene-name">{active?.name ?? 'Scenes'}</span>
        <ChevronDown size={14} />
      </button>
      {open && (
        <div className="popover scene-popover" role="menu">
          <div className="popover-title">Scenes</div>
          <p className="muted small">A scene remembers every connection, volume and effect so you can switch setups instantly.</p>
          {state.scenes.length === 0 && <div className="muted small pad">No scenes saved yet.</div>}
          {state.scenes.map((s) => (
            <div key={s.id} className="menu-row">
              <button
                className="menu-item grow"
                role="menuitem"
                onClick={() => {
                  void act(engine.call('scene.load', { id: s.id }));
                  setOpen(false);
                }}
              >
                {s.id === state.activeScene ? <Check size={14} /> : <span style={{ width: 14 }} />}
                {s.name}
                {s.hotkey && <kbd>{s.hotkey}</kbd>}
              </button>
              <button
                className="icon-btn subtle"
                title="Overwrite with current setup"
                onClick={() => void act(engine.call('scene.save', { id: s.id, name: s.name }))}
              >
                <Bookmark size={13} />
              </button>
              <button
                className="icon-btn subtle"
                title="Delete scene"
                onClick={() => void act(engine.call('scene.delete', { id: s.id }))}
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
          <div className="menu-row new-scene">
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="New scene name"
              aria-label="New scene name"
              onKeyDown={(e) => e.key === 'Enter' && void save()}
            />
            <button className="btn primary small" onClick={() => void save()}>
              <Plus size={13} /> Save
            </button>
          </div>
          <div className="popover-title">Presets</div>
          <p className="muted small">Presets are routing maps you can share. They find devices by name, so they work on any PC.</p>
          <div className="menu-row">
            <button className="btn small" onClick={() => void importPreset()}>
              <Upload size={13} /> Import preset…
            </button>
            <button className="btn small" onClick={() => void exportPreset()}>
              <Download size={13} /> Export as preset…
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
