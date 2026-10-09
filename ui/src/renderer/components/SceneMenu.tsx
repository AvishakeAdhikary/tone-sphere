import { useEffect, useRef, useState } from 'react';
import { Bookmark, Check, ChevronDown, Plus, Trash2 } from 'lucide-react';
import { act, engine, useStore } from '../store';

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
        </div>
      )}
    </div>
  );
}
