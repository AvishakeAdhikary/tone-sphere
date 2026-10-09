import { useRef, useState } from 'react';
import { ArrowRight, Check } from 'lucide-react';
import { isDestination, type Bin, type NodeInfo } from '@shared/protocol';
import { BIN_META, DEST_ORDER, nodeIcon } from '../bins';
import { act, engine, findRoute, useStore } from '../store';
import { Popover } from './Popover';

/** Keyboard-friendly alternative to drag & drop: tick the destinations a source should feed. */
export function SendToMenu({ node }: { node: NodeInfo }) {
  const state = useStore((s) => s.state)!;
  const [open, setOpen] = useState(false);
  const btn = useRef<HTMLButtonElement>(null);
  const count = state.routes.filter((r) => r.from === node.id).length;

  const groups: Bin[] = ['bus', ...DEST_ORDER];
  const toggle = (dest: NodeInfo) => {
    const r = findRoute(state, node.id, dest.id);
    if (r) void act(engine.call('route.disconnect', { id: r.id }));
    else void act(engine.call('route.connect', { from: node.id, to: dest.id, mode: 'auto' }));
  };

  return (
    <>
      <button
        ref={btn}
        className={`send-btn ${count ? 'has' : ''}`}
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        aria-haspopup="dialog"
        title="Choose where this sound goes"
      >
        <ArrowRight size={13} />
        <span>{count || 'Send'}</span>
      </button>
      {open && (
        <Popover anchor={btn.current} onClose={() => setOpen(false)} label={`Send ${node.name} to`} width={320}>
          <div className="popover-title">Send “{node.name}” to…</div>
          <div className="send-list">
            {groups.map((bin) => {
              const dests = state.nodes.filter((n) => n.bin === bin && n.id !== node.id && isDestination(n.bin));
              if (!dests.length) return null;
              return (
                <div key={bin} className="send-group">
                  <div className="send-group-title">{BIN_META[bin].title}</div>
                  {dests.map((d) => {
                    const on = !!findRoute(state, node.id, d.id);
                    const Icon = nodeIcon(d);
                    return (
                      <button
                        key={d.id}
                        className={`send-item ${on ? 'on' : ''}`}
                        role="menuitemcheckbox"
                        aria-checked={on}
                        onClick={() => toggle(d)}
                      >
                        <span className="check">{on && <Check size={13} />}</span>
                        {d.icon ? <img src={d.icon} alt="" className="mini-icon" /> : <Icon size={14} />}
                        <span className="grow ellipsis">{d.name}</span>
                        {d.state !== 'active' && <span className="dim small">{d.state === 'unavailable' ? 'offline' : ''}</span>}
                      </button>
                    );
                  })}
                </div>
              );
            })}
          </div>
        </Popover>
      )}
    </>
  );
}
