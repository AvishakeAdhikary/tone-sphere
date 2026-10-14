import { useRef, useState } from 'react';
import { ChevronDown, Globe, Info, Plus } from 'lucide-react';
import type { Bin, NodeInfo } from '@shared/protocol';
import { BIN_META, DEST_ORDER, SOURCE_ORDER } from '../bins';
import { act, engine, useStore } from '../store';
import { NodeCard } from './NodeCard';
import { ConnectionLines } from './ConnectionLines';
import { VbanDialog } from './VbanDialog';

function useCollapsed(key: string): [boolean, () => void] {
  const [c, setC] = useState(() => {
    try {
      return localStorage.getItem(`tonesphere.collapsed.${key}`) === '1';
    } catch {
      return false;
    }
  });
  return [
    c,
    () => {
      setC(!c);
      try {
        localStorage.setItem(`tonesphere.collapsed.${key}`, c ? '0' : '1');
      } catch {
        /* ignore */
      }
    },
  ];
}

function Section({
  id,
  title,
  hint,
  empty,
  icon: Icon,
  hue,
  nodes,
  actions,
}: {
  id: string;
  title: string;
  hint: string;
  empty: string;
  icon: React.ComponentType<{ size?: number }>;
  hue: string;
  nodes: NodeInfo[];
  actions?: React.ReactNode;
}) {
  const [collapsed, toggle] = useCollapsed(id);
  return (
    <section className={`bin hue-${hue}`} aria-labelledby={`bin-${id}`}>
      <header className="bin-header">
        <button className="bin-toggle" onClick={toggle} aria-expanded={!collapsed} id={`bin-${id}`}>
          <ChevronDown size={14} className={`chev ${collapsed ? 'closed' : ''}`} />
          <span className="bin-icon">
            <Icon size={14} />
          </span>
          <span className="bin-title">{title}</span>
          <span className="bin-count">{nodes.length}</span>
        </button>
        <span className="bin-hint" title={hint} aria-label={hint}>
          <Info size={13} />
        </span>
        {actions}
      </header>
      {!collapsed && (
        <div className="bin-body">
          {nodes.length === 0 ? <div className="bin-empty">{empty}</div> : nodes.map((n) => <NodeCard key={n.id} node={n} />)}
        </div>
      )}
    </section>
  );
}

function sortNodes(a: NodeInfo, b: NodeInfo): number {
  const rank = (n: NodeInfo) => (n.state === 'unavailable' ? 2 : n.isDefault ? 0 : 1);
  return rank(a) - rank(b) || a.name.localeCompare(b.name);
}

export function SimpleView() {
  const state = useStore((s) => s.state)!;
  const gridRef = useRef<HTMLDivElement>(null);
  const [newMix, setNewMix] = useState<string | null>(null);
  const [vban, setVban] = useState<'in' | 'out' | null>(null);

  const byBin = (bin: Bin) =>
    state.nodes.filter((n) => n.bin === bin && n.kind !== 'vban' && n.kind !== 'tone' && !n.hidden).sort(sortNodes);
  const network = (dir: 'in' | 'out') =>
    state.nodes.filter((n) => n.kind === 'vban' && n.bin === (dir === 'in' ? 'virtual-out' : 'virtual-in'));
  const buses = state.nodes.filter((n) => n.bin === 'bus');

  const createMix = async () => {
    const name = (newMix ?? '').trim() || `Mix ${buses.length + 1}`;
    setNewMix(null);
    await act(engine.call('bus.create', { name, channels: 2 }));
  };

  return (
    <div className="simple-view">
      <div className="simple-grid" ref={gridRef}>
        <ConnectionLines container={gridRef} />
        <div className="column sources">
          <div className="column-title">
            <span>Sources</span>
            <span className="column-sub">where sound comes from</span>
          </div>
          {SOURCE_ORDER.map((bin) => (
            <Section key={bin} id={bin} {...BIN_META[bin]} nodes={byBin(bin)} />
          ))}
          <Section
            id="net-in"
            title="Network inputs"
            hint="Audio received from other computers on your network (VBAN, compatible with VoiceMeeter)."
            empty="No network streams. Add one to receive audio from another computer."
            icon={Globe}
            hue="violet"
            nodes={network('in')}
            actions={
              <button className="icon-btn subtle" title="Receive a network stream" onClick={() => setVban('in')}>
                <Plus size={14} />
              </button>
            }
          />
        </div>

        <div className="column mixes">
          <div className="column-title">
            <span>Mixes</span>
            <span className="column-sub">optional blending</span>
          </div>
          <Section
            id="bus"
            {...BIN_META.bus}
            nodes={buses}
            actions={
              <button className="icon-btn subtle" title="New mix" onClick={() => setNewMix('')}>
                <Plus size={14} />
              </button>
            }
          />
          {newMix !== null && (
            <div className="inline-create">
              <input
                autoFocus
                value={newMix}
                placeholder="Mix name (e.g. Stream mix)"
                aria-label="New mix name"
                onChange={(e) => setNewMix(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') void createMix();
                  if (e.key === 'Escape') setNewMix(null);
                }}
              />
              <button className="btn primary small" onClick={() => void createMix()}>
                Create
              </button>
            </div>
          )}
          <div className="drag-help">
            <strong>Tip:</strong> drag any source card onto a destination (or a mix) to connect it. Click a card to
            fine-tune volume, effects and channels.
          </div>
        </div>

        <div className="column destinations">
          <div className="column-title">
            <span>Destinations</span>
            <span className="column-sub">where sound goes</span>
          </div>
          {DEST_ORDER.map((bin) => (
            <Section key={bin} id={bin} {...BIN_META[bin]} nodes={byBin(bin)} />
          ))}
          <Section
            id="net-out"
            title="Network outputs"
            hint="Send audio to other computers on your network (VBAN, compatible with VoiceMeeter)."
            empty="No network streams. Add one to send audio to another computer."
            icon={Globe}
            hue="violet"
            nodes={network('out')}
            actions={
              <button className="icon-btn subtle" title="Send a network stream" onClick={() => setVban('out')}>
                <Plus size={14} />
              </button>
            }
          />
        </div>
      </div>
      {vban && <VbanDialog direction={vban} onClose={() => setVban(null)} />}
    </div>
  );
}
