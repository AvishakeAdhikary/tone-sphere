import { memo } from 'react';
import { isDestination, isSource, type NodeInfo } from '@shared/protocol';
import { Headphones, Pin, Volume2, VolumeX } from 'lucide-react';
import { BIN_META, kindLabel, nodeIcon } from '../bins';
import { act, engine, findRoute, useStore } from '../store';
import { Meter } from './Meter';
import { Fader } from './Fader';
import { SendToMenu } from './SendToMenu';
import { t } from '../i18n';

const DRAG_TYPE = 'application/x-tonesphere-node';

function statusLabel(n: NodeInfo): string | null {
  if (n.state === 'unavailable') return n.statusText ?? 'Disconnected';
  if (n.state === 'error') return n.statusText ?? 'Error';
  if (!n.enabled) return 'Off';
  if (n.state === 'idle') return n.statusText ?? (n.kind === 'app' ? 'Silent' : 'Idle');
  return null;
}

export const NodeCard = memo(function NodeCard({ node }: { node: NodeInfo }) {
  const selected = useStore((s) => s.selection?.type === 'node' && s.selection.id === node.id);
  const hover = useStore((s) => s.hoverNode);
  const dragFrom = useStore((s) => s.dragFrom);
  const state = useStore((s) => s.state);
  const connectedToHover =
    hover != null &&
    hover !== node.id &&
    !!state?.routes.some(
      (r) => (r.from === hover && r.to === node.id) || (r.to === hover && r.from === node.id),
    );

  const src = isSource(node.bin);
  const dst = isDestination(node.bin);
  const meta = BIN_META[node.bin];
  const Icon = nodeIcon(node);
  const status = statusLabel(node);

  // Drop-target logic: a dragged source can connect to any destination except itself.
  const canDrop = dragFrom != null && dragFrom !== node.id && dst;
  const alreadyConnected = canDrop && !!findRoute(state, dragFrom!, node.id);

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const from = e.dataTransfer.getData(DRAG_TYPE);
    useStore.getState().setDragFrom(null);
    if (!from || from === node.id) return;
    const existing = findRoute(useStore.getState().state, from, node.id);
    if (existing) {
      useStore.getState().select({ type: 'route', id: existing.id });
      return;
    }
    void act(engine.call('route.connect', { from, to: node.id, mode: 'auto' })).then((r) => {
      if (r) useStore.getState().toast('info', 'Connected');
    });
  };

  return (
    <div
      className={[
        'node-card',
        `hue-${meta.hue}`,
        selected ? 'selected' : '',
        connectedToHover ? 'linked' : '',
        canDrop ? (alreadyConnected ? 'drop-existing' : 'drop-ok') : '',
        node.state !== 'active' || !node.enabled ? 'inactive' : '',
      ].join(' ')}
      data-node-id={node.id}
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      aria-label={`${node.name}, ${meta.title}${status ? `, ${status}` : ''}`}
      draggable={src}
      onDragStart={(e) => {
        e.dataTransfer.setData(DRAG_TYPE, node.id);
        e.dataTransfer.effectAllowed = 'link';
        useStore.getState().setDragFrom(node.id);
      }}
      onDragEnd={() => useStore.getState().setDragFrom(null)}
      onDragOver={(e) => {
        if (canDrop) {
          e.preventDefault();
          e.dataTransfer.dropEffect = 'link';
        }
      }}
      onDrop={onDrop}
      onMouseEnter={() => useStore.getState().setHover(node.id)}
      onMouseLeave={() => useStore.getState().setHover(null)}
      onClick={() => useStore.getState().select(selected ? null : { type: 'node', id: node.id })}
      onKeyDown={(e) => {
        if (e.target !== e.currentTarget) return;
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          useStore.getState().select({ type: 'node', id: node.id });
        }
      }}
    >
      {dst && <span className="port port-in" aria-hidden />}
      {src && <span className="port port-out" aria-hidden />}
      <div className="node-head">
        <div className="node-icon">
          {node.icon ? <img src={node.icon} alt="" draggable={false} /> : <Icon size={18} />}
        </div>
        <div className="node-titles">
          <div className="node-name" title={node.name}>
            {node.name}
            {node.isDefault && (
              <span className="badge" title={t("Windows default device")}>
                Default
              </span>
            )}
            {node.app?.pinned && <Pin size={11} className="pin" aria-label={t("Remembered app")} />}
          </div>
          <div className="node-sub">
            {status ? <span className={`status s-${node.state}`}>{status} ·</span> : null}
            <span>{node.subtitle ?? kindLabel(node)}</span>
            <span className="dim">· {node.channels === 1 ? 'mono' : node.channels === 2 ? 'stereo' : `${node.channels} ch`}</span>
          </div>
        </div>
      </div>
      <Meter slot={node.meter} />
      <div className="node-controls" onClick={(e) => e.stopPropagation()}>
        <button
          className={`icon-btn subtle mute ${node.muted ? 'on' : ''}`}
          onClick={() => void act(engine.call('node.update', { id: node.id, muted: !node.muted }))}
          aria-pressed={node.muted}
          title={node.muted ? 'Unmute' : 'Mute'}
          aria-label={node.muted ? `Unmute ${node.name}` : `Mute ${node.name}`}
        >
          {node.muted ? <VolumeX size={15} /> : <Volume2 size={15} />}
        </button>
        <Fader
          compact
          valueDb={node.gainDb}
          label={`${node.name} volume`}
          onChange={(db) => void act(engine.call('node.update', { id: node.id, gainDb: db }))}
        />
        {node.bin === 'bus' && (
          <button
            className={`icon-btn subtle ${node.solo ? 'on' : ''}`}
            title={t("Listen (solo this mix on monitor outputs)")}
            aria-pressed={node.solo}
            onClick={() => void act(engine.call('node.update', { id: node.id, solo: !node.solo }))}
          >
            <Headphones size={14} />
          </button>
        )}
        {src && <SendToMenu node={node} />}
      </div>
    </div>
  );
});

export { DRAG_TYPE };
