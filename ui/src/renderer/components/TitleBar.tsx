import { Circle, LayoutGrid, Settings, Square, Workflow } from 'lucide-react';
import { act, engine, useStore } from '../store';
import { SceneMenu } from './SceneMenu';

export function TitleBar() {
  const state = useStore((s) => s.state);
  const status = useStore((s) => s.status);
  const view = useStore((s) => s.view);
  const setView = useStore((s) => s.setView);
  const setSettingsOpen = useStore((s) => s.setSettingsOpen);
  const st = state?.status;
  const recording = state?.recorder.recording ?? false;

  const toggleRecord = () => {
    if (!state) return;
    if (recording) {
      void act(engine.call('recorder.stop')).then((r) => {
        if (r?.files.length) useStore.getState().toast('info', `Saved ${r.files.length} recording(s)`);
      });
    } else {
      const sel = useStore.getState().selection;
      const ids =
        sel?.type === 'node'
          ? [sel.id]
          : state.nodes.filter((n) => n.bin === 'bus' || n.bin === 'physical-out').map((n) => n.id);
      void act(engine.call('recorder.start', { nodeIds: ids, format: 'wav' }));
    }
  };

  return (
    <header className="titlebar">
      <div className="titlebar-brand">
        <span className="brand-mark" aria-hidden>
          <svg viewBox="0 0 24 24" width="20" height="20">
            <path
              d="M3 12c2.5 0 2.5-6 5-6s2.5 12 5 12 2.5-12 5-12 2.5 6 3 6"
              fill="none"
              stroke="currentColor"
              strokeWidth="2.4"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </span>
        <span className="brand-name">NexTone</span>
      </div>

      <div className="segmented no-drag" role="tablist" aria-label="View">
        <button
          role="tab"
          aria-selected={view === 'simple'}
          className={view === 'simple' ? 'active' : ''}
          onClick={() => setView('simple')}
          title="Simple view: drag a source onto a destination"
        >
          <Workflow size={15} /> Routing
        </button>
        <button
          role="tab"
          aria-selected={view === 'matrix'}
          className={view === 'matrix' ? 'active' : ''}
          onClick={() => setView('matrix')}
          title="Matrix view: every source × every destination"
        >
          <LayoutGrid size={15} /> Matrix
        </button>
      </div>

      <div className="titlebar-spacer" />

      {st && status === 'connected' && (
        <div className="engine-pill no-drag" title={st.message ?? 'Engine running'}>
          <span className={`dot ${st.running ? (st.xruns > 0 ? 'warn' : 'ok') : 'bad'}`} />
          <span>{(st.sampleRate / 1000).toFixed(st.sampleRate % 1000 ? 1 : 0)} kHz</span>
          <span className="sep" />
          <span>{st.blockSize} smp</span>
          <span className="sep" />
          <span>{st.latencyMs.toFixed(1)} ms</span>
          <span className="sep" />
          <span>CPU {st.cpuPercent.toFixed(0)}%</span>
          {st.xruns > 0 && (
            <>
              <span className="sep" />
              <span className="warn-text">{st.xruns} dropouts</span>
            </>
          )}
        </div>
      )}

      {state && <SceneMenu />}

      {state && (
        <button
          className={`icon-btn no-drag record ${recording ? 'on' : ''}`}
          onClick={toggleRecord}
          aria-pressed={recording}
          title={recording ? 'Stop recording' : 'Record (selected item, or all mixes and outputs)'}
        >
          {recording ? <Square size={14} fill="currentColor" /> : <Circle size={14} fill="currentColor" />}
          <span>{recording ? 'Stop' : 'Rec'}</span>
        </button>
      )}

      <button
        className="icon-btn no-drag"
        onClick={() => setSettingsOpen(true)}
        title="Settings"
        aria-label="Settings"
        disabled={!state}
      >
        <Settings size={17} />
      </button>
      <div className="window-controls-space" />
    </header>
  );
}
