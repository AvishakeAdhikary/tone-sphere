import { Circle, Headphones, LayoutGrid, Settings, Square, Workflow } from 'lucide-react';
import { act, engine, useStore } from '../store';
import { SceneMenu } from './SceneMenu';
import logo from '../assets/logo.png';
import { t } from '../i18n';

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
        <img className="brand-mark" src={logo} alt="" aria-hidden width={26} height={26} />
        <span className="brand-name">{t("ToneSphere")}</span>
      </div>

      <div className="segmented no-drag" role="tablist" aria-label={t("View")}>
        <button
          role="tab"
          aria-selected={view === 'simple'}
          className={view === 'simple' ? 'active' : ''}
          onClick={() => setView('simple')}
          title={t("Simple view: drag a source onto a destination")}
        >
          <Workflow size={15} /> {t('Routing')}
        </button>
        <button
          role="tab"
          aria-selected={view === 'matrix'}
          className={view === 'matrix' ? 'active' : ''}
          onClick={() => setView('matrix')}
          title={t("Matrix view: every source × every destination")}
        >
          <LayoutGrid size={15} /> {t('Matrix')}
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
          <span>{recording ? t('Stop') : t('Rec')}</span>
        </button>
      )}

      <button
        className="icon-btn no-drag"
        onClick={() => useStore.getState().setWizardOpen(true)}
        title={t("Hear my instrument or microphone")}
        aria-label={t("Hear my instrument or microphone")}
        disabled={!state}
      >
        <Headphones size={17} />
      </button>
      <button
        className="icon-btn no-drag"
        onClick={() => setSettingsOpen(true)}
        title={t("Settings")}
        aria-label={t("Settings")}
        disabled={!state}
      >
        <Settings size={17} />
      </button>
      <div className="window-controls-space" />
    </header>
  );
}
