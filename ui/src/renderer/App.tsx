import { useEffect } from 'react';
import { TitleBar } from './components/TitleBar';
import { SimpleView } from './components/SimpleView';
import { MatrixView } from './components/MatrixView';
import { Inspector } from './components/Inspector';
import { SettingsDialog } from './components/SettingsDialog';
import { Toasts } from './components/Toasts';
import { EngineMissing } from './components/EngineMissing';
import { CrashNotice } from './components/CrashNotice';
import { MonitorWizard } from './components/MonitorWizard';
import { exportPresetFile, importPresetFile } from './presetFiles';
import { act, engine, useStore } from './store';
import { bridge } from './engine/bridge';
import { setMeterLayout } from './engine/meterLoop';
import { applySkin, findSkin } from './skins';
import { resolveLanguage, setLanguage } from './i18n';

export function App() {
  const status = useStore((s) => s.status);
  const state = useStore((s) => s.state);
  const view = useStore((s) => s.view);
  const selection = useStore((s) => s.selection);
  const settingsOpen = useStore((s) => s.settingsOpen);
  const wizardOpen = useStore((s) => s.wizardOpen);

  useEffect(() => {
    if (state) setMeterLayout(state.layoutVersion);
  }, [state]);

  const theme = state?.settings.theme ?? 'system';
  const skinId = state?.settings.skin ?? 'fluent';
  // The state object is rebuilt on every engine update: depend on the skins' content, not identity.
  const customKey = JSON.stringify(state?.settings.customSkins ?? []);
  useEffect(() => {
    const skin = findSkin(skinId, JSON.parse(customKey));
    const apply = () => {
      const mode = applySkin(skin, theme);
      document.documentElement.dataset.theme = mode;
      void bridge.setTheme(mode);
    };
    apply();
    // Per-mode skin tokens must follow Windows when the theme is "Match Windows".
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    mq.addEventListener('change', apply);
    return () => mq.removeEventListener('change', apply);
  }, [theme, skinId, customKey]);

  useEffect(
    () =>
      bridge.onCommand((cmd) => {
        const s = useStore.getState().state;
        if (cmd === 'mute-all' && s) {
          for (const n of s.nodes) {
            if (n.bin === 'physical-out' && !n.muted) void act(engine.call('node.update', { id: n.id, muted: true }));
          }
        }
        if (cmd === 'shutdown-engine') void engine.call('engine.shutdown').catch(() => {});
      }),
    [],
  );

  // Escape clears the selection / closes the inspector. Window shortcuts (from the old ToneSphere):
  // Ctrl+S export a preset, Ctrl+O import one, Ctrl+, settings, Ctrl+M hear my input,
  // Ctrl+1 / Ctrl+2 routing / matrix view, F5 rescan devices, Ctrl+Shift+M mute every output.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const st = useStore.getState();
      if (e.key === 'Escape') st.select(null);
      const el = e.target as HTMLElement | null;
      // Never steal keys from text fields, lists or the on-screen keyboard.
      if (el?.closest('input, textarea, select, [contenteditable="true"], .keyboard')) return;
      if (!st.state || st.status !== 'connected') return;
      const ctrl = e.ctrlKey || e.metaKey;
      const key = e.key.toLowerCase();
      let done = true;
      if (ctrl && !e.shiftKey && key === 's') void exportPresetFile();
      else if (ctrl && !e.shiftKey && key === 'o') void importPresetFile();
      else if (ctrl && key === ',') st.setSettingsOpen(true);
      else if (ctrl && !e.shiftKey && key === 'm') st.setWizardOpen(true);
      else if (ctrl && key === '1') st.setView('simple');
      else if (ctrl && key === '2') st.setView('matrix');
      else if (e.key === 'F5' && !ctrl) {
        void act(engine.call('devices.refresh'));
        st.toast('info', 'Looking for devices…');
      } else if (ctrl && e.shiftKey && key === 'm') {
        for (const n of st.state.nodes) if (n.bin === 'physical-out' && !n.muted) void act(engine.call('node.update', { id: n.id, muted: true }));
        st.toast('info', 'All outputs muted.');
      } else done = false;
      if (done) e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const ready = status === 'connected' && state;
  const lang = resolveLanguage(state?.settings.language);
  setLanguage(lang);

  return (
    <div className="app" key={lang}>
      <TitleBar />
      <div className="workspace">
        {ready ? (
          <>
            <main className="main" aria-label={view === 'simple' ? 'Routing overview' : 'Routing matrix'}>
              {view === 'simple' ? <SimpleView /> : <MatrixView />}
            </main>
            {selection && <Inspector />}
          </>
        ) : (
          <EngineMissing status={status} />
        )}
      </div>
      {ready && <CrashNotice />}
      {settingsOpen && ready && <SettingsDialog />}
      {wizardOpen && ready && <MonitorWizard />}
      <Toasts />
    </div>
  );
}
