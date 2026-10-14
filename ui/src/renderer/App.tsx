import { useEffect } from 'react';
import { TitleBar } from './components/TitleBar';
import { SimpleView } from './components/SimpleView';
import { MatrixView } from './components/MatrixView';
import { Inspector } from './components/Inspector';
import { SettingsDialog } from './components/SettingsDialog';
import { Toasts } from './components/Toasts';
import { EngineMissing } from './components/EngineMissing';
import { act, engine, useStore } from './store';
import { bridge } from './engine/bridge';
import { setMeterLayout } from './engine/meterLoop';
import { applySkin, findSkin } from './skins';

export function App() {
  const status = useStore((s) => s.status);
  const state = useStore((s) => s.state);
  const view = useStore((s) => s.view);
  const selection = useStore((s) => s.selection);
  const settingsOpen = useStore((s) => s.settingsOpen);

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

  // Escape clears the selection / closes the inspector.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') useStore.getState().select(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const ready = status === 'connected' && state;

  return (
    <div className="app">
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
      {settingsOpen && ready && <SettingsDialog />}
      <Toasts />
    </div>
  );
}
