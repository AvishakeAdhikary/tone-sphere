import { useEffect, useRef, useState } from 'react';
import { CheckCircle2, FolderOpen, X, XCircle } from 'lucide-react';
import type { EngineSettings } from '@shared/protocol';
import { act, engine, useStore } from '../store';
import { bridge } from '../engine/bridge';

const RATES = [44100, 48000, 88200, 96000, 176400, 192000];
const BLOCKS = [32, 64, 128, 256, 512, 1024, 2048];
const TABS = ['Audio', 'Devices', 'Startup', 'Recording', 'Plugins', 'Shortcuts', 'Appearance', 'About'] as const;
type Tab = (typeof TABS)[number];

const HOTKEY_ACTIONS: [string, string][] = [
  ['mute-mic', 'Mute / unmute microphones'],
  ['push-to-talk', 'Push to talk (hold)'],
  ['mute-all', 'Mute all outputs'],
  ['save-replay', 'Save instant replay'],
  ['toggle-record', 'Start / stop recording'],
  ['next-scene', 'Next scene'],
];

function acceleratorFromEvent(e: React.KeyboardEvent): string | null {
  if (['Control', 'Shift', 'Alt', 'Meta'].includes(e.key)) return null;
  const parts = [];
  if (e.ctrlKey) parts.push('Ctrl');
  if (e.altKey) parts.push('Alt');
  if (e.shiftKey) parts.push('Shift');
  if (e.metaKey) parts.push('Win');
  const k = e.key.length === 1 ? e.key.toUpperCase() : e.key;
  parts.push(k === ' ' ? 'Space' : k);
  return parts.join('+');
}

export function SettingsDialog() {
  const state = useStore((s) => s.state)!;
  const close = () => useStore.getState().setSettingsOpen(false);
  const [tab, setTab] = useState<Tab>('Audio');
  const [storeBuild, setStoreBuild] = useState(false);
  const dlg = useRef<HTMLDivElement>(null);
  const s = state.settings;
  const st = state.status;
  const set = (p: Partial<EngineSettings>) => void act(engine.call('engine.setSettings', p));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && close();
    window.addEventListener('keydown', onKey);
    dlg.current?.focus();
    void bridge.platform().then((p) => setStoreBuild(!!p.store));
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const clockCandidates = state.nodes.filter(
    (n) => (n.kind === 'asio' || n.kind === 'wasapi' || n.kind === 'pipewire' || n.kind === 'coreaudio') && n.state !== 'unavailable',
  );

  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className="modal settings" role="dialog" aria-modal="true" aria-label="Settings" tabIndex={-1} ref={dlg}>
        <nav className="settings-nav" aria-label="Settings sections">
          <div className="settings-title">Settings</div>
          {TABS.map((t) => (
            <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)} aria-current={tab === t}>
              {t}
            </button>
          ))}
        </nav>
        <div className="settings-body">
          <button className="icon-btn close" onClick={close} aria-label="Close settings">
            <X size={16} />
          </button>
          {tab === 'Audio' && (
            <>
              <h2>Audio engine</h2>
              <label className="field">
                <span>Sample rate</span>
                <select value={s.sampleRate} onChange={(e) => set({ sampleRate: Number(e.target.value) })}>
                  {RATES.map((r) => (
                    <option key={r} value={r}>
                      {(r / 1000).toFixed(r % 1000 ? 1 : 0)} kHz
                    </option>
                  ))}
                </select>
                <span className="hint">48 kHz is the best choice for most setups (video, streaming, calls).</span>
              </label>
              <label className="field">
                <span>Buffer size</span>
                <select value={s.blockSize} onChange={(e) => set({ blockSize: Number(e.target.value) })}>
                  {BLOCKS.map((b) => (
                    <option key={b} value={b}>
                      {b} samples ({((b / s.sampleRate) * 1000).toFixed(1)} ms)
                    </option>
                  ))}
                </select>
                <span className="hint">Smaller = less delay but more CPU. Increase it if you hear crackles.</span>
              </label>
              <label className="field">
                <span>Master clock</span>
                <select value={s.clockMaster} onChange={(e) => set({ clockMaster: e.target.value })}>
                  <option value="auto">Automatic (recommended)</option>
                  <option value="internal">Internal timer</option>
                  {clockCandidates.map((n) => (
                    <option key={n.id} value={n.id}>
                      {n.name}
                    </option>
                  ))}
                </select>
                <span className="hint">Every other device is kept in sync automatically (drift compensation).</span>
              </label>
              <div className="field-row">
                <div className="muted small">
                  Currently running at {(st.sampleRate / 1000).toFixed(1)} kHz, {st.blockSize} samples, total latency{' '}
                  {st.latencyMs.toFixed(1)} ms.
                </div>
                <button className="btn small" onClick={() => void act(engine.call('engine.restart'))}>
                  Restart audio
                </button>
              </div>
            </>
          )}
          {tab === 'Devices' && (
            <>
              <h2>Devices &amp; drivers</h2>
              <div className="status-list">
                <div className="status-item">
                  {st.driverPack.vad ? <CheckCircle2 className="ok" size={18} /> : <XCircle className="bad" size={18} />}
                  <div>
                    <div>NexTone virtual devices</div>
                    <div className="hint">
                      {st.driverPack.vad
                        ? 'Installed: virtual inputs/outputs are provided by the NexTone driver.'
                        : 'Not installed: install the NexTone Driver Pack for native virtual devices.'}
                    </div>
                  </div>
                </div>
                <div className="status-item">
                  {st.driverPack.asio ? <CheckCircle2 className="ok" size={18} /> : <XCircle className="bad" size={18} />}
                  <div>
                    <div>NexTone ASIO driver</div>
                    <div className="hint">
                      {st.driverPack.asio
                        ? 'Installed: DAWs can select “NexTone ASIO” and several can run at once.'
                        : 'Not installed: part of the NexTone Driver Pack.'}
                    </div>
                  </div>
                </div>
              </div>
              <label className="toggle-row">
                <span className="grow">
                  <span>Compatibility mode</span>
                  <span className="hint">
                    Use already-installed virtual cables (VB-CABLE, VoiceMeeter, VAC) as virtual devices when the NexTone
                    driver isn’t available.
                  </span>
                </span>
                <input type="checkbox" role="switch" className="switch" checked={s.compatMode} onChange={(e) => set({ compatMode: e.target.checked })} />
              </label>
              <div className="field-row">
                <button className="btn small" onClick={() => void act(engine.call('devices.refresh'))}>
                  Rescan devices
                </button>
                <button className="btn small" onClick={() => void bridge.openSoundSettings()}>
                  Windows sound settings
                </button>
              </div>
            </>
          )}
          {tab === 'Startup' && (
            <>
              <h2>Startup</h2>
              {storeBuild ? (
                <div className="toggle-row">
                  <span className="grow">
                    <span>Start NexTone when I sign in</span>
                    <span className="hint">Windows manages this for apps from the Microsoft Store. It is on by default.</span>
                  </span>
                  <button className="btn small" onClick={() => void bridge.openStartupSettings()}>
                    Startup apps…
                  </button>
                </div>
              ) : (
              <label className="toggle-row">
                <span className="grow">
                  <span>Start NexTone when I sign in</span>
                  <span className="hint">Your routing is active right after sign-in, even before you open this window.</span>
                </span>
                <input
                  type="checkbox"
                  role="switch"
                  className="switch"
                  checked={s.startWithSystem}
                  onChange={(e) => {
                    set({ startWithSystem: e.target.checked });
                    void bridge.setLoginItem(e.target.checked);
                  }}
                />
              </label>
              )}
              <label className="toggle-row">
                <span className="grow">
                  <span>Start minimized to the tray</span>
                </span>
                <input type="checkbox" role="switch" className="switch" checked={s.startMinimized} onChange={(e) => set({ startMinimized: e.target.checked })} />
              </label>
            </>
          )}
          {tab === 'Recording' && (
            <>
              <h2>Recording</h2>
              <div className="field">
                <span>Save recordings to</span>
                <div className="field-row">
                  <input readOnly value={s.recordFolder} className="grow mono small" aria-label="Recording folder" />
                  <button
                    className="btn small"
                    onClick={async () => {
                      const f = await bridge.pickFolder();
                      if (f) set({ recordFolder: f });
                    }}
                  >
                    <FolderOpen size={13} /> Change
                  </button>
                  <button className="btn small" onClick={() => void bridge.openPath(s.recordFolder)}>
                    Open
                  </button>
                </div>
              </div>
              <label className="field">
                <span>Instant replay buffer</span>
                <select value={s.replayBufferSeconds} onChange={(e) => set({ replayBufferSeconds: Number(e.target.value) })}>
                  <option value={0}>Off</option>
                  {[30, 60, 120, 300, 600].map((v) => (
                    <option key={v} value={v}>
                      Last {v >= 60 ? `${v / 60} min` : `${v} s`}
                    </option>
                  ))}
                </select>
                <span className="hint">Continuously keeps the last minutes of every mix so you can save something that already happened.</span>
              </label>
              {s.replayBufferSeconds > 0 && (
                <button className="btn small" onClick={() => void act(engine.call('recorder.saveReplay', { seconds: s.replayBufferSeconds })).then((r) => r && useStore.getState().toast('info', `Saved ${r.file}`))}>
                  Save replay now
                </button>
              )}
            </>
          )}
          {tab === 'Plugins' && (
            <>
              <h2>Plugins</h2>
              <p className="hint">
                NexTone loads VST3 effects from the standard VST3 folders. Add a folder here if your plugins are installed somewhere else.
              </p>
              <div className="field">
                <span>Extra plugin folders</span>
                {(s.pluginFolders ?? []).length === 0 && <span className="muted small">None</span>}
                {(s.pluginFolders ?? []).map((f) => (
                  <div className="field-row" key={f}>
                    <input readOnly value={f} className="grow mono small" aria-label="Plugin folder" />
                    <button
                      className="btn small"
                      aria-label={`Remove ${f}`}
                      onClick={() => set({ pluginFolders: (s.pluginFolders ?? []).filter((x) => x !== f) })}
                    >
                      <X size={13} /> Remove
                    </button>
                  </div>
                ))}
                <div className="field-row">
                  <button
                    className="btn small"
                    onClick={async () => {
                      const f = await bridge.pickFolder();
                      if (f && !(s.pluginFolders ?? []).includes(f)) set({ pluginFolders: [...(s.pluginFolders ?? []), f] });
                    }}
                  >
                    <FolderOpen size={13} /> Add folder…
                  </button>
                </div>
              </div>
              <div className="field-row">
                <button className="btn small" disabled={state.pluginsScanning} onClick={() => void act(engine.call('plugins.scan'))}>
                  {state.pluginsScanning ? 'Scanning…' : 'Scan now'}
                </button>
                <span className="muted small" role="status">
                  {state.plugins.length === 1 ? '1 plugin found' : `${state.plugins.length} plugins found`}
                </span>
              </div>
              <p className="hint">
                Each plugin is tested in a separate process first, so a broken plugin can't crash NexTone while scanning.
              </p>
            </>
          )}
          {tab === 'Shortcuts' && (
            <>
              <h2>Keyboard shortcuts</h2>
              <p className="muted small">Shortcuts work everywhere, even while you’re in a game or another app. Click a box and press the keys.</p>
              {HOTKEY_ACTIONS.map(([id, label]) => (
                <label key={id} className="num-row">
                  <span className="grow">{label}</span>
                  <input
                    readOnly
                    className="hotkey-input"
                    value={s.hotkeys[id] ?? ''}
                    placeholder="Not set"
                    onKeyDown={(e) => {
                      e.preventDefault();
                      if (e.key === 'Backspace' || e.key === 'Delete') {
                        const h = { ...s.hotkeys };
                        delete h[id];
                        set({ hotkeys: h });
                        return;
                      }
                      const acc = acceleratorFromEvent(e);
                      if (acc) set({ hotkeys: { ...s.hotkeys, [id]: acc } });
                    }}
                  />
                </label>
              ))}
            </>
          )}
          {tab === 'Appearance' && (
            <>
              <h2>Appearance</h2>
              <div className="choice-group horizontal" role="radiogroup" aria-label="Theme">
                {(['system', 'light', 'dark'] as const).map((t) => (
                  <label key={t} className={`choice ${s.theme === t ? 'on' : ''}`}>
                    <input type="radio" name="theme" checked={s.theme === t} onChange={() => set({ theme: t })} />
                    <span className="choice-label">{t === 'system' ? 'Match Windows' : t === 'light' ? 'Light' : 'Dark'}</span>
                  </label>
                ))}
              </div>
              <label className="toggle-row">
                <span className="grow">
                  <span>Show advanced options by default</span>
                  <span className="hint">Expands device buffers, delay, polarity and statistics in the details panel.</span>
                </span>
                <input type="checkbox" role="switch" className="switch" checked={s.advancedMode} onChange={(e) => set({ advancedMode: e.target.checked })} />
              </label>
            </>
          )}
          {tab === 'About' && (
            <>
              <h2>About NexTone</h2>
              <p>
                Engine {state.version} · protocol v{state.protocol} · {st.platform}
              </p>
              <p className="muted small">
                NexTone is free and open-source software, licensed under the GNU GPL v3. ASIO is a trademark and software of
                Steinberg Media Technologies GmbH. VST is a registered trademark of Steinberg Media Technologies GmbH.
              </p>
              {st.testMode && (
                <p className="notice s-idle">
                  This PC runs in Windows test-signing mode, which the development build of the NexTone virtual audio driver
                  needs.
                </p>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
