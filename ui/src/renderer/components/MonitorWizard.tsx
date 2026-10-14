// "Monitor my input" (from the old ToneSphere): plug in a guitar or a microphone and hear it in
// your headphones in one step. Picks the audio interface and input 1 for you, sends that one
// channel to both ears at a safe level.
import { useEffect, useMemo, useRef, useState } from 'react';
import { Headphones, X } from 'lucide-react';
import type { NodeInfo } from '@shared/protocol';
import { act, engine, useStore } from '../store';

const START_DB = -12; // start quietly: the ears come first

function guessInput(nodes: NodeInfo[]): NodeInfo | undefined {
  const ins = nodes.filter((n) => n.bin === 'physical-in' && n.state !== 'unavailable');
  // An audio interface (ASIO first) rather than the laptop's microphone.
  return ins.find((n) => n.kind === 'asio') ?? ins.find((n) => !n.isDefault && !/microphone array|webcam/i.test(n.name)) ?? ins[0];
}
function guessOutput(nodes: NodeInfo[]): NodeInfo | undefined {
  const outs = nodes.filter((n) => n.bin === 'physical-out' && n.state !== 'unavailable');
  return outs.find((n) => /headphone|headset|earphone/i.test(`${n.name} ${n.subtitle ?? ''}`)) ?? outs.find((n) => n.isDefault) ?? outs[0];
}

export function MonitorWizard() {
  const state = useStore((s) => s.state)!;
  const close = () => useStore.getState().setWizardOpen(false);
  const inputs = useMemo(() => state.nodes.filter((n) => n.bin === 'physical-in' && n.state !== 'unavailable'), [state.nodes]);
  const outputs = useMemo(() => state.nodes.filter((n) => n.bin === 'physical-out' && n.state !== 'unavailable'), [state.nodes]);
  const [inputId, setInputId] = useState(() => guessInput(state.nodes)?.id ?? '');
  const [outputId, setOutputId] = useState(() => guessOutput(state.nodes)?.id ?? '');
  const [channel, setChannel] = useState(0); // -1 = all channels (stereo)
  const [busy, setBusy] = useState(false);
  const dlg = useRef<HTMLDivElement>(null);
  const input = inputs.find((n) => n.id === inputId);
  const output = outputs.find((n) => n.id === outputId);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && close();
    window.addEventListener('keydown', onKey);
    dlg.current?.focus();
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const start = async () => {
    if (!input || !output) return;
    setBusy(true);
    try {
      const existing = state.routes.find((r) => r.from === input.id && r.to === output.id);
      const route = existing ?? (await engine.call('route.connect', { from: input.id, to: output.id, mode: 'one-to-one' }));
      if (channel >= 0) {
        // One input channel (a guitar or a microphone) into every output channel: both ears.
        await engine.call('route.applyPreset', { id: route.id, mode: 'clear' });
        for (let ch = 0; ch < output.channels; ch++) await engine.call('route.setCrosspoint', { id: route.id, fromCh: channel, toCh: ch, gainDb: 0 });
      } else {
        await engine.call('route.applyPreset', { id: route.id, mode: 'one-to-one' });
      }
      await engine.call('route.update', { id: route.id, gainDb: START_DB, muted: false, enabled: true });
      useStore
        .getState()
        .toast('info', `You should hear ${channel >= 0 ? `input ${channel + 1} of ${input.name}` : input.name} in ${output.name} now. Turn it up on the connection when you're ready.`);
      close();
    } catch (e) {
      useStore.getState().toast('error', (e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && close()}>
      <div className="modal wizard" role="dialog" aria-modal="true" aria-labelledby="wizard-title" tabIndex={-1} ref={dlg}>
        <button className="icon-btn close" onClick={close} aria-label="Close">
          <X size={16} />
        </button>
        <h2 id="wizard-title">
          <Headphones size={18} aria-hidden /> Hear my instrument or microphone
        </h2>
        <p className="muted small">Plug it in, pick it here, and listen in your headphones. You can change everything later.</p>
        {inputs.length === 0 || outputs.length === 0 ? (
          <p className="hint" role="status">
            {inputs.length === 0 ? 'No input is connected. Plug in your audio interface or microphone.' : 'No output is connected. Plug in headphones or speakers.'}
          </p>
        ) : (
          <>
            <label className="field">
              <span>Plugged into</span>
              <select value={inputId} onChange={(e) => (setInputId(e.target.value), setChannel(0))}>
                {inputs.map((n) => (
                  <option key={n.id} value={n.id}>
                    {n.name}
                    {n.subtitle ? ` (${n.subtitle})` : ''}
                  </option>
                ))}
              </select>
            </label>
            {input && input.channels > 1 && (
              <label className="field">
                <span>Input</span>
                <select value={channel} onChange={(e) => setChannel(Number(e.target.value))}>
                  {Array.from({ length: input.channels }, (_, i) => (
                    <option key={i} value={i}>
                      Input {i + 1}
                      {i === 0 ? ' (guitar, usually)' : ''}
                    </option>
                  ))}
                  <option value={-1}>All inputs (stereo)</option>
                </select>
                <span className="hint">One input plays in both ears.</span>
              </label>
            )}
            <label className="field">
              <span>Listen on</span>
              <select value={outputId} onChange={(e) => setOutputId(e.target.value)}>
                {outputs.map((n) => (
                  <option key={n.id} value={n.id}>
                    {n.name}
                    {n.subtitle ? ` (${n.subtitle})` : ''}
                  </option>
                ))}
              </select>
            </label>
            <p className="hint">
              It starts at {START_DB} dB, so it won't be loud. For the least delay, pick your interface's ASIO driver and a small buffer in Settings ›
              Audio.
            </p>
            <div className="field-row end">
              <button className="btn" onClick={close}>
                Cancel
              </button>
              <button className="btn primary" disabled={busy || !input || !output} onClick={() => void act(start())}>
                {busy ? 'Connecting…' : 'Start listening'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
