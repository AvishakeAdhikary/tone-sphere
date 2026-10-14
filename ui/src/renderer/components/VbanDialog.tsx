import { useState } from 'react';
import { X } from 'lucide-react';
import { act, engine } from '../store';

export function VbanDialog({ direction, onClose }: { direction: 'in' | 'out'; onClose: () => void }) {
  const [name, setName] = useState(direction === 'in' ? 'Network in' : 'Network out');
  const [streamName, setStreamName] = useState('Stream1');
  const [host, setHost] = useState(direction === 'in' ? '0.0.0.0' : '192.168.1.');
  const [port, setPort] = useState(6980);
  const [channels, setChannels] = useState(2);

  const submit = async () => {
    const r = await act(engine.call('vban.add', { direction, name, streamName, host, port, channels }));
    if (r) onClose();
  };

  return (
    <div className="modal-backdrop" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <form
        className="modal small-modal"
        role="dialog"
        aria-modal="true"
        aria-label={direction === 'in' ? 'Receive network audio' : 'Send network audio'}
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <div className="modal-head">
          <h2>{direction === 'in' ? 'Receive audio from the network' : 'Send audio to the network'}</h2>
          <button type="button" className="icon-btn" onClick={onClose} aria-label="Close">
            <X size={16} />
          </button>
        </div>
        <p className="muted small">
          Uses the VBAN protocol, compatible with VoiceMeeter, VBAN apps and other ToneSphere computers on your network.
        </p>
        <label className="field">
          <span>Name in ToneSphere</span>
          <input value={name} onChange={(e) => setName(e.target.value)} required />
        </label>
        <label className="field">
          <span>VBAN stream name</span>
          <input value={streamName} maxLength={16} onChange={(e) => setStreamName(e.target.value)} required />
          <span className="hint">Must match on both computers (up to 16 characters).</span>
        </label>
        <div className="field-row">
          <label className="field grow">
            <span>{direction === 'in' ? 'Accept from IP' : 'Destination IP'}</span>
            <input value={host} onChange={(e) => setHost(e.target.value)} required />
            {direction === 'in' && <span className="hint">0.0.0.0 accepts the stream from any computer.</span>}
          </label>
          <label className="field">
            <span>Port</span>
            <input type="number" min={1} max={65535} value={port} onChange={(e) => setPort(Number(e.target.value))} />
          </label>
        </div>
        <label className="field">
          <span>Channels</span>
          <select value={channels} onChange={(e) => setChannels(Number(e.target.value))}>
            {[1, 2, 4, 6, 8, 16].map((c) => (
              <option key={c} value={c}>
                {c === 1 ? 'Mono' : c === 2 ? 'Stereo' : `${c} channels`}
              </option>
            ))}
          </select>
        </label>
        <div className="modal-actions">
          <button type="button" className="btn" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="btn primary">
            Add stream
          </button>
        </div>
      </form>
    </div>
  );
}
