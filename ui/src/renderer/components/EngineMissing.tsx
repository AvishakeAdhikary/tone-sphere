import { Loader2, PlugZap } from 'lucide-react';
import type { ConnectionStatus } from '../engine/client';
import { engine } from '../store';

export function EngineMissing({ status }: { status: ConnectionStatus }) {
  const busy = status === 'connecting' || status === 'connected';
  return (
    <div className="engine-missing">
      <div className="engine-missing-card">
        {busy ? <Loader2 className="spin" size={32} /> : <PlugZap size={32} />}
        <h2>{busy ? 'Starting the audio engine…' : status === 'disconnected' ? 'Reconnecting to the audio engine…' : 'The audio engine isn’t running'}</h2>
        <p className="muted">
          {busy
            ? 'This only takes a moment.'
            : 'ToneSphere keeps working in the background. If this message stays, the engine may have been closed or failed to start.'}
        </p>
        {!busy && (
          <button className="btn primary" onClick={() => void engine.connect()}>
            Try again
          </button>
        )}
      </div>
    </div>
  );
}
