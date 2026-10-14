// WebSocket client for the ToneSphere engine: typed RPC, state events, binary meters.
import {
  METER_MAGIC,
  type EngineDiscovery,
  type EngineState,
  type RpcMethods,
  type RpcResponse,
} from '@shared/protocol';

export type ConnectionStatus = 'connecting' | 'connected' | 'no-engine' | 'disconnected';

type Pending = { resolve: (v: unknown) => void; reject: (e: Error) => void; timer: number };

/** Meter values live outside React; components read them in a shared rAF loop. */
export const meters = {
  layoutVersion: -1,
  values: new Float32Array(0),
  /** Smoothed/held peaks for display, keyed by value index. */
  display: new Float32Array(0),
  lastFrameAt: 0,
};

export class EngineClient {
  private ws: WebSocket | null = null;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private retryTimer: number | null = null;
  private closedByUser = false;

  onState: (s: EngineState) => void = () => {};
  onStatus: (s: ConnectionStatus) => void = () => {};
  onLog: (level: string, message: string) => void = () => {};

  constructor(private discover: () => Promise<EngineDiscovery | null>) {}

  async connect(): Promise<void> {
    this.closedByUser = false;
    this.onStatus('connecting');
    let d: EngineDiscovery | null = null;
    try {
      d = await this.discover();
    } catch {
      d = null;
    }
    if (!d) {
      this.onStatus('no-engine');
      this.scheduleRetry(2000);
      return;
    }
    const ws = new WebSocket(`ws://127.0.0.1:${d.port}/?token=${encodeURIComponent(d.token)}`);
    ws.binaryType = 'arraybuffer';
    this.ws = ws;
    ws.onopen = () => this.onStatus('connected');
    ws.onmessage = (ev) => this.handleMessage(ev.data);
    ws.onclose = () => {
      this.ws = null;
      for (const [, p] of this.pending) {
        clearTimeout(p.timer);
        p.reject(new Error('Engine disconnected'));
      }
      this.pending.clear();
      if (!this.closedByUser) {
        this.onStatus('disconnected');
        this.scheduleRetry(1000);
      }
    };
    ws.onerror = () => ws.close();
  }

  close(): void {
    this.closedByUser = true;
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.ws?.close();
  }

  private scheduleRetry(ms: number): void {
    if (this.retryTimer) clearTimeout(this.retryTimer);
    this.retryTimer = window.setTimeout(() => void this.connect(), ms);
  }

  private handleMessage(data: unknown): void {
    if (data instanceof ArrayBuffer) {
      this.handleMeters(data);
      return;
    }
    let msg: RpcResponse & { event?: string; data?: unknown };
    try {
      msg = JSON.parse(String(data));
    } catch {
      return;
    }
    if (msg.event === 'state') {
      this.onState(msg.data as EngineState);
    } else if (msg.event === 'log') {
      const d = msg.data as { level: string; message: string };
      this.onLog(d.level, d.message);
    } else if (typeof msg.id === 'number') {
      const p = this.pending.get(msg.id);
      if (!p) return;
      this.pending.delete(msg.id);
      clearTimeout(p.timer);
      if (msg.error) p.reject(new Error(msg.error.message));
      else p.resolve(msg.result);
    }
  }

  private handleMeters(buf: ArrayBuffer): void {
    if (buf.byteLength < 12) return;
    const dv = new DataView(buf);
    if (dv.getUint32(0, true) !== METER_MAGIC) return;
    const layout = dv.getUint32(4, true);
    const count = dv.getUint32(8, true);
    if (buf.byteLength < 12 + count * 4) return;
    if (meters.values.length !== count) {
      meters.values = new Float32Array(count);
      meters.display = new Float32Array(count);
    }
    meters.values.set(new Float32Array(buf, 12, count));
    meters.layoutVersion = layout;
    meters.lastFrameAt = performance.now();
  }

  call<M extends keyof RpcMethods>(
    method: M,
    ...args: RpcMethods[M]['params'] extends void ? [] : [RpcMethods[M]['params']]
  ): Promise<RpcMethods[M]['result']> {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return Promise.reject(new Error('Not connected to the ToneSphere engine'));
    }
    const id = this.nextId++;
    const params = args[0];
    return new Promise((resolve, reject) => {
      const timer = window.setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Engine did not answer "${method}" in time`));
      }, 10000);
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, timer });
      ws.send(JSON.stringify({ id, method, params }));
    });
  }
}
