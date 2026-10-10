// Mock NexTone engine: speaks the real protocol with simulated devices and meters.
// Used for UI development and automated UI tests. Run: node mock-engine/server.mjs [--port 47999]
import { WebSocketServer } from 'ws';
import crypto from 'node:crypto';

const portArg = process.argv.indexOf('--port');
const PORT = portArg > 0 ? Number(process.argv[portArg + 1]) : 47999;
const TOKEN = process.env.NEXTONE_MOCK_TOKEN ?? 'mock';

const uid = () => crypto.randomBytes(4).toString('hex');
let layoutVersion = 1;

function node(id, bin, kind, name, channels, extra = {}) {
  return {
    id,
    bin,
    kind,
    name,
    channels,
    channelNames: channels === 2 ? ['L', 'R'] : channels === 1 ? ['Mono'] : Array.from({ length: channels }, (_, i) => `${i + 1}`),
    state: 'active',
    enabled: true,
    gainDb: 0,
    muted: false,
    solo: false,
    sampleRate: 48000,
    latencyMs: 3.2,
    driftPpm: 0,
    xruns: 0,
    cpuUs: 12,
    meter: { offset: 0, channels },
    fx: [],
    caps: { canDisable: true, canRename: false, canDelete: false, hasControlPanel: kind === 'asio', exclusiveMode: kind === 'wasapi' },
    settings: { exclusive: false, bufferMs: 10, safetyMs: 0, delayMs: 0, pan: 0, invert: false },
    ...extra,
  };
}

const state = {
  version: '0.1.0-mock',
  protocol: 1,
  layoutVersion,
  settings: {
    sampleRate: 48000,
    blockSize: 128,
    clockMaster: 'auto',
    compatMode: true,
    startWithSystem: true,
    startMinimized: true,
    hotkeys: { 'mute-mic': 'Ctrl+Shift+M' },
    replayBufferSeconds: 60,
    recordFolder: 'C:\\Users\\you\\Music\\NexTone',
    theme: 'system',
    advancedMode: false,
  pluginFolders: [],
  oscPort: 0,
  skin: 'fluent',
  customSkins: [],
  midiMappings: [],
  },
  status: {
    running: true,
    sampleRate: 48000,
    blockSize: 128,
    cpuPercent: 3,
    xruns: 0,
    latencyMs: 5.3,
    clockMaster: 'wasapi:out:speakers',
    platform: 'windows',
    driverPack: { vad: false, asio: false },
  },
  nodes: [
    node('wasapi:in:mic', 'physical-in', 'wasapi', 'Microphone Array', 2, { subtitle: 'Intel Smart Sound', isDefault: true }),
    node('wasapi:in:line', 'physical-in', 'wasapi', 'Line In', 2, { subtitle: 'Realtek HD Audio' }),
    node('asio:in:FlexASIO', 'physical-in', 'asio', 'FlexASIO Inputs', 2, { subtitle: 'ASIO' }),
    node('wasapi:out:speakers', 'physical-out', 'wasapi', 'Speakers', 2, { subtitle: 'Realtek HD Audio', isDefault: true }),
    node('wasapi:out:headphones', 'physical-out', 'wasapi', 'Headphones', 2, { subtitle: 'USB Headset' }),
    node('wasapi:out:hdmi', 'physical-out', 'wasapi', 'LG Monitor', 2, { subtitle: 'NVIDIA HDMI', state: 'unavailable', statusText: 'Disconnected' }),
    node('compat:out:cable', 'virtual-out', 'compat', 'CABLE Input', 2, { subtitle: 'VB-Audio Virtual Cable' }),
    node('compat:in:cable', 'virtual-in', 'compat', 'CABLE Output', 2, { subtitle: 'VB-Audio Virtual Cable' }),
    node('app:out:spotify.exe', 'app-out', 'app', 'Spotify', 2, {
      app: { exe: 'spotify.exe', pids: [1234], mode: 'copy', pinned: true, playing: true },
      caps: { canDisable: false, canRename: false, canDelete: false, hasControlPanel: false, canIsolate: true },
    }),
    node('app:out:chrome.exe', 'app-out', 'app', 'Google Chrome', 2, {
      state: 'idle',
      app: { exe: 'chrome.exe', pids: [4321], mode: 'copy', pinned: false, playing: false },
      caps: { canDisable: false, canRename: false, canDelete: false, hasControlPanel: false, canIsolate: true },
    }),
    node('app:in:discord.exe', 'app-in', 'app', 'Discord', 2, {
      app: { exe: 'discord.exe', pids: [999], mode: 'off', pinned: true, playing: true },
      caps: { canDisable: false, canRename: false, canDelete: false, hasControlPanel: false },
    }),
    node('bus:stream', 'bus', 'bus', 'Stream mix', 2, {
      caps: { canDisable: false, canRename: true, canDelete: true, hasControlPanel: false },
    }),
  ],
  routes: [],
  scenes: [{ id: 'sc1', name: 'Streaming', hotkey: 'Ctrl+1' }],
  activeScene: 'sc1',
  plugins: [{ id: 'vst3:demo', name: 'Demo EQ', vendor: 'Example', category: 'Fx|EQ', path: 'C:\\Program Files\\Common Files\\VST3\\Demo.vst3' }],
  recorder: { recording: false, files: [], folder: 'C:\\Users\\you\\Music\\NexTone', replaySeconds: 60 },
};

function relayout() {
  let off = 0;
  for (const n of state.nodes) {
    n.meter = { offset: off, channels: n.channels };
    off += n.channels * 2;
  }
  state.layoutVersion = ++layoutVersion;
}

function crosspointsFor(from, to, mode = 'auto') {
  const a = state.nodes.find((n) => n.id === from);
  const b = state.nodes.find((n) => n.id === to);
  const out = [];
  const nf = a.channels;
  const nt = b.channels;
  if (mode === 'clear') return out;
  if (mode === 'mono-sum') {
    for (let f = 0; f < nf; f++) for (let t = 0; t < nt; t++) out.push({ f, t, g: nf > 1 ? -3 : 0 });
  } else if (mode === 'spread' || (mode === 'auto' && nf === 1)) {
    for (let f = 0; f < nf; f++) for (let t = 0; t < nt; t++) if (nf === 1 || f % nt === t) out.push({ f, t, g: 0 });
  } else {
    for (let i = 0; i < Math.min(nf, nt); i++) out.push({ f: i, t: i, g: 0 });
  }
  return out;
}

function connect(from, to, mode) {
  const r = { id: `r_${uid()}`, from, to, gainDb: 0, muted: false, enabled: true, crosspoints: crosspointsFor(from, to, mode) };
  state.routes.push(r);
  return r;
}

relayout();
connect('wasapi:in:mic', 'bus:stream');
connect('app:out:spotify.exe', 'bus:stream');
connect('app:out:spotify.exe', 'wasapi:out:speakers');
connect('bus:stream', 'compat:in:cable');
connect('wasapi:in:mic', 'app:in:discord.exe');

const FX_DEFS = {
  gate: ['Noise gate', [['threshold', 'Threshold', -80, 0, -45, 'dB'], ['attack', 'Attack', 0.1, 50, 2, 'ms', 'log'], ['release', 'Release', 5, 1000, 120, 'ms', 'log']]],
  denoise: ['Noise suppression', [['amount', 'Amount', 0, 100, 100, '%']]],
  eq: ['Equalizer', [['low', 'Bass', -15, 15, 0, 'dB'], ['mid', 'Mids', -15, 15, 0, 'dB'], ['midFreq', 'Mid freq', 200, 5000, 1000, 'Hz', 'log'], ['high', 'Treble', -15, 15, 0, 'dB'], ['hpf', 'Low cut', 20, 400, 20, 'Hz', 'log']]],
  compressor: ['Compressor', [['threshold', 'Threshold', -60, 0, -18, 'dB'], ['ratio', 'Ratio', 1, 20, 3, ':1'], ['attack', 'Attack', 0.1, 100, 10, 'ms', 'log'], ['release', 'Release', 10, 1000, 120, 'ms', 'log'], ['makeup', 'Make-up', 0, 24, 0, 'dB']]],
  limiter: ['Limiter', [['ceiling', 'Ceiling', -12, 0, -1, 'dB'], ['release', 'Release', 10, 500, 60, 'ms', 'log']]],
  ducker: ['Auto-duck', [['amount', 'Reduce by', 0, 40, 12, 'dB'], ['threshold', 'Trigger level', -60, 0, -35, 'dB'], ['release', 'Release', 50, 2000, 400, 'ms', 'log']]],
  delay: ['Delay', [['time', 'Time', 0, 2000, 0, 'ms']]],
};

function makeFx(type, pluginId) {
  if (type === 'vst3') {
    const p = state.plugins.find((x) => x.id === pluginId);
    return { id: `fx_${uid()}`, type, name: p?.name ?? 'Plugin', bypass: false, params: [], pluginId, hasEditor: true };
  }
  const [name, params] = FX_DEFS[type];
  return {
    id: `fx_${uid()}`,
    type,
    name,
    bypass: false,
    sidechain: type === 'ducker' ? null : undefined,
    params: params.map(([id, pname, min, max, value, unit, scale]) => ({ id, name: pname, min, max, value, unit, scale })),
  };
}

const sceneSnapshots = new Map();

const handlers = {
  'engine.getState': () => state,
  'engine.setSettings': (p) => {
    Object.assign(state.settings, p);
    state.status.sampleRate = state.settings.sampleRate;
    state.status.blockSize = state.settings.blockSize;
    state.status.latencyMs = (state.settings.blockSize * 2 / state.settings.sampleRate) * 1000 + 2.6;
    return state.settings;
  },
  'engine.restart': () => {},
  'engine.shutdown': () => {},
  'devices.refresh': () => {},
  'node.update': (p) => {
    const n = mustNode(p.id);
    for (const k of ['gainDb', 'muted', 'solo', 'enabled', 'name']) if (p[k] !== undefined) n[k] = p[k];
    if (p.settings) Object.assign(n.settings, p.settings);
  },
  'node.openControlPanel': () => {},
  'route.connect': (p) => {
    mustNode(p.from);
    mustNode(p.to);
    if (p.from === p.to) throw new Error('A device can’t be connected to itself');
    const existing = state.routes.find((r) => r.from === p.from && r.to === p.to);
    if (existing) return existing;
    return connect(p.from, p.to, p.mode);
  },
  'route.disconnect': (p) => {
    state.routes = state.routes.filter((r) => r.id !== p.id);
  },
  'route.update': (p) => {
    const r = mustRoute(p.id);
    for (const k of ['gainDb', 'muted', 'enabled']) if (p[k] !== undefined) r[k] = p[k];
  },
  'route.setCrosspoint': (p) => {
    const r = mustRoute(p.id);
    r.crosspoints = r.crosspoints.filter((c) => !(c.f === p.fromCh && c.t === p.toCh));
    if (p.gainDb !== null) r.crosspoints.push({ f: p.fromCh, t: p.toCh, g: p.gainDb });
  },
  'route.applyPreset': (p) => {
    const r = mustRoute(p.id);
    r.crosspoints = crosspointsFor(r.from, r.to, p.mode === 'one-to-one' ? 'one-to-one' : p.mode);
  },
  'bus.create': (p) => {
    const n = node(`bus:${uid()}`, 'bus', 'bus', p.name, p.channels, {
      caps: { canDisable: false, canRename: true, canDelete: true, hasControlPanel: false },
    });
    state.nodes.push(n);
    relayout();
    return n;
  },
  'bus.delete': (p) => {
    state.nodes = state.nodes.filter((n) => n.id !== p.id);
    state.routes = state.routes.filter((r) => r.from !== p.id && r.to !== p.id);
    relayout();
  },
  'fx.add': (p) => {
    const n = mustNode(p.nodeId);
    const fx = makeFx(p.type, p.pluginId);
    n.fx.splice(p.index ?? n.fx.length, 0, fx);
    return fx;
  },
  'fx.remove': (p) => {
    const n = mustNode(p.nodeId);
    n.fx = n.fx.filter((f) => f.id !== p.fxId);
  },
  'fx.update': (p) => {
    const fx = mustNode(p.nodeId).fx.find((f) => f.id === p.fxId);
    if (!fx) throw new Error('Effect not found');
    if (p.bypass !== undefined) fx.bypass = p.bypass;
    if (p.sidechain !== undefined) fx.sidechain = p.sidechain;
    if (p.params) for (const [k, v] of Object.entries(p.params)) {
      const prm = fx.params.find((x) => x.id === k);
      if (prm) prm.value = Math.min(prm.max, Math.max(prm.min, v));
    }
  },
  'fx.move': (p) => {
    const n = mustNode(p.nodeId);
    const i = n.fx.findIndex((f) => f.id === p.fxId);
    const [fx] = n.fx.splice(i, 1);
    n.fx.splice(Math.max(0, Math.min(n.fx.length, p.index)), 0, fx);
  },
  'fx.openEditor': () => {},
  'plugins.scan': () => {},
  'app.setMode': (p) => {
    for (const n of state.nodes) if (n.app?.exe === p.exe && n.bin === 'app-out') n.app.mode = p.mode;
  },
  'app.setRule': (p) => {
    for (const n of state.nodes) if (n.app?.exe === p.exe && p.pinned !== undefined) n.app.pinned = p.pinned;
  },
  'vban.add': (p) => {
    const n = node(`vban:${p.direction}:${uid()}`, p.direction === 'in' ? 'virtual-out' : 'virtual-in', 'vban', p.name, p.channels, {
      subtitle: `${p.streamName} · ${p.host}:${p.port}`,
      caps: { canDisable: true, canRename: true, canDelete: true, hasControlPanel: false },
      settings: { host: p.host, port: p.port, streamName: p.streamName, safetyMs: 20 },
    });
    state.nodes.push(n);
    relayout();
    return n;
  },
  'vban.remove': (p) => handlers['bus.delete'](p),
  'scene.save': (p) => {
    const id = p.id ?? `sc_${uid()}`;
    sceneSnapshots.set(id, JSON.stringify({ routes: state.routes }));
    let s = state.scenes.find((x) => x.id === id);
    if (!s) state.scenes.push((s = { id, name: p.name }));
    s.name = p.name;
    state.activeScene = id;
    return s;
  },
  'scene.load': (p) => {
    const snap = sceneSnapshots.get(p.id);
    if (snap) state.routes = JSON.parse(snap).routes;
    state.activeScene = p.id;
  },
  'scene.delete': (p) => {
    state.scenes = state.scenes.filter((s) => s.id !== p.id);
    if (state.activeScene === p.id) state.activeScene = undefined;
  },
  'recorder.start': () => {
    state.recorder.recording = true;
    state.recorder.startedAt = Date.now();
  },
  'recorder.stop': () => {
    state.recorder.recording = false;
    const f = `${state.recorder.folder}\\NexTone-${new Date().toISOString().replace(/[:.]/g, '-')}.wav`;
    state.recorder.files.push(f);
    return { files: [f] };
  },
  'recorder.saveReplay': () => ({ file: `${state.recorder.folder}\\Replay.wav` }),
  'tone.set': () => {},
  'midi.devices': () => [],
  'node.forget': ({ id }) => { state.nodes = state.nodes.filter((n) => n.id !== id); state.routes = state.routes.filter((r) => r.from !== id && r.to !== id); },
  'preset.import': ({ preset }) => ({ sceneId: 'sc_mock', name: preset?.name ?? 'Preset', connections: (preset?.routes ?? []).length, waiting: [], skipped: 0 }),
  'preset.export': ({ name }) => ({ format: 'nextone-preset', version: 1, name, devices: [], mixes: [], routes: [] }),
  'midi.learn': () => { throw new Error('No MIDI control moved.'); },
  'latency.measure': () => ({ latencyMs: 7.4, samples: 355.2, sampleRate: 48000, confidence: 0.98 }),
};

function mustNode(id) {
  const n = state.nodes.find((x) => x.id === id);
  if (!n) throw new Error(`Unknown device or app: ${id}`);
  return n;
}
function mustRoute(id) {
  const r = state.routes.find((x) => x.id === id);
  if (!r) throw new Error('Connection not found');
  return r;
}

const wss = new WebSocketServer({ host: '127.0.0.1', port: PORT });
const clients = new Set();

wss.on('connection', (ws, req) => {
  const url = new URL(req.url, 'http://x');
  if (url.searchParams.get('token') !== TOKEN) {
    ws.close(1008, 'bad token');
    return;
  }
  clients.add(ws);
  ws.send(JSON.stringify({ event: 'state', data: state }));
  ws.on('message', (raw) => {
    let msg;
    try {
      msg = JSON.parse(String(raw));
    } catch {
      return;
    }
    const h = handlers[msg.method];
    try {
      if (!h) throw Object.assign(new Error(`Unknown method ${msg.method}`), { code: -32601 });
      const result = h(msg.params ?? {});
      ws.send(JSON.stringify({ id: msg.id, result: result ?? null }));
      broadcastState();
    } catch (e) {
      ws.send(JSON.stringify({ id: msg.id, error: { code: e.code ?? -32000, message: e.message } }));
    }
  });
  ws.on('close', () => clients.delete(ws));
});

let stateTimer = null;
function broadcastState() {
  if (stateTimer) return;
  stateTimer = setTimeout(() => {
    stateTimer = null;
    const s = JSON.stringify({ event: 'state', data: state });
    for (const c of clients) c.send(s);
  }, 30);
}

// Simulated meters: sources generate signal, destinations sum their inputs.
let t = 0;
setInterval(() => {
  t += 1 / 30;
  const levels = new Map();
  for (const n of state.nodes) {
    let base = 0;
    if (n.id.startsWith('wasapi:in:mic')) base = 0.15 + 0.12 * Math.abs(Math.sin(t * 3.1)) * (Math.sin(t * 0.7) > 0 ? 1 : 0.1);
    else if (n.kind === 'app' && n.bin === 'app-out' && n.app?.playing) base = 0.35 + 0.2 * Math.abs(Math.sin(t * 5.3));
    else if (n.kind === 'asio' && n.bin === 'physical-in') base = 0.05 * Math.random();
    if (n.muted || !n.enabled || n.state !== 'active') base = 0;
    levels.set(n.id, base * Math.pow(10, n.gainDb / 20));
  }
  for (let pass = 0; pass < 2; pass++) {
    for (const r of state.routes) {
      if (r.muted || !r.enabled) continue;
      const src = levels.get(r.from) ?? 0;
      const dst = state.nodes.find((n) => n.id === r.to);
      if (!dst || dst.muted) continue;
      const add = src * Math.pow(10, r.gainDb / 20) * Math.pow(10, dst.gainDb / 20);
      levels.set(r.to, Math.max(levels.get(r.to) ?? 0, add));
    }
  }
  const count = state.nodes.reduce((a, n) => a + n.channels * 2, 0);
  const buf = Buffer.alloc(12 + count * 4);
  buf.writeUInt32LE(0x314d544e, 0);
  buf.writeUInt32LE(state.layoutVersion, 4);
  buf.writeUInt32LE(count, 8);
  for (const n of state.nodes) {
    const l = levels.get(n.id) ?? 0;
    for (let c = 0; c < n.channels; c++) {
      const v = l * (0.85 + 0.15 * Math.random());
      buf.writeFloatLE(v, 12 + (n.meter.offset + c * 2) * 4);
      buf.writeFloatLE(v * 0.6, 12 + (n.meter.offset + c * 2 + 1) * 4);
    }
  }
  for (const c of clients) c.send(buf);
}, 33);

console.log(`NexTone mock engine on ws://127.0.0.1:${PORT}/?token=${TOKEN}`);
