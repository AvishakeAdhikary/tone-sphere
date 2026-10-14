// ToneSphere engine <-> UI protocol types (v1). See shared/protocol/PROTOCOL.md.
// The C++ engine serialises exactly these shapes (engine/src/ipc/state_json.cpp).

export const PROTOCOL_VERSION = 1;
export const METER_MAGIC = 0x314d5354; // "TSM1"

/** The six user-facing bins, plus internal mix buses. */
export type Bin =
  | 'physical-in'
  | 'physical-out'
  | 'virtual-in'
  | 'virtual-out'
  | 'app-in'
  | 'app-out'
  | 'bus';

export type NodeKind =
  | 'wasapi'
  | 'asio'
  | 'vad'
  | 'compat'
  | 'app'
  | 'asioclient'
  | 'bus'
  | 'vban'
  | 'tone'
  | 'pipewire'
  | 'coreaudio';

export type NodeState = 'active' | 'idle' | 'unavailable' | 'error';

/** Sources feed audio into ToneSphere; destinations receive it. Buses are both. */
export const SOURCE_BINS: Bin[] = ['physical-in', 'virtual-out', 'app-out'];
export const DEST_BINS: Bin[] = ['physical-out', 'virtual-in', 'app-in'];

export function isSource(bin: Bin): boolean {
  return bin === 'bus' || SOURCE_BINS.includes(bin);
}
export function isDestination(bin: Bin): boolean {
  return bin === 'bus' || DEST_BINS.includes(bin);
}

export interface MeterSlot {
  offset: number;
  channels: number;
}

export type FxType =
  | 'gate'
  | 'denoise'
  | 'eq'
  | 'compressor'
  | 'limiter'
  | 'ducker'
  | 'delay'
  | 'vst3';

export interface FxParamInfo {
  id: string;
  name: string;
  min: number;
  max: number;
  value: number;
  unit?: string;
  /** 'log' parameters use a logarithmic knob/slider mapping. */
  scale?: 'lin' | 'log';
  /** Discrete choices for enumerated parameters. */
  choices?: string[];
}

export interface FxInfo {
  id: string;
  type: FxType;
  name: string;
  bypass: boolean;
  params: FxParamInfo[];
  /** Ducker sidechain source node id. */
  sidechain?: string | null;
  pluginId?: string;
  hasEditor?: boolean;
  /** Gain reduction in dB (dynamics processors), updated with state. */
  reductionDb?: number;
  /** A problem with this effect (plugin missing or stopped), shown to the user. */
  error?: string;
  /** A VST3 instrument (has a MIDI input): played with fx.midi or a MIDI device. */
  instrument?: boolean;
}

export interface NodeCaps {
  canDisable: boolean;
  canRename: boolean;
  canDelete: boolean;
  hasControlPanel: boolean;
  /** Application nodes: can be isolated (redirected to a private virtual output). */
  canIsolate?: boolean;
  exclusiveMode?: boolean;
}

export interface AppInfo {
  exe: string;
  pids: number[];
  /** Output capture mode for app-out nodes. */
  mode: 'off' | 'copy' | 'isolate';
  /** Persistent rule: app automatically appears/captures when it starts. */
  pinned: boolean;
  /** Whether the app currently has an active audio session. */
  playing: boolean;
}

export interface NodeSettings {
  /** WASAPI exclusive mode (physical devices). */
  exclusive?: boolean;
  /** Extra safety buffer in ms (virtual + network devices). */
  safetyMs?: number;
  /** Device buffer request in ms (WASAPI) / samples (ASIO). */
  bufferMs?: number;
  /** Network: remote IP / port / stream name. */
  host?: string;
  port?: number;
  streamName?: string;
  /** Delay compensation in ms applied to this node's signal. */
  delayMs?: number;
  /** Pan (-1..1) for stereo destinations of mono sources (balance for stereo). */
  pan?: number;
  /** Invert polarity per node. */
  invert?: boolean;
  channelNames?: string[];
}

export interface NodeInfo {
  id: string;
  bin: Bin;
  kind: NodeKind;
  name: string;
  subtitle?: string;
  /** PNG data URL for app icons. */
  icon?: string;
  channels: number;
  channelNames: string[];
  state: NodeState;
  statusText?: string;
  /** Not shown in lists (e.g. an app's plain card while it uses the ToneSphere ASIO driver). */
  hidden?: boolean;
  enabled: boolean;
  gainDb: number;
  muted: boolean;
  solo: boolean;
  isDefault?: boolean;
  isDefaultComms?: boolean;
  sampleRate?: number;
  latencyMs?: number;
  driftPpm?: number;
  xruns?: number;
  cpuUs?: number;
  meter: MeterSlot;
  fx: FxInfo[];
  caps: NodeCaps;
  settings: NodeSettings;
  app?: AppInfo;
}

export interface Crosspoint {
  /** source channel */
  f: number;
  /** destination channel */
  t: number;
  /** gain in dB */
  g: number;
}

export interface RouteInfo {
  id: string;
  from: string;
  to: string;
  gainDb: number;
  muted: boolean;
  enabled: boolean;
  crosspoints: Crosspoint[];
}

export interface SceneInfo {
  id: string;
  name: string;
  hotkey?: string;
}

export interface MidiMapping {
  type: 'cc' | 'note';
  /** 1..16, 0 = any channel. */
  channel: number;
  number: number;
  action: 'volume' | 'gain' | 'mute' | 'solo' | 'scene' | 'record' | 'replay';
  /** Node id (volume/gain/mute/solo) or scene id. */
  target: string;
}

export interface PluginInfo {
  id: string;
  name: string;
  vendor: string;
  category: string;
  path: string;
  /** Instruments (synths, samplers) go on a mix and are played by MIDI. */
  instrument?: boolean;
}

export interface RecorderState {
  recording: boolean;
  startedAt?: number;
  files: string[];
  folder: string;
  replaySeconds: number;
}

export interface EngineSettings {
  sampleRate: number;
  blockSize: number;
  /** Node id of the clock master, or 'auto' / 'internal'. */
  clockMaster: string;
  /** Use already-installed signed virtual cables when the ToneSphere driver is missing. */
  compatMode: boolean;
  startWithSystem: boolean;
  startMinimized: boolean;
  /** Global hotkeys: action -> accelerator. */
  hotkeys: Record<string, string>;
  replayBufferSeconds: number;
  recordFolder: string;
  theme: 'system' | 'light' | 'dark';
  /** Show advanced controls by default. */
  advancedMode: boolean;
  /** Extra folders searched for VST3 plugins (besides the standard VST3 folders). */
  pluginFolders?: string[];
  /** OSC control over UDP (0 = off). */
  oscPort?: number;
  /** MIDI controller mappings. */
  midiMappings?: MidiMapping[];
  /** UI skin id (built-in or imported). */
  skin?: string;
  /** Imported skins (see ui/src/renderer/skins.ts). */
  customSkins?: Array<Record<string, unknown>>;
  /** DSP cores: 0 = auto (several cores only when the graph is heavy), 1 = one core, N = N cores. */
  dspThreads?: number;
  /** The instrument that connected MIDI devices play, or null. */
  midiInstrument?: { nodeId: string; fxId: string } | null;
  /** Interface language (UI only): 'system' follows Windows. */
  language?: 'system' | 'en' | 'hi';
}

export interface EngineStatus {
  running: boolean;
  sampleRate: number;
  blockSize: number;
  cpuPercent: number;
  /** Cores the audio graph can use, and how many the last block used. */
  dspCores?: number;
  dspThreadsInUse?: number;
  xruns: number;
  latencyMs: number;
  clockMaster: string;
  message?: string;
  platform: 'windows' | 'linux' | 'macos';
  /** Installed drivers: ToneSphere virtual audio, ToneSphere ASIO, VB-CABLE (VB-Audio's virtual cable). */
  driverPack: { vad: boolean; asio: boolean; vbcable?: boolean };
  testMode?: boolean;
}

export interface EngineState {
  version: string;
  protocol: number;
  layoutVersion: number;
  settings: EngineSettings;
  status: EngineStatus;
  nodes: NodeInfo[];
  routes: RouteInfo[];
  scenes: SceneInfo[];
  activeScene?: string;
  plugins: PluginInfo[];
  /** A plugin scan is running. */
  pluginsScanning?: boolean;
  recorder: RecorderState;
}

export interface VbanStreamConfig {
  direction: 'in' | 'out';
  name: string;
  streamName: string;
  host: string;
  port: number;
  channels: number;
}

export type ConnectMode = 'auto' | 'one-to-one' | 'mono-sum' | 'spread';

/** Request method -> params/result typing. */
export interface RpcMethods {
  'engine.getState': { params: void; result: EngineState };
  'engine.setSettings': { params: Partial<EngineSettings>; result: EngineSettings };
  'engine.restart': { params: void; result: void };
  'engine.shutdown': { params: void; result: void };
  'devices.refresh': { params: void; result: void };
  'node.update': {
    params: {
      id: string;
      gainDb?: number;
      muted?: boolean;
      solo?: boolean;
      enabled?: boolean;
      name?: string;
      settings?: Partial<NodeSettings>;
    };
    result: void;
  };
  'node.openControlPanel': { params: { id: string }; result: void };
  'route.connect': { params: { from: string; to: string; mode?: ConnectMode }; result: RouteInfo };
  'route.disconnect': { params: { id: string }; result: void };
  'route.update': {
    params: { id: string; gainDb?: number; muted?: boolean; enabled?: boolean };
    result: void;
  };
  'route.setCrosspoint': {
    params: { id: string; fromCh: number; toCh: number; gainDb: number | null };
    result: void;
  };
  'route.applyPreset': {
    params: { id: string; mode: ConnectMode | 'clear' };
    result: void;
  };
  'bus.create': { params: { name: string; channels: number }; result: NodeInfo };
  'bus.delete': { params: { id: string }; result: void };
  'fx.add': {
    params: { nodeId: string; type: FxType; pluginId?: string; index?: number };
    result: FxInfo;
  };
  'fx.remove': { params: { nodeId: string; fxId: string }; result: void };
  'fx.update': {
    params: {
      nodeId: string;
      fxId: string;
      bypass?: boolean;
      params?: Record<string, number>;
      sidechain?: string | null;
    };
    result: void;
  };
  'fx.move': { params: { nodeId: string; fxId: string; index: number }; result: void };
  'fx.openEditor': { params: { nodeId: string; fxId: string }; result: void };
  /** MIDI short messages [status, data1, data2] for an instrument (notes, CCs, pitch bend). */
  'fx.midi': { params: { nodeId: string; fxId: string; events: number[][] }; result: void };
  /** This connection wants binary meter frames (default) or not (hidden window). */
  'meters.subscribe': { params: { on: boolean }; result: void };
  'plugins.scan': { params: void; result: void };
  'app.setMode': { params: { exe: string; mode: 'off' | 'copy' | 'isolate' }; result: void };
  'app.setRule': { params: { exe: string; pinned?: boolean }; result: void };
  'vban.add': { params: VbanStreamConfig; result: NodeInfo };
  'vban.remove': { params: { id: string }; result: void };
  'scene.save': { params: { name: string; id?: string }; result: SceneInfo };
  'scene.load': { params: { id: string }; result: void };
  /** Waits for the next MIDI knob/fader/pad move (MIDI learn). */
  'midi.learn': { params: { seconds?: number }; result: { type: 'cc' | 'note'; channel: number; number: number; value: number } };
  'midi.devices': { params: void; result: string[] };
  /** Removes a disconnected device together with its connections. */
  'node.forget': { params: { id: string }; result: void };
  /** Applies a portable preset (see engine/src/core/presets.h) and keeps it as a scene. */
  'preset.import': { params: { preset: unknown }; result: { sceneId: string; name: string; connections: number; waiting: string[]; skipped: number } };
  /** The current routing as a portable preset. */
  'preset.export': { params: { name: string; description?: string }; result: Record<string, unknown> };
  'scene.delete': { params: { id: string }; result: void };
  'recorder.start': {
    params: { nodeIds: string[]; folder?: string; format: 'wav' | 'flac' };
    result: void;
  };
  'recorder.stop': { params: void; result: { files: string[] } };
  'recorder.saveReplay': { params: { seconds: number }; result: { file: string } };
  'tone.set': {
    params: { enabled: boolean; freq: number; levelDb: number; target?: string };
    result: void;
  };
  /** Plays a short chirp into output and finds it in input (round trip through a cable/loopback). */
  'latency.measure': { params: { output: string; input: string }; result: { latencyMs: number; samples: number; sampleRate: number; confidence: number } };
}

export type RpcMethod = keyof RpcMethods;

export interface RpcRequest {
  id: number;
  method: string;
  params?: unknown;
}
export interface RpcResponse {
  id: number;
  result?: unknown;
  error?: { code: number; message: string };
}
export interface RpcEvent {
  event: 'state' | 'log';
  data: unknown;
}

export interface EngineDiscovery {
  port: number;
  token: string;
  pid: number;
  version: string;
  protocol: number;
}

/** Linear gain -> dBFS, clamped. */
export function linToDb(v: number): number {
  return v <= 1e-6 ? -120 : 20 * Math.log10(v);
}
export function dbToLin(db: number): number {
  return db <= -120 ? 0 : Math.pow(10, db / 20);
}
