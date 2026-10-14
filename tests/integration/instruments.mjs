// VST3 instruments played by MIDI, against the running engine, with the ToneSphere Test Synth
// (build/x64-release/bin/test-plugins): the scan lists it as an instrument, it goes on a mix,
// notes sent like the on-screen keyboard sends them sound at the right level, MIDI CC 7 reaches
// the parameter the plugin maps it to, note off silences it, and effects that aren't
// instruments refuse notes clearly.
//   node tests/integration/instruments.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const pluginDir = path.join(root, 'build', 'x64-release', 'bin', 'test-plugins');
const TEST_SYNTH = '54535465737453796E74683100000001';
const TEST_GAIN = '4E78546F6E65546573744761696E3031';
const disc = JSON.parse(fs.readFileSync(path.join(process.env.LOCALAPPDATA ?? os.homedir(), 'ToneSphere', 'engine.json'), 'utf8'));
const ws = new WebSocket(`ws://127.0.0.1:${disc.port}/?token=${disc.token}`);
ws.binaryType = 'arraybuffer';
let state = null, meters = null, id = 1;
const pending = new Map();
ws.onmessage = (ev) => {
  if (ev.data instanceof ArrayBuffer) {
    meters = new Float32Array(ev.data, 12, new DataView(ev.data).getUint32(8, true)).slice();
    return;
  }
  const m = JSON.parse(ev.data);
  if (m.event === 'state') state = m.data;
  else if (pending.has(m.id)) {
    const p = pending.get(m.id);
    pending.delete(m.id);
    m.error ? p.reject(new Error(m.error.message)) : p.resolve(m.result);
  }
};
const call = (method, params = {}) =>
  new Promise((resolve, reject) => {
    const i = id++;
    pending.set(i, { resolve, reject });
    ws.send(JSON.stringify({ id: i, method, params }));
  });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];
const check = (name, ok, detail) => {
  results.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${JSON.stringify(detail)}`);
};

// Peak level of a node over a short window, in dBFS (meters hold peaks ~120 ms).
async function levelDb(nodeId, ms = 700) {
  await sleep(ms);
  const n = state.nodes.find((x) => x.id === nodeId);
  let peak = 0;
  for (let i = 0; i < 6; i++) {
    await sleep(60);
    peak = Math.max(peak, meters?.[n.meter.offset] ?? 0);
  }
  return 20 * Math.log10(peak || 1e-7);
}

async function main() {
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
  while (!state) await sleep(20);
  const prevFolders = state.settings.pluginFolders ?? [];
  await call('engine.setSettings', { pluginFolders: [pluginDir] });
  for (let i = 0; i < 100 && (state.pluginsScanning || !state.plugins.some((p) => p.id === TEST_SYNTH)); i++) await sleep(200);
  const synth = state.plugins.find((p) => p.id === TEST_SYNTH);
  check('scan lists the test synth as an instrument', !!synth?.instrument && synth.category === 'Instrument', { synth });

  const bus = await call('bus.create', { name: 'Synth', channels: 2 });
  const fx = await call('fx.add', { nodeId: bus.id, type: 'vst3', pluginId: TEST_SYNTH });
  await sleep(300);
  const info = state.nodes.find((n) => n.id === bus.id)?.fx?.find((f) => f.id === fx.id);
  check('the instrument goes on a mix and says it is one', !!info?.instrument, { fx: info?.name });

  const silent = await levelDb(bus.id);
  // A4 at full velocity: amplitude 0.5 (-6 dBFS).
  await call('fx.midi', { nodeId: bus.id, fxId: fx.id, events: [[0x90, 69, 127]] });
  const playing = await levelDb(bus.id);
  check('a note plays at the expected level', silent < -80 && Math.abs(playing + 6.02) < 0.5, { silent: +silent.toFixed(1), playing: +playing.toFixed(2) });

  // CC 7 (volume) at 64: the plugin maps it to its Volume parameter -> about -6 dB more.
  await call('fx.midi', { nodeId: bus.id, fxId: fx.id, events: [[0xb0, 7, 64]] });
  const quieter = await levelDb(bus.id, 1200);
  check('MIDI CC 7 reaches the parameter the plugin maps it to', Math.abs(quieter - playing - 20 * Math.log10(64 / 127)) < 0.6, { quieter: +quieter.toFixed(2) });

  // Note off: silence (after the peak meter's fall).
  await call('fx.midi', { nodeId: bus.id, fxId: fx.id, events: [[0x80, 69, 0]] });
  const off = await levelDb(bus.id, 2500);
  check('note off silences it', off < -60, { off: +off.toFixed(1) });

  // MIDI devices play the chosen instrument: the setting is stored and reported.
  await call('engine.setSettings', { midiInstrument: { nodeId: bus.id, fxId: fx.id } });
  await sleep(300);
  check('a MIDI keyboard can be pointed at the instrument', state.settings.midiInstrument?.fxId === fx.id, { midiInstrument: state.settings.midiInstrument });
  await call('engine.setSettings', { midiInstrument: null });

  // Effects that aren't instruments refuse notes with a clear message.
  const gain = await call('fx.add', { nodeId: bus.id, type: 'vst3', pluginId: TEST_GAIN });
  let msg = '';
  try {
    await call('fx.midi', { nodeId: bus.id, fxId: gain.id, events: [[0x90, 60, 100]] });
  } catch (e) {
    msg = e.message;
  }
  check('an effect that is not an instrument refuses notes', /isn't an instrument/.test(msg), { msg });

  await call('bus.delete', { id: bus.id });
  await call('engine.setSettings', { pluginFolders: prevFolders });
}

main()
  .catch((e) => check('harness', false, e.message))
  .finally(() => {
    const failed = results.filter((r) => !r.ok).length;
    console.log(`\n${results.length - failed}/${results.length} passed`);
    ws.close();
    process.exit(failed ? 1 : 0);
  });
