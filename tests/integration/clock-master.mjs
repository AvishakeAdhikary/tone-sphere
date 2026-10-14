// Device clock master against the running engine: the engine paces itself by the chosen device's
// own position. FlexASIO (an ASIO device, so hardware-clocked from the engine's view) plays into
// VB-CABLE and audio-probe records the cable: the master is reported, audio through it is clean,
// a shared-mode device can't lead (it keeps the internal timer), and when the master goes away
// the engine carries on, on its own timer.
//   node tests/integration/clock-master.mjs [seconds]
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const probe = path.join(root, 'build', 'x64-release', 'bin', 'audio-probe.exe');
const SECS = Number(process.argv[2] ?? 8);
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
const node = (pred) => state.nodes.find(pred);
const peakDb = (n) => {
  const v = meters?.[n.meter.offset] ?? 0;
  return v > 1e-7 ? 20 * Math.log10(v) : -140;
};
async function clear() {
  for (const r of [...state.routes]) await call('route.disconnect', { id: r.id }).catch(() => {});
  await call('tone.set', { enabled: false });
}

async function main() {
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
  while (!state) await sleep(20);
  const original = state.settings.clockMaster;
  const cable = node((n) => n.kind === 'compat' && n.bin === 'virtual-in');
  const asio = node((n) => n.id === 'asio:out:FlexASIO');
  if (!cable || !asio) return check('VB-CABLE and FlexASIO present', false, { cable: !!cable, flexasio: !!asio });
  await clear();
  try {
    // 0) A shared-mode device can't lead: its stream isn't paced by the hardware clock.
    await call('engine.setSettings', { clockMaster: cable.id });
    await sleep(1500);
    await call('tone.set', { enabled: true, freq: 1000, levelDb: -12, target: cable.id });
    await sleep(2000);
    check('a shared-mode device keeps the internal timer', state.status.clockMaster === 'internal', { clockMaster: state.status.clockMaster });
    await clear();

    // 1) FlexASIO becomes the clock master.
    await call('engine.setSettings', { clockMaster: asio.id });
    await sleep(1500);
    await call('tone.set', { enabled: true, freq: 1000, levelDb: -12, target: asio.id });
    await sleep(2500);
    check('the chosen ASIO device is the clock master', state.status.clockMaster === asio.id, { clockMaster: state.status.clockMaster });

    // 2) Audio through the master, recorded by an app on the other end of the cable.
    // FlexASIO plays into the cable's input; an app records the cable's output.
    const cap = JSON.parse(execFileSync(probe, ['capture', cable.name, String(SECS)], { encoding: 'utf8' }).trim());
    check('following the master: frequency', Math.abs(cap.freqHz - 1000) < 5, cap.freqHz);
    check('following the master: level', Math.abs(cap.peakDb + 12) < 0.3, cap.peakDb);
    check('following the master: no clicks or gaps', cap.clicks === 0 && cap.gaps === 0, { clicks: cap.clicks, gaps: cap.gaps });
    const c = node((n) => n.id === asio.id);
    check('following the master: no dropouts, no drift correction', (c.xruns ?? 0) === 0 && Math.abs(c.driftPpm ?? 0) < 1, { xruns: c.xruns, ppm: c.driftPpm, latencyMs: c.latencyMs });

    // 3) The master goes away (nothing routed to it, so its stream closes): the engine carries on.
    await clear();
    const bus = await call('bus.create', { name: 'Clock test', channels: 2 });
    await call('tone.set', { enabled: true, freq: 500, levelDb: -6, target: bus.id });
    await sleep(2500);
    const lv = [];
    for (let i = 0; i < 12; i++) {
      await sleep(250);
      lv.push(peakDb(node((n) => n.id === bus.id)));
    }
    check('without its master the engine runs on its own timer', state.status.clockMaster === 'internal' && state.status.running, { clockMaster: state.status.clockMaster });
    check('and keeps processing audio', Math.min(...lv) > -7 && Math.max(...lv) < -5, { min: Math.min(...lv), max: Math.max(...lv) });
    await call('bus.delete', { id: bus.id });
  } finally {
    await clear();
    await call('engine.setSettings', { clockMaster: original });
  }
}

main()
  .catch((e) => check('harness', false, e.message))
  .finally(() => {
    const failed = results.filter((r) => !r.ok).length;
    console.log(`\n${results.length - failed}/${results.length} passed`);
    ws.close();
    process.exit(failed ? 1 : 0);
  });
