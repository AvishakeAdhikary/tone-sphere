// End-to-end test of the ToneSphere virtual audio driver + engine (run inside the test VM).
//   node vad-audio.mjs <path-to-audio-probe.exe> [seconds]
// 1) engine tone -> "ToneSphere Mic 1"           : recorded by a Windows app (audio-probe)
// 2) app plays into "ToneSphere Output 1" -> mix  : level reaches ToneSphere
// 3) app -> "ToneSphere Output 1" -> ToneSphere -> "ToneSphere Mic 2" -> app : full virtual cable through ToneSphere
import { execFileSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const probe = process.argv[2];
const SECS = Number(process.argv[3] ?? 10);
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
const node = (pred) => state.nodes.find(pred);
const peakDb = (n) => {
  const v = meters?.[n.meter.offset] ?? 0;
  return v > 1e-7 ? 20 * Math.log10(v) : -140;
};
const results = [];
const check = (name, ok, detail) => {
  results.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${JSON.stringify(detail)}`);
};
const capture = (dev, secs) => JSON.parse(execFileSync(probe, ['capture', dev, String(secs)], { encoding: 'utf8' }).trim());
async function clearRoutes() {
  for (const r of [...state.routes]) await call('route.disconnect', { id: r.id }).catch(() => {});
  await call('tone.set', { enabled: false, freq: 1000, levelDb: -12 });
}

async function main() {
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
  while (!state) await sleep(20);
  await clearRoutes();
  check('driver present', state.status.driverPack.vad === true, state.status.driverPack);
  const vads = state.nodes.filter((n) => n.kind === 'vad');
  check('virtual devices discovered', vads.length >= 8, vads.map((n) => `${n.name} [${n.bin}]`));
  const out1 = node((n) => n.id === 'vad:out:1');
  const mic1 = node((n) => n.id === 'vad:in:1');
  const mic2 = node((n) => n.id === 'vad:in:2');
  if (!out1 || !mic1 || !mic2) return;

  // 1) ToneSphere -> virtual microphone -> app
  await call('tone.set', { enabled: true, freq: 1000, levelDb: -12, target: mic1.id });
  await sleep(1500);
  const c1 = capture('ToneSphere Mic 1', SECS);
  check('mic path: frequency', Math.abs(c1.freqHz - 1000) < 5, c1.freqHz);
  check('mic path: level', Math.abs(c1.peakDb + 12) < 0.5, c1.peakDb);
  check('mic path: no clicks', c1.clicks === 0, c1.clicks);
  check('mic path: no gaps', c1.gaps === 0, c1.gaps);
  check('mic path: engine dropouts', (node((n) => n.id === mic1.id).xruns ?? 0) === 0, node((n) => n.id === mic1.id).diag);
  await clearRoutes();

  // 2) app -> virtual output -> ToneSphere mix
  const bus = await call('bus.create', { name: 'VAD test', channels: 2 });
  await call('route.connect', { from: out1.id, to: bus.id });
  const player = spawn(probe, ['play', 'ToneSphere Output 1', '440', '-6', String(SECS * 2 + 15)]);
  await sleep(2500);
  const lv = [];
  for (let i = 0; i < SECS * 2; i++) {
    await sleep(250);
    lv.push(peakDb(node((n) => n.id === bus.id)));
  }
  check('output path: level reaches mix', Math.max(...lv) > -7 && Math.max(...lv) < -5, { max: Math.max(...lv) });
  check('output path: continuous', Math.min(...lv) > -9, { min: Math.min(...lv) });

  // 3) app -> ToneSphere Output 1 -> ToneSphere -> ToneSphere Mic 2 -> app
  await call('route.connect', { from: out1.id, to: mic2.id });
  await sleep(1500);
  const c3 = capture('ToneSphere Mic 2', SECS);
  check('cable path: frequency', Math.abs(c3.freqHz - 440) < 4, c3.freqHz);
  check('cable path: level', Math.abs(c3.peakDb + 6) < 0.6, c3.peakDb);
  check('cable path: no clicks', c3.clicks === 0, c3.clicks);
  check('cable path: no gaps', c3.gaps === 0, c3.gaps);
  const o1 = node((n) => n.id === out1.id);
  const m2 = node((n) => n.id === mic2.id);
  check('cable path: engine dropouts', (o1.xruns ?? 0) === 0 && (m2.xruns ?? 0) === 0, { out: o1.diag, mic: m2.diag });
  player.kill();
  await call('bus.delete', { id: bus.id });
  await clearRoutes();
}

main()
  .catch((e) => check('harness', false, e.message))
  .finally(() => {
    const failed = results.filter((r) => !r.ok).length;
    console.log(`\n${results.length - failed}/${results.length} passed`);
    fs.writeFileSync(path.join(path.dirname(probe), 'vad-audio-report.json'), JSON.stringify(results, null, 2));
    ws.close();
    process.exit(failed ? 1 : 0);
  });
