// Engine hosting a real ASIO driver (FlexASIO configured as a silent VB-CABLE loopback).
// 1) engine tone -> ASIO outputs -> (loopback) -> ASIO inputs -> mix meter
// 2) DAW A (ToneSphere ASIO) -> ASIO outputs -> loopback -> ASIO inputs -> DAW B (ToneSphere ASIO)
//   node tests/integration/asio-host.mjs [driver] [seconds]
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const host = path.join(root, 'build', 'x64-release', 'bin', 'asio-test-host.exe');
const DRIVER = process.argv[2] ?? 'FlexASIO';
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
const runHost = (args) => {
  const p = spawn(host, args);
  let out = '';
  p.stdout.on('data', (d) => (out += d));
  return new Promise((resolve) => p.on('exit', () => resolve(out.trim())));
};
async function clear() {
  for (const r of [...state.routes]) await call('route.disconnect', { id: r.id }).catch(() => {});
  await call('tone.set', { enabled: false, freq: 1000, levelDb: -12 });
  for (const b of state.nodes.filter((n) => n.bin === 'bus')) await call('bus.delete', { id: b.id }).catch(() => {});
  await sleep(500);
}

async function main() {
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
  while (!state) await sleep(20);
  await clear();
  const aOut = node((n) => n.id === `asio:out:${DRIVER}`);
  const aIn = node((n) => n.id === `asio:in:${DRIVER}`);
  check('ASIO driver listed as physical device', !!aOut && !!aIn, [aOut?.name, aIn?.name]);
  if (!aOut || !aIn) return;

  // 1) tone -> ASIO out -> loopback -> ASIO in -> mix
  const bus = await call('bus.create', { name: 'ASIO test', channels: 2 });
  await call('tone.set', { enabled: true, freq: 1000, levelDb: -12, target: aOut.id });
  await call('route.connect', { from: aIn.id, to: bus.id });
  await sleep(3000);
  const lv = [];
  for (let i = 0; i < SECS * 2; i++) {
    await sleep(250);
    lv.push(peakDb(node((n) => n.id === bus.id)));
  }
  const ain = node((n) => n.id === aIn.id), aout = node((n) => n.id === aOut.id);
  check('hosted ASIO: loopback level', Math.max(...lv) > -12.6 && Math.max(...lv) < -11.4, { max: Math.max(...lv) });
  check('hosted ASIO: continuous', Math.min(...lv) > -14, { min: Math.min(...lv) });
  check('hosted ASIO: channel counts from driver', ain.channels >= 1 && aout.channels >= 1, { in: ain.channels, out: aout.channels });
  check('hosted ASIO: no dropouts', (ain.xruns ?? 0) === 0 && (aout.xruns ?? 0) === 0, { in: ain.diag, out: aout.diag });
  await clear();

  // 2) DAW A -> ToneSphere ASIO -> hosted interface -> loopback -> hosted interface -> ToneSphere ASIO -> DAW B
  const a = runHost(['run', 'ToneSphere ASIO', String(SECS + 4), '440', '-6', '256']);
  await sleep(700);
  const b = runHost(['run', 'ToneSphere ASIO', String(SECS + 3), '0', '-120', '256']);
  let dawA, dawB;
  for (let i = 0; i < 40 && !(dawA && dawB); i++) {
    await sleep(100);
    dawA = node((n) => n.id === 'asioclient:out:asio-test-host.exe');
    dawB = node((n) => n.id === 'asioclient:in:asio-test-host.exe#2');
  }
  if (dawA && dawB) {
    await call('route.connect', { from: dawA.id, to: aOut.id });
    await call('route.connect', { from: aIn.id, to: dawB.id });
  }
  const [ra, rb] = (await Promise.all([a, b])).map((s) => JSON.parse(s));
  console.log('DAW B', JSON.stringify(rb));
  check('shared interface: DAW B hears DAW A', Math.abs(rb.inFreqHz - 440) < 4 && Math.abs(rb.inPeakDb + 6) < 0.7, { f: rb.inFreqHz, db: rb.inPeakDb });
  check('shared interface: no clicks', rb.inClicks === 0, rb.inClicks);
  check('shared interface: both DAWs on time', ra.switches > ra.expectedSwitches * 0.97 && rb.switches > rb.expectedSwitches * 0.97, {
    a: ra.switches,
    b: rb.switches,
  });
  await clear();
}

main()
  .catch((e) => check('harness', false, e.message))
  .finally(() => {
    const failed = results.filter((r) => !r.ok).length;
    console.log(`\n${results.length - failed}/${results.length} passed`);
    ws.close();
    process.exit(failed ? 1 : 0);
  });
