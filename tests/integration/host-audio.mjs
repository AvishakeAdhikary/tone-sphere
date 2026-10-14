// Host integration tests against the real running engine and real Windows audio endpoints.
// Uses VB-CABLE (if installed) as a silent loopback so nothing plays through the speakers.
//   node tests/integration/host-audio.mjs [--soak SECONDS]
// Writes a JSON report to build/test-reports/host-audio.json and exits non-zero on failure.
import { execFileSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const probe = path.join(root, 'build', 'x64-release', 'bin', 'audio-probe.exe');
const soakIdx = process.argv.indexOf('--soak');
const SOAK = soakIdx > 0 ? Number(process.argv[soakIdx + 1]) : 20;

const disc = JSON.parse(fs.readFileSync(path.join(process.env.LOCALAPPDATA ?? os.homedir(), 'ToneSphere', 'engine.json'), 'utf8'));
const ws = new WebSocket(`ws://127.0.0.1:${disc.port}/?token=${disc.token}`);
ws.binaryType = 'arraybuffer';
let state = null;
let meters = null;
let id = 1;
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
function check(name, ok, detail) {
  results.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${JSON.stringify(detail)}`);
}

async function clearRoutes() {
  for (const r of [...state.routes]) await call('route.disconnect', { id: r.id }).catch(() => {});
  await call('tone.set', { enabled: false, freq: 1000, levelDb: -12 });
  await sleep(300);
}

async function main() {
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
  while (!state) await sleep(20);
  await clearRoutes();

  const cableIn = node((n) => n.kind === 'compat' && n.bin === 'virtual-in'); // ToneSphere renders into the cable
  const cableOut = node((n) => n.kind === 'compat' && n.bin === 'virtual-out'); // ToneSphere records the cable
  if (!cableIn || !cableOut) {
    check('virtual cable present (compatibility mode)', false, 'install VB-CABLE to run loopback tests');
    return;
  }

  // 1) Render path: tone -> virtual input (CABLE Input) -> probe records CABLE Output.
  await call('tone.set', { enabled: true, freq: 1000, levelDb: -12, target: cableIn.id });
  await sleep(1000);
  const cap = JSON.parse(execFileSync(probe, ['capture', 'CABLE Output', String(SOAK)], { encoding: 'utf8' }).trim());
  check('render path: frequency', Math.abs(cap.freqHz - 1000) < 5, cap.freqHz);
  check('render path: level', Math.abs(cap.peakDb + 12) < 0.3, cap.peakDb);
  check('render path: no clicks', cap.clicks === 0, { clicks: cap.clicks, firstGapAt: cap.firstGapAt, captureDiscontinuities: cap.discontinuities });
  check('render path: no gaps', cap.gaps === 0, cap.gaps);
  const ci = node((n) => n.id === cableIn.id);
  check('render path: no engine dropouts', (ci.xruns ?? 0) === 0, { xruns: ci.xruns, driftPpm: ci.driftPpm, latencyMs: ci.latencyMs });
  await clearRoutes();

  // 2) Capture path: probe plays into CABLE Input -> engine records it -> mix bus meter.
  const bus = await call('bus.create', { name: 'Test mix', channels: 2 });
  await call('route.connect', { from: cableOut.id, to: bus.id });
  const player = spawn(probe, ['play', 'CABLE Input', '440', '-6', String(SOAK + 20)]);
  await sleep(2500);
  const levels = [];
  for (let i = 0; i < SOAK * 4; i++) {
    await sleep(250);
    levels.push(peakDb(node((n) => n.id === bus.id)));
  }
  const minL = Math.min(...levels), maxL = Math.max(...levels);
  check('capture path: level reaches mix', maxL > -7 && maxL < -5, { maxL });
  check('capture path: continuous (no dropouts to silence)', minL > -9, { minL });
  const co = node((n) => n.id === cableOut.id);
  check('capture path: no engine dropouts', (co.xruns ?? 0) === 0, { xruns: co.xruns, driftPpm: co.driftPpm, latencyMs: co.latencyMs });

  // 3) Application capture (process loopback): the playing probe process appears as an app.
  await sleep(1500);
  const app = node((n) => n.bin === 'app-out' && n.app?.exe === 'audio-probe.exe');
  check('app discovery: playing app appears in Application outputs', !!app, app?.name);
  if (app) {
    const bus2 = await call('bus.create', { name: 'App mix', channels: 2 });
    await call('route.connect', { from: app.id, to: bus2.id });
    const lv = [];
    for (let i = 0; i < 12; i++) {
      await sleep(250);
      lv.push(peakDb(node((n) => n.id === bus2.id)));
    }
    const appNode = node((n) => n.id === app.id);
    check('app capture: app sound reaches mix', Math.max(...lv) > -8, { max: Math.max(...lv), state: appNode.state, status: appNode.statusText });
    await call('bus.delete', { id: bus2.id });
  }
  player.kill();
  await call('bus.delete', { id: bus.id });
  await clearRoutes();
}

main()
  .catch((e) => check('test harness', false, e.message))
  .finally(() => {
    const dir = path.join(root, 'build', 'test-reports');
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(path.join(dir, 'host-audio.json'), JSON.stringify({ when: new Date().toISOString(), results }, null, 2));
    const failed = results.filter((r) => !r.ok).length;
    console.log(`\n${results.length - failed}/${results.length} passed`);
    ws.close();
    process.exit(failed ? 1 : 0);
  });
