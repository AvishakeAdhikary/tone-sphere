// An ASIO driver on a real interface (FlexASIO or ASIO4ALL configured for e.g. the AI-04), hosted
// by the running engine, while two DAWs share the engine through ToneSphere ASIO:
//  - the driver's inputs and outputs open with the driver's channel counts, with no dropouts or
//    device glitches, and the interface's input reaches a mix;
//  - DAW A's tone goes through the engine to DAW B at the right pitch and level, without clicks,
//    both DAWs on time. The interface output carries a -60 dBFS tone only.
//   node tests/integration/interface-asio.mjs <ASIO driver name> [seconds]
// Hardware test, not part of run-all.
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
const results = [];
const check = (name, ok, detail) => {
  results.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${JSON.stringify(detail)}`);
};
const at = (nid) => state.nodes.find((n) => n.id === nid);
const peakDb = (n) => {
  const v = meters?.[n.meter.offset] ?? 0;
  return v > 1e-9 ? 20 * Math.log10(v) : -180;
};
const runHost = (args) => {
  const p = spawn(host, args);
  let out = '';
  p.stdout.on('data', (d) => (out += d));
  return new Promise((resolve) => p.on('exit', () => resolve(out.trim())));
};
async function clear() {
  for (const r of [...state.routes]) await call('route.disconnect', { id: r.id }).catch(() => {});
  await call('tone.set', { enabled: false });
  for (const b of state.nodes.filter((n) => n.bin === 'bus' && n.name === 'ASIO interface test')) await call('bus.delete', { id: b.id }).catch(() => {});
}

async function main() {
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
  while (!state) await sleep(20);
  const aOut = at(`asio:out:${DRIVER}`), aIn = at(`asio:in:${DRIVER}`);
  check(`${DRIVER} listed`, !!aOut && !!aIn, [aOut?.name, aIn?.name]);
  if (!aOut || !aIn) return;
  await clear();
  try {
    const bus = await call('bus.create', { name: 'ASIO interface test', channels: 2 });
    await call('route.connect', { from: aIn.id, to: bus.id });
    await call('tone.set', { enabled: true, freq: 1000, levelDb: -60, target: aOut.id });
    await sleep(3000);
    const x0 = { in: at(aIn.id)?.xruns ?? 0, out: at(aOut.id)?.xruns ?? 0 };
    // DAWs on ToneSphere ASIO at the same time.
    const a = runHost(['run', 'ToneSphere ASIO', String(SECS + 4), '440', '-6', '256']);
    await sleep(700);
    const b = runHost(['run', 'ToneSphere ASIO', String(SECS + 3), '0', '-120', '256']);
    let dawA, dawB;
    for (let i = 0; i < 40 && !(dawA && dawB); i++) {
      await sleep(100);
      dawA = at('asioclient:out:asio-test-host.exe');
      dawB = at('asioclient:in:asio-test-host.exe#2');
    }
    if (dawA && dawB) await call('route.connect', { from: dawA.id, to: dawB.id });
    let inPeak = -180;
    for (let i = 0; i < SECS * 4; i++) {
      await sleep(250);
      inPeak = Math.max(inPeak, peakDb(at(bus.id)));
    }
    const [ra, rb] = (await Promise.all([a, b])).map((s) => JSON.parse(s));
    const i1 = at(aIn.id), o1 = at(aOut.id);
    check(`${DRIVER}: streams open`, i1?.state === 'active' && o1?.state === 'active' && !i1.statusText && !o1.statusText, {
      in: `${i1?.channels} ch @ ${i1?.sampleRate}`, out: `${o1?.channels} ch @ ${o1?.sampleRate}`, period: o1?.diag?.devicePeriod,
    });
    check(`${DRIVER}: no dropouts`, (i1?.xruns ?? 0) - x0.in === 0 && (o1?.xruns ?? 0) - x0.out === 0, { in: i1?.diag, out: o1?.diag });
    check(`${DRIVER}: the interface's input reaches the mix`, inPeak > -150, { peakDb: +inPeak.toFixed(1) });
    check('DAWs alongside: DAW B hears DAW A', Math.abs(rb.inFreqHz - 440) < 4 && Math.abs(rb.inPeakDb + 6) < 0.7, { f: rb.inFreqHz, db: rb.inPeakDb });
    check('DAWs alongside: no clicks', rb.inClicks === 0, rb.inClicks);
    check('DAWs alongside: both on time', ra.switches > ra.expectedSwitches * 0.97 && rb.switches > rb.expectedSwitches * 0.97, { a: ra.switches, b: rb.switches });
    console.log(`latency: in ${i1?.latencyMs?.toFixed(1)} ms, out ${o1?.latencyMs?.toFixed(1)} ms; clock ${state.status.clockMaster}`);
  } finally {
    await clear();
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
