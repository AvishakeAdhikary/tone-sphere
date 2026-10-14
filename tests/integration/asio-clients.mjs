// Multi-client ToneSphere ASIO test: two ASIO hosts at once; host A's tone is routed by the
// engine into host B's inputs. Verifies timing, level, frequency and clicks.
//   node tests/integration/asio-clients.mjs [seconds]
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const host = process.env.ASIO_TEST_HOST ?? path.join(root, 'build', 'x64-release', 'bin', 'asio-test-host.exe');
const SECS = Number(process.argv[2] ?? 12);
const disc = JSON.parse(fs.readFileSync(path.join(process.env.LOCALAPPDATA ?? os.homedir(), 'ToneSphere', 'engine.json'), 'utf8'));
const ws = new WebSocket(`ws://127.0.0.1:${disc.port}/?token=${disc.token}`);
let state = null, id = 1;
const pending = new Map();
ws.onmessage = (ev) => {
  if (typeof ev.data !== 'string') return;
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
function runHost(args) {
  const p = spawn(host, args);
  let out = '';
  p.stdout.on('data', (d) => (out += d));
  return new Promise((resolve) => p.on('exit', () => resolve(out.trim())));
}

async function main() {
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
  while (!state) await sleep(20);
  for (const r of [...state.routes]) await call('route.disconnect', { id: r.id });

  const a = runHost(['run', 'ToneSphere ASIO', String(SECS), '1000', '-12', '256']);
  await sleep(700);
  const b = runHost(['run', 'ToneSphere ASIO', String(SECS - 1), '0', '-120', '128']);
  // Wait for both clients to show up, then route A's outputs into B's inputs.
  let aOut, bIn;
  for (let i = 0; i < 40 && !(aOut && bIn); i++) {
    await sleep(100);
    aOut = state.nodes.find((n) => n.id === 'asioclient:out:asio-test-host.exe');
    bIn = state.nodes.find((n) => n.id === 'asioclient:in:asio-test-host.exe#2');
  }
  check('both ASIO clients appear in ToneSphere', !!aOut && !!bIn, state.nodes.filter((n) => n.kind === 'asioclient').map((n) => n.name));
  if (aOut && bIn) await call('route.connect', { from: aOut.id, to: bIn.id });
  const [ra, rb] = (await Promise.all([a, b])).map((s) => JSON.parse(s));
  console.log('host A', JSON.stringify(ra));
  console.log('host B', JSON.stringify(rb));
  for (const [nm, r] of [['A', ra], ['B', rb]]) {
    check(`host ${nm}: buffer switches on time`, r.switches > r.expectedSwitches * 0.97, { switches: r.switches, expected: r.expectedSwitches });
    check(`host ${nm}: timing`, Math.abs(r.avgIntervalMs - (r.buffer / r.rate) * 1000) < 0.2 && r.maxIntervalMs < 30, {
      avg: r.avgIntervalMs,
      max: r.maxIntervalMs,
    });
    check(`host ${nm}: float32 samples`, r.sampleType === 19, r.sampleType);
  }
  check('A -> ToneSphere -> B: frequency', Math.abs(rb.inFreqHz - 1000) < 5, rb.inFreqHz);
  check('A -> ToneSphere -> B: level', Math.abs(rb.inPeakDb + 12) < 0.5, rb.inPeakDb);
  check('A -> ToneSphere -> B: no clicks', rb.inClicks === 0, rb.inClicks);
  await sleep(1500);
  // Clients still referenced by a connection stay listed as unavailable (routes survive a DAW restart).
  const left = state.nodes.filter((n) => n.kind === 'asioclient');
  check('closed clients become unavailable', left.every((n) => n.state === 'unavailable'), left.map((n) => `${n.name}: ${n.state}`));
}

main()
  .catch((e) => check('harness', false, e.message))
  .finally(() => {
    const failed = results.filter((r) => !r.ok).length;
    console.log(`\n${results.length - failed}/${results.length} passed`);
    ws.close();
    process.exit(failed ? 1 : 0);
  });
