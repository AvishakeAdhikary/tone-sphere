// The Guitar Rig situation on a real interface: the interface (default AI-04) in WASAPI exclusive
// mode, so in "auto" it leads the engine's clock, while two DAWs share the engine through
// ToneSphere ASIO (DAW A plays a 440 Hz tone, the engine routes it to DAW B). DAW B must hear the
// tone at the right pitch and level, without clicks, and both DAWs must stay on time.
//   node tests/integration/interface-daws.mjs [name fragment] [seconds] [runs]
// Hardware test, not part of run-all.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const host = path.join(root, 'build', 'x64-release', 'bin', 'asio-test-host.exe');
const NAME = process.argv[2] ?? 'AI-04';
const SECS = Number(process.argv[3] ?? 10);
const RUNS = Number(process.argv[4] ?? 3);
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
const runHost = (args) => {
  const p = spawn(host, args);
  let out = '';
  p.stdout.on('data', (d) => (out += d));
  return new Promise((resolve) => p.on('exit', () => resolve(out.trim())));
};

async function main() {
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
  while (!state) await sleep(20);
  const out = state.nodes.find((n) => n.bin === 'physical-out' && n.kind === 'wasapi' && `${n.name} ${n.subtitle ?? ''}`.includes(NAME));
  if (!out) return check(`${NAME} present`, false, null);
  const orig = { ...out.settings };
  try {
    await call('node.update', { id: out.id, settings: { exclusive: true, bufferMs: 3 } });
    for (let run = 1; run <= RUNS; run++) {
      for (const r of [...state.routes]) await call('route.disconnect', { id: r.id }).catch(() => {});
      // Keep the interface's stream open (and so the clock master) with an inaudible tone.
      await call('tone.set', { enabled: true, freq: 1000, levelDb: -60, target: out.id });
      await sleep(1500);
      const a = runHost(['run', 'ToneSphere ASIO', String(SECS + 4), '440', '-6', '256']);
      await sleep(700);
      const b = runHost(['run', 'ToneSphere ASIO', String(SECS + 3), '0', '-120', '256']);
      let dawA, dawB;
      for (let i = 0; i < 40 && !(dawA && dawB); i++) {
        await sleep(100);
        dawA = state.nodes.find((n) => n.id === 'asioclient:out:asio-test-host.exe');
        dawB = state.nodes.find((n) => n.id === 'asioclient:in:asio-test-host.exe#2');
      }
      if (dawA && dawB) await call('route.connect', { from: dawA.id, to: dawB.id });
      const master = state.status.clockMaster;
      const [ra, rb] = (await Promise.all([a, b])).map((s) => JSON.parse(s));
      const label = `run ${run} (clock ${master === out.id ? NAME : master})`;
      check(`${label}: DAW B hears DAW A`, Math.abs(rb.inFreqHz - 440) < 4 && Math.abs(rb.inPeakDb + 6) < 0.7, { f: rb.inFreqHz, db: rb.inPeakDb });
      check(`${label}: no clicks`, rb.inClicks === 0, rb.inClicks);
      check(`${label}: both DAWs on time`, ra.switches > ra.expectedSwitches * 0.97 && rb.switches > rb.expectedSwitches * 0.97, { a: ra.switches, b: rb.switches });
    }
  } finally {
    for (const r of [...state.routes]) await call('route.disconnect', { id: r.id }).catch(() => {});
    await call('tone.set', { enabled: false });
    await call('node.update', { id: out.id, settings: orig }).catch(() => {});
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
