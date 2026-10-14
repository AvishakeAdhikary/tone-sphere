// Round-trip latency measurement (latency.measure) against the running engine, through the
// VB-CABLE loop: ToneSphere plays a chirp into "CABLE Input" and records "CABLE Output".
//   node tests/integration/latency.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

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

async function main() {
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
  while (!state) await sleep(20);
  for (const r of [...state.routes]) await call('route.disconnect', { id: r.id });
  const out = state.nodes.find((n) => n.kind === 'compat' && n.bin === 'virtual-in'); // ToneSphere plays into the cable
  const inp = state.nodes.find((n) => n.kind === 'compat' && n.bin === 'virtual-out'); // ToneSphere records the cable
  if (!out || !inp) return check('virtual cable present', false, 'install VB-CABLE');

  const runs = [];
  for (let i = 0; i < 3; i++) runs.push(await call('latency.measure', { output: out.id, input: inp.id }));
  const ms = runs.map((r) => +r.latencyMs.toFixed(2));
  check('loop measured', runs.every((r) => r.confidence > 0.5), { ms, confidence: runs.map((r) => +r.confidence.toFixed(2)) });
  check('plausible round trip (2..400 ms)', ms.every((v) => v > 2 && v < 400), ms);
  const spread = Math.max(...ms) - Math.min(...ms);
  // Each run reopens both ends of the cable; VB-CABLE's internal buffer fills differently on each start,
  // so real round trips vary by tens of ms. The estimator's own precision is unit-tested (< 0.5 sample).
  check('consistent across device restarts (spread < 30 ms)', spread < 30, { spread: +spread.toFixed(2) });
  check('no streams left open afterwards', state.routes.length === 0 && !state.nodes.some((n) => n.id === out.id && n.state === 'active' && n.streaming), {
    out: state.nodes.find((n) => n.id === out.id)?.state,
  });

  // An input that doesn't receive the output: a clear error, not a bogus number.
  const mic = state.nodes.find((n) => n.bin === 'physical-in' && n.kind === 'wasapi' && n.state !== 'unavailable');
  if (mic) {
    let msg = '';
    try {
      const r = await call('latency.measure', { output: out.id, input: mic.id });
      msg = `measured ${r.latencyMs} ms`;
    } catch (e) {
      msg = e.message;
    }
    check('no loop: explained, not measured', /didn't come back/.test(msg), { mic: mic.name, msg });
  }
  let bad = '';
  try {
    await call('latency.measure', { output: inp.id, input: out.id });
  } catch (e) {
    bad = e.message;
  }
  check('wrong direction is refused', /must be/.test(bad), { bad });
}

main()
  .catch((e) => check('harness', false, e.message))
  .finally(() => {
    const failed = results.filter((r) => !r.ok).length;
    console.log(`\n${results.length - failed}/${results.length} passed`);
    ws.close();
    process.exit(failed ? 1 : 0);
  });
