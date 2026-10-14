// Live VBAN test over the loopback interface: tone -> VBAN out -> UDP -> VBAN in -> mix.
//   node tests/integration/vban.mjs [seconds]
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const SECS = Number(process.argv[2] ?? 10);
const disc = JSON.parse(fs.readFileSync(path.join(process.env.LOCALAPPDATA ?? os.homedir(), 'NexTone', 'engine.json'), 'utf8'));
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

async function main() {
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
  while (!state) await sleep(20);
  const port = 6990 + Math.floor(Math.random() * 9);
  const out = await call('vban.add', { direction: 'out', name: 'Net out', streamName: 'NTTest', host: '127.0.0.1', port, channels: 2 });
  const inn = await call('vban.add', { direction: 'in', name: 'Net in', streamName: 'NTTest', host: '0.0.0.0', port, channels: 2 });
  const bus = await call('bus.create', { name: 'VBAN test', channels: 2 });
  check('VBAN nodes created in the virtual bins', out.bin === 'virtual-in' && inn.bin === 'virtual-out', { out: out.bin, in: inn.bin });
  await call('tone.set', { enabled: true, freq: 1000, levelDb: -12, target: out.id });
  await call('route.connect', { from: inn.id, to: bus.id });
  await sleep(2500);
  const lv = [];
  for (let i = 0; i < SECS * 4; i++) {
    await sleep(250);
    lv.push(peakDb(node((n) => n.id === bus.id)));
  }
  const ni = node((n) => n.id === inn.id), no = node((n) => n.id === out.id);
  check('VBAN: level arrives (16-bit PCM)', Math.max(...lv) > -12.3 && Math.max(...lv) < -11.7, { max: Math.max(...lv) });
  check('VBAN: continuous', Math.min(...lv) > -14, { min: Math.min(...lv) });
  check('VBAN: no dropouts', (ni.xruns ?? 0) === 0 && (no.xruns ?? 0) === 0, { in: ni.diag, out: no.diag });
  for (const r of [...state.routes]) await call('route.disconnect', { id: r.id });
  await call('tone.set', { enabled: false, freq: 1000, levelDb: -12 });
  await call('vban.remove', { id: out.id });
  await call('vban.remove', { id: inn.id });
  await call('bus.delete', { id: bus.id });
}

main()
  .catch((e) => check('harness', false, e.message))
  .finally(() => {
    const failed = results.filter((r) => !r.ok).length;
    console.log(`\n${results.length - failed}/${results.length} passed`);
    ws.close();
    process.exit(failed ? 1 : 0);
  });
