// How low can an interface's latency go without dropouts? Sweeps WASAPI exclusive device buffers
// and engine block sizes on a real interface (default AI-04) and reports latency and dropouts for
// each, running input -> mix and a -60 dBFS tone -> output for a while at each setting.
//   node tests/integration/interface-latency.mjs [name fragment] [seconds per case] [rate]
// Hardware test, not part of run-all.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const NAME = process.argv[2] ?? 'AI-04';
const SECS = Number(process.argv[3] ?? 15);
const RATE = Number(process.argv[4] ?? 48000);
const BUFFERS_MS = [10, 5, 3, 2, 1.5];
const BLOCKS = [128, 64, 32];
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
const find = (bin) => state.nodes.find((n) => n.bin === bin && n.kind === 'wasapi' && `${n.name} ${n.subtitle ?? ''}`.includes(NAME));
const at = (nid) => state.nodes.find((n) => n.id === nid);

await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
while (!state) await sleep(20);
const original = { sampleRate: state.settings.sampleRate, blockSize: state.settings.blockSize };
const inNode = find('physical-in'), outNode = find('physical-out');
if (!inNode || !outNode) throw new Error(`${NAME} not found`);
const orig = { in: { ...inNode.settings }, out: { ...outNode.settings } };
const rows = [];
try {
  for (const block of BLOCKS) {
    for (const bufferMs of BUFFERS_MS) {
      await call('engine.setSettings', { sampleRate: RATE, blockSize: block });
      await sleep(1000);
      for (const n of [inNode, outNode]) await call('node.update', { id: n.id, settings: { exclusive: true, bufferMs } });
      await sleep(500);
      for (const r of [...state.routes]) await call('route.disconnect', { id: r.id }).catch(() => {});
      const bus = await call('bus.create', { name: 'Latency sweep', channels: 2 });
      await call('route.connect', { from: inNode.id, to: bus.id });
      await call('tone.set', { enabled: true, freq: 1000, levelDb: -60, target: outNode.id });
      await sleep(3000);
      const x0 = { in: at(inNode.id)?.xruns ?? 0, out: at(outNode.id)?.xruns ?? 0, late: state.status.xruns };
      await sleep(SECS * 1000);
      const i1 = at(inNode.id), o1 = at(outNode.id);
      rows.push({
        block, bufferMs,
        inPeriod: i1?.diag?.devicePeriod, outPeriod: o1?.diag?.devicePeriod,
        inMs: i1?.latencyMs, outMs: o1?.latencyMs,
        xin: (i1?.xruns ?? 0) - x0.in, xout: (o1?.xruns ?? 0) - x0.out,
        cpu: state.status.cpuPercent,
        master: state.status.clockMaster === outNode.id ? NAME : state.status.clockMaster,
        ok: i1?.state === 'active' && o1?.state === 'active' && !i1?.statusText && !o1?.statusText,
      });
      const r = rows.at(-1);
      console.log(`block ${block}, buffer ${bufferMs} ms: periods ${r.inPeriod}/${r.outPeriod}, latency in ${r.inMs?.toFixed(1)} out ${r.outMs?.toFixed(1)} ms (through ${(r.inMs + r.outMs).toFixed(1)} ms), dropouts ${r.xin}/${r.xout}, cpu ${r.cpu?.toFixed(1)} %, clock ${r.master}${r.ok ? '' : ' (stream problem)'}`);
      await call('tone.set', { enabled: false });
      await call('bus.delete', { id: bus.id });
    }
  }
} finally {
  for (const r of [...state.routes]) await call('route.disconnect', { id: r.id }).catch(() => {});
  await call('tone.set', { enabled: false });
  await call('node.update', { id: inNode.id, settings: orig.in }).catch(() => {});
  await call('node.update', { id: outNode.id, settings: orig.out }).catch(() => {});
  await call('engine.setSettings', original);
}
const clean = rows.filter((r) => r.ok && r.xin === 0 && r.xout === 0).sort((a, b) => a.inMs + a.outMs - (b.inMs + b.outMs));
console.log(clean.length ? `\nlowest clean: block ${clean[0].block}, buffer ${clean[0].bufferMs} ms -> ${(clean[0].inMs + clean[0].outMs).toFixed(1)} ms through` : '\nno clean setting');
ws.close();
