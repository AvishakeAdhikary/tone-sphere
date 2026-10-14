// A real USB audio interface (default: the Audio Array AI-04) through the running engine, at
// every sample rate, in WASAPI shared and exclusive mode: the input feeds a mix and the output
// plays a -60 dBFS tone (inaudible in practice: the test can run with headphones plugged in).
// Evidence per case comes from the engine: the stream opened (in exclusive mode at exactly the
// engine rate), no dropouts or device glitches, drift locked, latency.
//   node tests/integration/interface.mjs [name fragment] [seconds per case]
// Hardware test: it needs the interface connected, so it isn't part of run-all.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const NAME = process.argv[2] ?? 'AI-04';
const SECS = Number(process.argv[3] ?? 8);
const RATES = (process.env.RATES ?? '44100,48000,96000,192000').split(',').map(Number);
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
const find = (bin) => state.nodes.find((n) => n.bin === bin && n.kind === 'wasapi' && `${n.name} ${n.subtitle ?? ''}`.includes(NAME));
const peakDb = (n) => {
  const v = meters?.[n.meter.offset] ?? 0;
  return v > 1e-9 ? 20 * Math.log10(v) : -180;
};

async function clear() {
  for (const r of [...state.routes]) await call('route.disconnect', { id: r.id }).catch(() => {});
  await call('tone.set', { enabled: false });
  for (const b of state.nodes.filter((n) => n.bin === 'bus' && n.name === 'Interface test')) await call('bus.delete', { id: b.id }).catch(() => {});
}

async function main() {
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
  while (!state) await sleep(20);
  const original = { sampleRate: state.settings.sampleRate };
  const inNode = find('physical-in'), outNode = find('physical-out');
  check(`${NAME} found (input and output)`, !!inNode && !!outNode, [inNode?.name, outNode?.name]);
  if (!inNode || !outNode) return;
  const originalExclusive = { in: !!inNode.settings.exclusive, out: !!outNode.settings.exclusive };
  await clear();
  const rows = [];
  try {
    for (const rate of RATES) {
      for (const exclusive of [false, true]) {
        await call('engine.setSettings', { sampleRate: rate });
        await sleep(1500);
        for (const n of [inNode, outNode]) await call('node.update', { id: n.id, settings: { exclusive } });
        await sleep(500);
        const bus = await call('bus.create', { name: 'Interface test', channels: 2 });
        await call('route.connect', { from: inNode.id, to: bus.id });
        await call('tone.set', { enabled: true, freq: 1000, levelDb: -60, target: outNode.id });
        await sleep(2500); // streams open and bridges settle
        const at = (nid) => state.nodes.find((n) => n.id === nid);
        const x0 = { in: at(inNode.id)?.xruns ?? 0, out: at(outNode.id)?.xruns ?? 0 };
        let inPeak = -180;
        for (let i = 0; i < SECS * 4; i++) {
          await sleep(250);
          inPeak = Math.max(inPeak, peakDb(at(bus.id)));
        }
        const i1 = at(inNode.id), o1 = at(outNode.id);
        const label = `${rate / 1000} kHz ${exclusive ? 'exclusive' : 'shared'}`;
        const opened = i1?.state === 'active' && o1?.state === 'active' && !i1.statusText && !o1.statusText;
        // Exclusive mode runs the device at exactly the engine's rate; shared mode at Windows' mix rate.
        const exactRate = !exclusive || (i1?.sampleRate === rate && o1?.sampleRate === rate);
        const xr = { in: (i1?.xruns ?? 0) - x0.in, out: (o1?.xruns ?? 0) - x0.out };
        rows.push({ label, inRate: i1?.sampleRate, outRate: o1?.sampleRate, inLat: i1?.latencyMs, outLat: o1?.latencyMs, inPpm: i1?.driftPpm, outPpm: o1?.driftPpm, xr, inPeak });
        check(`${label}: both streams open`, opened, { in: i1?.statusText ?? i1?.state, out: o1?.statusText ?? o1?.state });
        check(`${label}: device rate`, exactRate, { in: i1?.sampleRate, out: o1?.sampleRate });
        check(`${label}: no dropouts in ${SECS} s`, xr.in === 0 && xr.out === 0, { xruns: xr, inDiag: i1?.diag, outDiag: o1?.diag });
        check(`${label}: input signal reaches the mix`, inPeak > -150, { peakDb: +inPeak.toFixed(1) });
        await clear();
      }
    }
  } finally {
    await clear();
    for (const [n, ex] of [[inNode, originalExclusive.in], [outNode, originalExclusive.out]]) await call('node.update', { id: n.id, settings: { exclusive: ex } }).catch(() => {});
    await call('engine.setSettings', original);
  }
  console.log('\nrate/mode               in Hz   out Hz  in ms  out ms  in ppm  out ppm  in peak');
  for (const r of rows)
    console.log(`${r.label.padEnd(22)} ${String(r.inRate).padStart(6)} ${String(r.outRate).padStart(7)} ${(r.inLat ?? 0).toFixed(1).padStart(6)} ${(r.outLat ?? 0).toFixed(1).padStart(7)} ${Math.round(r.inPpm ?? 0).toString().padStart(7)} ${Math.round(r.outPpm ?? 0).toString().padStart(8)} ${r.inPeak.toFixed(1).padStart(8)}`);
}

main()
  .catch((e) => check('harness', false, e.message))
  .finally(() => {
    const failed = results.filter((r) => !r.ok).length;
    console.log(`\n${results.length - failed}/${results.length} passed`);
    ws.close();
    process.exit(failed ? 1 : 0);
  });
