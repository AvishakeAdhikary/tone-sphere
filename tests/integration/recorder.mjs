// Recording + instant replay test: tone -> mix, record 3 s, save replay, validate the WAV files.
//   node tests/integration/recorder.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const disc = JSON.parse(fs.readFileSync(path.join(process.env.LOCALAPPDATA ?? os.homedir(), 'NexTone', 'engine.json'), 'utf8'));
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

function readWav(file) {
  const b = fs.readFileSync(file);
  const riff = b.toString('ascii', 0, 4);
  let off = 12, fmt = null, data = null;
  while (off + 8 <= b.length) {
    const id = b.toString('ascii', off, off + 4);
    const size = b.readUInt32LE(off + 4);
    if (id === 'fmt ') fmt = { tag: b.readUInt16LE(off + 8), ch: b.readUInt16LE(off + 10), rate: b.readUInt32LE(off + 12), bits: b.readUInt16LE(off + 22) };
    if (id === 'data') data = { off: off + 8, size };
    off += 8 + size + (size & 1);
  }
  const frames = data.size / (fmt.ch * 4);
  let peak = 0;
  for (let i = 0; i < frames * fmt.ch; i++) peak = Math.max(peak, Math.abs(b.readFloatLE(data.off + i * 4)));
  return { riff, ...fmt, seconds: frames / fmt.rate, peakDb: 20 * Math.log10(peak || 1e-9) };
}

async function main() {
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
  while (!state) await sleep(20);
  const folder = path.join(os.tmpdir(), `nextone-rec-${Date.now()}`);
  await call('engine.setSettings', { recordFolder: folder, replayBufferSeconds: 30 });
  const bus = await call('bus.create', { name: 'Rec test', channels: 2 });
  await call('tone.set', { enabled: true, freq: 1000, levelDb: -12, target: bus.id });
  await sleep(1500);
  await call('recorder.start', { nodeIds: [bus.id], format: 'wav' });
  await sleep(300); // state broadcasts are coalesced
  check('state shows recording', state.recorder.recording === true, state.recorder.recording);
  await sleep(3000);
  const { files } = await call('recorder.stop');
  check('one WAV file written', files.length === 1 && fs.existsSync(files[0]), files);
  if (files[0]) {
    const w = readWav(files[0]);
    check('WAV is 32-bit float stereo 48 kHz', w.riff === 'RIFF' && w.tag === 3 && w.ch === 2 && w.bits === 32 && w.rate === 48000, w);
    check('recording length ~3 s', Math.abs(w.seconds - 3) < 0.25, w.seconds);
    check('recording level -12 dB', Math.abs(w.peakDb + 12) < 0.2, w.peakDb);
  }
  const rep = await call('recorder.saveReplay', { seconds: 30 });
  const r = readWav(rep.file);
  check('replay saved with the recent past (>= 4 s)', r.seconds >= 4 && Math.abs(r.peakDb + 12) < 0.2, { seconds: r.seconds, peakDb: r.peakDb });
  await call('tone.set', { enabled: false, freq: 1000, levelDb: -12 });
  await call('bus.delete', { id: bus.id });
  await call('engine.setSettings', { replayBufferSeconds: 0 });
  fs.rmSync(folder, { recursive: true, force: true });
}

main()
  .catch((e) => check('harness', false, e.message))
  .finally(() => {
    const failed = results.filter((r) => !r.ok).length;
    console.log(`\n${results.length - failed}/${results.length} passed`);
    ws.close();
    process.exit(failed ? 1 : 0);
  });
