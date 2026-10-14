// Linux (PipeWire) end-to-end test, run from Windows against WSL Ubuntu:
//   node tests/integration/linux-pipewire.mjs [build dir in WSL, default ~/ts-build]
// Starts PipeWire + WirePlumber (headless) and the Linux engine inside WSL, then checks:
//  - ToneSphere's virtual devices are real PipeWire nodes other apps can use
//  - tone -> "ToneSphere Mic 1" -> recorded by a PipeWire app (pw-record)
//  - a PipeWire app (pw-play) -> "ToneSphere Output 1" -> ToneSphere mix
//  - tone -> a sink device (null sink) -> its monitor recorded by pw-record
//  - an app's playback stream shows up in Application outputs and can be captured
//  - removing a device is handled (node becomes unavailable, engine keeps running)
import { execFileSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const buildDir = process.argv[2] ?? '~/ts-build';
const tmpWin = fs.mkdtempSync(path.join(os.tmpdir(), 'ts-linux-'));
const tmpWsl = execFileSync('wsl', ['-e', 'wslpath', '-a', tmpWin], { encoding: 'utf8' }).trim();
const wsl = (cmd, opts = {}) => execFileSync('wsl', ['-e', 'bash', '-lc', cmd], { encoding: 'utf8', ...opts });
const wslBg = (cmd) => spawn('wsl', ['-e', 'bash', '-lc', cmd], { stdio: 'ignore' });

const results = [];
const check = (name, ok, detail) => {
  results.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${JSON.stringify(detail)}`);
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---- WAV helpers (float32 or int16 PCM) ----
function writeSineWav(file, freq, db, seconds, rate = 48000) {
  const n = Math.round(seconds * rate), amp = 10 ** (db / 20);
  const data = Buffer.alloc(n * 2 * 4);
  for (let i = 0; i < n; i++) {
    const v = amp * Math.sin((2 * Math.PI * freq * i) / rate);
    data.writeFloatLE(v, i * 8);
    data.writeFloatLE(v, i * 8 + 4);
  }
  const h = Buffer.alloc(44);
  h.write('RIFF', 0); h.writeUInt32LE(36 + data.length, 4); h.write('WAVE', 8); h.write('fmt ', 12);
  h.writeUInt32LE(16, 16); h.writeUInt16LE(3, 20); h.writeUInt16LE(2, 22); h.writeUInt32LE(rate, 24);
  h.writeUInt32LE(rate * 8, 28); h.writeUInt16LE(8, 32); h.writeUInt16LE(32, 34); h.write('data', 36); h.writeUInt32LE(data.length, 40);
  fs.writeFileSync(file, Buffer.concat([h, data]));
}
function analyseWav(file) {
  const b = fs.readFileSync(file);
  let p = 12, fmt = null, data = null;
  while (p + 8 <= b.length) {
    const id = b.toString('ascii', p, p + 4), size = b.readUInt32LE(p + 4);
    if (id === 'fmt ') fmt = { tag: b.readUInt16LE(p + 8), ch: b.readUInt16LE(p + 10), rate: b.readUInt32LE(p + 12), bits: b.readUInt16LE(p + 22) };
    if (id === 'data') data = b.subarray(p + 8, p + 8 + Math.min(size, b.length - p - 8));
    p += 8 + size + (size & 1);
  }
  const bps = fmt.bits / 8, frames = Math.floor(data.length / (bps * fmt.ch));
  const x = new Float64Array(frames);
  for (let i = 0; i < frames; i++) {
    const o = i * bps * fmt.ch;
    x[i] = fmt.tag === 3 || fmt.bits === 32 && fmt.tag === 0xfffe ? data.readFloatLE(o) : fmt.bits === 32 ? data.readInt32LE(o) / 2 ** 31 : data.readInt16LE(o) / 32768;
  }
  const skip = Math.floor(fmt.rate * 0.5);
  let peak = 0, crossings = 0, n = 0;
  for (let i = skip; i < frames; i++) {
    peak = Math.max(peak, Math.abs(x[i]));
    if (i > skip && x[i - 1] < 0 && x[i] >= 0) crossings++;
    n++;
  }
  const freq = n ? (crossings * fmt.rate) / n : 0;
  const k = 2 * Math.cos((2 * Math.PI * freq) / fmt.rate), thr = Math.max(0.02, peak * 0.08);
  let clicks = 0, gaps = 0, run = 0, firstClickAt = -1;
  for (let i = skip + 2; i < frames; i++) {
    if (Math.abs(x[i] - (k * x[i - 1] - x[i - 2])) > thr) { if (firstClickAt < 0) firstClickAt = +(i / fmt.rate).toFixed(3); clicks++; }
    if (Math.abs(x[i]) < 1e-5) { if (++run === Math.round(fmt.rate * 0.002)) gaps++; } else run = 0;
  }
  return { seconds: +(n / fmt.rate).toFixed(2), freq: +freq.toFixed(1), peakDb: +(20 * Math.log10(peak || 1e-9)).toFixed(2), clicks, gaps, firstClickAt, total: +(frames / fmt.rate).toFixed(3) };
}

// ---- engine connection ----
let ws, state = null, meters = null, rid = 1;
const pending = new Map();
const call = (method, params = {}) =>
  new Promise((resolve, reject) => {
    const i = rid++;
    pending.set(i, { resolve, reject });
    ws.send(JSON.stringify({ id: i, method, params }));
  });
const node = (pred) => state.nodes.find(pred);
async function levelDb(id, ms = 1200) {
  await sleep(ms);
  let peak = 0;
  for (let i = 0; i < 8; i++) {
    await sleep(80);
    const n = node((x) => x.id === id);
    peak = Math.max(peak, meters?.[n.meter.offset] ?? 0);
  }
  return +(20 * Math.log10(peak || 1e-7)).toFixed(2);
}

async function main() {
  // PipeWire session (headless) + engine inside WSL.
  wsl('pgrep -x pipewire >/dev/null || (setsid pipewire >/tmp/pw.log 2>&1 &); sleep 1; pgrep -x wireplumber >/dev/null || (setsid wireplumber >/tmp/wp.log 2>&1 &); sleep 2; pkill -x tonesphere-engine; sleep 1; rm -f $XDG_RUNTIME_DIR/tonesphere/engine.json; true');
  wsl(`pw-cli destroy ts-test-speakers >/dev/null 2>&1; true`);
  wslBg(`setsid ${buildDir}/bin/tonesphere-engine >/tmp/engine.out 2>&1`);
  let disc = null;
  for (let i = 0; i < 40 && !disc; i++) {
    await sleep(250);
    try { disc = JSON.parse(wsl('cat $XDG_RUNTIME_DIR/tonesphere/engine.json 2>/dev/null')); } catch {}
  }
  if (!disc) throw new Error('engine did not start in WSL');
  // WSL forwards new localhost ports to Windows after a short delay: retry the connection.
  for (let attempt = 0; ; attempt++) {
    ws = new WebSocket(`ws://127.0.0.1:${disc.port}/?token=${disc.token}`);
    const ok = await new Promise((r) => ((ws.onopen = () => r(true)), (ws.onerror = () => r(false))));
    if (ok) break;
    if (attempt >= 20) throw new Error(`cannot reach the WSL engine on port ${disc.port}`);
    await sleep(500);
  }
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) { meters = new Float32Array(ev.data, 12, new DataView(ev.data).getUint32(8, true)).slice(); return; }
    const m = JSON.parse(ev.data);
    if (m.event === 'state') state = m.data;
    else if (pending.has(m.id)) { const p = pending.get(m.id); pending.delete(m.id); m.error ? p.reject(new Error(m.error.message)) : p.resolve(m.result); }
  };
  while (!state) await sleep(20);
  check('engine runs on Linux with the PipeWire backend', state.status.platform === 'linux', { platform: state.status.platform, nodes: state.nodes.length });

  // 1) Virtual devices are PipeWire nodes.
  const status = wsl('wpctl status');
  const vout = node((n) => n.id === 'pw:vout:1'), vmic = node((n) => n.id === 'pw:vmic:1');
  check('virtual devices exist for other apps', !!vout && !!vmic && /ToneSphere Output 1/.test(status) && /ToneSphere Mic 1/.test(status), {
    vout: vout?.bin, vmic: vmic?.bin,
  });

  // 2) Mic path: tone -> ToneSphere Mic 1 -> pw-record.
  await call('tone.set', { enabled: true, freq: 1000, levelDb: -12, target: vmic.id });
  await sleep(1500);
  wsl(`timeout 5 pw-record --target tonesphere.mic.1 --rate 48000 --channels 2 --format f32 ${tmpWsl}/mic.wav; true`);
  const mic = analyseWav(path.join(tmpWin, 'mic.wav'));
  check('mic path: an app records the tone', Math.abs(mic.freq - 1000) < 5 && Math.abs(mic.peakDb + 12) < 0.3, mic);
  check('mic path: clean (no clicks or gaps)', mic.clicks === 0 && mic.gaps === 0, mic);
  for (const r of [...state.routes]) await call('route.disconnect', { id: r.id });

  // 3) Output path: pw-play -> ToneSphere Output 1 -> mix.
  writeSineWav(path.join(tmpWin, 'sine440.wav'), 440, -6, 40); // outlasts the output + app checks
  const bus = await call('bus.create', { name: 'Linux test', channels: 2 });
  await call('route.connect', { from: vout.id, to: bus.id });
  const player = wslBg(`pw-play --target tonesphere.output.1 ${tmpWsl}/sine440.wav`);
  await sleep(1500);
  const levels = [];
  for (let i = 0; i < 16; i++) { await sleep(250); levels.push(await levelDb(bus.id, 0)); }
  const vo = node((n) => n.id === vout.id);
  check('output path: app audio reaches the mix', Math.max(...levels) > -7 && Math.max(...levels) < -5, { max: Math.max(...levels) });
  check('output path: continuous, no dropouts', Math.min(...levels) > -9 && (vo.xruns ?? 0) === 0, { min: Math.min(...levels), xruns: vo.xruns, diag: vo.diag });

  // 5) App capture: the playing pw-play stream appears in Application outputs.
  await sleep(500);
  const app = node((n) => n.bin === 'app-out' && /pw-(play|cat)/i.test(n.id + n.name));
  check('app discovery: a playing app appears in Application outputs', !!app, { app: app?.name, id: app?.id });
  if (app) {
    const bus2 = await call('bus.create', { name: 'App capture', channels: 2 });
    await call('route.connect', { from: app.id, to: bus2.id });
    const lvl = await levelDb(bus2.id, 2000);
    check('app capture: the app sound reaches a mix', lvl > -7 && lvl < -5, { lvl });
    await call('bus.delete', { id: bus2.id });
  }
  player.kill();
  wsl('pkill -x pw-play; pkill -x pw-cat; true');
  await call('bus.delete', { id: bus.id });

  // 4) Physical output: tone -> a sink device -> its monitor.
  wsl(`pw-cli create-node adapter '{ factory.name=support.null-audio-sink node.name=ts-test-speakers node.description="Test speakers" media.class=Audio/Sink object.linger=true audio.position=[FL FR] }' >/dev/null`);
  let spk = null;
  for (let i = 0; i < 20 && !spk; i++) { await sleep(250); spk = node((n) => n.id === 'pw:out:ts-test-speakers' && n.state !== 'unavailable'); }
  check('a new sink appears in Physical outputs (hot-plug)', !!spk && spk.bin === 'physical-out', { spk: spk?.name, bin: spk?.bin });
  if (spk) {
    await call('tone.set', { enabled: true, freq: 1000, levelDb: -12, target: spk.id });
    await sleep(1500);
    wsl(`timeout 5 pw-record --target ts-test-speakers -P '{ stream.capture.sink=true }' --rate 48000 --channels 2 --format f32 ${tmpWsl}/spk.wav; true`);
    const out = analyseWav(path.join(tmpWin, 'spk.wav'));
    check('physical output: tone plays on the device', Math.abs(out.freq - 1000) < 5 && Math.abs(out.peakDb + 12) < 0.3 && out.gaps === 0, out);
    // 6) Unplug: the device goes away, the engine keeps running.
    wsl('pw-cli destroy ts-test-speakers >/dev/null 2>&1; true');
    let gone = null;
    for (let i = 0; i < 20; i++) { await sleep(250); gone = node((n) => n.id === spk.id); if (!gone || gone.state === 'unavailable') break; }
    const alive = await call('engine.getState').then(() => true, () => false);
    check('device removal is handled', (!gone || gone.state === 'unavailable') && alive, { state: gone?.state ?? 'removed', alive });
  }
  await call('tone.set', { enabled: false });
  for (const r of [...state.routes]) await call('route.disconnect', { id: r.id });
}

main()
  .catch((e) => check('harness', false, e.message))
  .finally(() => {
    try { wsl('pkill -x tonesphere-engine; pkill -x pw-play; true'); } catch {}
    const failed = results.filter((r) => !r.ok).length;
    console.log(`\n${results.length - failed}/${results.length} passed`);
    ws?.close();
    process.exit(failed ? 1 : 0);
  });
