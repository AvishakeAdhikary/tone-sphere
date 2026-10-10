// Guitar Rig live scenario, run inside the test VM's desktop session (see vm/Test-GuitarRig.ps1):
//   guitar (stand-in) → NexTone → Guitar Rig 7 (NexTone ASIO) → NexTone → speakers + OBS (stand-ins)
// Checks the sound at both destinations, then that unplugging devices and restarting Guitar Rig
// keep the routing and the sound comes back by itself.
//   node guitar-rig-scenario.mjs <preset.json> <audio-probe.exe>
import { execFileSync, spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const [presetFile, probe] = process.argv.slice(2);
const GR = 'C:\\Program Files\\Native Instruments\\Guitar Rig 7\\Guitar Rig 7.exe';
const report = path.join(path.dirname(presetFile), 'guitar-rig-report.json');
const results = [];
const check = (name, ok, detail) => {
  results.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${JSON.stringify(detail)}`);
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let ws, state = null, id = 1;
const pending = new Map();
const call = (method, params = {}) =>
  new Promise((resolve, reject) => {
    const i = id++;
    pending.set(i, { resolve, reject });
    ws.send(JSON.stringify({ id: i, method, params }));
  });
const node = (nid) => state.nodes.find((n) => n.id === nid);
async function until(pred, ms) {
  for (let t = 0; t < ms; t += 250) {
    if (pred()) return true;
    await sleep(250);
  }
  return pred();
}
const capture = (dev, secs = 5) => JSON.parse(execFileSync(probe, ['capture', dev, String(secs)], { encoding: 'utf8' }).trim());
let player = null;
const play = () => {
  player?.kill();
  player = spawn(probe, ['play', 'NexTone Output 1', '220', '-18', '600'], { stdio: 'ignore' });
};
const audible = (c) => c.peakDb > -45 && c.gaps === 0;
const ps = (cmd) => execFileSync('powershell', ['-NoProfile', '-Command', cmd], { encoding: 'utf8' }).trim();

async function main() {
  const discPath = path.join(process.env.LOCALAPPDATA ?? os.homedir(), 'NexTone', 'engine.json');
  await until(() => fs.existsSync(discPath), 60000);
  const disc = JSON.parse(fs.readFileSync(discPath, 'utf8'));
  ws = new WebSocket(`ws://127.0.0.1:${disc.port}/?token=${disc.token}`);
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
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not reachable')))));
  await until(() => state && state.nodes.some((n) => n.kind === 'vad'), 30000);

  // 1) Import the preset before Guitar Rig runs: Guitar Rig's ASIO ports wait as placeholders.
  const imp = await call('preset.import', { preset: JSON.parse(fs.readFileSync(presetFile, 'utf8')) });
  check('preset imported; Guitar Rig not running yet, so its ports wait', imp.connections === 3 && imp.waiting.length === 2, imp);

  // 2) Start Guitar Rig (set to the NexTone ASIO driver): its ports bind to the preset.
  const t0 = Date.now();
  spawn(GR, [], { detached: true, stdio: 'ignore' }).unref();
  const rigIn = 'asioclient:in:guitar rig 7.exe', rigOut = 'asioclient:out:guitar rig 7.exe';
  const up = await until(() => node(rigIn)?.state !== 'unavailable' && node(rigOut)?.state !== 'unavailable' && node(rigIn) && node(rigOut), 180000);
  check('Guitar Rig opened NexTone ASIO and joined the preset', up && state.routes.length === 3, {
    seconds: Math.round((Date.now() - t0) / 1000), in: node(rigIn)?.state, out: node(rigOut)?.state, routes: state.routes.length,
  });
  if (!up) return;

  // 3) Play the "guitar" and listen at both destinations.
  play();
  await sleep(8000); // streams that just started settle in the first seconds
  const obs1 = capture('NexTone Mic 1'), spk1 = capture('NexTone Mic 2');
  check('OBS hears Guitar Rig', audible(obs1), obs1);
  check('speakers hear Guitar Rig', audible(spk1), spk1);
  const st = state.status;
  check('no dropouts in the engine', (st.xruns ?? 0) === 0, { xruns: st.xruns, cpu: st.cpuPercent });

  // 4) Unplug all virtual devices (driver disabled) while NexTone and Guitar Rig run, then plug them back in.
  //    Windows refuses to disable a device an app still has open, so the player app stops first
  //    (a real USB unplug can't be refused, so this is the gentler case for the device, not for NexTone).
  const routesBefore = JSON.stringify(state.routes.map((r) => [r.from, r.to]).sort());
  player.kill();
  await sleep(1500);
  for (let i = 0; i < 5; i++) {
    try {
      ps("Get-PnpDevice | Where-Object { $_.HardwareID -contains 'ROOT\\NexToneVad' } | Disable-PnpDevice -Confirm:$false");
      break;
    } catch {
      await sleep(2000);
    }
  }
  const gone = await until(() => !state.nodes.some((n) => n.kind === 'vad' && n.state !== 'unavailable'), 20000);
  const keptWhileGone = JSON.stringify(state.routes.map((r) => [r.from, r.to]).sort()) === routesBefore;
  check('unplugged: devices shown as disconnected, connections kept', gone && keptWhileGone, { routes: state.routes.length });
  ps("Get-PnpDevice | Where-Object { $_.HardwareID -contains 'ROOT\\NexToneVad' } | Enable-PnpDevice -Confirm:$false");
  const back = await until(() => state.nodes.filter((n) => n.kind === 'vad' && n.state !== 'unavailable').length >= 8, 40000);
  await sleep(3000);
  play(); // the player app lost its device too, like any app would
  await sleep(12000); // Windows re-creates all 8 endpoints, then buffers re-settle: judge the steady state
  const obs2 = capture('NexTone Mic 1'), spk2 = capture('NexTone Mic 2');
  const sameRoutes = JSON.stringify(state.routes.map((r) => [r.from, r.to]).sort()) === routesBefore;
  check('plugged back in: same connections, sound returns by itself', back && sameRoutes && audible(obs2) && audible(spk2), { obs: obs2, speakers: spk2 });

  // 5) Guitar Rig closed and reopened: its ports reconnect to the same routing.
  ps("Get-Process 'Guitar Rig 7' -ErrorAction SilentlyContinue | Stop-Process -Force");
  const closed = await until(() => node(rigOut)?.state === 'unavailable', 20000);
  check('Guitar Rig closed: its connections are kept', closed && state.routes.length === 3, { out: node(rigOut)?.state });
  spawn(GR, [], { detached: true, stdio: 'ignore' }).unref();
  const reopened = await until(() => node(rigIn)?.state !== 'unavailable' && node(rigOut)?.state !== 'unavailable', 180000);
  await sleep(8000);
  const obs3 = capture('NexTone Mic 1');
  check('Guitar Rig reopened: sound returns by itself', reopened && audible(obs3), obs3);
}

main()
  .catch((e) => check('harness', false, e.message))
  .finally(() => {
    player?.kill();
    const failed = results.filter((r) => !r.ok).length;
    console.log(`\n${results.length - failed}/${results.length} passed`);
    fs.writeFileSync(report, JSON.stringify(results, null, 2));
    ws?.close();
    process.exit(failed ? 1 : 0);
  });
