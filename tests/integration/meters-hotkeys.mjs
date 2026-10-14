// Idle meters (app level visible without routing) and global hotkeys (mute-mic toggle).
//   node tests/integration/meters-hotkeys.mjs
import { spawn, execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const probe = path.join(root, 'build', 'x64-release', 'bin', 'audio-probe.exe');
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
// Presses Ctrl+Shift+F9 through SendInput (PowerShell + user32).
function pressHotkey() {
  const ps = `
Add-Type -TypeDefinition 'using System;using System.Runtime.InteropServices;public static class K{[DllImport("user32.dll")]public static extern void keybd_event(byte v,byte s,uint f,UIntPtr e);}'
[K]::keybd_event(0x11,0,0,[UIntPtr]::Zero);[K]::keybd_event(0x10,0,0,[UIntPtr]::Zero);[K]::keybd_event(0x78,0,0,[UIntPtr]::Zero)
Start-Sleep -Milliseconds 60
[K]::keybd_event(0x78,0,2,[UIntPtr]::Zero);[K]::keybd_event(0x10,0,2,[UIntPtr]::Zero);[K]::keybd_event(0x11,0,2,[UIntPtr]::Zero)`;
  execFileSync('powershell', ['-NoProfile', '-Command', ps]);
}

async function main() {
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
  while (!state) await sleep(20);
  for (const r of [...state.routes]) await call('route.disconnect', { id: r.id });

  // Idle meter: an unrouted app playing a tone shows its level.
  const player = spawn(probe, ['play', 'CABLE Input', '500', '-10', '12']);
  let level = -140, app;
  for (let i = 0; i < 40; i++) {
    await sleep(250);
    app = state.nodes.find((n) => n.app?.exe === 'audio-probe.exe' && n.bin === 'app-out');
    if (app && meters) level = Math.max(level, 20 * Math.log10(meters[app.meter.offset] || 1e-7));
    if (level > -20) break;
  }
  check('idle app meter shows level without routing', level > -14 && level < -6, { app: app?.name, level, routes: state.routes.length });
  player.kill();

  // Hotkey: Ctrl+Shift+F9 -> mute-mic toggle.
  const prev = state.settings.hotkeys;
  await call('engine.setSettings', { hotkeys: { ...prev, 'mute-mic': 'Ctrl+Shift+F9' } });
  await sleep(2600); // shortcut map reloads every 2 s
  const mics = () => state.nodes.filter((n) => n.bin === 'physical-in' && n.kind !== 'tone' && n.state !== 'unavailable');
  const before = mics().every((n) => n.muted);
  pressHotkey();
  await sleep(800);
  const after1 = mics().every((n) => n.muted);
  pressHotkey();
  await sleep(800);
  const after2 = mics().every((n) => n.muted);
  check('hotkey toggles microphone mute', mics().length > 0 && after1 !== before && after2 === before, { mics: mics().length, before, after1, after2 });
  await call('engine.setSettings', { hotkeys: prev });
}

main()
  .catch((e) => check('harness', false, e.message))
  .finally(() => {
    const failed = results.filter((r) => !r.ok).length;
    console.log(`\n${results.length - failed}/${results.length} passed`);
    ws.close();
    process.exit(failed ? 1 : 0);
  });
