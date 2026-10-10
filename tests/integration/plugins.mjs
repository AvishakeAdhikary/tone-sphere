// VST3 hosting test against the running engine, using the NexTone Test Gain plugin
// (build/x64-release/bin/test-plugins): scan, insert on a mix (-6 dB), open the plugin
// window and move its slider (edit reaches the audio thread: +12 dB), save/restore the
// plugin state through a scene, close the window, remove the plugin.
//   node tests/integration/plugins.mjs
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', '..');
const pluginDir = path.join(root, 'build', 'x64-release', 'bin', 'test-plugins');
const TEST_GAIN = '4E78546F6E65546573744761696E3031';
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
const results = [];
const check = (name, ok, detail) => {
  results.push({ name, ok, detail });
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${JSON.stringify(detail)}`);
};

// Win32 helpers (PowerShell + user32) to drive the plugin window like a user would.
const WIN32 = `Add-Type -TypeDefinition 'using System;using System.Text;using System.Runtime.InteropServices;public static class W{
[DllImport("user32.dll",CharSet=CharSet.Unicode)]public static extern IntPtr FindWindow(string c,string t);
[DllImport("user32.dll",CharSet=CharSet.Unicode)]public static extern IntPtr FindWindowEx(IntPtr p,IntPtr a,string c,string t);
[DllImport("user32.dll")]public static extern IntPtr SendMessage(IntPtr h,uint m,IntPtr w,IntPtr l);
[DllImport("user32.dll")]public static extern bool PostMessage(IntPtr h,uint m,IntPtr w,IntPtr l);
[DllImport("user32.dll")]public static extern bool IsWindowVisible(IntPtr h);}'`;
const ps = (body) => execFileSync('powershell', ['-NoProfile', '-Command', `${WIN32}\n${body}`], { encoding: 'utf8' }).trim();
const editorOpen = () => ps(`$h=[W]::FindWindow('NexTonePluginWindow','NexTone Test Gain'); [int]($h -ne [IntPtr]::Zero -and [W]::IsWindowVisible($h))`) === '1';
const moveSlider = (pos) =>
  ps(`$h=[W]::FindWindow('NexTonePluginWindow','NexTone Test Gain')
$v=[W]::FindWindowEx($h,[IntPtr]::Zero,'NexToneTestGainView',$null)
$s=[W]::FindWindowEx($v,[IntPtr]::Zero,'msctls_trackbar32',$null)
[void][W]::SendMessage($s,0x405,[IntPtr]1,[IntPtr]${pos})   # TBM_SETPOS
[void][W]::SendMessage($v,0x114,[IntPtr]8,$s)               # WM_HSCROLL / TB_ENDTRACK
[int]($s -ne [IntPtr]::Zero)`);
const closeEditor = () => ps(`$h=[W]::FindWindow('NexTonePluginWindow','NexTone Test Gain'); [void][W]::PostMessage($h,0x10,[IntPtr]::Zero,[IntPtr]::Zero)`);

async function levelDb(nodeId, ms = 900) {
  await sleep(ms);
  const n = state.nodes.find((x) => x.id === nodeId);
  let peak = 0;
  for (let i = 0; i < 6; i++) {
    await sleep(60);
    peak = Math.max(peak, meters?.[n.meter.offset] ?? 0);
  }
  return 20 * Math.log10(peak || 1e-7);
}

async function main() {
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
  while (!state) await sleep(20);
  for (const r of [...state.routes]) await call('route.disconnect', { id: r.id });
  const prevFolders = state.settings.pluginFolders ?? [];

  // 1) Scan finds the test plugin (out-of-process scanner).
  await call('engine.setSettings', { pluginFolders: [pluginDir] });
  for (let i = 0; i < 100 && (state.pluginsScanning || !state.plugins.some((p) => p.id === TEST_GAIN)); i++) await sleep(200);
  const plug = state.plugins.find((p) => p.id === TEST_GAIN);
  check('scan lists the test plugin', !!plug, { plugins: state.plugins.map((p) => p.name) });

  // 2) Insert on a mix fed by the tone: level drops by 6 dB.
  const bus = await call('bus.create', { name: 'Plugin test', channels: 2 });
  await call('tone.set', { enabled: true, freq: 1000, levelDb: -12, target: bus.id });
  const dry = await levelDb(bus.id, 1200);
  const fx = await call('fx.add', { nodeId: bus.id, type: 'vst3', pluginId: TEST_GAIN });
  const wet = await levelDb(bus.id);
  check('plugin processes audio (-6 dB)', Math.abs(wet - dry + 6) < 0.3, { dry: +dry.toFixed(2), wet: +wet.toFixed(2), fx: fx.name });

  // 3) Plugin window: opens, and an edit in it reaches the audio thread (+12 dB).
  await call('fx.openEditor', { nodeId: bus.id, fxId: fx.id });
  await sleep(500);
  check('plugin window opens', editorOpen(), {});
  const moved = moveSlider(1000);
  const loud = await levelDb(bus.id);
  check('edit in the plugin window changes the sound (+12 dB)', moved === '1' && Math.abs(loud - dry - 12) < 0.3, { loud: +loud.toFixed(2) });

  // 4) State survives a scene save/load (plugin is recreated from saved state).
  const scene = await call('scene.save', { name: 'Plugin state' });
  await call('fx.remove', { nodeId: bus.id, fxId: fx.id });
  await sleep(300);
  const removed = await levelDb(bus.id);
  await call('scene.load', { id: scene.id });
  const restored = await levelDb(bus.id, 1200);
  const fxAfter = state.nodes.find((n) => n.id === bus.id)?.fx ?? [];
  check('plugin and its settings restore from a scene', fxAfter.length === 1 && Math.abs(restored - loud) < 0.3 && Math.abs(removed - dry) < 0.3, {
    removed: +removed.toFixed(2), restored: +restored.toFixed(2), fx: fxAfter.map((f) => f.name),
  });

  // 5) Closing the window keeps the engine and the plugin running.
  if (!editorOpen()) await call('fx.openEditor', { nodeId: bus.id, fxId: fxAfter[0].id });
  await sleep(300);
  closeEditor();
  await sleep(500);
  const still = await levelDb(bus.id, 300);
  check('closing the plugin window is clean', !editorOpen() && Math.abs(still - loud) < 0.3, { still: +still.toFixed(2) });

  // 6) Unknown plugin ids fail with a clear message.
  let msg = '';
  try {
    await call('fx.add', { nodeId: bus.id, type: 'vst3', pluginId: '00000000000000000000000000000000' });
  } catch (e) {
    msg = e.message;
  }
  check('unknown plugin is refused clearly', /not installed/i.test(msg), { msg });

  await call('tone.set', { enabled: false });
  await call('scene.delete', { id: scene.id });
  await call('bus.delete', { id: bus.id });
  await call('engine.setSettings', { pluginFolders: prevFolders });
}

main()
  .catch((e) => check('harness', false, e.message))
  .finally(() => {
    const failed = results.filter((r) => !r.ok).length;
    console.log(`\n${results.length - failed}/${results.length} passed`);
    ws.close();
    process.exit(failed ? 1 : 0);
  });
