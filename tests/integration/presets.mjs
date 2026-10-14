// Portable presets against the running engine: import (with a device that isn't connected),
// placeholder kept, export → re-import round trip, forget a disconnected device.
//   node tests/integration/presets.mjs
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

const PRESET = {
  format: 'nextone-preset',
  version: 1,
  name: 'Preset test',
  devices: [
    { key: 'cable', label: 'Cable (apps play into it)', bin: 'virtual-out', kind: 'compat', match: ['CABLE Input (VB-Audio Virtual Cable)', 'CABLE Input'] },
    { key: 'ghost', label: 'Imaginary Speaker 9000', bin: 'physical-out', kind: 'wasapi', match: ['Imaginary Speaker 9000'], settings: { bufferMs: 4 } },
  ],
  mixes: [{ key: 'mix', name: 'Preset test mix', channels: 2, gainDb: -3 }],
  routes: [
    { from: 'cable', to: 'mix', channels: [[0, 0, 0], [1, 1, 0]] },
    { from: 'mix', to: 'ghost', gainDb: -6 },
  ],
};

async function main() {
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
  while (!state) await sleep(20);
  const before = { scenes: state.scenes.map((s) => s.id), routes: state.routes.length };

  const r = await call('preset.import', { preset: PRESET });
  await sleep(500);
  const ghost = state.nodes.find((n) => n.name === 'Imaginary Speaker 9000');
  const mix = state.nodes.find((n) => n.bin === 'bus' && n.name === 'Preset test mix');
  const cable = state.nodes.find((n) => n.kind === 'compat' && n.bin === 'virtual-out');
  check('import connects present devices and waits for missing ones', r.connections === 2 && r.waiting.includes('Imaginary Speaker 9000'), r);
  check('missing device is a placeholder that keeps its connection', ghost?.state === 'unavailable' && state.routes.some((x) => x.from === mix?.id && x.to === ghost?.id), {
    ghost: ghost?.id, state: ghost?.state,
  });
  check('mix created with preset settings', mix && Math.abs(mix.gainDb + 3) < 0.01, { gainDb: mix?.gainDb });
  check('present device matched by name', state.routes.some((x) => x.from === cable?.id && x.to === mix?.id), { cable: cable?.name });
  const scene = state.scenes.find((s) => s.name === 'Preset test');
  check('preset kept as the active scene', scene && state.activeScene === scene.id, { scene: scene?.name });

  const exported = await call('preset.export', { name: 'Round trip' });
  const ok = exported.format === 'nextone-preset' && exported.devices.length === 2 && exported.mixes.some((m) => m.name === 'Preset test mix') && exported.routes.length === 2;
  const ghostRef = exported.devices.find((d) => d.label === 'Imaginary Speaker 9000');
  check('export describes devices portably (no machine ids)', ok && ghostRef?.match?.includes('Imaginary Speaker 9000') && !JSON.stringify(exported).includes('{0.0.'), {
    devices: exported.devices.map((d) => ({ label: d.label, match: d.match })),
  });

  const again = await call('preset.import', { preset: { ...exported, name: 'Preset test' } });
  await sleep(300);
  const mixes = state.nodes.filter((n) => n.bin === 'bus' && n.name === 'Preset test mix');
  const ghosts = state.nodes.filter((n) => n.name === 'Imaginary Speaker 9000');
  check('re-import is idempotent (no duplicate mixes or placeholders)', again.connections === 2 && mixes.length === 1 && ghosts.length === 1, {
    mixes: mixes.length, ghosts: ghosts.length, connections: again.connections,
  });

  let refused = '';
  try {
    await call('preset.import', { preset: { name: 'x' } });
  } catch (e) {
    refused = e.message;
  }
  check('non-presets are refused clearly', /isn't a NexTone preset/.test(refused), { refused });

  await call('node.forget', { id: ghosts[0].id });
  await sleep(300);
  check('a disconnected device can be forgotten', !state.nodes.some((n) => n.name === 'Imaginary Speaker 9000') && !state.routes.some((x) => x.to === ghosts[0].id), {});

  // Clean up: the test's routes, mix and scenes.
  for (const x of [...state.routes]) await call('route.disconnect', { id: x.id });
  for (const m of mixes) await call('bus.delete', { id: m.id });
  for (const s of state.scenes) if (!before.scenes.includes(s.id)) await call('scene.delete', { id: s.id });
}

main()
  .catch((e) => check('harness', false, e.message))
  .finally(() => {
    const failed = results.filter((r) => !r.ok).length;
    console.log(`\n${results.length - failed}/${results.length} passed`);
    ws.close();
    process.exit(failed ? 1 : 0);
  });
