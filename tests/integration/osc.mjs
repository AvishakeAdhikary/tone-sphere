// OSC remote control against the running engine: enables OSC, sends UDP messages like a
// control surface (TouchOSC, Stream Deck plugin, …) would, and checks the engine state.
//   node tests/integration/osc.mjs
import dgram from 'node:dgram';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const PORT = 9123;
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

// ---- OSC encoding
const pad = (b) => Buffer.concat([b, Buffer.alloc((4 - (b.length % 4)) % 4)]);
const str = (s) => pad(Buffer.concat([Buffer.from(s, 'utf8'), Buffer.alloc(1)]));
function osc(address, ...args) {
  const tags = ',' + args.map((a) => (typeof a === 'string' ? 's' : Number.isInteger(a) ? 'i' : 'f')).join('');
  const parts = [str(address), str(tags)];
  for (const a of args) {
    if (typeof a === 'string') parts.push(str(a));
    else {
      const b = Buffer.alloc(4);
      Number.isInteger(a) ? b.writeInt32BE(a) : b.writeFloatBE(a);
      parts.push(b);
    }
  }
  return Buffer.concat(parts);
}
const udp = dgram.createSocket('udp4');
const send = (msg) => new Promise((r) => udp.send(msg, PORT, '127.0.0.1', r));
const node = (nid) => state.nodes.find((n) => n.id === nid);
async function until(pred, ms = 2000) {
  for (let t = 0; t < ms; t += 50) {
    if (pred()) return true;
    await sleep(50);
  }
  return false;
}

async function main() {
  await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
  while (!state) await sleep(20);
  const prevPort = state.settings.oscPort ?? 0;
  await call('engine.setSettings', { oscPort: PORT });
  await sleep(2600); // the remote-control worker picks up settings every 2 s
  const bus = await call('bus.create', { name: 'OSC test mix', channels: 2 });

  await send(osc('/nextone/volume', 'OSC test mix', 0.5));
  check('volume by name (fader 0.5 = -27 dB)', await until(() => Math.abs((node(bus.id)?.gainDb ?? 0) + 27) < 0.01), { gainDb: node(bus.id)?.gainDb });

  await send(osc('/nextone/gain', bus.id, -12.5));
  check('gain by id in dB', await until(() => Math.abs((node(bus.id)?.gainDb ?? 0) + 12.5) < 0.01), { gainDb: node(bus.id)?.gainDb });

  await send(osc('/nextone/mute', 'OSC test mix'));
  const muted = await until(() => node(bus.id)?.muted === true);
  await send(osc('/nextone/mute', 'OSC test mix'));
  const unmuted = await until(() => node(bus.id)?.muted === false);
  check('mute toggles', muted && unmuted, { muted, unmuted });

  await send(osc('/nextone/solo', 'osc test', 1)); // partial, case-insensitive name
  check('solo set by partial name', await until(() => node(bus.id)?.solo === true), { solo: node(bus.id)?.solo });
  await send(osc('/nextone/solo', 'osc test', 0));
  await until(() => node(bus.id)?.solo === false);

  await send(Buffer.from('not osc at all'));
  await send(osc('/nextone/volume', 'No such device', 0.9));
  await sleep(300);
  const alive = await call('engine.getState').then(() => true, () => false);
  check('bad packets are ignored', alive && Math.abs(node(bus.id).gainDb + 12.5) < 0.01, { alive });

  await call('bus.delete', { id: bus.id });
  await call('engine.setSettings', { oscPort: prevPort });
}

main()
  .catch((e) => check('harness', false, e.message))
  .finally(() => {
    const failed = results.filter((r) => !r.ok).length;
    console.log(`\n${results.length - failed}/${results.length} passed`);
    udp.close();
    ws.close();
    process.exit(failed ? 1 : 0);
  });
