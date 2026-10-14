// Meters only flow to connections that want them: the app unsubscribes while its window is hidden
// (usually in the tray), and with nobody watching the engine skips meter work on the audio thread.
//   node tests/integration/meters-subscribe.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const disc = JSON.parse(fs.readFileSync(path.join(process.env.LOCALAPPDATA ?? os.homedir(), 'ToneSphere', 'engine.json'), 'utf8'));
const ws = new WebSocket(`ws://127.0.0.1:${disc.port}/?token=${disc.token}`);
ws.binaryType = 'arraybuffer';
let frames = 0, id = 1;
const pending = new Map();
ws.onmessage = (ev) => {
  if (ev.data instanceof ArrayBuffer) {
    frames++;
    return;
  }
  const m = JSON.parse(ev.data);
  if (pending.has(m.id)) {
    pending.get(m.id)(m);
    pending.delete(m.id);
  }
};
const call = (method, params = {}) =>
  new Promise((resolve) => {
    const i = id++;
    pending.set(i, resolve);
    ws.send(JSON.stringify({ id: i, method, params }));
  });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
const check = (name, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${JSON.stringify(detail)}`);
  if (!ok) failures++;
};
const countFor = async (ms) => {
  frames = 0;
  await sleep(ms);
  return frames;
};

await new Promise((r, j) => ((ws.onopen = r), (ws.onerror = () => j(new Error('engine not running')))));
const on = await countFor(1000);
check('meter frames arrive by default (~30 per second)', on >= 20 && on <= 40, { frames: on });
const r = await call('meters.subscribe', { on: false });
check('meters.subscribe answers', r.result === null && !r.error, r);
await sleep(100); // a frame may already be on its way
const off = await countFor(1000);
check('no meter frames while unsubscribed', off === 0, { frames: off });
await call('meters.subscribe', { on: true });
const back = await countFor(1000);
check('frames resume when subscribed again', back >= 20, { frames: back });
ws.close();
console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
