#!/usr/bin/env node
// nextone-ctl — command-line control of the running NexTone engine.
//   node tools/nextone-ctl.mjs state                 # summary of devices and connections
//   node tools/nextone-ctl.mjs json                  # full state as JSON
//   node tools/nextone-ctl.mjs call <method> [json]  # raw RPC, e.g. call route.connect '{"from":"tone","to":"..."}'
//   node tools/nextone-ctl.mjs connect <from> <to>   # ids or unique name fragments
//   node tools/nextone-ctl.mjs disconnect <from> <to>
//   node tools/nextone-ctl.mjs tone on|off [freq] [levelDb] [target]
//   node tools/nextone-ctl.mjs meters [seconds]      # prints live peak levels
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

function discoveryPath() {
  if (process.env.NEXTONE_DISCOVERY) return process.env.NEXTONE_DISCOVERY;
  if (process.platform === 'win32') {
    const local = process.env.LOCALAPPDATA ?? path.join(os.homedir(), 'AppData', 'Local');
    const desktop = path.join(local, 'NexTone', 'engine.json');
    if (fs.existsSync(desktop)) return desktop;
    // The Store (MSIX) build's AppData writes land in the package's private LocalCache.
    const pkgs = path.join(local, 'Packages');
    for (const p of fs.existsSync(pkgs) ? fs.readdirSync(pkgs) : []) {
      const f = path.join(pkgs, p, 'LocalCache', 'Local', 'NexTone', 'engine.json');
      if (/nextone/i.test(p) && fs.existsSync(f)) return f;
    }
    return desktop;
  }
  if (process.platform === 'darwin') return path.join(os.homedir(), 'Library', 'Application Support', 'NexTone', 'engine.json');
  return path.join(process.env.XDG_RUNTIME_DIR ?? path.join(os.homedir(), '.local', 'state'), 'nextone', 'engine.json');
}

const d = JSON.parse(fs.readFileSync(discoveryPath(), 'utf8'));
const ws = new WebSocket(`ws://127.0.0.1:${d.port}/?token=${d.token}`);
ws.binaryType = 'arraybuffer';
let state = null;
let nextId = 1;
const pending = new Map();
const meterListeners = [];

ws.onmessage = (ev) => {
  if (ev.data instanceof ArrayBuffer) {
    for (const l of meterListeners) l(ev.data);
    return;
  }
  const msg = JSON.parse(ev.data);
  if (msg.event === 'state') state = msg.data;
  else if (msg.event === 'log') console.error(`[engine ${msg.data.level}] ${msg.data.message}`);
  else if (pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id);
    pending.delete(msg.id);
    msg.error ? reject(new Error(msg.error.message)) : resolve(msg.result);
  }
};

function call(method, params) {
  const id = nextId++;
  ws.send(JSON.stringify({ id, method, params }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

function find(q) {
  const exact = state.nodes.find((n) => n.id === q);
  if (exact) return exact;
  const ql = q.toLowerCase();
  const hits = state.nodes.filter((n) => n.name.toLowerCase().includes(ql) || n.id.toLowerCase().includes(ql));
  if (hits.length !== 1) throw new Error(`"${q}" matches ${hits.length} nodes: ${hits.map((h) => `${h.name} [${h.bin}]`).join(', ')}`);
  return hits[0];
}

const db = (v) => (v <= 1e-6 ? '  -inf' : (20 * Math.log10(v)).toFixed(1).padStart(6));

async function main() {
  await new Promise((r, j) => {
    ws.onopen = r;
    ws.onerror = () => j(new Error('Cannot connect to the NexTone engine'));
  });
  while (!state) await new Promise((r) => setTimeout(r, 20));
  const [cmd = 'state', ...args] = process.argv.slice(2);

  if (cmd === 'json') {
    console.log(JSON.stringify(await call('engine.getState'), null, 2));
  } else if (cmd === 'state') {
    const s = await call('engine.getState');
    console.log(`Engine ${s.version} · ${s.status.sampleRate} Hz · block ${s.status.blockSize} · CPU ${s.status.cpuPercent.toFixed(1)}% · xruns ${s.status.xruns}`);
    for (const bin of ['physical-in', 'virtual-out', 'app-out', 'bus', 'physical-out', 'virtual-in', 'app-in']) {
      const ns = s.nodes.filter((n) => n.bin === bin);
      if (!ns.length) continue;
      console.log(`\n${bin}`);
      for (const n of ns) {
        const extra = n.latencyMs != null ? ` lat ${n.latencyMs.toFixed(1)}ms drift ${n.driftPpm?.toFixed(0)}ppm xruns ${n.xruns}` : '';
        console.log(`  ${n.state.padEnd(11)} ${n.name} (${n.kind}, ${n.channels}ch)${n.statusText ? ' — ' + n.statusText : ''}${extra}\n              ${n.id}`);
      }
    }
    console.log('\nconnections');
    for (const r of s.routes) {
      const a = s.nodes.find((n) => n.id === r.from)?.name ?? r.from;
      const b = s.nodes.find((n) => n.id === r.to)?.name ?? r.to;
      console.log(`  ${a} -> ${b}  ${r.gainDb} dB${r.muted ? ' (muted)' : ''}`);
    }
  } else if (cmd === 'call') {
    console.log(JSON.stringify(await call(args[0], args[1] ? JSON.parse(args[1]) : {}), null, 2));
  } else if (cmd === 'connect') {
    const r = await call('route.connect', { from: find(args[0]).id, to: find(args[1]).id, mode: args[2] ?? 'auto' });
    console.log('connected', r.id);
  } else if (cmd === 'disconnect') {
    const a = find(args[0]).id, b = find(args[1]).id;
    const r = state.routes.find((x) => x.from === a && x.to === b);
    if (!r) throw new Error('not connected');
    await call('route.disconnect', { id: r.id });
    console.log('disconnected');
  } else if (cmd === 'tone') {
    const on = args[0] !== 'off';
    const p = { enabled: on, freq: Number(args[1] ?? 1000), levelDb: Number(args[2] ?? -12) };
    if (args[3]) p.target = find(args[3]).id;
    await call('tone.set', p);
    console.log(`tone ${on ? 'on' : 'off'}`);
  } else if (cmd === 'meters') {
    const secs = Number(args[0] ?? 3);
    meterListeners.push((buf) => {
      const dv = new DataView(buf);
      const count = dv.getUint32(8, true);
      const vals = new Float32Array(buf, 12, count);
      const line = state.nodes
        .filter((n) => vals[n.meter.offset] > 1e-4)
        .map((n) => `${n.name}:${db(vals[n.meter.offset])}`)
        .join('  ');
      process.stdout.write(`\r${line.padEnd(150).slice(0, 150)}`);
    });
    await new Promise((r) => setTimeout(r, secs * 1000));
    process.stdout.write('\n');
  } else {
    throw new Error(`unknown command ${cmd}`);
  }
  ws.close();
}

main().catch((e) => {
  console.error(e.message);
  process.exit(1);
});
