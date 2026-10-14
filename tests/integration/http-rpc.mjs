// HTTP JSON-RPC for scripts, against the running engine: POST /rpc with the discovery token
// answers like the WebSocket does; a wrong token, a request from a web page (Origin header) and
// broken JSON are refused clearly.
//   node tests/integration/http-rpc.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const disc = JSON.parse(
  fs.readFileSync(process.env.TONESPHERE_DISCOVERY ?? path.join(process.env.LOCALAPPDATA ?? os.homedir(), 'ToneSphere', 'engine.json'), 'utf8'),
);
const url = `http://127.0.0.1:${disc.port}/rpc`;
let failures = 0;
const check = (name, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${JSON.stringify(detail)}`);
  if (!ok) failures++;
};
const post = async (body, headers = {}) => {
  const r = await fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json', ...headers }, body });
  return { status: r.status, json: await r.json() };
};

const ok = await post(JSON.stringify({ id: 7, method: 'engine.getState', params: {} }), { Authorization: `Bearer ${disc.token}` });
check('getState over HTTP with the token', ok.status === 200 && ok.json.id === 7 && ok.json.result?.version === disc.version, {
  status: ok.status,
  version: ok.json.result?.version,
  nodes: ok.json.result?.nodes?.length,
});

const q = await fetch(`${url}?token=${disc.token}`, { method: 'POST', body: JSON.stringify({ id: 1, method: 'engine.getState' }) });
check('token in the query string works too', q.status === 200 && !!(await q.json()).result, { status: q.status });

const call = await post(JSON.stringify({ id: 2, method: 'route.connect', params: { from: 'missing', to: 'missing' } }), { Authorization: `Bearer ${disc.token}` });
check('errors come back as JSON-RPC errors', call.status === 200 && !!call.json.error?.message, { error: call.json.error?.message });

const bad = await post(JSON.stringify({ id: 3, method: 'engine.getState' }), { Authorization: 'Bearer wrong' });
check('a wrong token is refused', bad.status === 403, { status: bad.status });

const web = await post(JSON.stringify({ id: 4, method: 'engine.getState' }), { Authorization: `Bearer ${disc.token}`, Origin: 'https://example.com' });
check('a request from a web page is refused', web.status === 403, { status: web.status });

const broken = await post('{not json', { Authorization: `Bearer ${disc.token}` });
check('broken JSON gets a parse error', broken.status === 200 && broken.json.error?.code === -32700, { error: broken.json.error });

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
