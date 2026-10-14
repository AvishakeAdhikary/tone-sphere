// Dev runner: Vite dev server + esbuild watch for main/preload + Electron.
// Pass --mock to also start the mock engine (no C++ engine needed).
import { spawn } from 'node:child_process';
import { createServer } from 'vite';

const useMock = process.argv.includes('--mock');
const children = [];
const run = (cmd, args, env = {}) => {
  const merged = { ...process.env, ...env };
  delete merged.ELECTRON_RUN_AS_NODE; // set by VS Code terminals; breaks Electron
  const p = spawn(cmd, args, { stdio: 'inherit', shell: true, env: merged });
  children.push(p);
  return p;
};

if (useMock) run('node', ['mock-engine/server.mjs']);
await new Promise((r) => run('node', ['scripts/build-electron.mjs']).on('exit', r));
run('node', ['scripts/build-electron.mjs', '--watch']);

const server = await createServer({ configFile: 'vite.config.ts' });
await server.listen();
const url = `http://localhost:${server.config.server.port}`;
const electron = run('npx', ['electron', '.'], { NEXTONE_DEV_URL: url });
electron.on('exit', () => {
  for (const c of children) c.kill();
  server.close();
  process.exit(0);
});
