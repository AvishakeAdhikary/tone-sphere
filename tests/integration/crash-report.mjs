// Crash reports: an engine that faults leaves a readable report (and on Windows a minidump) in
// <data dir>/crashes, which the app shows on its next start. Runs the engine's --crash-test in a
// throwaway data folder, so it doesn't touch a running ToneSphere.
//   node tests/integration/crash-report.mjs [build\x64-release\bin]
import { spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const bin = path.resolve(process.argv[2] ?? 'build/x64-release/bin');
const exe = path.join(bin, process.platform === 'win32' ? 'tonesphere-engine.exe' : 'tonesphere-engine');
const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'ts-crash-'));
let failures = 0;
const check = (name, ok, detail) => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  ${JSON.stringify(detail)}`);
  if (!ok) failures++;
};

const r = spawnSync(exe, ['--crash-test'], { env: { ...process.env, TONESPHERE_DATA_DIR: dataDir }, timeout: 30_000 });
check('the engine ends with a fault, not a clean exit', r.status !== 0, { status: r.status, signal: r.signal });
const crashes = path.join(dataDir, 'crashes');
const files = fs.existsSync(crashes) ? fs.readdirSync(crashes) : [];
const txt = files.find((f) => /^crash-\d{8}-\d{6}\.txt$/.test(f));
check('a readable crash report is written', !!txt, { files });
const text = txt ? fs.readFileSync(path.join(crashes, txt), 'utf8') : '';
check('the report names the version and the fault', /ToneSphere engine \d+\.\d+\.\d+ stopped unexpectedly/.test(text) && /(Fault: 0xC0000005|Signal: 11)/.test(text), {
  text: text.split(/\r?\n/).slice(0, 3),
});
if (process.platform === 'win32') {
  const dmp = files.find((f) => f.endsWith('.dmp'));
  const size = dmp ? fs.statSync(path.join(crashes, dmp)).size : 0;
  // A minidump starts with "MDMP".
  const magic = dmp ? fs.readFileSync(path.join(crashes, dmp)).subarray(0, 4).toString('latin1') : '';
  check('a minidump is written next to it', magic === 'MDMP' && size > 10_000, { dmp, size });
}
fs.rmSync(dataDir, { recursive: true, force: true });
console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
