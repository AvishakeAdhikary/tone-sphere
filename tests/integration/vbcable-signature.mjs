// "Install drivers" only runs VB-CABLE's setup when it really is VB-Audio's signed, unmodified file.
// Downloads the official package (as the app does), then asks tonesphere-setup to verify:
//   the genuine setup (accepted), a copy with one byte changed (refused), and a validly signed
//   file from another publisher (refused). Needs no administrator rights and installs nothing.
//   node tests/integration/vbcable-signature.mjs [build\x64-release\bin]
import { execFileSync, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const bin = path.resolve(process.argv[2] ?? 'build/x64-release/bin');
const setupTool = path.join(bin, 'tonesphere-setup.exe');
const URL = 'https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip';
let failures = 0;
const check = (name, ok, detail = '') => {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
  if (!ok) failures++;
};
const verify = (file) => {
  const r = spawnSync(setupTool, ['verify-signature', file], { encoding: 'utf8' });
  return { code: r.status, out: JSON.parse(r.stdout.trim().split('\n').at(-1)) };
};

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ts-vbcable-test-'));
try {
  const res = await fetch(URL);
  check('official package downloads', res.ok, `${res.status}`);
  fs.writeFileSync(path.join(dir, 'pack.zip'), Buffer.from(await res.arrayBuffer()));
  fs.mkdirSync(path.join(dir, 'x'));
  execFileSync(path.join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'tar.exe'), ['-xf', path.join(dir, 'pack.zip'), '-C', path.join(dir, 'x')]);
  const genuine = path.join(dir, 'x', 'VBCABLE_Setup_x64.exe');
  check('package contains VBCABLE_Setup_x64.exe', fs.existsSync(genuine));

  const g = verify(genuine);
  check('genuine setup is accepted', g.code === 0 && g.out.ok, `${g.out.message}; signer: ${g.out.signer}`);

  const tampered = path.join(dir, 'tampered.exe');
  const bytes = fs.readFileSync(genuine);
  bytes[4096] ^= 0xff;
  fs.writeFileSync(tampered, bytes);
  const t = verify(tampered);
  check('a copy with one byte changed is refused', t.code !== 0 && !t.out.ok, t.out.message);

  // Validly signed, but by someone else: Microsoft's vswhere (VS Build Tools), else the setup tool's own
  // Windows component signed with an embedded signature.
  const candidates = [
    path.join(process.env['ProgramFiles(x86)'] ?? '', 'Microsoft Visual Studio', 'Installer', 'vswhere.exe'),
    path.join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell_ise.exe'),
    path.join(process.env.SystemRoot ?? 'C:\\Windows', 'explorer.exe'),
  ];
  const foreign = candidates.find((f) => fs.existsSync(f) && verifyTrusted(f));
  if (foreign) {
    const f = verify(foreign);
    check('a validly signed file from another publisher is refused', f.code !== 0 && /isn't signed by VB-Audio/.test(f.out.message), `${path.basename(foreign)}: ${f.out.signer}`);
  } else {
    console.log('SKIP  no embedded-signed file from another publisher found on this PC');
  }

  const missing = verify(path.join(dir, 'missing.exe'));
  check('a missing setup is refused', missing.code !== 0 && !missing.out.ok, missing.out.message);
} finally {
  fs.rmSync(dir, { recursive: true, force: true });
}

// An embedded Authenticode signature PowerShell trusts (catalog-signed system files don't count).
function verifyTrusted(file) {
  const r = spawnSync('powershell.exe', ['-NoProfile', '-Command', `(Get-AuthenticodeSignature '${file}').Status -eq 'Valid' -and (Get-AuthenticodeSignature '${file}').SignatureType -eq 'Authenticode'`], {
    encoding: 'utf8',
  });
  return r.stdout.trim() === 'True';
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed');
process.exit(failures ? 1 : 0);
