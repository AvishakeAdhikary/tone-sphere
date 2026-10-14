// The version a push to main releases, and the files that must carry it.
//
// Every green push to main is a release (.github/workflows/ci.yml): the latest `v*` tag's patch
// number goes up by one, or its minor or major number when the head commit's message says
// `[minor]` or `[major]`. The VERSION file holds the floor (set it ahead of the tags to start a new
// series) and is otherwise left alone in the repository: CI stamps the computed version into the
// build's own copy of every file that reports one.
//
//   node scripts/release-version.mjs next                     # print the version to release
//   node scripts/release-version.mjs stamp 1.0.1              # write it into this tree
//   node scripts/release-version.mjs notes 1.0.1 v1.0.0       # release notes (Markdown)
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const SEMVER = /^v?(\d+)\.(\d+)\.(\d+)$/;

export function parse(version) {
  const m = SEMVER.exec(String(version).trim());
  if (!m) throw new Error(`not a version: ${JSON.stringify(version)}`);
  return [Number(m[1]), Number(m[2]), Number(m[3])];
}

const cmp = (a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2];

/** The version after the newest of `tags`, bumped as `message` asks; never below `floor`. */
export function nextVersion(tags, floor, message) {
  const parsed = tags.filter((t) => SEMVER.test(t.trim())).map(parse);
  const lowest = parse(floor);
  if (!parsed.length) return lowest.join('.');
  const [major, minor, patch] = parsed.sort(cmp).at(-1);
  let bumped;
  if (message.includes('[major]')) bumped = [major + 1, 0, 0];
  else if (message.includes('[minor]')) bumped = [major, minor + 1, 0];
  else bumped = [major, minor, patch + 1];
  return (cmp(bumped, lowest) >= 0 ? bumped : lowest).join('.');
}

const git = (...args) => execFileSync('git', args, { cwd: ROOT, encoding: 'utf8' });

export function floorVersion(root = ROOT) {
  return fs.readFileSync(path.join(root, 'VERSION'), 'utf8').trim();
}

/** Writes `version` into every file that reports it, in this tree only. */
export function stamp(version, root = ROOT) {
  parse(version);
  fs.writeFileSync(path.join(root, 'VERSION'), `${version}\n`);
  for (const name of ['ui/package.json', 'ui/package-lock.json']) {
    const file = path.join(root, name);
    if (!fs.existsSync(file)) continue;
    const text = fs.readFileSync(file, 'utf8');
    // The package's own version: the first "version" key (and the root package in the lock file).
    let count = 0;
    const stamped = text.replace(/("version":\s*")[^"]+(")/g, (all, a, b) => (count++ < (name.endsWith('lock.json') ? 2 : 1) ? a + version + b : all));
    if (!count) throw new Error(`${name}: no version to stamp`);
    fs.writeFileSync(file, stamped);
  }
}

/** The release page: what changed, how to install, and the GPL source. */
export function notes(version, previous) {
  const span = previous ? `${previous}..HEAD` : 'HEAD';
  const log = git('log', span, '--no-merges', '--format=- %s').trim() || '- Maintenance.';
  return `## What's new

${log}

## Install

**Windows 10/11 (64-bit):** download \`ToneSphere-Setup-${version}.exe\` and run it. Windows
SmartScreen may say "Windows protected your PC" because the installer isn't code-signed yet:
choose **More info → Run anyway**. Prefer no installer? \`ToneSphere-${version}-win-x64-portable.zip\`
runs from any folder: unzip it and open \`ToneSphere.exe\`.

Virtual devices need a virtual cable: open **Settings › Devices › Install drivers…** and ToneSphere
downloads and installs the free VB-CABLE (donationware by VB-Audio) together with the ToneSphere ASIO
driver, after one administrator prompt.

**Linux (x86_64, PipeWire):** download \`ToneSphere-${version}-x86_64.AppImage\`, make it executable
(\`chmod +x\`) and open it.

\`ToneSphere-${version}.appx\` is the unsigned Microsoft Store package, for Store submission only.
\`SHA256SUMS.txt\` lists every file's checksum.

## Licence and source

ToneSphere is free software under the GNU GPL v3 (or later). The complete corresponding source of
this release is \`ToneSphere-${version}-source.zip\`.
`;
}

function main(argv) {
  const [command, a, b] = argv;
  if (command === 'next') {
    const tags = git('tag', '--list', 'v*').split(/\s+/).filter(Boolean);
    console.log(nextVersion(tags, floorVersion(), git('log', '-1', '--format=%B')));
  } else if (command === 'stamp' && a) {
    stamp(a);
  } else if (command === 'notes' && a) {
    process.stdout.write(notes(a, b || null));
  } else {
    console.error('usage: release-version.mjs next | stamp <version> | notes <version> [<previous tag>]');
    return 2;
  }
  return 0;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) process.exitCode = main(process.argv.slice(2));
