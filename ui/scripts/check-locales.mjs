// Every string the UI translates must be in every catalog (falling back to English is a bug).
//   node scripts/check-locales.mjs        (also run by the CI)
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..', 'src', 'renderer');
const files = [];
(function walk(d) {
  for (const e of fs.readdirSync(d, { withFileTypes: true })) {
    const p = path.join(d, e.name);
    if (e.isDirectory() && e.name !== 'locales') walk(p);
    else if (/\.tsx?$/.test(e.name) && e.name !== 'skins.ts' && e.name !== 'i18n.ts') files.push(p);
  }
})(root);
const keys = new Set(['Audio', 'Devices', 'Startup', 'Recording', 'Plugins', 'Shortcuts', 'Control', 'Appearance', 'About', 'Machine translation, not yet reviewed']);
const re = /\bt\(("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g;
for (const f of files) for (const m of fs.readFileSync(f, 'utf8').matchAll(re)) keys.add(m[1][0] === '"' ? JSON.parse(m[1]) : m[1].slice(1, -1).replace(/\\'/g, "'"));

let failed = 0;
for (const file of fs.readdirSync(path.join(root, 'locales'))) {
  const src = fs.readFileSync(path.join(root, 'locales', file), 'utf8');
  // Catalog keys: 'key': or "key": or bare identifiers at the start of an entry.
  const have = new Set();
  for (const m of src.matchAll(/^\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)"|([A-Za-z][\w]*)):/gm)) have.add((m[1] ?? m[2] ?? m[3]).replace(/\\'/g, "'"));
  const missing = [...keys].filter((k) => !have.has(k));
  const unused = [...have].filter((k) => !keys.has(k));
  console.log(`${file}: ${keys.size - missing.length}/${keys.size} strings${unused.length ? `, ${unused.length} unused` : ''}`);
  for (const k of missing) console.log(`  missing: ${k}`);
  for (const k of unused) console.log(`  unused:  ${k}`);
  failed += missing.length;
}
process.exit(failed ? 1 : 0);
