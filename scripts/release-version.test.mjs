// node --test scripts/
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import { nextVersion, parse, stamp } from './release-version.mjs';

test('the first release of the new engine is the VERSION floor (v1.0.0 after the Python v0.2.1)', () => {
  assert.equal(nextVersion(['v0.1.0', 'v0.2.0', 'v0.2.1', 'gpl-source', 'legacy-python'], '1.0.0', 'Rename'), '1.0.0');
});

test('every later push bumps the patch; [minor] and [major] bump more', () => {
  const tags = ['v0.2.1', 'v1.0.0', 'v1.0.9', 'v1.0.10'];
  assert.equal(nextVersion(tags, '1.0.0', 'Fix a click'), '1.0.11');
  assert.equal(nextVersion(tags, '1.0.0', 'New mixer [minor]'), '1.1.0');
  assert.equal(nextVersion(tags, '1.0.0', 'Break things [major]'), '2.0.0');
});

test('with no tags at all, the floor is released', () => {
  assert.equal(nextVersion([], '1.0.0', ''), '1.0.0');
});

test('a floor ahead of the tags starts a new series', () => {
  assert.equal(nextVersion(['v1.4.2'], '2.0.0', ''), '2.0.0');
});

test('parse rejects anything that is not x.y.z', () => {
  assert.deepEqual(parse('v1.2.3'), [1, 2, 3]);
  assert.throws(() => parse('1.2'));
  assert.throws(() => parse('1.2.3-beta'));
});

test('stamp writes VERSION and the UI package versions, and nothing else', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ts-stamp-'));
  fs.mkdirSync(path.join(root, 'ui'));
  fs.writeFileSync(path.join(root, 'VERSION'), '1.0.0\n');
  fs.writeFileSync(path.join(root, 'ui/package.json'), '{\n  "name": "tonesphere",\n  "version": "1.0.0",\n  "devDependencies": { "x": { "version": "9.9.9" } }\n}\n');
  fs.writeFileSync(
    path.join(root, 'ui/package-lock.json'),
    '{\n  "name": "tonesphere",\n  "version": "1.0.0",\n  "packages": {\n    "": { "version": "1.0.0" },\n    "node_modules/x": { "version": "9.9.9" }\n  }\n}\n',
  );
  stamp('1.2.3', root);
  assert.equal(fs.readFileSync(path.join(root, 'VERSION'), 'utf8'), '1.2.3\n');
  const pkg = fs.readFileSync(path.join(root, 'ui/package.json'), 'utf8');
  assert.match(pkg, /"version": "1.2.3"/);
  assert.match(pkg, /"version": "9.9.9"/);
  const lock = fs.readFileSync(path.join(root, 'ui/package-lock.json'), 'utf8');
  assert.equal(lock.match(/1\.2\.3/g)?.length, 2);
  assert.match(lock, /"version": "9.9.9"/);
  fs.rmSync(root, { recursive: true });
});
