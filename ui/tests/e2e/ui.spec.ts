// End-to-end UI test: launches the real Electron app against the mock engine
// (or a real engine when TONESPHERE_E2E_REAL=1) and exercises the main flows.
import { test, expect, _electron as electron, type ElectronApplication, type Page } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const PORT = 47900 + Math.floor(Math.random() * 90);
let mock: ChildProcess | null = null;
let app: ElectronApplication;
let page: Page;
let splashUrl = '';
let profile = '';
let launchEnv: Record<string, string> = {};
const shots = path.join(process.cwd(), 'test-results', 'screens');

test.beforeAll(async () => {
  fs.mkdirSync(shots, { recursive: true });
  const discovery = path.join(os.tmpdir(), `tonesphere-e2e-${PORT}.json`);
  if (process.env.TONESPHERE_E2E_REAL !== '1') {
    mock = spawn(process.execPath, ['mock-engine/server.mjs', '--port', String(PORT)], { stdio: 'inherit' });
    fs.writeFileSync(discovery, JSON.stringify({ port: PORT, token: 'mock', pid: 0, version: 'mock', protocol: 1 }));
    await new Promise((r) => setTimeout(r, 600));
  }
  const env: Record<string, string> = { ...(process.env as Record<string, string>) };
  // VS Code terminals set this, which would make Electron run as plain Node.
  delete env.ELECTRON_RUN_AS_NODE;
  if (process.env.TONESPHERE_E2E_REAL !== '1') env.TONESPHERE_DISCOVERY = discovery;
  // An engine crash report from "an hour ago" in a throwaway data folder: the app must mention it.
  const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tonesphere-e2e-data-'));
  fs.mkdirSync(path.join(dataDir, 'crashes'));
  fs.writeFileSync(path.join(dataDir, 'crashes', 'crash-20261014-120000.txt'), 'ToneSphere engine 1.0.0 stopped unexpectedly.');
  const hourAgo = new Date(Date.now() - 3600_000);
  fs.utimesSync(path.join(dataDir, 'crashes', 'crash-20261014-120000.txt'), hourAgo, hourAgo);
  env.TONESPHERE_DATA_DIR = dataDir;
  // Own profile: the tests must not collide with a ToneSphere the user has open (single-instance lock).
  profile = fs.mkdtempSync(path.join(os.tmpdir(), 'tonesphere-e2e-profile-'));
  launchEnv = env;
  app = await electron.launch({ args: ['.', `--user-data-dir=${profile}`], env });
  // The splash (the ToneSphere banner) is the first window; the main window follows.
  const splash = await app.firstWindow();
  splashUrl = splash.url();
  await splash.waitForLoadState().then(() => splash.screenshot({ path: path.join(shots, '00-splash.png') })).catch((e) => console.log('splash shot', e));
  const isMain = (p: Page) => p.url().includes('index.html') || (!!process.env.TONESPHERE_DEV_URL && !p.url().includes('splash'));
  page = app.windows().find(isMain) ?? (await app.waitForEvent('window', { predicate: isMain, timeout: 20_000 }));
  await page.setViewportSize({ width: 1400, height: 880 });
  // The app remembers the last view; tests start from the Routing view.
  await page.locator('.segmented button', { hasText: 'Routing' }).click({ timeout: 20_000 });
  await page.waitForSelector('.node-card', { timeout: 20_000 });
  // Web fonts (JetBrains Mono) can shift the layout when they finish loading: wait first.
  await page.evaluate(() => document.fonts.ready);
});

test.afterAll(async () => {
  await app?.close();
  mock?.kill();
});

test('shows the six bins plus mixes', async () => {
  for (const t of ['Physical inputs', 'Virtual outputs', 'Application outputs', 'Physical outputs', 'Virtual inputs', 'Application inputs', 'Mixes']) {
    await expect(page.locator('.bin-title', { hasText: t }).first()).toBeVisible();
  }
  await page.waitForTimeout(500);
  await page.screenshot({ path: path.join(shots, '01-routing.png') });
});

test('drag a source onto a destination creates a connection', async () => {
  const before = await page.locator('.conn').count();
  const src = page.locator('[data-node-id="wasapi:in:line"]');
  const dst = page.locator('[data-node-id="wasapi:out:headphones"]');
  await src.dragTo(dst);
  await expect.poll(async () => page.locator('.conn').count()).toBeGreaterThan(before);
});

test('send-to menu toggles a connection', async () => {
  const card = page.locator('[data-node-id="wasapi:in:mic"]');
  await card.locator('.send-btn').click();
  const item = page.locator('.send-item', { hasText: 'Headphones' });
  const was = await item.getAttribute('aria-checked');
  await item.click();
  await expect(item).toHaveAttribute('aria-checked', was === 'true' ? 'false' : 'true');
  await page.screenshot({ path: path.join(shots, '02-send-to.png') });
  await page.keyboard.press('Escape');
});

test('inspector opens with effects and connections', async () => {
  await page.locator('[data-node-id="wasapi:in:mic"] .node-name').click();
  await expect(page.locator('.inspector')).toBeVisible();
  await page.locator('.inspector button', { hasText: 'Add effect' }).click();
  await page.locator('.menu-item', { hasText: 'Noise suppression' }).click();
  await expect(page.locator('.fx-item', { hasText: 'Noise suppression' })).toBeVisible();
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(shots, '03-inspector.png') });
});

test('matrix view lists every source and destination', async () => {
  await page.locator('.segmented button', { hasText: 'Matrix' }).click();
  await expect(page.locator('table.matrix')).toBeVisible();
  const cell = page.locator('td.cell').first();
  await expect(cell).toBeVisible();
  await page.screenshot({ path: path.join(shots, '04-matrix.png') });
  await page.locator('.segmented button', { hasText: 'Routing' }).click();
});

test('settings dialog opens', async () => {
  await page.locator('button[aria-label="Settings"]').click();
  await expect(page.locator('.modal.settings')).toBeVisible();
  await page.screenshot({ path: path.join(shots, '05-settings.png') });
  await expect(page.locator('.modal.settings h2', { hasText: 'Measure latency' })).toBeVisible();
  await page.locator('.settings-nav button', { hasText: 'Plugins' }).click();
  await expect(page.locator('.modal.settings', { hasText: 'Extra plugin folders' })).toBeVisible();
  await expect(page.locator('.modal.settings', { hasText: '2 plugins found' })).toBeVisible();
  await page.locator('.settings-nav button', { hasText: 'Control' }).click();
  await page.locator('.modal.settings input[role=switch]').first().click(); // state comes back from the engine
  await expect(page.locator('.modal.settings', { hasText: '/tonesphere/volume' })).toBeVisible();
  await page.screenshot({ path: path.join(shots, '07-control.png') });
  await page.locator('.settings-nav button', { hasText: 'Appearance' }).click();
  await page.locator('.skin-card', { hasText: 'Studio' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-skin', 'studio');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'dark');
  await page.screenshot({ path: path.join(shots, '08-skin-studio.png') });
  await page.locator('.skin-card', { hasText: 'Fluent' }).click();
  await expect(page.locator('html')).toHaveAttribute('data-skin', 'fluent');
  await page.keyboard.press('Escape');
});

test('scene menu offers preset import and export', async () => {
  await page.locator('button[title="Scenes"]').click();
  await expect(page.locator('.scene-popover button', { hasText: 'Import preset' })).toBeVisible();
  await expect(page.locator('.scene-popover button', { hasText: 'Export as preset' })).toBeVisible();
  await page.locator('button[title="Scenes"]').click();
});

test('dark theme renders', async () => {
  await page.evaluate(() => (document.documentElement.dataset.theme = 'dark'));
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(shots, '06-dark.png') });
});

test('the splash shows the banner, then hands over to the main window', async () => {
  expect(splashUrl).toContain('splash.html');
  // The splash closes once the main window has rendered and the engine answered.
  await expect.poll(() => app.windows().filter((w) => w.url().includes('splash.html')).length, { timeout: 15_000 }).toBe(0);
  const visible = await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows().filter((w) => w.isVisible()).length);
  expect(visible).toBe(1);
});

test('a second launch focuses the running window instead of starting another ToneSphere', async () => {
  const electronBin = (await import('electron')).default as unknown as string;
  const second = spawn(electronBin, ['.', `--user-data-dir=${profile}`], { env: launchEnv, stdio: 'ignore' });
  const code = await new Promise<number | null>((resolve) => {
    const timer = setTimeout(() => {
      second.kill();
      resolve(-1);
    }, 15_000);
    second.on('exit', (c) => {
      clearTimeout(timer);
      resolve(c);
    });
  });
  expect(code).toBe(0); // the second process gave up at once (single-instance lock)
  const windows = await app.evaluate(({ BrowserWindow }) =>
    BrowserWindow.getAllWindows().map((w) => ({ visible: w.isVisible(), title: w.getTitle() })),
  );
  expect(windows.filter((w) => w.visible)).toHaveLength(1);
});

test('devices tab offers the ASIO driver and VB-CABLE in one install', async () => {
  await page.locator('button[aria-label="Settings"]').click();
  await page.locator('.settings-nav button', { hasText: 'Devices' }).click();
  await expect(page.locator('[data-driver="asio"]')).toContainText('ToneSphere ASIO driver');
  await expect(page.locator('[data-driver="vbcable"]')).toContainText('VB-CABLE');
  const cable = page.locator('.check-row input[type=checkbox]');
  await expect(cable).toBeChecked(); // on by default while no virtual cable is installed
  await expect(page.locator('.install-drivers')).toContainText('vb-audio.com');
  await expect(page.locator('button', { hasText: 'Install drivers…' })).toBeEnabled();
  await page.screenshot({ path: path.join(shots, '09-devices.png') });
  await page.keyboard.press('Escape');
});

test('processor cores: automatic by default, and the choice reaches the engine', async () => {
  await page.locator('button[aria-label="Settings"]').click();
  const select = page.locator('label.field', { hasText: 'Processor cores' }).locator('select');
  await expect(select).toHaveValue('0');
  await expect(select.locator('option')).toHaveCount(9); // auto, one core, 2..8 cores (mock: 8 cores)
  await select.selectOption('4');
  await expect(select).toHaveValue('4');
  await select.selectOption('0');
  await expect(page.locator('.settings-body [role=status]').first()).toContainText('of 8 cores');
  await page.keyboard.press('Escape');
});

// Asks the mock engine directly (a second client) for its state.
async function mockState(): Promise<Record<string, unknown>> {
  const sock = new WebSocket(`ws://127.0.0.1:${PORT}/?token=mock`);
  await new Promise((r) => (sock.onopen = r));
  const result = await new Promise<Record<string, unknown>>((resolve) => {
    sock.onmessage = (ev) => {
      if (typeof ev.data !== 'string') return; // binary meter frames
      const m = JSON.parse(String(ev.data));
      if (m.id === 1) resolve(m.result);
    };
    sock.send(JSON.stringify({ id: 1, method: 'engine.getState', params: {} }));
  });
  sock.close();
  return result;
}

test('an instrument plugin gets an on-screen keyboard that plays notes', async () => {
  await page.locator('[data-node-id="bus:stream"] .node-name').click();
  await page.locator('.inspector button', { hasText: 'Add effect' }).click();
  await page.locator('.menu-item', { hasText: 'Demo Synth' }).click();
  const item = page.locator('.fx-item', { hasText: 'Demo Synth' });
  await expect(item).toBeVisible();
  await expect(item.locator('.toggle-row', { hasText: 'My MIDI keyboard plays this' })).toBeVisible();
  await item.locator('button[title="On-screen keyboard"]').click();
  const kb = item.locator('.keyboard');
  await expect(kb).toBeVisible();
  // Computer keyboard: "a" is C4 (MIDI note 60) while the keyboard has focus.
  await kb.focus();
  await page.keyboard.down('a');
  await expect.poll(async () => JSON.stringify((await mockState()).lastMidi)).toBe('[[144,60,100]]');
  await page.screenshot({ path: path.join(shots, '10-keyboard.png') });
  await page.keyboard.up('a');
  await expect.poll(async () => JSON.stringify((await mockState()).lastMidi)).toBe('[[128,60,0]]');
  // Z/X change the octave: "a" is now C3.
  await page.keyboard.press('z');
  await page.keyboard.down('a');
  await expect.poll(async () => JSON.stringify((await mockState()).lastMidi)).toBe('[[144,48,100]]');
  await page.keyboard.up('a');
  // Clicking a key plays it too.
  await kb.locator('.key[aria-label="E3"]').dispatchEvent('pointerdown', { pointerId: 1 });
  await expect.poll(async () => JSON.stringify((await mockState()).lastMidi)).toBe('[[144,52,100]]');
  await kb.locator('.key[aria-label="E3"]').dispatchEvent('pointerup', { pointerId: 1 });
  // A MIDI keyboard can play it.
  await item.locator('.toggle-row input[role=switch]').click();
  await expect.poll(async () => JSON.stringify((await mockState()).settings && (((await mockState()).settings as Record<string, unknown>).midiInstrument as { nodeId: string })?.nodeId)).toBe('"bus:stream"');
  await page.keyboard.press('Escape');
});

test('a crash report from the last run is mentioned once', async () => {
  const notice = page.locator('.crash-notice');
  await expect(notice).toBeVisible();
  await expect(notice).toContainText('stopped unexpectedly');
  await expect(notice.locator('button', { hasText: 'Show crash report' })).toBeVisible();
  await notice.locator('button[aria-label="Dismiss"]').click();
  await expect(notice).toHaveCount(0);
  await page.reload();
  await page.waitForSelector('.node-card', { timeout: 20_000 });
  await expect(page.locator('.crash-notice')).toHaveCount(0); // seen: not shown again
});

test('"hear my instrument" connects input 1 of the interface to both ears of the headphones', async () => {
  await page.locator('button[aria-label="Hear my instrument or microphone"]').click();
  const dlg = page.locator('.modal.wizard');
  await expect(dlg).toBeVisible();
  // It guesses the audio interface (ASIO) and the headphones.
  await expect(dlg.locator('label.field', { hasText: 'Plugged into' }).locator('select')).toHaveValue('asio:in:FlexASIO');
  await expect(dlg.locator('label.field', { hasText: 'Listen on' }).locator('select')).toHaveValue('wasapi:out:headphones');
  await expect(dlg.locator('label.field', { hasText: 'Input' }).last().locator('select')).toHaveValue('0');
  await page.screenshot({ path: path.join(shots, '11-wizard.png') });
  await dlg.locator('button', { hasText: 'Start listening' }).click();
  await expect(dlg).toHaveCount(0);
  const st = (await mockState()) as { routes: Array<{ from: string; to: string; gainDb: number; crosspoints: Array<{ f: number; t: number }> }> };
  const r = st.routes.find((x) => x.from === 'asio:in:FlexASIO' && x.to === 'wasapi:out:headphones');
  expect(r?.gainDb).toBe(-12);
  expect(r?.crosspoints.map((c) => `${c.f}->${c.t}`).sort()).toEqual(['0->0', '0->1']);
});

test('window shortcuts switch views and open dialogs, but not while typing', async () => {
  await page.locator('body').click({ position: { x: 5, y: 300 } });
  await page.keyboard.press('Control+2');
  await expect(page.locator('table.matrix')).toBeVisible();
  await page.keyboard.press('Control+1');
  await expect(page.locator('.node-card').first()).toBeVisible();
  await page.keyboard.press('Control+,');
  await expect(page.locator('.modal.settings')).toBeVisible();
  // In a text field, Ctrl+2 is left alone.
  await page.locator('.settings-nav button', { hasText: 'Plugins' }).click();
  const input = page.locator('.modal.settings input[type=text]').first();
  if (await input.count()) {
    await input.focus();
    await page.keyboard.press('Control+2');
    await expect(page.locator('.modal.settings')).toBeVisible();
  }
  await page.keyboard.press('Escape');
  await page.keyboard.press('Control+m');
  await expect(page.locator('.modal.wizard')).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(page.locator('.modal.wizard')).toHaveCount(0);
});
