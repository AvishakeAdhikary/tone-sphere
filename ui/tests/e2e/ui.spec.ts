// End-to-end UI test: launches the real Electron app against the mock engine
// (or a real engine when NEXTONE_E2E_REAL=1) and exercises the main flows.
import { test, expect, _electron as electron, type ElectronApplication, type Page } from '@playwright/test';
import { spawn, type ChildProcess } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const PORT = 47900 + Math.floor(Math.random() * 90);
let mock: ChildProcess | null = null;
let app: ElectronApplication;
let page: Page;
const shots = path.join(process.cwd(), 'test-results', 'screens');

test.beforeAll(async () => {
  fs.mkdirSync(shots, { recursive: true });
  const discovery = path.join(os.tmpdir(), `nextone-e2e-${PORT}.json`);
  if (process.env.NEXTONE_E2E_REAL !== '1') {
    mock = spawn(process.execPath, ['mock-engine/server.mjs', '--port', String(PORT)], { stdio: 'inherit' });
    fs.writeFileSync(discovery, JSON.stringify({ port: PORT, token: 'mock', pid: 0, version: 'mock', protocol: 1 }));
    await new Promise((r) => setTimeout(r, 600));
  }
  const env: Record<string, string> = { ...(process.env as Record<string, string>) };
  // VS Code terminals set this, which would make Electron run as plain Node.
  delete env.ELECTRON_RUN_AS_NODE;
  if (process.env.NEXTONE_E2E_REAL !== '1') env.NEXTONE_DISCOVERY = discovery;
  app = await electron.launch({ args: ['.'], env });
  page = await app.firstWindow();
  await page.setViewportSize({ width: 1400, height: 880 });
  // The app remembers the last view; tests start from the Routing view.
  await page.locator('.segmented button', { hasText: 'Routing' }).click({ timeout: 20_000 });
  await page.waitForSelector('.node-card', { timeout: 20_000 });
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
  await page.locator('.settings-nav button', { hasText: 'Plugins' }).click();
  await expect(page.locator('.modal.settings', { hasText: 'Extra plugin folders' })).toBeVisible();
  await expect(page.locator('.modal.settings', { hasText: '1 plugin found' })).toBeVisible();
  await page.keyboard.press('Escape');
});

test('dark theme renders', async () => {
  await page.evaluate(() => (document.documentElement.dataset.theme = 'dark'));
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(shots, '06-dark.png') });
});
