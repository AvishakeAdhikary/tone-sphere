// ToneSphere UI — Electron main process.
// Owns the window + tray, discovers (or launches) the always-running audio engine,
// and exposes a small, explicit API to the renderer through preload.
import {
  app,
  BrowserWindow,
  Menu,
  Tray,
  dialog,
  ipcMain,
  nativeImage,
  nativeTheme,
  shell,
} from 'electron';
import { execFile, spawn } from 'node:child_process';
import fs from 'node:fs';
import net from 'node:net';
import os from 'node:os';
import path from 'node:path';
import { downloadVbCable } from './vbcable';

const DEV_URL = process.env.TONESPHERE_DEV_URL;
const isMac = process.platform === 'darwin';
const isWin = process.platform === 'win32';

let win: BrowserWindow | null = null;
let tray: Tray | null = null;
let quitting = false;

interface Discovery {
  port: number;
  token: string;
  pid: number;
  version: string;
  protocol: number;
}

function discoveryPath(): string {
  if (process.env.TONESPHERE_DISCOVERY) return process.env.TONESPHERE_DISCOVERY;
  if (isWin) {
    const base = process.env.LOCALAPPDATA ?? path.join(os.homedir(), 'AppData', 'Local');
    return path.join(base, 'ToneSphere', 'engine.json');
  }
  if (isMac) return path.join(os.homedir(), 'Library', 'Application Support', 'ToneSphere', 'engine.json');
  const runtime = process.env.XDG_RUNTIME_DIR;
  return runtime
    ? path.join(runtime, 'tonesphere', 'engine.json')
    : path.join(os.homedir(), '.local', 'state', 'tonesphere', 'engine.json');
}

function readDiscovery(): Discovery | null {
  try {
    const d = JSON.parse(fs.readFileSync(discoveryPath(), 'utf8')) as Discovery;
    if (typeof d.port === 'number' && typeof d.token === 'string') return d;
  } catch {
    /* engine not started yet */
  }
  return null;
}

function portOpen(port: number): Promise<boolean> {
  return new Promise((resolve) => {
    const s = net.connect({ host: '127.0.0.1', port }, () => {
      s.destroy();
      resolve(true);
    });
    s.on('error', () => resolve(false));
    s.setTimeout(500, () => {
      s.destroy();
      resolve(false);
    });
  });
}

function engineExecutable(): string | null {
  const exe = isWin ? 'tonesphere-engine.exe' : 'tonesphere-engine';
  const candidates = [
    process.env.TONESPHERE_ENGINE_PATH,
    path.join(process.resourcesPath ?? '', 'engine', exe),
    path.join(app.getAppPath(), '..', 'build', 'engine', exe),
    path.join(app.getAppPath(), '..', 'build', 'engine', 'Release', exe),
  ].filter((p): p is string => !!p);
  return candidates.find((p) => fs.existsSync(p)) ?? null;
}

/** Ensures an engine is reachable; launches it detached if needed. */
async function ensureEngine(): Promise<Discovery | null> {
  let d = readDiscovery();
  if (d && (await portOpen(d.port))) return d;
  const exe = engineExecutable();
  if (!exe) return null;
  setSplashStatus('Starting the audio engine…');
  const child = spawn(exe, ['--background'], { detached: true, stdio: 'ignore', windowsHide: true });
  child.unref();
  for (let i = 0; i < 50; i++) {
    await new Promise((r) => setTimeout(r, 100));
    d = readDiscovery();
    if (d && (await portOpen(d.port))) return d;
  }
  return null;
}

// ---- Splash: the ToneSphere banner while the engine and the window start ----
let splash: BrowserWindow | null = null;
let firstEngine: Promise<Discovery | null> | null = null;
const SPLASH_MIN_MS = 900; // long enough to read, short enough not to be in the way

function createSplash(): void {
  splash = new BrowserWindow({
    width: 660,
    height: 360,
    frame: false,
    resizable: false,
    movable: true,
    center: true,
    show: false,
    skipTaskbar: false,
    alwaysOnTop: false,
    title: 'ToneSphere',
    icon: appIcon(),
    backgroundColor: '#000000',
    webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false, spellcheck: false },
  });
  splash.setMenuBarVisibility(false);
  splash.once('ready-to-show', () => splash?.show());
  splash.on('closed', () => {
    splash = null;
  });
  void splash.loadFile(path.join(app.getAppPath(), 'assets', 'splash.html'));
}

function setSplashStatus(text: string): void {
  if (!splash || splash.isDestroyed()) return;
  void splash.webContents
    .executeJavaScript(`document.getElementById('status').textContent = ${JSON.stringify(text)}`)
    .catch(() => {});
}

/** Shows the main window once it has rendered and the engine answered (or gave up), then closes the splash. */
function handOver(mainReady: Promise<void>, shownAt: number): void {
  const engineDone = (firstEngine ?? Promise.resolve(null)).then(() => setSplashStatus('Opening…'));
  const floor = new Promise((r) => setTimeout(r, Math.max(0, SPLASH_MIN_MS - (Date.now() - shownAt))));
  // Never keep the user waiting on the splash: after 10 s the window opens regardless.
  const cap = new Promise((r) => setTimeout(r, 10_000));
  void Promise.race([Promise.all([mainReady, engineDone, floor]), cap]).then(() => {
    win?.show();
    win?.focus();
    splash?.close();
  });
}

function appIcon(): Electron.NativeImage {
  const p = path.join(app.getAppPath(), 'assets', isWin ? 'icon.ico' : 'icon.png');
  const img = fs.existsSync(p) ? nativeImage.createFromPath(p) : nativeImage.createEmpty();
  return img;
}

function trayIcon(): Electron.NativeImage {
  const p = path.join(app.getAppPath(), 'assets', 'tray.png');
  return fs.existsSync(p) ? nativeImage.createFromPath(p) : appIcon();
}

function overlayColors() {
  const dark = nativeTheme.shouldUseDarkColors;
  return { color: '#00000000', symbolColor: dark ? '#ffffff' : '#1a1a1a', height: 44 };
}

function createWindow(): void {
  win = new BrowserWindow({
    width: 1360,
    height: 860,
    minWidth: 940,
    minHeight: 600,
    show: false,
    title: 'ToneSphere',
    icon: appIcon(),
    backgroundColor: isWin ? '#00000000' : nativeTheme.shouldUseDarkColors ? '#1c1c1e' : '#f5f5f7',
    backgroundMaterial: isWin ? 'mica' : undefined,
    titleBarStyle: 'hidden',
    titleBarOverlay: isMac ? undefined : overlayColors(),
    trafficLightPosition: isMac ? { x: 14, y: 14 } : undefined,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true,
      sandbox: true,
      nodeIntegration: false,
      spellcheck: false,
    },
  });
  win.setMenuBarVisibility(false);
  if (DEV_URL) void win.loadURL(DEV_URL);
  else void win.loadFile(path.join(app.getAppPath(), 'dist', 'index.html'));

  const startHidden = process.argv.includes('--hidden');
  const mainReady = new Promise<void>((resolve) => win?.once('ready-to-show', () => resolve()));
  if (splash) handOver(mainReady, Date.now());
  else if (!startHidden) void mainReady.then(() => win?.show());
  win.on('close', (e) => {
    if (!quitting && tray) {
      e.preventDefault();
      win?.hide();
    }
  });
  // External links open in the browser, never inside the app.
  win.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:\/\//.test(url)) void shell.openExternal(url);
    return { action: 'deny' };
  });
  win.webContents.on('will-navigate', (e, url) => {
    if (DEV_URL && url.startsWith(DEV_URL)) return;
    e.preventDefault();
  });
}

function showWindow(): void {
  if (!win) createWindow();
  win?.show();
  win?.focus();
}

function createTray(): void {
  tray = new Tray(trayIcon());
  tray.setToolTip('ToneSphere');
  tray.on('click', showWindow);
  tray.setContextMenu(
    Menu.buildFromTemplate([
      { label: 'Open ToneSphere', click: showWindow },
      { type: 'separator' },
      {
        label: 'Mute all outputs',
        click: () => win?.webContents.send('tonesphere:command', 'mute-all'),
      },
      { type: 'separator' },
      {
        label: 'Close window (audio keeps running)',
        click: () => {
          quitting = true;
          app.quit();
        },
      },
      {
        label: 'Quit ToneSphere completely',
        click: () => {
          win?.webContents.send('tonesphere:command', 'shutdown-engine');
          setTimeout(() => {
            quitting = true;
            app.quit();
          }, 300);
        },
      },
    ]),
  );
}

// ---- IPC API for the renderer (kept deliberately small) -------------------
// The first discovery was started with the splash; later ones (reconnects) look again.
ipcMain.handle('tonesphere:discover', async () => {
  const first = firstEngine;
  firstEngine = null;
  return first ?? ensureEngine();
});
ipcMain.handle('tonesphere:platform', () => ({
  platform: process.platform,
  version: app.getVersion(),
  arch: process.arch,
  // Microsoft Store (MSIX) build: sign-in startup is a package startup task managed by Windows.
  store: !!process.windowsStore,
}));
ipcMain.handle('tonesphere:set-theme', (_e, theme: 'system' | 'light' | 'dark') => {
  nativeTheme.themeSource = theme;
  if (win && !isMac) win.setTitleBarOverlay(overlayColors());
});
ipcMain.handle('tonesphere:pick-folder', async () => {
  if (!win) return null;
  const r = await dialog.showOpenDialog(win, { properties: ['openDirectory', 'createDirectory'] });
  return r.canceled ? null : r.filePaths[0];
});
ipcMain.handle('tonesphere:open-path', async (_e, p: string) => {
  if (typeof p === 'string' && fs.existsSync(p)) await shell.openPath(p);
});
ipcMain.handle('tonesphere:open-sound-settings', async () => {
  if (isWin) await shell.openExternal('ms-settings:sound');
});
ipcMain.handle('tonesphere:open-startup-settings', async () => {
  if (isWin) await shell.openExternal('ms-settings:startupapps');
});
// Desktop build: installs the ToneSphere ASIO driver (and the virtual audio driver when the build
// carries it) from resources/drivers, plus VB-CABLE when asked, after one UAC prompt. The Store
// build doesn't contain drivers (MSIX can't install them).
ipcMain.handle('tonesphere:install-drivers', async (_e, opts?: { vbcable?: boolean }): Promise<{ ok: boolean; message: string }> => {
  if (!isWin || process.windowsStore) return { ok: false, message: 'Drivers come with the ToneSphere desktop installer.' };
  const dir = path.join(process.resourcesPath, 'drivers');
  const setup = path.join(dir, 'tonesphere-setup.exe');
  if (!fs.existsSync(setup)) return { ok: false, message: 'This build does not include the drivers.' };
  const progress = (text: string) => win?.webContents.send('tonesphere:install-progress', text);
  let vbDir = '';
  if (opts?.vbcable) {
    progress('Downloading VB-CABLE from vb-audio.com…');
    try {
      vbDir = await downloadVbCable();
    } catch (err) {
      return { ok: false, message: (err as Error).message };
    }
  }
  const engineExe = path.join(process.resourcesPath, 'engine', 'tonesphere-engine.exe');
  const resultFile = path.join(os.tmpdir(), `tonesphere-setup-${process.pid}-${Date.now()}.jsonl`);
  const q = (s: string) => `"${s}"`.replace(/'/g, "''");
  const args = ['install-all', q(dir), '--engine', q(engineExe), '--ui', q(process.execPath), '--result', q(resultFile)];
  if (vbDir) args.push('--vbcable', q(vbDir));
  const ps = `try { $p = Start-Process -FilePath '${setup.replace(/'/g, "''")}' -ArgumentList '${args.join(' ')}' -Verb RunAs -Wait -PassThru -WindowStyle Hidden; exit $p.ExitCode } catch { exit 1223 }`;
  progress(vbDir ? 'Windows asks for permission, then VB-Audio’s setup opens: choose “Install Driver”.' : 'Windows asks for permission…');
  const code = await new Promise<number>((resolve) => {
    const child = execFile('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', ps], { windowsHide: true }, () => {});
    child.on('exit', (c) => resolve(c ?? 1));
  });
  // Each step wrote one JSON line: report them in plain words.
  let steps: Array<{ ok: boolean; message: string; reboot?: boolean }> = [];
  try {
    steps = fs.readFileSync(resultFile, 'utf8').split('\n').filter(Boolean).map((l) => JSON.parse(l));
    fs.rmSync(resultFile, { force: true });
  } catch {
    /* nothing ran (cancelled) */
  }
  if (vbDir) fs.rmSync(path.dirname(vbDir), { recursive: true, force: true });
  if (code === 1223) return { ok: false, message: 'Installation was cancelled.' };
  const failed = steps.filter((s) => !s.ok);
  if (code !== 0 || failed.length) return { ok: false, message: failed.map((s) => s.message).join(' ') || `Driver installation failed (code ${code}).` };
  return { ok: true, message: steps.map((s) => s.message).join('. ') || 'Drivers installed.' };
});// Presets and skins: the user picks the file; only that file is read or written.
const fileFilters = (extensions: string[]) => [
  { name: 'ToneSphere files', extensions: extensions.filter((e) => /^[a-z0-9]+$/i.test(e)) },
  { name: 'All files', extensions: ['*'] },
];
ipcMain.handle('tonesphere:open-file', async (_e, title: string, extensions: string[]) => {
  if (!win) return null;
  const r = await dialog.showOpenDialog(win, { title: String(title), properties: ['openFile'], filters: fileFilters(extensions) });
  const p = r.filePaths[0];
  if (r.canceled || !p) return null;
  if (fs.statSync(p).size > 2_000_000) throw new Error('That file is too large to be a ToneSphere file.');
  return { name: path.basename(p), text: fs.readFileSync(p, 'utf8') };
});
ipcMain.handle('tonesphere:save-file', async (_e, title: string, defaultName: string, text: string, extensions: string[]) => {
  if (!win || typeof text !== 'string') return null;
  const safeName = String(defaultName).replace(/[\\/:*?"<>|]+/g, '-');
  const r = await dialog.showSaveDialog(win, {
    title: String(title),
    defaultPath: path.join(app.getPath('documents'), safeName),
    filters: fileFilters(extensions),
  });
  if (r.canceled || !r.filePath) return null;
  fs.writeFileSync(r.filePath, text, 'utf8');
  return r.filePath;
});
ipcMain.handle('tonesphere:set-login-item', (_e, enabled: boolean) => {
  app.setLoginItemSettings({ openAtLogin: !!enabled, args: ['--hidden'] });
});

nativeTheme.on('updated', () => {
  if (win && !isMac) win.setTitleBarOverlay(overlayColors());
});

// ToneSphere was called NexTone before. Its window profile (view choice, collapsed sections) lived
// in a folder named after the old product; carry it over once, into a profile that doesn't exist yet.
// (The engine migrates its own settings, presets and plugin cache.)
function migrateLegacyProfile(): void {
  try {
    const userData = app.getPath('userData');
    const legacy = path.join(path.dirname(userData), isWin || isMac ? 'NexTone' : 'nextone');
    const from = path.join(legacy, 'Local Storage');
    const to = path.join(userData, 'Local Storage');
    if (!fs.existsSync(from) || fs.existsSync(to)) return;
    fs.mkdirSync(userData, { recursive: true });
    fs.cpSync(from, to, { recursive: true });
  } catch {
    /* best effort: a fresh profile only loses the remembered view */
  }
}

// ---- Lifecycle -------------------------------------------------------------
if (!app.requestSingleInstanceLock()) {
  app.quit();
} else {
  app.on('second-instance', showWindow);
  app.setAppUserModelId('com.neuralnexusstudios.tonesphere');
  migrateLegacyProfile();
  void app.whenReady().then(() => {
    // The splash shows on a normal start, not when signing in starts ToneSphere in the tray.
    if (!process.argv.includes('--hidden') && !process.argv.includes('--no-splash')) createSplash();
    firstEngine = ensureEngine();
    createWindow();
    createTray();
  });
  app.on('before-quit', () => {
    quitting = true;
  });
  // Session logout / `kill` on Linux and macOS: quit for real (the tray would otherwise keep the window hidden).
  for (const sig of ['SIGTERM', 'SIGINT', 'SIGHUP'] as const) {
    process.on(sig, () => {
      quitting = true;
      app.quit();
    });
  }
  app.on('window-all-closed', () => {
    if (!tray) app.quit();
  });
  app.on('activate', showWindow);
}
