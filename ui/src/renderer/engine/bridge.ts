// Access to the Electron preload bridge, with a browser fallback so the renderer
// can run in a plain browser against the mock engine (used by tests and design work).
import type { EngineDiscovery } from '@shared/protocol';

export interface NexToneBridge {
  discover(): Promise<EngineDiscovery | null>;
  platform(): Promise<{ platform: string; version: string; arch: string; store?: boolean }>;
  openStartupSettings(): Promise<void>;
  installDrivers(): Promise<{ ok: boolean; message: string }>;
  /** Native open dialog; returns the file's name and text, or null when cancelled. */
  openFile(title: string, extensions: string[]): Promise<{ name: string; text: string } | null>;
  /** Native save dialog; returns the saved path, or null when cancelled. */
  saveFile(title: string, defaultName: string, text: string, extensions: string[]): Promise<string | null>;
  setTheme(theme: 'system' | 'light' | 'dark'): Promise<void>;
  pickFolder(): Promise<string | null>;
  openPath(p: string): Promise<void>;
  openSoundSettings(): Promise<void>;
  setLoginItem(enabled: boolean): Promise<void>;
  onCommand(cb: (cmd: string) => void): () => void;
}

declare global {
  interface Window {
    nextone?: NexToneBridge;
  }
}

const MOCK_PORT = 47999;

const browserFallback: NexToneBridge = {
  async discover() {
    const q = new URLSearchParams(location.search);
    const port = Number(q.get('port') ?? MOCK_PORT);
    const token = q.get('token') ?? 'mock';
    return { port, token, pid: 0, version: 'browser', protocol: 1 };
  },
  async platform() {
    return { platform: 'browser', version: '0.0.0', arch: 'x64' };
  },
  async setTheme() {},
  async pickFolder() {
    return null;
  },
  async openPath() {},
  async openSoundSettings() {},
  async openStartupSettings() {},
  async installDrivers() {
    return { ok: false, message: 'Not available in the browser preview.' };
  },
  async openFile() {
    return null;
  },
  async saveFile() {
    return null;
  },
  async setLoginItem() {},
  onCommand() {
    return () => {};
  },
};

export const bridge: NexToneBridge = window.nextone ?? browserFallback;
export const inElectron = !!window.nextone;
