// Access to the Electron preload bridge, with a browser fallback so the renderer
// can run in a plain browser against the mock engine (used by tests and design work).
import type { EngineDiscovery } from '@shared/protocol';

export interface ToneSphereBridge {
  discover(): Promise<EngineDiscovery | null>;
  platform(): Promise<{ platform: string; version: string; arch: string; store?: boolean }>;
  openStartupSettings(): Promise<void>;
  /** Installs the ToneSphere drivers, plus VB-CABLE (downloaded from vb-audio.com) when asked. */
  installDrivers(opts?: { vbcable?: boolean }): Promise<{ ok: boolean; message: string }>;
  onInstallProgress(cb: (text: string) => void): () => void;
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
    tonesphere?: ToneSphereBridge;
  }
}

const MOCK_PORT = 47999;

const browserFallback: ToneSphereBridge = {
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
  onInstallProgress() {
    return () => {};
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

export const bridge: ToneSphereBridge = window.tonesphere ?? browserFallback;
export const inElectron = !!window.tonesphere;
