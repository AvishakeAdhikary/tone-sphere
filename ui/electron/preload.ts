// Exposes a minimal, typed bridge to the renderer. No Node APIs leak through.
import { contextBridge, ipcRenderer } from 'electron';

const api = {
  discover: () => ipcRenderer.invoke('tonesphere:discover'),
  platform: () => ipcRenderer.invoke('tonesphere:platform'),
  setTheme: (theme: 'system' | 'light' | 'dark') => ipcRenderer.invoke('tonesphere:set-theme', theme),
  pickFolder: () => ipcRenderer.invoke('tonesphere:pick-folder') as Promise<string | null>,
  openPath: (p: string) => ipcRenderer.invoke('tonesphere:open-path', p),
  openSoundSettings: () => ipcRenderer.invoke('tonesphere:open-sound-settings'),
  openStartupSettings: () => ipcRenderer.invoke('tonesphere:open-startup-settings'),
  installDrivers: (opts?: { vbcable?: boolean }) => ipcRenderer.invoke('tonesphere:install-drivers', opts),
  onInstallProgress: (cb: (text: string) => void) => {
    const listener = (_e: unknown, text: string) => cb(text);
    ipcRenderer.on('tonesphere:install-progress', listener);
    return () => ipcRenderer.removeListener('tonesphere:install-progress', listener);
  },
  openFile: (title: string, extensions: string[]) => ipcRenderer.invoke('tonesphere:open-file', title, extensions),
  saveFile: (title: string, defaultName: string, text: string, extensions: string[]) =>
    ipcRenderer.invoke('tonesphere:save-file', title, defaultName, text, extensions),
  setLoginItem: (enabled: boolean) => ipcRenderer.invoke('tonesphere:set-login-item', enabled),
  onCommand: (cb: (cmd: string) => void) => {
    const listener = (_e: unknown, cmd: string) => cb(cmd);
    ipcRenderer.on('tonesphere:command', listener);
    return () => ipcRenderer.removeListener('tonesphere:command', listener);
  },
};

contextBridge.exposeInMainWorld('tonesphere', api);

export type ToneSphereBridge = typeof api;
