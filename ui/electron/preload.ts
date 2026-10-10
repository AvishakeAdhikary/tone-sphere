// Exposes a minimal, typed bridge to the renderer. No Node APIs leak through.
import { contextBridge, ipcRenderer } from 'electron';

const api = {
  discover: () => ipcRenderer.invoke('nextone:discover'),
  platform: () => ipcRenderer.invoke('nextone:platform'),
  setTheme: (theme: 'system' | 'light' | 'dark') => ipcRenderer.invoke('nextone:set-theme', theme),
  pickFolder: () => ipcRenderer.invoke('nextone:pick-folder') as Promise<string | null>,
  openPath: (p: string) => ipcRenderer.invoke('nextone:open-path', p),
  openSoundSettings: () => ipcRenderer.invoke('nextone:open-sound-settings'),
  openStartupSettings: () => ipcRenderer.invoke('nextone:open-startup-settings'),
  setLoginItem: (enabled: boolean) => ipcRenderer.invoke('nextone:set-login-item', enabled),
  onCommand: (cb: (cmd: string) => void) => {
    const listener = (_e: unknown, cmd: string) => cb(cmd);
    ipcRenderer.on('nextone:command', listener);
    return () => ipcRenderer.removeListener('nextone:command', listener);
  },
};

contextBridge.exposeInMainWorld('nextone', api);

export type NexToneBridge = typeof api;
