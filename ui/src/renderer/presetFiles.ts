// Preset files (shareable routing maps): import from / export to a file the user picks.
// Used by the Scenes menu and the Ctrl+O / Ctrl+S shortcuts.
import { bridge } from './engine/bridge';
import { engine, useStore } from './store';

export async function importPresetFile(): Promise<boolean> {
  const toast = useStore.getState().toast;
  try {
    const f = await bridge.openFile('Import a ToneSphere preset', ['json']);
    if (!f) return false;
    let preset: unknown;
    try {
      preset = JSON.parse(f.text);
    } catch {
      throw new Error("This file isn't a ToneSphere preset (it isn't valid JSON).");
    }
    const r = await engine.call('preset.import', { preset });
    toast('info', `Preset “${r.name}” is active: ${r.connections} connection${r.connections === 1 ? '' : 's'}.`);
    if (r.waiting.length) toast('info', `Waiting for: ${r.waiting.join(', ')}. They connect automatically as soon as they appear.`);
    return true;
  } catch (e) {
    toast('error', (e as Error).message);
    return false;
  }
}

export async function exportPresetFile(): Promise<void> {
  const { state, toast } = useStore.getState();
  const active = state?.scenes.find((s) => s.id === state.activeScene);
  const name = active?.name ?? 'My ToneSphere setup';
  try {
    const p = await engine.call('preset.export', { name });
    const file = `${name.replace(/[^\w -]+/g, '').trim() || 'preset'}.tonesphere.json`;
    const path = await bridge.saveFile('Export preset', file, JSON.stringify(p, null, 2) + '\n', ['json']);
    if (path) toast('info', `Saved ${path}`);
  } catch (e) {
    toast('error', (e as Error).message);
  }
}
