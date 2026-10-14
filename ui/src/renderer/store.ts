import { create } from 'zustand';
import type { EngineState, NodeInfo, RouteInfo } from '@shared/protocol';
import { EngineClient, type ConnectionStatus } from './engine/client';
import { bridge } from './engine/bridge';

export type View = 'simple' | 'matrix';
export type Selection =
  | { type: 'node'; id: string }
  | { type: 'route'; id: string }
  | null;

export interface Toast {
  id: number;
  level: 'info' | 'warn' | 'error';
  message: string;
}

interface Store {
  status: ConnectionStatus;
  state: EngineState | null;
  view: View;
  selection: Selection;
  hoverNode: string | null;
  dragFrom: string | null;
  settingsOpen: boolean;
  toasts: Toast[];
  setView(v: View): void;
  select(s: Selection): void;
  setHover(id: string | null): void;
  setDragFrom(id: string | null): void;
  setSettingsOpen(o: boolean): void;
  toast(level: Toast['level'], message: string): void;
  dismissToast(id: number): void;
}

let toastId = 1;

export const useStore = create<Store>((set) => ({
  status: 'connecting',
  state: null,
  view: (localStorage.getItem('tonesphere.view') as View) || 'simple',
  selection: null,
  hoverNode: null,
  dragFrom: null,
  settingsOpen: false,
  toasts: [],
  setView: (view) => {
    try {
      localStorage.setItem('tonesphere.view', view);
    } catch {
      /* storage unavailable */
    }
    set({ view });
  },
  select: (selection) => set({ selection }),
  setHover: (hoverNode) => set({ hoverNode }),
  setDragFrom: (dragFrom) => set({ dragFrom }),
  setSettingsOpen: (settingsOpen) => set({ settingsOpen }),
  toast: (level, message) => {
    const id = toastId++;
    set((s) => ({ toasts: [...s.toasts.slice(-3), { id, level, message }] }));
    setTimeout(() => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })), 5000);
  },
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}));

export const engine = new EngineClient(() => bridge.discover());
engine.onState = (state) => useStore.setState({ state });
engine.onStatus = (status) => useStore.setState({ status });
engine.onLog = (level, message) => {
  if (level === 'warn' || level === 'error') useStore.getState().toast(level, message);
};

/** Wraps an engine call and reports failures as a toast instead of throwing. */
export async function act<T>(p: Promise<T>): Promise<T | undefined> {
  try {
    return await p;
  } catch (e) {
    useStore.getState().toast('error', e instanceof Error ? e.message : String(e));
    return undefined;
  }
}

// ---- Selectors ---------------------------------------------------------------
export function nodeById(state: EngineState | null, id: string | undefined): NodeInfo | undefined {
  return id ? state?.nodes.find((n) => n.id === id) : undefined;
}

export function routesFrom(state: EngineState | null, id: string): RouteInfo[] {
  return state?.routes.filter((r) => r.from === id) ?? [];
}

export function routesTo(state: EngineState | null, id: string): RouteInfo[] {
  return state?.routes.filter((r) => r.to === id) ?? [];
}

export function findRoute(state: EngineState | null, from: string, to: string): RouteInfo | undefined {
  return state?.routes.find((r) => r.from === from && r.to === to);
}
