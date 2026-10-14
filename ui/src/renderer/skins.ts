// Skins: named sets of design-token overrides (see the :root tokens in styles.css).
// A skin may force light or dark mode. Skins are shared as small JSON files:
//   { "format": "tonesphere-skin", "version": 1, "id": "…", "name": "…", "mode": "dark",
//     "tokens": { "--accent": "#ffb020", … }, "light": { … }, "dark": { … } }
export interface Skin {
  format?: 'tonesphere-skin';
  version?: number;
  id: string;
  name: string;
  /** Forces light or dark mode; omitted = follows the Theme setting. */
  mode?: 'light' | 'dark';
  /** Tokens for every mode, then per-mode overrides. */
  tokens?: Record<string, string>;
  light?: Record<string, string>;
  dark?: Record<string, string>;
  /** Swatch colours for the skin picker (background, panel, accent, meter). */
  preview?: [string, string, string, string];
}

/** The tokens a skin may set (anything else in a skin file is ignored). */
export const SKIN_TOKENS = [
  '--bg', '--bg-solid', '--surface', '--surface-2', '--surface-hover', '--surface-sunken', '--stroke', '--stroke-strong',
  '--text', '--text-2', '--text-3', '--accent', '--accent-soft', '--accent-text', '--danger', '--danger-soft', '--warn', '--ok',
  '--hue-blue', '--hue-violet', '--hue-amber', '--hue-green', '--meter-bg', '--meter-low', '--meter-mid', '--meter-high',
  '--radius', '--radius-lg', '--shadow', '--shadow-pop',
] as const;

export const BUILTIN_SKINS: Skin[] = [
  { id: 'fluent', name: 'Fluent (default)', preview: ['#f3f3f3', '#202020', '#0a64d8', '#21b35c'] },
  {
    id: 'midnight',
    name: 'Midnight',
    mode: 'dark',
    tokens: {
      '--bg': '#0e1322', '--bg-solid': '#0e1322', '--surface': 'rgba(22, 30, 50, 0.78)', '--surface-2': 'rgba(28, 38, 62, 0.96)',
      '--surface-hover': 'rgba(36, 48, 78, 0.96)', '--surface-sunken': 'rgba(0, 0, 0, 0.28)', '--stroke': 'rgba(140, 170, 255, 0.10)',
      '--stroke-strong': 'rgba(140, 170, 255, 0.24)', '--text': '#e8edff', '--text-2': '#b3bedd', '--text-3': '#7d88a8',
      '--accent': '#7aa2ff', '--accent-soft': 'rgba(122, 162, 255, 0.18)', '--accent-text': '#0b1430', '--hue-blue': '#6ea8ff',
      '--meter-bg': 'rgba(255, 255, 255, 0.07)',
    },
  },
  {
    id: 'studio',
    name: 'Studio',
    mode: 'dark',
    tokens: {
      '--bg': '#1a1917', '--bg-solid': '#1a1917', '--surface': 'rgba(38, 36, 33, 0.85)', '--surface-2': 'rgba(46, 44, 40, 0.97)',
      '--surface-hover': 'rgba(56, 53, 48, 0.97)', '--surface-sunken': 'rgba(0, 0, 0, 0.30)', '--stroke': 'rgba(255, 220, 170, 0.08)',
      '--stroke-strong': 'rgba(255, 220, 170, 0.20)', '--text': '#f3ede4', '--text-2': '#c9bfb2', '--text-3': '#8f877c',
      '--accent': '#ffb020', '--accent-soft': 'rgba(255, 176, 32, 0.16)', '--accent-text': '#2a1a00', '--hue-amber': '#ffb020',
      '--meter-bg': 'rgba(255, 255, 255, 0.07)', '--meter-low': '#7bd88f', '--meter-mid': '#ffcf40', '--meter-high': '#ff5f4a',
    },
  },
  {
    id: 'neon',
    name: 'Neon',
    mode: 'dark',
    tokens: {
      '--bg': '#08080d', '--bg-solid': '#08080d', '--surface': 'rgba(18, 18, 30, 0.86)', '--surface-2': 'rgba(24, 24, 40, 0.97)',
      '--surface-hover': 'rgba(32, 32, 54, 0.97)', '--surface-sunken': 'rgba(0, 0, 0, 0.40)', '--stroke': 'rgba(0, 240, 255, 0.12)',
      '--stroke-strong': 'rgba(0, 240, 255, 0.30)', '--text': '#eafcff', '--text-2': '#a8c7d0', '--text-3': '#6f8b94',
      '--accent': '#00e5ff', '--accent-soft': 'rgba(0, 229, 255, 0.16)', '--accent-text': '#001a1f', '--hue-violet': '#ff2bd6',
      '--hue-blue': '#00b3ff', '--meter-bg': 'rgba(255, 255, 255, 0.07)', '--meter-low': '#00ff9c', '--meter-mid': '#fff35c',
      '--meter-high': '#ff2b6a',
    },
  },
  {
    id: 'paper',
    name: 'Paper',
    mode: 'light',
    tokens: {
      '--bg': '#f5efe3', '--bg-solid': '#f5efe3', '--surface': 'rgba(255, 251, 242, 0.88)', '--surface-2': '#fffaf0',
      '--surface-hover': '#fffdf7', '--surface-sunken': 'rgba(60, 40, 10, 0.05)', '--stroke': 'rgba(60, 40, 10, 0.10)',
      '--stroke-strong': 'rgba(60, 40, 10, 0.20)', '--text': '#2b2621', '--text-2': '#5e5448', '--text-3': '#8c8174',
      '--accent': '#b4532a', '--accent-soft': 'rgba(180, 83, 42, 0.12)', '--accent-text': '#ffffff',
      '--meter-bg': 'rgba(60, 40, 10, 0.08)',
    },
  },
];

const SAFE_VALUE = /^[#(),.%\-\s\w]{1,120}$/;

/** Validates a skin read from a file; throws a user-facing message. Unknown tokens and unsafe values are dropped. */
export function parseSkin(text: string): Skin {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new Error("This file isn't a ToneSphere skin (it isn't valid JSON).");
  }
  const o = raw as Partial<Skin>;
  // 'nextone-skin' files were exported before the rename and have the same layout.
  const format = (o as { format?: string } | null)?.format;
  if (!o || typeof o !== 'object' || (format !== 'tonesphere-skin' && format !== 'nextone-skin'))
    throw new Error("This file isn't a ToneSphere skin.");
  if (typeof o.name !== 'string' || !o.name.trim()) throw new Error('This skin has no name.');
  const clean = (m: unknown) => {
    const out: Record<string, string> = {};
    if (m && typeof m === 'object')
      for (const [k, v] of Object.entries(m as Record<string, unknown>))
        if ((SKIN_TOKENS as readonly string[]).includes(k) && typeof v === 'string' && SAFE_VALUE.test(v) && !/url\s*\(/i.test(v)) out[k] = v;
    return out;
  };
  const name = o.name.trim().slice(0, 40);
  return {
    format: 'tonesphere-skin',
    version: 1,
    id: `custom-${name.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`,
    name,
    mode: o.mode === 'light' || o.mode === 'dark' ? o.mode : undefined,
    tokens: clean(o.tokens),
    light: clean(o.light),
    dark: clean(o.dark),
  };
}

export function skinToFile(skin: Skin): string {
  const { preview: _preview, ...rest } = skin;
  return JSON.stringify({ ...rest, format: 'tonesphere-skin', version: 1 }, null, 2) + '\n';
}

/** Picker swatch colours: explicit preview, else the skin's own tokens. */
export function skinPreview(k: Skin): [string, string, string, string] {
  if (k.preview) return k.preview;
  const t = (name: string, fallback: string) => k.tokens?.[name] ?? k.dark?.[name] ?? k.light?.[name] ?? fallback;
  return [t('--bg', '#202020'), t('--surface-2', '#323232'), t('--accent', '#4cc2ff'), t('--meter-low', '#21b35c')];
}

let applied: string[] = [];

/** Applies a skin on top of the base theme. `theme` is the Theme setting (system/light/dark). */
export function applySkin(skin: Skin | undefined, theme: 'system' | 'light' | 'dark'): 'system' | 'light' | 'dark' {
  const root = document.documentElement;
  for (const k of applied) root.style.removeProperty(k);
  applied = [];
  const effective = skin?.mode ?? theme;
  const dark = effective === 'dark' || (effective === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches);
  const tokens = { ...(skin?.tokens ?? {}), ...((dark ? skin?.dark : skin?.light) ?? {}) };
  for (const [k, v] of Object.entries(tokens)) {
    root.style.setProperty(k, v);
    applied.push(k);
  }
  root.dataset.skin = skin?.id ?? 'fluent';
  return effective;
}

/** Every skin the user can pick: built-ins first, then imported ones. */
export function allSkins(custom?: Array<Record<string, unknown>>): Skin[] {
  const imported = (custom ?? []).filter((s): s is Record<string, unknown> & Skin => typeof s?.id === 'string' && typeof s?.name === 'string');
  return [...BUILTIN_SKINS, ...imported.filter((s) => !BUILTIN_SKINS.some((b) => b.id === s.id))];
}

export function findSkin(id: string, custom?: Array<Record<string, unknown>>): Skin | undefined {
  return allSkins(custom).find((s) => s.id === id) ?? BUILTIN_SKINS[0];
}