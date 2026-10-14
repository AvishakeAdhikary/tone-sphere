// Interface language. English text is the key (and the fallback); other languages are catalogs
// keyed by that text. The app re-renders from the top when the language changes (App's key).
//   t('Sources')                       -> 'स्रोत' in Hindi
//   t('{n} cores', { n: 4 })           -> placeholders in braces
import { hi } from './locales/hi';

export type Lang = 'en' | 'hi';

export const LANGUAGES: { id: 'system' | Lang; label: string; note?: string }[] = [
  { id: 'system', label: 'Match Windows' },
  { id: 'en', label: 'English' },
  // Like the old ToneSphere's catalog: written by a machine, not yet checked by a native speaker.
  { id: 'hi', label: 'हिन्दी (Hindi)', note: 'Machine translation, not yet reviewed' },
];

const CATALOGS: Record<Lang, Record<string, string>> = { en: {}, hi };
let current: Lang = 'en';

/** The language to use for a setting ('system' follows the OS / browser language). */
export function resolveLanguage(setting: string | undefined): Lang {
  if (setting === 'en' || setting === 'hi') return setting;
  return navigator.language.toLowerCase().startsWith('hi') ? 'hi' : 'en';
}

export function setLanguage(lang: Lang): void {
  current = lang;
  document.documentElement.lang = lang;
}

export function currentLanguage(): Lang {
  return current;
}

export function t(text: string, vars?: Record<string, string | number>): string {
  let s = CATALOGS[current][text] ?? text;
  if (vars) for (const [k, v] of Object.entries(vars)) s = s.split(`{${k}}`).join(String(v));
  return s;
}

/** Keys missing from a catalog (used by the catalog test). */
export function missingKeys(lang: Lang, keys: string[]): string[] {
  return keys.filter((k) => !(k in CATALOGS[lang]));
}
