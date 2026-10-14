// ToneSphere was called NexTone before: its remembered UI choices used "nextone." keys.
// Imported before the store (which reads them at load), it renames those keys once.
try {
  for (let i = localStorage.length - 1; i >= 0; i--) {
    const key = localStorage.key(i);
    if (!key?.startsWith('nextone.')) continue;
    const next = 'tonesphere.' + key.slice('nextone.'.length);
    const value = localStorage.getItem(key);
    if (localStorage.getItem(next) === null && value !== null) localStorage.setItem(next, value);
    localStorage.removeItem(key);
  }
} catch {
  /* storage unavailable */
}

export {};
