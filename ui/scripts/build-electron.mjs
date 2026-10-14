// Bundles the Electron main + preload scripts with esbuild.
import { build } from 'esbuild';

const watch = process.argv.includes('--watch');

const common = {
  bundle: true,
  platform: 'node',
  target: 'node22',
  format: 'cjs',
  sourcemap: true,
  external: ['electron'],
  logLevel: 'info',
};

const entries = [
  { entryPoints: ['electron/main.ts'], outfile: 'dist-electron/main.cjs' },
  { entryPoints: ['electron/preload.ts'], outfile: 'dist-electron/preload.cjs' },
];

if (watch) {
  const { context } = await import('esbuild');
  for (const e of entries) {
    const ctx = await context({ ...common, ...e });
    await ctx.watch();
  }
} else {
  await Promise.all(entries.map((e) => build({ ...common, ...e })));
}
