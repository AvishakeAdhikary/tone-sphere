import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

export default defineConfig({
  root: 'src/renderer',
  base: './',
  plugins: [react()],
  resolve: {
    alias: {
      '@shared': fileURLToPath(new URL('./src/shared', import.meta.url)),
    },
  },
  server: { port: 5199, strictPort: true },
  build: {
    outDir: '../../dist',
    emptyOutDir: true,
    target: 'chrome140',
  },
  test: {
    root: '.',
    include: ['src/**/*.test.ts'],
  },
} as never);
