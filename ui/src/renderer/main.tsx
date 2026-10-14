import './legacyStorage';
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { engine } from './store';
// JetBrains Mono (SIL OFL 1.1), bundled so it works offline: monospace text and numeric readouts.
import '@fontsource-variable/jetbrains-mono';
import './styles.css';

if (navigator.userAgent.includes('Windows')) document.documentElement.dataset.os = 'windows';
else if (navigator.userAgent.includes('Mac')) document.documentElement.dataset.os = 'mac';
else document.documentElement.dataset.os = 'linux';
if (window.tonesphere) document.documentElement.dataset.shell = 'electron';

void engine.connect();

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
