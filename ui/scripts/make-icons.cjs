// Renders the NexTone logo (SVG) to all icon sizes with Electron and writes:
//   assets/icon.png (512), assets/icon.ico (16-256), assets/tray.png (32), assets/tray@2x.png
//   build/appx/*.png (Store/MSIX tile assets)
// Run: npx electron scripts/make-icons.cjs
const { app, BrowserWindow } = require('electron');
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const logo = (size, { tile = false } = {}) => `
<svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size}" viewBox="0 0 100 100">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#3d7bff"/>
      <stop offset="0.55" stop-color="#8a5cf5"/>
      <stop offset="1" stop-color="#ef4f6b"/>
    </linearGradient>
  </defs>
  ${tile ? '' : '<rect x="4" y="4" width="92" height="92" rx="24" fill="url(#g)"/>'}
  <path d="M17 50 C27 50 27 26 37 26 S47 74 57 74 S67 26 77 26 S83 50 85 50"
        fill="none" stroke="#fff" stroke-width="${size <= 24 ? 11 : 9}" stroke-linecap="round" stroke-linejoin="round"/>
</svg>`;

async function render(win, svg, w, h, transparentTile = false) {
  const html = `<html><body style="margin:0;background:${transparentTile ? 'transparent' : 'transparent'}">
    <div style="width:${w}px;height:${h}px;display:flex;align-items:center;justify-content:center">${svg}</div></body></html>`;
  await win.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(html));
  win.setContentSize(w, h);
  await new Promise((r) => setTimeout(r, 80));
  const img = await win.webContents.capturePage({ x: 0, y: 0, width: w, height: h });
  return img.resize({ width: w, height: h, quality: 'best' }).toPNG();
}

function ico(pngs) {
  // ICO with embedded PNG images.
  const header = Buffer.alloc(6 + 16 * pngs.length);
  header.writeUInt16LE(0, 0);
  header.writeUInt16LE(1, 2);
  header.writeUInt16LE(pngs.length, 4);
  let offset = header.length;
  pngs.forEach(({ size, data }, i) => {
    const e = 6 + 16 * i;
    header.writeUInt8(size >= 256 ? 0 : size, e);
    header.writeUInt8(size >= 256 ? 0 : size, e + 1);
    header.writeUInt8(0, e + 2);
    header.writeUInt8(0, e + 3);
    header.writeUInt16LE(1, e + 4);
    header.writeUInt16LE(32, e + 6);
    header.writeUInt32LE(data.length, e + 8);
    header.writeUInt32LE(offset, e + 12);
    offset += data.length;
  });
  return Buffer.concat([header, ...pngs.map((p) => p.data)]);
}

app.disableHardwareAcceleration();
app.whenReady().then(async () => {
  const win = new BrowserWindow({ show: false, frame: false, transparent: true, width: 600, height: 600, webPreferences: { offscreen: true } });
  fs.mkdirSync(path.join(root, 'assets'), { recursive: true });
  fs.mkdirSync(path.join(root, 'build', 'appx'), { recursive: true });

  const sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256];
  const pngs = [];
  for (const s of sizes) pngs.push({ size: s, data: await render(win, logo(s), s, s) });
  fs.writeFileSync(path.join(root, 'assets', 'icon.ico'), ico(pngs));
  fs.writeFileSync(path.join(root, 'assets', 'icon.png'), await render(win, logo(512), 512, 512));
  fs.writeFileSync(path.join(root, 'assets', 'tray.png'), await render(win, logo(32), 32, 32));
  fs.writeFileSync(path.join(root, 'assets', 'tray@2x.png'), await render(win, logo(64), 64, 64));

  // MSIX assets (logo on transparent background; Windows draws the tile plate).
  const appx = [
    ['StoreLogo.png', 50, 50],
    ['Square44x44Logo.png', 44, 44],
    ['Square150x150Logo.png', 150, 150],
    ['Wide310x150Logo.png', 310, 150],
    ['LargeTile.png', 310, 310],
    ['SmallTile.png', 71, 71],
  ];
  for (const [name, w, h] of appx) {
    const s = Math.round(Math.min(w, h) * 0.8);
    fs.writeFileSync(path.join(root, 'build', 'appx', name), await render(win, logo(s), w, h));
  }
  console.log('icons written');
  app.quit();
});
