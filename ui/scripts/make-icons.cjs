// Derives every icon from the ToneSphere logo (assets/images/ToneSphere.png at the repo root) and writes:
//   assets/icon.png (512), assets/icon.ico (16-256), assets/tray.png (32), assets/tray@2x.png (64)
//   src/renderer/assets/logo.png (64, the title-bar mark)
//   build/appx/*.png (Store/MSIX tile assets)
// The splash (assets/splash.webm, splash-poster.jpg) is the README banner re-encoded:
//   ffmpeg -i ../assets/images/ToneSphereBanner.gif -vf "scale=660:330:flags=lanczos,format=yuv420p" \
//          -c:v libvpx-vp9 -b:v 0 -crf 34 -row-mt 1 -an assets/splash.webm
// Run: npx electron scripts/make-icons.cjs
const { app, nativeImage } = require('electron');
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const master = nativeImage.createFromPath(path.join(root, '..', 'assets', 'images', 'ToneSphere.png'));

const square = (size) => master.resize({ width: size, height: size, quality: 'best' }).toPNG();

// A tile: the logo centred on a transparent canvas (Windows draws the tile plate).
function tile(w, h, scale = 0.8) {
  const s = Math.round(Math.min(w, h) * scale);
  const logo = master.resize({ width: s, height: s, quality: 'best' }).toBitmap(); // BGRA
  const out = Buffer.alloc(w * h * 4);
  const ox = Math.floor((w - s) / 2);
  const oy = Math.floor((h - s) / 2);
  for (let y = 0; y < s; y++) logo.copy(out, ((oy + y) * w + ox) * 4, y * s * 4, (y + 1) * s * 4);
  return nativeImage.createFromBitmap(out, { width: w, height: h }).toPNG();
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

app.whenReady().then(() => {
  if (master.isEmpty()) throw new Error('assets/images/ToneSphere.png not found');
  for (const d of ['assets', 'build/appx', 'src/renderer/assets']) fs.mkdirSync(path.join(root, d), { recursive: true });

  const sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256];
  fs.writeFileSync(path.join(root, 'assets', 'icon.ico'), ico(sizes.map((size) => ({ size, data: square(size) }))));
  fs.writeFileSync(path.join(root, 'assets', 'icon.png'), square(512));
  fs.writeFileSync(path.join(root, 'assets', 'tray.png'), square(32));
  fs.writeFileSync(path.join(root, 'assets', 'tray@2x.png'), square(64));
  fs.writeFileSync(path.join(root, 'src', 'renderer', 'assets', 'logo.png'), square(64));

  const appx = [
    ['StoreLogo.png', 50, 50],
    ['Square44x44Logo.png', 44, 44],
    ['Square150x150Logo.png', 150, 150],
    ['Wide310x150Logo.png', 310, 150],
    ['LargeTile.png', 310, 310],
    ['SmallTile.png', 71, 71],
  ];
  for (const [name, w, h] of appx) fs.writeFileSync(path.join(root, 'build', 'appx', name), tile(w, h));
  console.log('icons written');
  app.quit();
});
