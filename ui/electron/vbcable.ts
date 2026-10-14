// VB-CABLE: VB-Audio's free (donationware) virtual audio cable, used for virtual devices while the
// ToneSphere virtual audio driver isn't signed by Microsoft. Its licence allows passing the package
// on unmodified but not building it into another installer, so ToneSphere only downloads the
// official package from vb-audio.com when the user asks; tonesphere-setup then checks VB-Audio's
// signature and opens VB-Audio's own installer.
import { execFile } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

export const VBCABLE_URL = 'https://download.vb-audio.com/Download_CABLE/VBCABLE_Driver_Pack45.zip';
export const VBCABLE_INFO_URL = 'https://vb-audio.com/Cable/';

/** Downloads and unpacks the official VB-CABLE package; returns the folder holding VBCABLE_Setup_x64.exe. */
export async function downloadVbCable(): Promise<string> {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tonesphere-vbcable-'));
  const zip = path.join(dir, 'VBCABLE_Driver_Pack.zip');
  let res: Response;
  try {
    res = await fetch(VBCABLE_URL, { redirect: 'follow', signal: AbortSignal.timeout(120_000) });
  } catch {
    throw new Error("VB-CABLE couldn't be downloaded. Check the internet connection and try again.");
  }
  if (!res.ok) throw new Error(`VB-CABLE couldn't be downloaded (the server answered ${res.status}).`);
  const data = Buffer.from(await res.arrayBuffer());
  if (data.length < 100_000 || data.length > 50_000_000) throw new Error("The VB-CABLE download doesn't look right, so it wasn't used.");
  fs.writeFileSync(zip, data);
  const out = path.join(dir, 'pack');
  fs.mkdirSync(out);
  // Windows 10+ ships bsdtar, which reads zip files.
  const tar = path.join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'tar.exe');
  await new Promise<void>((resolve, reject) =>
    execFile(tar, ['-xf', zip, '-C', out], { windowsHide: true }, (err) => (err ? reject(new Error("The VB-CABLE download couldn't be unpacked.")) : resolve())),
  );
  if (!fs.existsSync(path.join(out, 'VBCABLE_Setup_x64.exe'))) throw new Error("The VB-CABLE download doesn't contain its setup.");
  return out;
}
