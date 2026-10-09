# NexTone Virtual Audio Driver (Windows)

Kernel-mode WaveRT driver that creates **NexTone Output 1–4** (virtual playback devices apps
play into) and **NexTone Mic 1–4** (virtual microphones apps record from). Audio moves between
each endpoint and the NexTone engine through a named shared-memory ring
(`include/nextone_vad_shared.h`); no IOCTLs, so the engine can attach/detach at any time.

* License: **MS-PL** (derived from Microsoft's SimpleAudioSample, see `LICENSE-MS-PL.txt`).
  It is a separate component; the GPL engine talks to it only through the shared-memory layout.
* Build: `.\build.ps1` (CMake-free; uses the WDK headers/libs directly) → `build\package\`
  containing `nextonevad.sys/.inf/.cat` signed with a **self-signed test certificate**.
* Loading it requires Windows test-signing mode. During development it is installed and tested
  **only inside the Hyper-V test VM** (`vm\Deploy-Driver.ps1`, `vm\Test-InVM.ps1`).
* For distribution to other PCs the package must be attestation-signed by Microsoft
  (requires an EV code-signing certificate + Partner Center hardware program). No code changes needed.

Changes vs. the sample: WDF, tone generator and file-dump code removed; endpoints built at
runtime with unique names (pin Name GUIDs registered under MediaCategories by the INF); render
data is copied into the ring in `ReadBytes`, capture data comes from the ring in `WriteBytes`.
