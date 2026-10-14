# Publishing NexTone on the Microsoft Store

The Store build is an MSIX package with the Electron app and the audio engine. The Store signs
it for free, so no code-signing certificate is needed. Kernel drivers and HKLM ASIO
registration cannot ship in an MSIX. Those come from the separate desktop installer, the
"Driver Pack" (`ui/release/NexTone-Setup-*.exe`), which is optional.

## What works from the Store package alone
- Physical inputs and outputs (WASAPI, plus ASIO interfaces already installed on the PC).
- Application bins: per-app capture and per-app output/mic redirection.
- Mixes, effects, scenes, hotkeys, recording and the instant-replay buffer.
- VBAN network audio.
- Virtual devices in **Compatibility mode**, which uses any installed signed virtual cable
  such as VB-CABLE.
- Package wiring, verified in the test VM (`vm/Test-Msix.ps1`):
  - **Sign-in.** The package starts the engine at sign-in (startup task), and the engine
    opens the app in the tray.
  - **ASIO launch.** The `nextone-engine.exe` execution alias lets the NexTone ASIO driver
    (from the Driver Pack) start the engine.
  - **Control panel.** The `nextone:` link lets the ASIO driver's control-panel button open
    the app.

## One-time Partner Center setup (you)
1. In Partner Center, go to **Apps and games → New product → MSIX or PWA app** and
   reserve a name, e.g. "NexTone". If it is taken, use something like "NexTone Audio Router".
2. Open **Product management → Product identity** and copy three values into
   `ui/electron-builder.store.json` under `appx`:
   - `Package/Identity/Name` goes into `identityName`.
   - `Package/Identity/Publisher` goes into `publisher` (it looks like `CN=XXXXXXXX-XXXX-...`).
   - `Package/Properties/PublisherDisplayName` goes into `publisherDisplayName`.
3. Build the package:
   ```powershell
   .\scripts\build.ps1            # engine (x64 release)
   cd ui; npm run dist:store      # → ui\release-store\NexTone-<version>.appx
   ```
   Upload it unsigned; the Store signs it.
4. Fill in the submission:
   - **Pricing:** Free. Markets: all.
   - **Properties:** Category *Music*, subcategory *Music production* (or *Utilities & tools*).
   - **Privacy policy URL:** host [PRIVACY.md](PRIVACY.md) somewhere public, e.g. GitHub, and
     link it. NexTone collects no data.
   - **Age ratings:** complete the questionnaire. Answer "no" to everything, giving the
     lowest rating.
   - **Store listing:** use the text in [listing.md](listing.md). Take screenshots with
     `cd ui; npx playwright test`; they appear in `ui/test-results/screens/`. Store
     screenshots must be at least 1366×768.
   - **Restricted capability `runFullTrust`:** paste the justification from
     [listing.md](listing.md).
   - **Submission options → Notes for certification:** paste the tester notes from
     [listing.md](listing.md).
5. **GPLv3 obligation.** Publish the full source, e.g. a public GitHub repository, and link
   it in the description and the About page. Every Store release must have matching
   source available.

## Local testing before upload
```powershell
# Sign with the local dev certificate (CN=NexTone Dev, created on this PC) and test in the VM:
signtool sign /fd SHA256 /f packaging\msix\nextone-dev.pfx /p nextone-dev ui\release-store\NexTone-0.1.0.appx
.\vm\Test-Msix.ps1     # install, launch, sign-in startup, alias, protocol, CLI
.\vm\Test-Wack.ps1     # Windows App Certification Kit
```
The local `publisher` must stay `CN=NexTone Dev` for the dev certificate to sign the package.
Change it to the Partner Center value only for the upload build.

## Driver Pack (outside the Store)
`cd ui; npm run dist:win` builds `ui\release\NexTone-Setup-<version>.exe`. It contains:
- the NexTone ASIO driver (x64 + x86, user-mode; works on every PC);
- the NexTone virtual audio driver.

The virtual audio driver is **test-signed**. It only loads with test signing enabled, so
public use needs Microsoft attestation signing, which requires an EV certificate. Until then,
the Store app uses Compatibility mode for virtual devices.
