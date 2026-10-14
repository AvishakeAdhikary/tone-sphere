# Publishing ToneSphere on the Microsoft Store

The Store build is an MSIX package with the Electron app and the audio engine. The Store signs
it for free, so no code-signing certificate is needed. Kernel drivers and HKLM ASIO
registration cannot ship in an MSIX. Those come from the separate desktop installer
(`ToneSphere-Setup-<version>.exe` on GitHub Releases), which is optional.

CI builds the unsigned Store package on every push: `ToneSphere-<version>.appx` is attached to
each GitHub release, ready to upload once the identity below is filled in.

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
  - **ASIO launch.** The `tonesphere-engine.exe` execution alias lets the ToneSphere ASIO driver
    (from the desktop installer) start the engine.
  - **Control panel.** The `tonesphere:` link lets the ASIO driver's control-panel button open
    the app.

## One-time Partner Center setup (you)
1. In Partner Center, go to **Apps and games → New product → MSIX or PWA app** and
   reserve a name, e.g. "ToneSphere". If it is taken, use something like "ToneSphere Audio Router".
2. Open **Product management → Product identity** and copy three values into
   `ui/electron-builder.store.json` under `appx`:
   - `Package/Identity/Name` goes into `identityName`.
   - `Package/Identity/Publisher` goes into `publisher` (it looks like `CN=XXXXXXXX-XXXX-...`).
   - `Package/Properties/PublisherDisplayName` goes into `publisherDisplayName`.
3. Build the package:
   ```powershell
   .\scripts\build.ps1            # engine (x64 release)
   cd ui; npm run dist:store      # → ui\release-store\ToneSphere-<version>.appx
   ```
   Upload it unsigned; the Store signs it.
4. Fill in the submission:
   - **Pricing:** Free. Markets: all.
   - **Properties:** Category *Music*, subcategory *Music production* (or *Utilities & tools*).
   - **Privacy policy URL:** <https://avishakeadhikary.github.io/tone-sphere/legal/privacy-policy/>
     (published from `docs/legal/` by `.github/workflows/pages.yml`). ToneSphere collects no data.
   - **Age ratings:** complete the questionnaire. Answer "no" to everything, giving the
     lowest rating.
   - **Store listing:** use the text in [listing.md](listing.md). Take screenshots with
     `cd ui; npx playwright test`; they appear in `ui/test-results/screens/`. Store
     screenshots must be at least 1366×768.
   - **Restricted capability `runFullTrust`:** paste the justification from
     [listing.md](listing.md).
   - **Submission options → Notes for certification:** paste the tester notes from
     [listing.md](listing.md).
5. **GPLv3 obligation.** The source is public at <https://github.com/AvishakeAdhikary/tone-sphere>,
   and every GitHub release carries the matching `ToneSphere-<version>-source.zip`. Upload the
   `.appx` from the same release, so the Store version always has matching source.

## Local testing before upload
```powershell
# A local test certificate (git-ignored, never committed), then sign and test in the VM:
New-SelfSignedCertificate -Type CodeSigningCert -Subject 'CN=Neural Nexus Studios' -CertStoreLocation Cert:\CurrentUser\My |
  Export-PfxCertificate -FilePath packaging\msix\tonesphere-dev.pfx -Password (Read-Host -AsSecureString 'pfx password')
signtool sign /fd SHA256 /f packaging\msix\tonesphere-dev.pfx /p <password> ui\release-store\ToneSphere-<version>.appx
.\vm\Test-Msix.ps1     # install, launch, sign-in startup, alias, protocol, CLI
.\vm\Test-Wack.ps1     # Windows App Certification Kit
```
The local `publisher` must stay `CN=Neural Nexus Studios` for the dev certificate to sign the package.
Change it to the Partner Center value only for the upload build.

## Desktop installer (outside the Store)
`cd ui; npm run dist:win` builds `ui\release\ToneSphere-Setup-<version>.exe`. It contains:
- the ToneSphere ASIO driver (x64 + x86, user-mode; works on every PC);
- in development builds only, the ToneSphere virtual audio driver.

The virtual audio driver is **test-signed**. It only loads with test signing enabled, so public
builds leave it out until Microsoft attestation-signs it (which requires an EV certificate).
Until then virtual devices come from VB-CABLE in Compatibility mode, which the desktop app
installs from Settings > Devices > Install drivers.
