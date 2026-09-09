# Microsoft Store submission

ToneSphere as an MSIX package, for the Microsoft Store, published by **Neural Nexus
Studios** (individual developer account, Kolkata, West Bengal, India).

Read this first, because it governs everything below:

> **Nothing in this repository has been submitted to, accepted by, or installed from the
> Microsoft Store.** No Store identity exists yet. No certificate exists yet. The package
> this repository builds is a locally-packable, locally-sideloadable MSIX carrying
> **placeholder identity values**, and the Store will reject it as-is. The section
> [What is genuinely untested](#what-is-genuinely-untested) is not a disclaimer at the
> bottom; it is the honest half of this document.

## Status

| | |
|---|---|
| `packaging/msix/AppxManifest.xml` — Desktop Bridge manifest, schema-valid | **done**, validated by `makeappx pack` |
| `packaging/msix/generate_assets.py` — the 40-file logo set from one brand PNG | **done**, run for real |
| `packaging/msix/build_msix.ps1` — freeze, stage, pack, optionally sign | **done** through the pack step: run end to end on Windows 11 build 26200 / SDK 10.0.26100.8249, producing an 87.8 MB unsigned x64 `.msix` (317 entries, `ToneSphere/ToneSphere.exe` at the declared path). The `signtool` branch has never run — no certificate exists |
| `tests/test_msix_packaging.py` — manifest, capability set, version, asset coverage | **done**, in the CI-safe suite |
| Real Store `Identity` values in the manifest | **owner**, Partner Center only |
| Signed package installed and launched from an MSIX | **not done** — see below |
| Partner Center product, age rating, privacy policy URL, submission | **owner** |

## 1. Partner Center account

1. Register at <https://partner.microsoft.com/dashboard> for a **Windows & Xbox** developer
   account, account type **Individual**.
   - Publisher display name: `Neural Nexus Studios`. This is the string shown on the Store
     listing and it must match `<PublisherDisplayName>` in the manifest, which is already
     set to it. Microsoft verifies the name; an individual account cannot claim a name that
     reads as a registered company it cannot prove it owns, so if `Neural Nexus Studios` is
     rejected during verification, the manifest value has to change with it.
   - Country/region: India. There is a one-time registration fee for individual accounts in
     some regions; Partner Center shows the current amount for India at sign-up. **This
     document does not state a figure, because none was verified.**
2. Complete the **payout and tax profile** before the first submission if the app will ever
   be paid or carry ads — for India that means PAN and the W-8BEN equivalent Partner Center
   walks through. A free app with no ads does not need it to publish, but a submission that
   later becomes paid stalls without it.
3. **Reserve the app name**: Dashboard → *Apps and games* → *New product* → *MSIX or PWA
   app* → reserve `ToneSphere`. The reservation is what mints the identity values in step 2.

## 2. The three placeholder values in the manifest

Open `packaging/msix/AppxManifest.xml`. The `<Identity>` element holds three values, all
marked with `PLACEHOLDER` comments in the file itself.

In Partner Center, go to the reserved product → **Product management** → **Product
identity**. That page shows exactly these strings; copy them verbatim.

| Manifest attribute | Current placeholder | Replace with |
|---|---|---|
| `Identity/@Name` | `NeuralNexusStudios.ToneSphere` | **Package/Identity/Name** from Product identity — a Store-assigned string, typically `<12 digits>.ToneSphere` |
| `Identity/@Publisher` | `CN=Neural Nexus Studios` | **Package/Identity/Publisher** from Product identity — a `CN=<guid>` string, *not* the display name |
| `Identity/@Version` | `0.1.0.0` | Nothing by hand. `build_msix.ps1` stamps it from `pyproject.toml` at pack time (see [Versioning](#5-versioning)) |

`Package/Identity/Publisher Display Name` on that same page must equal
`<PublisherDisplayName>` in the manifest, which is already `Neural Nexus Studios`.

Two values that are *not* placeholders and should not be changed:

- `<DisplayName>ToneSphere</DisplayName>` and `uap:VisualElements/@DisplayName` — the app
  name as Windows shows it. Partner Center's "Store-assigned display name" is the reserved
  name and must match this.
- `Application/@Id` is `ToneSphere`. It is internal, it is not the Store identity, and
  changing it after the first published version breaks upgrades and Start-menu pins.

**Keeping the placeholders out of a submitted package.** The manifest in git is deliberately
the sideload-testable version. Substitute the real values in a local working copy at
submission time (or keep a private copy of the file and pass it in); do not commit the
Store-assigned `Publisher` GUID if the repository is public — it is not a secret, but it is
noise that invites someone to build a confusable package. `tests/test_msix_packaging.py`
asserts only the *shape* of `Name`/`Publisher`, so substituting real values does not break
the suite.

## 3. Building the package

Prerequisites, on Windows:

- **PowerShell 7** (`pwsh`). The script declares `#Requires -Version 7.0`; Windows'
  built-in 5.1 will refuse it rather than fail halfway.
- **Windows 10/11 SDK**, for `makeappx.exe` and `signtool.exe`:
  `winget install --id Microsoft.WindowsSDK.10.0.26100`, or Visual Studio Installer →
  *Individual components* → *Windows 11 SDK*. The SDK does not put its tools on PATH, so
  `build_msix.ps1` also looks under
  `%ProgramFiles(x86)%\Windows Kits\10\bin\<version>\x64\`. If it finds neither it stops
  with the install command and does not leave a half-made package behind.
- `uv sync --all-groups`.

```powershell
# freeze, generate logos, stage, pack -- unsigned
pwsh -File packaging/msix/build_msix.ps1

# ...and sign it for local sideload testing
pwsh -File packaging/msix/build_msix.ps1 -CertificatePath .\ToneSphereTest.pfx -CertificatePassword <pw>

# iterate on the manifest/logos without re-freezing (reuses dist/ToneSphere)
pwsh -File packaging/msix/build_msix.ps1 -SkipFreeze
```

Output: `build/msix/ToneSphere-<version>-x64.msix`, with the staged layout left in
`build/msix/layout/` for inspection. It comes out around **88 MB** — PySide6 and pedalboard
dominate that, and it is well inside the Store's limits, but it is the number to watch if
`tonesphere.spec`'s `excludes` list is ever trimmed.

The layout is:

```
AppxManifest.xml
assets\                     <- 40 generated logo files
ToneSphere\ToneSphere.exe   <- the PyInstaller one-folder output, verbatim
ToneSphere\_internal\...
```

The frozen app sits in its own subdirectory so the package-root `assets\` (Store logos)
cannot collide with the app's own bundled `assets\images\`. `Executable=` in the manifest
points at that path, and a test asserts the two agree — get this wrong and the package
installs cleanly and then does nothing when launched.

### Logo assets

`packaging/msix/generate_assets.py` derives all 40 files from `assets/images/ToneSphere.png`
(500x500). They are **not committed** — `packaging/msix/assets/` is gitignored, and
`build_msix.ps1` regenerates the set before every pack.

| Logo | 100% size | Variants generated |
|---|---|---|
| `StoreLogo` | 50x50 | `scale-125/150/200/400` |
| `Square44x44Logo` | 44x44 | `scale-125/150/200/400`, plus `targetsize-16/24/32/48/256` and the `_altform-unplated` form of each |
| `Square71x71Logo` (small tile) | 71x71 | `scale-125/150/200/400` |
| `Square150x150Logo` (medium tile) | 150x150 | `scale-125/150/200/400` |
| `Square310x310Logo` (large tile) | 310x310 | `scale-125/150/200/400` |
| `Wide310x150Logo` (wide tile) | 310x150 | `scale-125/150/200/400` |

The unqualified file (`Square150x150Logo.png`) is the 100% form and doubles as the neutral
resource; `makeappx` checks that literal path exists, and Windows' resource loader picks a
`.scale-NNN` sibling on a higher-DPI display.

**Known defect:** the 500x500 source is too small for one variant.
`Square310x310Logo.scale-400.png` needs 818px of artwork and is upscaled from 500px, so it
will look soft on a 400%-scale display. The script prints exactly which files it had to
upscale. Fix it by replacing `assets/images/ToneSphere.png` with a 1024x1024-or-larger
master before submitting; nothing else in the pipeline changes.

No `uap:SplashScreen` is declared and no splash image is generated: a
`Windows.FullTrustApplication` never shows one, so shipping the asset would mean the package
carries a logo it can never use.

The Store *listing* images — screenshots (`assets/images/screenshot.png` is a start), the
2400x1200 hero art, the promotional tiles — are uploaded in Partner Center and are not part
of the package. Nothing here generates them.

### Local signing, for sideload testing only

The Store re-signs every submitted package with its own certificate, so **the certificate
below has nothing to do with the published app.** It exists only so Windows will install
the package on your own machine.

```powershell
# 1. a self-signed code-signing cert whose subject EXACTLY equals the manifest Publisher
$cert = New-SelfSignedCertificate -Type Custom `
  -Subject "CN=Neural Nexus Studios" `
  -KeyUsage DigitalSignature -FriendlyName "ToneSphere sideload test" `
  -CertStoreLocation "Cert:\CurrentUser\My" `
  -TextExtension @("2.5.29.37={text}1.3.6.1.5.5.7.3.3", "2.5.29.19={text}")

# 2. export it
$pw = ConvertTo-SecureString -String "<pick one>" -Force -AsPlainText
Export-PfxCertificate -Cert "Cert:\CurrentUser\My\$($cert.Thumbprint)" `
  -FilePath .\ToneSphereTest.pfx -Password $pw

# 3. trust it (admin) -- Windows will not install an MSIX signed by an untrusted cert
Import-PfxCertificate -FilePath .\ToneSphereTest.pfx `
  -CertStoreLocation "Cert:\LocalMachine\TrustedPeople" -Password $pw

# 4. build + sign, then install
pwsh -File packaging/msix/build_msix.ps1 -CertificatePath .\ToneSphereTest.pfx -CertificatePassword "<pick one>"
Add-AppxPackage -Path .\build\msix\ToneSphere-0.1.0.0-x64.msix
```

The certificate subject must match `Identity/@Publisher` **exactly**. `build_msix.ps1`
compares the two before calling signtool, because signtool's own failure for this is an
opaque `0x8007000B`. Note the consequence: once you substitute the Store-assigned
`CN=<guid>` publisher into the manifest, the old test certificate no longer matches it, and
you must issue a new one with that GUID subject to keep sideloading.

Never commit the `.pfx` or its password.

### Store signing versus this certificate

| | Local sideload | Store submission |
|---|---|---|
| Who signs | you, with the self-signed `.pfx` above | Microsoft, during ingestion |
| `Identity/@Publisher` must equal | your certificate's subject | the Partner Center `CN=<guid>` |
| Trust | only on machines where you imported the cert into `TrustedPeople` | every Windows machine |
| Certificate needed for submission | — | **none of your own** |

You do not need a paid code-signing certificate to publish through the Store. (An EV
certificate *is* required for the Windows kernel-mode virtual audio driver route, which this
project deliberately does not attempt — see
[VIRTUAL_AUDIO_DRIVER.md](VIRTUAL_AUDIO_DRIVER.md). Do not conflate the two.)

Whether Partner Center accepts a completely **unsigned** `.msix` upload, or requires a
signature whose subject matches the Store publisher, was **not verified** while writing
this. If the upload is rejected on signature grounds, sign it with a self-signed certificate
whose subject is exactly the Store-assigned `CN=<guid>` and re-upload; that is the
documented path.

## 4. Capabilities, and what a reviewer will ask about

The package declares exactly two capabilities. The full reasoning is in the manifest's own
comments; the short version, which is also the text to paste into Partner Center's
*Notes for certification*:

**`runFullTrust` (restricted capability).** ToneSphere is a full-trust Win32 desktop
application packaged with the Desktop Bridge, not a UWP app. It loads native audio libraries
(PortAudio through cffi, a VST3 plug-in host), calls WASAPI COM interfaces directly through
`ctypes`, and enumerates Windows audio sessions. None of that is possible inside an
AppContainer. Partner Center flags restricted capabilities and may ask for a justification
before certification; this is it.

**`microphone` (device capability).** The app's purpose is routing live audio, so it opens
real capture streams (`tonesphere/engine/host.py`) — an audio interface, a USB microphone,
a line input. On Windows 10 build 20348 and later it also captures a single application's
output through `ActivateAudioInterfaceAsync` with `VIRTUAL_AUDIO_DEVICE_PROCESS_LOOPBACK`
(`tonesphere/engine/process_capture.py`); Windows classifies that as microphone access and
surfaces it under *Settings → Privacy → Microphone* alongside ordinary capture. Audio is
processed in-process and sent only where the user patches it. Nothing is recorded to disk or
transmitted anywhere without an explicit route.

Deliberately **not** declared, and not to be added speculatively:

- `internetClient` / `internetClientServer` / `privateNetworkClientServer` — the REST and
  WebSocket API (`tonesphere/api`) and the TCP audio router (`tonesphere/network`) do open
  sockets, but a `runFullTrust` process is not in an AppContainer, so these capabilities
  gate nothing for this app. Declaring them would widen the permission prompt the user sees
  and hand review an extra question, for no behaviour.
- `broadFileSystemAccess` — presets are YAML in the user's own profile.

`tests/test_msix_packaging.py` asserts the capability set **exactly**, and names the
over-broad capabilities it must not contain. Adding one means arguing for it in that test
first.

## 5. Versioning

The version now lives in three places, and MSIX has its own rules:

1. `pyproject.toml` → `version = "0.1.0"` — the source of truth. CI's auto-tag job releases
   on a change here (`.github/workflows/ci.yml`).
2. `tonesphere/__init__.py` → `__version__ = "0.1.0"`.
3. `packaging/msix/AppxManifest.xml` → `Version="0.1.0.0"`.

`build_msix.ps1` reads (1) and (2), refuses to build if they disagree, converts to the
four-part MSIX form with a **zero fourth part** (the Store rejects a non-zero one), and
stamps that into the staged manifest — so the packed `.msix` always carries the project
version even if (3) drifted. It warns loudly when it had to.
`tests/test_msix_packaging.py::TestVersionTracksTheProject` then fails if (3) does not match
(1) and (2), so the checked-in file cannot stay wrong.

The Store also requires each submission's version to be **strictly greater** than the last
published one. It never goes backwards; a rejected submission still burns its version
number in practice, so bump `pyproject.toml` rather than re-uploading the same version.

## 6. Submission checklist in Partner Center

Per-submission, on the reserved product:

- **Packages** — upload `build/msix/ToneSphere-<version>-x64.msix` (built with the real
  identity values substituted). A single x64 package is fine; there is no arm64 build,
  because PyInstaller here freezes for the machine it runs on and no arm64 Windows machine
  has built or tested this.
- **Age ratings** — Partner Center runs the **IARC** questionnaire. ToneSphere has no user
  generated content, no ads, no in-app purchase, no data sharing, no violence and no chat,
  so the questionnaire is short and the outcome should be the lowest rating in every region
  including India. Answer it honestly rather than copying an expected outcome from here; the
  rating is issued by IARC, not chosen.
- **Privacy policy URL** — *required*, and required specifically because the app declares
  the `microphone` device capability: any package that can access a capture device must
  point at a published privacy policy. The policy text is written in this repository at
  **`docs/legal/privacy-policy.md`** (authored separately; if that file is not there yet,
  the submission is blocked on it). It is intended to be served from this repository's
  GitHub Pages site, so the Partner Center field takes the **published URL of that page**.
  Get the URL from the repository's *Settings → Pages* panel after enabling Pages — do not
  guess it: it depends on the Pages source (branch root vs `/docs`) and on whether a custom
  domain is configured, and Pages is **not enabled on this repository yet**. Paste the URL
  that panel shows, then load it in a browser and confirm it renders before submitting.
- **Support contact info** — an email address that receives mail. Partner Center requires
  it and reviewers do use it.
- **Category** — *Music* (sub-category *Music creation and editing* if offered).
- **Description / features / screenshots** — README.md's opening and its honest feature
  list are the right source. Do not restore any claim README.md has already retracted; the
  Store listing is subject to the same rule as the code
  (see [AGENTS.md](../AGENTS.md)). In particular do not advertise a Windows virtual audio
  *device* — there isn't one (`docs/VIRTUAL_AUDIO_DRIVER.md`); per-process capture is the
  real feature and is worth describing as what it is.
- **Notes for certification** — paste the two capability justifications from section 4, and
  say plainly that the app requires an audio device and that a reviewer on a VM with no
  audio hardware will see it report no devices rather than crash.
- **Windows App Certification Kit** — run it before submitting:
  `"%ProgramFiles(x86)%\Windows Kits\10\App Certification Kit\appcert.exe" test -appxpackagepath <msix> -reportoutputpath report.xml`.
  It is the same suite Store ingestion runs. **It has not been run here.**

## What is genuinely untested

Everything in this section is a thing this repository does *not* establish. Read it before
telling anyone the app is Store-ready.

1. **No package has ever been installed.** `makeappx pack` succeeded here — the manifest is
   schema-valid and the layout is well-formed — but no MSIX has been signed, installed with
   `Add-AppxPackage`, or launched. The frozen app is smoke-tested by CI as a plain Win32
   folder; that is not the same as running from an installed MSIX.
2. **The `signtool` path in `build_msix.ps1` has never run.** No certificate exists. The
   subject/publisher comparison and the signtool invocation are written but unexercised. The
   `makeappx pack` step, by contrast, has been run for real on Windows 11 build 26200 with
   SDK 10.0.26100.
3. **Whether audio works from inside the package is unknown**, and it is the biggest open
   risk here, not a formality. Specifically unverified:
   - that PortAudio/WASAPI can open exclusive-mode streams from a packaged process;
   - that the microphone consent prompt appears and that a denial surfaces as the actionable
     error `AudioHost._explain_open_failure` produces rather than as a silent dead stream —
     which is exactly the "reports healthy, moves nothing" failure this project exists to
     prevent;
   - that per-process WASAPI loopback (`process_capture.py`) still activates under package
     identity;
   - that `app_capture.py`'s **`powershell.exe` subprocess** still runs. It should, since a
     full-trust package is not in an AppContainer, but "should" is not "was observed".
   Run `main.py test` from the *installed* app and check the measured latency and xrun
   figures before believing any of it.
4. **VST3 plug-in loading from a packaged install is unverified.** MSIX installs to a
   read-only, partly virtualised `WindowsApps` location; pedalboard's plug-in host loading
   third-party VST3 binaries from the user's own plug-in directories has not been tried
   there.
5. **Preset write paths are unverified under MSIX filesystem virtualisation.** Writes to the
   install directory are redirected or refused; presets are meant to live in the user
   profile, but no packaged run has confirmed where they actually land.
6. **No Partner Center account, product reservation, or identity exists.** Steps 1 and 2
   above are written from Microsoft's documented flow, not from having walked it. Field
   names and page locations in Partner Center change.
7. **Nothing in CI builds or checks the MSIX.** `.github/workflows/ci.yml` builds and
   releases the plain PyInstaller archives only, and was deliberately not modified. The MSIX
   is a local, manual build today.
8. **`MaxVersionTested="10.0.26100.0"`** in the manifest is the schema's required
   declaration, not a record of testing — see item 1. Raise it after a real installed test.
9. **One logo variant is a visible-quality defect**, not a passing detail:
   `Square310x310Logo.scale-400.png` is upscaled from a 500px source.
10. **No arm64 package.** x64 only.
