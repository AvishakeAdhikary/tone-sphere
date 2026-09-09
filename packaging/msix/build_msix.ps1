<#
.SYNOPSIS
    Builds an MSIX package for ToneSphere from the existing PyInstaller output.

.DESCRIPTION
    Freeze -> generate logos -> stage the package layout -> makeappx pack -> (optionally)
    signtool sign. It builds on top of tonesphere.spec rather than replacing it: the frozen
    app is exactly what CI already produces and smoke-tests, and this script only wraps it.

    The package this produces is sideload-testable and is NOT submittable as-is: the
    Identity in AppxManifest.xml still holds placeholder values, and only Partner Center can
    supply the real ones. See docs/MICROSOFT_STORE.md.

.PARAMETER CertificatePath
    A .pfx to sign the package with, for local sideload testing only. Its subject must equal
    the manifest's Publisher exactly; the script checks that before invoking signtool,
    because signtool's own failure for this is an opaque hex code. Omit to leave the package
    unsigned, which is enough for a Store upload but not enough to install locally.

.PARAMETER CertificatePassword
    Password for the .pfx. Omit for a passwordless one.

.PARAMETER SkipFreeze
    Reuse whatever is already in dist/ToneSphere instead of re-running PyInstaller. Only
    useful when iterating on the manifest or the logos.

.PARAMETER OutputDirectory
    Where the .msix lands. Defaults to build/msix/ under the repository root.

.EXAMPLE
    pwsh -File packaging/msix/build_msix.ps1

.EXAMPLE
    pwsh -File packaging/msix/build_msix.ps1 -CertificatePath .\ToneSphereTest.pfx -CertificatePassword hunter2
#>

# PowerShell 7, not the 5.1 that ships with Windows: this uses `Join-Path a b c` and
# `-Encoding utf8NoBOM`, both of which 5.1 rejects. Failing on the shebang line is better
# than failing three minutes into a PyInstaller run.
#Requires -Version 7.0

[CmdletBinding()]
param(
    [string]$CertificatePath,
    [string]$CertificatePassword = '',
    [switch]$SkipFreeze,
    [string]$OutputDirectory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$packagingDir = $PSScriptRoot
$repoRoot = (Resolve-Path (Join-Path $packagingDir '..' '..')).Path

function Fail([string]$message) {
    # PowerShell's error formatting collapses newlines, which turns a carefully written
    # multi-line remedy into one unreadable paragraph. Print it verbatim first, then throw a
    # one-liner so the exit code and the stack frame still come out right.
    [Console]::Error.WriteLine()
    [Console]::Error.WriteLine($message)
    [Console]::Error.WriteLine()
    throw 'MSIX build aborted; see the message above.'
}

function Find-SdkTool([string]$name) {
    # The Windows SDK does not put its tools on PATH, so "not on PATH" is the normal case
    # rather than an error -- but a stale or missing SDK is a real failure, and guessing at
    # a half-built package instead of saying so is exactly what this project forbids.
    $onPath = Get-Command $name -CommandType Application -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    $roots = @(
        (Join-Path ${env:ProgramFiles(x86)} 'Windows Kits\10\bin'),
        (Join-Path $env:ProgramFiles 'Windows Kits\10\bin')
    ) | Where-Object { $_ -and (Test-Path $_) }

    $candidates = foreach ($root in $roots) {
        foreach ($arch in @('x64', 'x86')) {
            Get-ChildItem -Path $root -Directory -ErrorAction SilentlyContinue |
                Where-Object { $_.Name -match '^10\.\d+\.\d+\.\d+$' } |
                ForEach-Object {
                    $candidate = Join-Path $_.FullName "$arch\$name"
                    if (Test-Path $candidate) {
                        [pscustomobject]@{ Version = [version]$_.Name; Path = $candidate }
                    }
                }
        }
    }

    $best = $candidates | Sort-Object Version -Descending | Select-Object -First 1
    if ($best) { return $best.Path }

    Fail @"
$name was not found.

It ships with the Windows 10/11 SDK, which is not on PATH by default. This script already
looked on PATH and under:
  ${env:ProgramFiles(x86)}\Windows Kits\10\bin\<version>\x64\$name
  $env:ProgramFiles\Windows Kits\10\bin\<version>\x64\$name

Fix it by either:
  * installing the SDK    -> winget install --id Microsoft.WindowsSDK.10.0.26100
    (or Visual Studio Installer > Individual components > "Windows 11 SDK"), or
  * running this script from a "Developer PowerShell for VS" prompt, which puts the SDK
    tools on PATH itself.

Nothing has been packed. dist/ and build/msix/layout/ may hold a staged layout from the
steps that did run; it is not a package.
"@
}

function Get-ProjectVersion {
    $pyprojectPath = Join-Path $repoRoot 'pyproject.toml'
    $initPath = Join-Path $repoRoot 'tonesphere\__init__.py'

    $pyprojectMatch = [regex]::Match((Get-Content $pyprojectPath -Raw), '(?m)^version\s*=\s*"([^"]+)"')
    if (-not $pyprojectMatch.Success) { Fail "could not read the 'version' key from $pyprojectPath" }

    $initMatch = [regex]::Match((Get-Content $initPath -Raw), '(?m)^__version__\s*=\s*"([^"]+)"')
    if (-not $initMatch.Success) { Fail "could not read '__version__' from $initPath" }

    $fromPyproject = $pyprojectMatch.Groups[1].Value
    $fromInit = $initMatch.Groups[1].Value

    if ($fromPyproject -ne $fromInit) {
        Fail "version mismatch: pyproject.toml says $fromPyproject, tonesphere/__init__.py says $fromInit. Fix both before packaging."
    }

    return $fromPyproject
}

function ConvertTo-MsixVersion([string]$projectVersion) {
    $parts = $projectVersion.Split('.')
    if ($parts.Count -lt 1 -or $parts.Count -gt 3 -or ($parts | Where-Object { $_ -notmatch '^\d+$' })) {
        Fail "project version '$projectVersion' is not 1-3 numeric parts; an MSIX version needs x.y.z with a zero fourth part."
    }
    while ($parts.Count -lt 3) { $parts += '0' }
    # The Store rejects a non-zero fourth part outright, so it is always literally 0.
    return ($parts -join '.') + '.0'
}

Write-Host '== ToneSphere MSIX build ==' -ForegroundColor Cyan

$makeappx = Find-SdkTool 'makeappx.exe'
Write-Host "makeappx: $makeappx"

$signtool = $null
if ($CertificatePath) {
    $signtool = Find-SdkTool 'signtool.exe'
    Write-Host "signtool: $signtool"
}

$projectVersion = Get-ProjectVersion
$msixVersion = ConvertTo-MsixVersion $projectVersion
Write-Host "version:  $projectVersion -> $msixVersion"

$manifestSource = Join-Path $packagingDir 'AppxManifest.xml'
if (-not (Test-Path $manifestSource)) { Fail "manifest not found: $manifestSource" }

# Loaded as XML, not text. Stamping the version with a regex was tried and was wrong:
# `MinVersion="10.0.17763.0"` also ends in `Version="` and is also four numeric parts, so
# the substitution silently rewrote the manifest's Windows floor to the app version and
# makeappx packed it without complaint. Addressing the attribute by path cannot do that.
$manifest = [xml](Get-Content $manifestSource -Raw)

$declaredVersion = $manifest.Package.Identity.Version
if ($declaredVersion -ne $msixVersion) {
    Write-Warning "AppxManifest.xml declares Version=`"$declaredVersion`"; packing $msixVersion from pyproject.toml instead. Update the manifest so tests/test_msix_packaging.py agrees."
}

$distApp = Join-Path $repoRoot 'dist\ToneSphere'
$exePath = Join-Path $distApp 'ToneSphere.exe'

if ($SkipFreeze) {
    Write-Host '-- skipping PyInstaller (-SkipFreeze)' -ForegroundColor Yellow
} else {
    Write-Host '-- freezing the app (uv run pyinstaller tonesphere.spec)'
    Push-Location $repoRoot
    try {
        & uv run pyinstaller --noconfirm tonesphere.spec
        if ($LASTEXITCODE -ne 0) { Fail "PyInstaller failed with exit code $LASTEXITCODE. Nothing has been packed." }
    } finally {
        Pop-Location
    }
}

if (-not (Test-Path $exePath)) {
    Fail "$exePath does not exist. Run this without -SkipFreeze, or run 'uv run pyinstaller tonesphere.spec' by hand first."
}

Write-Host '-- generating the logo set'
Push-Location $repoRoot
try {
    & uv run python (Join-Path $packagingDir 'generate_assets.py')
    if ($LASTEXITCODE -ne 0) { Fail "generate_assets.py failed with exit code $LASTEXITCODE. Nothing has been packed." }
} finally {
    Pop-Location
}

$assetsDir = Join-Path $packagingDir 'assets'
if (-not (Test-Path (Join-Path $assetsDir 'StoreLogo.png'))) {
    Fail "generate_assets.py reported success but $assetsDir does not hold the logo set."
}

$buildDir = if ($OutputDirectory) { $OutputDirectory } else { Join-Path $repoRoot 'build\msix' }
$layoutDir = Join-Path $buildDir 'layout'

Write-Host "-- staging the package layout in $layoutDir"
if (Test-Path $layoutDir) { Remove-Item -Recurse -Force $layoutDir }
New-Item -ItemType Directory -Force -Path $layoutDir | Out-Null

# The frozen app goes in its own subdirectory so that the package-root `assets\` (Store
# logos) cannot collide with the app's own bundled `assets\images\`, which PyInstaller puts
# under _internal/. Executable= in the manifest points at this path.
Copy-Item -Recurse -Path $distApp -Destination (Join-Path $layoutDir 'ToneSphere')
Copy-Item -Recurse -Path $assetsDir -Destination (Join-Path $layoutDir 'assets')

$manifest.Package.Identity.Version = $msixVersion
$manifest.Save((Join-Path $layoutDir 'AppxManifest.xml'))

$packagePath = Join-Path $buildDir "ToneSphere-$msixVersion-x64.msix"

Write-Host "-- packing $packagePath"
& $makeappx pack /d $layoutDir /p $packagePath /o
if ($LASTEXITCODE -ne 0) {
    Fail "makeappx pack failed with exit code $LASTEXITCODE. The layout in $layoutDir is intact; the output at $packagePath is not a usable package."
}

if ($CertificatePath) {
    $resolvedCert = (Resolve-Path $CertificatePath).Path
    $publisher = $manifest.Package.Identity.Publisher

    $certificate = New-Object System.Security.Cryptography.X509Certificates.X509Certificate2 -ArgumentList $resolvedCert, $CertificatePassword
    $normalise = { param($subject) ($subject -replace '\s', '').ToUpperInvariant() }

    if ((& $normalise $certificate.Subject) -ne (& $normalise $publisher)) {
        Fail @"
Certificate subject does not match the manifest Publisher, so signtool would fail with an
opaque error (0x8007000B / "publisher name does not match").

  manifest Publisher : $publisher
  certificate subject: $($certificate.Subject)

Either re-issue the test certificate with that exact subject:

  New-SelfSignedCertificate -Type Custom -Subject "$publisher" ``
    -KeyUsage DigitalSignature -FriendlyName "ToneSphere sideload test" ``
    -CertStoreLocation "Cert:\CurrentUser\My" ``
    -TextExtension @("2.5.29.37={text}1.3.6.1.5.5.7.3.3", "2.5.29.19={text}")

or change Publisher in packaging/msix/AppxManifest.xml to match the certificate you have.

The unsigned package at $packagePath is intact and can still be uploaded to Partner Center;
it just cannot be installed locally.
"@
    }

    Write-Host '-- signing'
    $signArgs = @('sign', '/fd', 'SHA256', '/f', $resolvedCert)
    if ($CertificatePassword) { $signArgs += @('/p', $CertificatePassword) }
    $signArgs += $packagePath

    & $signtool @signArgs
    if ($LASTEXITCODE -ne 0) { Fail "signtool failed with exit code $LASTEXITCODE. $packagePath is packed but unsigned." }
}

$sizeMb = [math]::Round((Get-Item $packagePath).Length / 1MB, 1)
Write-Host ''
Write-Host "Packed $packagePath ($sizeMb MB)" -ForegroundColor Green

if (-not $CertificatePath) {
    Write-Host 'Unsigned. Windows will refuse to install it until it is signed with a trusted certificate;'
    Write-Host 'pass -CertificatePath to sign a sideload build. See docs/MICROSOFT_STORE.md.'
} else {
    Write-Host "Signed. To install: Add-AppxPackage -Path `"$packagePath`""
    Write-Host '(the signing certificate must first be trusted in Cert:\LocalMachine\TrustedPeople)'
}

Write-Host ''
Write-Host 'Not done, and not doable from here: the Identity Name/Publisher in the manifest are'
Write-Host 'placeholders. Substitute the Partner Center values before submitting -- docs/MICROSOFT_STORE.md.'
