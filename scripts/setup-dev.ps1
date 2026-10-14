# Installs everything needed to build and test ToneSphere on Windows, from a fresh clone.
#   .\scripts\setup-dev.ps1            (run in an elevated PowerShell; safe to run again)
#   .\scripts\setup-dev.ps1 -NoVm      (skip the Hyper-V test VM)
# Installs: Visual Studio 2022 Build Tools (C++, Windows SDK 26100, CMake, Ninja), the Windows
# Driver Kit 10.0.26100, Node.js LTS, Git, FlexASIO (+ a FlexASIO.toml for the tests), then the
# UI's npm packages. VB-CABLE (the tests' audio loopback) is installed from the app or from
# https://vb-audio.com/Cable/ by hand: its licence doesn't allow scripted installs.
param([switch]$NoVm)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

function Install([string]$id, [string]$override) {
  $wargs = @('install', '--id', $id, '--exact', '--silent', '--accept-package-agreements', '--accept-source-agreements')
  if ($override) { $wargs += @('--override', $override) }
  Write-Host "== $id"
  winget @wargs | Select-Object -Last 2
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$haveVs = (Test-Path $vswhere) -and (& $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
if (-not $haveVs) {
  Install 'Microsoft.VisualStudio.2022.BuildTools' ('--quiet --wait --norestart --nocache ' +
    '--add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ' +
    '--add Microsoft.VisualStudio.Component.Windows11SDK.26100 --add Microsoft.VisualStudio.Component.VC.CMake.Project ' +
    '--add Microsoft.VisualStudio.Component.VC.ATL --includeRecommended')
}
if (-not (Test-Path "${env:ProgramFiles(x86)}\Windows Kits\10\Include\10.0.26100.0\km")) { Install 'Microsoft.WindowsWDK.10.0.26100' }
if (-not (Get-Command node -ErrorAction SilentlyContinue)) { Install 'OpenJS.NodeJS.LTS' }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { Install 'Git.Git' }
if (-not (Test-Path "$env:ProgramFiles\FlexASIO")) { Install 'dechamps.FlexASIO' }

# FlexASIO as a silent loopback on VB-CABLE, which tests/integration/asio-host.mjs expects.
# An existing configuration is left alone.
$toml = Join-Path $env:USERPROFILE 'FlexASIO.toml'
if (-not (Test-Path $toml)) {
  @'
# ToneSphere tests: FlexASIO plays into VB-CABLE and records from it (a silent loopback).
backend = "Windows WASAPI"
bufferSizeSamples = 480

[input]
device = "CABLE Output (VB-Audio Virtual Cable)"

[output]
device = "CABLE Input (VB-Audio Virtual Cable)"
'@ | Set-Content -Path $toml -Encoding utf8
  Write-Host "Wrote $toml"
}

# New tools are on PATH for new shells; this one gets them too.
$env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
Push-Location (Join-Path $root 'ui')
Remove-Item env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
npm ci --no-audit --no-fund
Pop-Location

if (-not $NoVm) {
  if ((Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All).State -ne 'Enabled') {
    Write-Warning 'Hyper-V is off. Driver tests need it: Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All (restart required), then run vm\New-TestVM.ps1.'
  } else {
    Write-Host 'Create the driver-test VM when you need it (about 30 minutes, downloads ~8 GB): .\vm\New-TestVM.ps1'
  }
}
if (-not (Test-Path 'HKLM:\SYSTEM\CurrentControlSet\Services\VBAudioVACMME')) {
  Write-Warning 'VB-CABLE is not installed. The integration tests need it: install it from https://vb-audio.com/Cable/ (or the app: Settings > Devices > Install drivers).'
}
Write-Host "`nReady. Build with .\scripts\build.ps1 -Test, then .\tests\run-all.ps1"
