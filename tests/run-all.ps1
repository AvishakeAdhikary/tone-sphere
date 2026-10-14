# Runs every automated ToneSphere test that is safe on the host PC (no kernel driver needed).
#   .\tests\run-all.ps1            # unit + integration
#   .\tests\run-all.ps1 -SkipBuild
# Note: the suites change the engine's routing (the user's configuration): don't run them while
# relying on a setup in ToneSphere.
# Integration suites use VB-CABLE (+ FlexASIO configured on it) as a silent loopback.
param([switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
Get-Process tonesphere-engine -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Milliseconds 500
if (-not $SkipBuild) { .\scripts\build.ps1 -Test }

$engine = Start-Process .\build\x64-release\bin\tonesphere-engine.exe -PassThru
Start-Sleep -Seconds 3
$suites = @(
  @('host-audio', 'tests\integration\host-audio.mjs', '--soak', '10'),
  @('asio-clients', 'tests\integration\asio-clients.mjs', '8'),
  @('asio-host', 'tests\integration\asio-host.mjs', 'FlexASIO', '8'),
  @('vban', 'tests\integration\vban.mjs', '8'),
  @('recorder', 'tests\integration\recorder.mjs'),
  @('meters-hotkeys', 'tests\integration\meters-hotkeys.mjs'),
  @('plugins', 'tests\integration\plugins.mjs'),
  @('instruments', 'tests\integration\instruments.mjs'),
  @('latency', 'tests\integration\latency.mjs'),
  @('osc', 'tests\integration\osc.mjs'),
  @('presets', 'tests\integration\presets.mjs'),
  @('http-rpc', 'tests\integration\http-rpc.mjs'),
  @('crash-report', 'tests\integration\crash-report.mjs'),
  @('vbcable-signature', 'tests\integration\vbcable-signature.mjs')
)
$summary = @()
foreach ($s in $suites) {
  Write-Host "`n=== $($s[0]) ===" -ForegroundColor Cyan
  & node @($s[1..($s.Length - 1)])
  $summary += [pscustomobject]@{ Suite = $s[0]; Result = $(if ($LASTEXITCODE -eq 0) { 'PASS' } else { 'FAIL' }) }
}
Stop-Process -Id $engine.Id -Force -ErrorAction SilentlyContinue
Push-Location ui
npx playwright test
$summary += [pscustomobject]@{ Suite = 'ui-e2e'; Result = $(if ($LASTEXITCODE -eq 0) { 'PASS' } else { 'FAIL' }) }
Pop-Location
Write-Host ''
$summary | Format-Table -AutoSize
if ($summary.Result -contains 'FAIL') { exit 1 }
