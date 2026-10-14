# Installs the Windows download on a clean machine (a fresh CI runner: no Node, no compilers,
# no VC++ runtime) the way a user would, launches it, checks that the app started its engine and
# answers, then uninstalls it and checks nothing is left behind.
#   .github/scripts/smoke-windows.ps1 -Installer ToneSphere-Setup-1.0.0.exe -Version 1.0.0
param([Parameter(Mandatory)] [string]$Installer, [Parameter(Mandatory)] [string]$Version)
$ErrorActionPreference = 'Stop'
$failures = 0
function Check([string]$name, [bool]$ok, [string]$detail = '') {
  Write-Host ("{0}  {1}  {2}" -f ($(if ($ok) { 'PASS' } else { 'FAIL' })), $name, $detail)
  if (-not $ok) { $script:failures++ }
}

# Windows Server runners ship with the audio service off; a desktop PC has it on.
try { Set-Service Audiosrv -StartupType Automatic; Start-Service Audiosrv } catch { Write-Host "audio service: $_" }

$p = Start-Process $Installer -ArgumentList '/S' -Wait -PassThru
Check 'silent install succeeds' ($p.ExitCode -eq 0) "exit $($p.ExitCode)"
$dir = Join-Path $env:ProgramFiles 'ToneSphere'
foreach ($f in 'ToneSphere.exe', 'resources\engine\tonesphere-engine.exe', 'resources\engine\tonesphere-plugscan.exe',
  'resources\drivers\ToneSphereASIO64.dll', 'resources\drivers\ToneSphereASIO32.dll', 'resources\drivers\tonesphere-setup.exe') {
  Check "installed $f" (Test-Path (Join-Path $dir $f))
}
Check 'no test-signed driver in a public build' (-not (Test-Path (Join-Path $dir 'resources\drivers\vad')))
Check 'ToneSphere ASIO is registered for DAWs' (Test-Path 'HKLM:\SOFTWARE\ASIO\ToneSphere ASIO')
$status = & (Join-Path $dir 'resources\drivers\tonesphere-setup.exe') status | ConvertFrom-Json
Check 'setup tool reports the ASIO driver' ($status.asio -eq $true) ($status | ConvertTo-Json -Compress)

# Launch it like a user (no arguments). The app starts the engine, which writes its discovery file.
Remove-Item env:ELECTRON_RUN_AS_NODE -ErrorAction SilentlyContinue
$disc = Join-Path $env:LOCALAPPDATA 'ToneSphere\engine.json'
Start-Process (Join-Path $dir 'ToneSphere.exe')
$deadline = (Get-Date).AddSeconds(90)
while (-not (Test-Path $disc) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 500 }
Check 'the app starts the audio engine' (Test-Path $disc)
if (Test-Path $disc) {
  Start-Sleep -Seconds 2
  $d = Get-Content $disc -Raw | ConvertFrom-Json
  $body = '{"id":1,"method":"engine.getState","params":{}}'
  $r = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$($d.port)/rpc" -Headers @{ Authorization = "Bearer $($d.token)" } -ContentType 'application/json' -Body $body
  Check 'the engine answers with the released version' ($r.result.version -eq $Version) "version $($r.result.version), $(@($r.result.nodes).Count) nodes"
  Check 'the engine runs its audio clock' ($r.result.status.running -eq $true) "cpu $($r.result.status.cpuPercent)%"
  $app = Get-Process ToneSphere -ErrorAction SilentlyContinue
  Check 'the window process is running' ($null -ne $app) "$(@($app).Count) process(es)"
  # A second launch must not start another copy.
  $before = @(Get-Process tonesphere-engine -ErrorAction SilentlyContinue).Count
  Start-Process (Join-Path $dir 'ToneSphere.exe'); Start-Sleep -Seconds 5
  Check 'single instance: still one engine after a second launch' (@(Get-Process tonesphere-engine -ErrorAction SilentlyContinue).Count -eq $before) "$before"
  try {
    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$($d.port)/rpc" -Headers @{ Authorization = "Bearer $($d.token)" } -ContentType 'application/json' -Body '{"id":2,"method":"engine.shutdown"}' | Out-Null
  } catch {}
}
Get-Process ToneSphere -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
Get-Process tonesphere-engine -ErrorAction SilentlyContinue | Stop-Process -Force

# The crash handler works in the installed build (in a throwaway data folder).
$crashDir = Join-Path $env:TEMP "ts-smoke-crash-$PID"
$env:TONESPHERE_DATA_DIR = $crashDir
Start-Process (Join-Path $dir 'resources\engine\tonesphere-engine.exe') -ArgumentList '--crash-test' -Wait
Remove-Item env:TONESPHERE_DATA_DIR
Check 'a crash leaves a report' (@(Get-ChildItem (Join-Path $crashDir 'crashes') -Filter 'crash-*.txt' -ErrorAction SilentlyContinue).Count -eq 1)

$un = Join-Path $dir 'Uninstall ToneSphere.exe'
$p = Start-Process $un -ArgumentList '/S' -Wait -PassThru
$deadline = (Get-Date).AddSeconds(60)
while ((Test-Path (Join-Path $dir 'ToneSphere.exe')) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 500 }
Check 'uninstall removes the app' (-not (Test-Path (Join-Path $dir 'ToneSphere.exe')))
Check 'uninstall unregisters the ASIO driver' (-not (Test-Path 'HKLM:\SOFTWARE\ASIO\ToneSphere ASIO'))

if ($failures) { throw "$failures smoke check(s) failed" }
Write-Host 'all smoke checks passed'
