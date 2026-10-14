# Installs the desktop installer in a pristine test VM, launches NexTone like a user would
# (the app starts the engine), and runs the virtual-device end-to-end test against it.
#   .\vm\Test-Installer.ps1 [-Installer ui\release\NexTone-Setup-0.1.0.exe]
param([string]$VmName = 'NexTone-DriverTest', [string]$Installer, [int]$Seconds = 8)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vmcommon.ps1')
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
if (-not $Installer) { $Installer = Get-ChildItem (Join-Path $root 'ui\release\NexTone-Setup-*.exe') | Sort-Object LastWriteTime | Select-Object -Last 1 }
Restore-TestVM -VmName $VmName
$s = New-PSSession -VMName $VmName -Credential (Get-TestVMCredential)
try {
  Invoke-Command -Session $s { New-Item -ItemType Directory -Force C:\NexToneTest | Out-Null }
  foreach ($f in @($Installer, "$root\build\x64-release\bin\audio-probe.exe", (Get-Command node).Source, "$root\tests\integration\vad-audio.mjs")) {
    Copy-Item -ToSession $s -Path $f -Destination 'C:\NexToneTest\' -Force
  }
  Invoke-Command -Session $s -ArgumentList (Split-Path $Installer -Leaf), $Seconds {
    param($setup, $secs)
    $ErrorActionPreference = 'Continue'
    $p = Start-Process "C:\NexToneTest\$setup" -ArgumentList '/S' -Wait -PassThru
    "installer exit code: $($p.ExitCode)"
    & 'C:\Program Files\NexTone\resources\drivers\nextone-setup.exe' status
    # Launch the app in the signed-in user's session (schtasks needs a path without spaces).
    Set-Content C:\NexToneTest\launch.cmd '@start "" "C:\Program Files\NexTone\NexTone.exe"'
    schtasks /create /f /tn NexToneApp /sc once /st 00:00 /ru nextone /it /rl highest /tr C:\NexToneTest\launch.cmd 2>$null | Out-Null
    schtasks /run /tn NexToneApp 2>$null | Out-Null
    # First start can take a while (antivirus scans the newly installed files).
    $deadline = (Get-Date).AddSeconds(120)
    do { Start-Sleep -Seconds 2 } while (-not (Test-Path C:\Users\nextone\AppData\Local\NexTone\engine.json) -and (Get-Date) -lt $deadline)
    Start-Sleep -Seconds 2
    Get-Process -ErrorAction SilentlyContinue | Where-Object Name -like '*exTone*' | Select-Object Name, SessionId -Unique | Format-Table | Out-String
    $cmd = "cmd /c C:\NexToneTest\node.exe C:\NexToneTest\vad-audio.mjs C:\NexToneTest\audio-probe.exe $secs > C:\NexToneTest\out.txt 2>&1"
    schtasks /create /f /tn NexToneT /sc once /st 00:00 /ru nextone /it /rl highest /tr $cmd 2>$null | Out-Null
    schtasks /run /tn NexToneT 2>$null | Out-Null
    $deadline = (Get-Date).AddMinutes(4)
    do { Start-Sleep -Seconds 3 } while (-not (Test-Path C:\NexToneTest\vad-audio-report.json) -and (Get-Date) -lt $deadline)
    Get-Content C:\NexToneTest\out.txt
  }
} finally {
  Remove-PSSession $s
}
