# Installs the desktop installer in a pristine test VM, launches ToneSphere like a user would
# (the app starts the engine), and runs the virtual-device end-to-end test against it.
#   .\vm\Test-Installer.ps1 [-Installer ui\release\ToneSphere-Setup-<version>.exe]
param([string]$VmName = 'ToneSphere-DriverTest', [string]$Installer, [int]$Seconds = 8)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vmcommon.ps1')
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
if (-not $Installer) { $Installer = Get-ChildItem (Join-Path $root 'ui\release\ToneSphere-Setup-*.exe') | Sort-Object LastWriteTime | Select-Object -Last 1 }
Restore-TestVM -VmName $VmName
$s = New-PSSession -VMName $VmName -Credential (Get-TestVMCredential)
try {
  Invoke-Command -Session $s { New-Item -ItemType Directory -Force C:\ToneSphereTest | Out-Null }
  foreach ($f in @($Installer, "$root\build\x64-release\bin\audio-probe.exe", (Get-Command node).Source, "$root\tests\integration\vad-audio.mjs")) {
    Copy-Item -ToSession $s -Path $f -Destination 'C:\ToneSphereTest\' -Force
  }
  Invoke-Command -Session $s -ArgumentList (Split-Path $Installer -Leaf), $Seconds {
    param($setup, $secs)
    $ErrorActionPreference = 'Continue'
    $p = Start-Process "C:\ToneSphereTest\$setup" -ArgumentList '/S' -Wait -PassThru
    "installer exit code: $($p.ExitCode)"
    & 'C:\Program Files\ToneSphere\resources\drivers\tonesphere-setup.exe' status
    # Launch the app in the signed-in user's session (schtasks needs a path without spaces).
    Set-Content C:\ToneSphereTest\launch.cmd '@start "" "C:\Program Files\ToneSphere\ToneSphere.exe"'
    schtasks /create /f /tn ToneSphereApp /sc once /st 00:00 /ru tonesphere /it /rl highest /tr C:\ToneSphereTest\launch.cmd 2>$null | Out-Null
    schtasks /run /tn ToneSphereApp 2>$null | Out-Null
    # First start can take a while (antivirus scans the newly installed files).
    $deadline = (Get-Date).AddSeconds(120)
    do { Start-Sleep -Seconds 2 } while (-not (Test-Path C:\Users\tonesphere\AppData\Local\ToneSphere\engine.json) -and (Get-Date) -lt $deadline)
    Start-Sleep -Seconds 2
    Get-Process -ErrorAction SilentlyContinue | Where-Object Name -like '*oneSphere*' | Select-Object Name, SessionId -Unique | Format-Table | Out-String
    $cmd = "cmd /c C:\ToneSphereTest\node.exe C:\ToneSphereTest\vad-audio.mjs C:\ToneSphereTest\audio-probe.exe $secs > C:\ToneSphereTest\out.txt 2>&1"
    schtasks /create /f /tn ToneSphereT /sc once /st 00:00 /ru tonesphere /it /rl highest /tr $cmd 2>$null | Out-Null
    schtasks /run /tn ToneSphereT 2>$null | Out-Null
    $deadline = (Get-Date).AddMinutes(4)
    do { Start-Sleep -Seconds 3 } while (-not (Test-Path C:\ToneSphereTest\vad-audio-report.json) -and (Get-Date) -lt $deadline)
    Get-Content C:\ToneSphereTest\out.txt
  }
} finally {
  Remove-PSSession $s
}
