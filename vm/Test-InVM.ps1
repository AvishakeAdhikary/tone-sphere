# Copies the engine + test tools into the Hyper-V test VM and runs the virtual-driver
# end-to-end test in the VM's interactive session (audio needs a real user session).
#   .\vm\Test-InVM.ps1 [-Seconds 10]
param([string]$VmName = 'ToneSphere-DriverTest', [int]$Seconds = 10)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vmcommon.ps1')
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
$bin = Join-Path $root 'build\x64-release\bin'
$node = (Get-Command node).Source

$s = New-PSSession -VMName $VmName -Credential (Get-TestVMCredential)
try {
  Invoke-Command -Session $s {
    Get-Process tonesphere-engine -ErrorAction SilentlyContinue | Stop-Process -Force
    New-Item -ItemType Directory -Force C:\ToneSphere\bin | Out-Null
    Remove-Item C:\ToneSphere\bin\vad-audio-report.json, C:\ToneSphere\bin\test-output.txt -ErrorAction SilentlyContinue
  }
  foreach ($f in @("$bin\tonesphere-engine.exe", "$bin\audio-probe.exe", $node, "$root\tools\tonesphere-ctl.mjs", "$root\tests\integration\vad-audio.mjs")) {
    Copy-Item -ToSession $s -Path $f -Destination 'C:\ToneSphere\bin\' -Force
  }
  Invoke-Command -Session $s -ArgumentList $Seconds {
    param($secs)
    $ErrorActionPreference = 'Continue'
    $user = 'tonesphere'
    # Run in the interactive session (session 1) via scheduled tasks.
    schtasks /create /f /tn ToneSphereEngine /sc once /st 00:00 /ru $user /it /rl highest /tr 'C:\ToneSphere\bin\tonesphere-engine.exe' 2>$null | Out-Null
    schtasks /run /tn ToneSphereEngine 2>$null | Out-Null
    Start-Sleep -Seconds 4
    $cmd = "cmd /c C:\ToneSphere\bin\node.exe C:\ToneSphere\bin\vad-audio.mjs C:\ToneSphere\bin\audio-probe.exe $secs > C:\ToneSphere\bin\test-output.txt 2>&1"
    schtasks /create /f /tn ToneSphereTest /sc once /st 00:00 /ru $user /it /rl highest /tr $cmd 2>$null | Out-Null
    schtasks /run /tn ToneSphereTest 2>$null | Out-Null
    $deadline = (Get-Date).AddMinutes(4)
    do { Start-Sleep -Seconds 3 } while (-not (Test-Path C:\ToneSphere\bin\vad-audio-report.json) -and (Get-Date) -lt $deadline)
    Start-Sleep -Seconds 1
    Get-Content C:\ToneSphere\bin\test-output.txt -ErrorAction SilentlyContinue
    '--- engine log (tail)'
    Get-Content "$env:LOCALAPPDATA\..\..\..\$user\AppData\Local\ToneSphere\engine.log" -Tail 25 -ErrorAction SilentlyContinue
  }
} finally {
  Remove-PSSession $s
}
