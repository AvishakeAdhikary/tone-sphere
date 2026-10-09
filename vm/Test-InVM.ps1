# Copies the engine + test tools into the Hyper-V test VM and runs the virtual-driver
# end-to-end test in the VM's interactive session (audio needs a real user session).
#   .\vm\Test-InVM.ps1 [-Seconds 10]
param([string]$VmName = 'NexTone-DriverTest', [int]$Seconds = 10)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vmcommon.ps1')
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
$bin = Join-Path $root 'build\x64-release\bin'
$node = (Get-Command node).Source

$s = New-PSSession -VMName $VmName -Credential (Get-TestVMCredential)
try {
  Invoke-Command -Session $s {
    Get-Process nextone-engine -ErrorAction SilentlyContinue | Stop-Process -Force
    New-Item -ItemType Directory -Force C:\NexTone\bin | Out-Null
    Remove-Item C:\NexTone\bin\vad-audio-report.json, C:\NexTone\bin\test-output.txt -ErrorAction SilentlyContinue
  }
  foreach ($f in @("$bin\nextone-engine.exe", "$bin\audio-probe.exe", $node, "$root\tools\nextone-ctl.mjs", "$root\tests\integration\vad-audio.mjs")) {
    Copy-Item -ToSession $s -Path $f -Destination 'C:\NexTone\bin\' -Force
  }
  Invoke-Command -Session $s -ArgumentList $Seconds {
    param($secs)
    $ErrorActionPreference = 'Continue'
    $user = 'nextone'
    # Run in the interactive session (session 1) via scheduled tasks.
    schtasks /create /f /tn NexToneEngine /sc once /st 00:00 /ru $user /it /rl highest /tr 'C:\NexTone\bin\nextone-engine.exe' 2>$null | Out-Null
    schtasks /run /tn NexToneEngine 2>$null | Out-Null
    Start-Sleep -Seconds 4
    $cmd = "cmd /c C:\NexTone\bin\node.exe C:\NexTone\bin\vad-audio.mjs C:\NexTone\bin\audio-probe.exe $secs > C:\NexTone\bin\test-output.txt 2>&1"
    schtasks /create /f /tn NexToneTest /sc once /st 00:00 /ru $user /it /rl highest /tr $cmd 2>$null | Out-Null
    schtasks /run /tn NexToneTest 2>$null | Out-Null
    $deadline = (Get-Date).AddMinutes(4)
    do { Start-Sleep -Seconds 3 } while (-not (Test-Path C:\NexTone\bin\vad-audio-report.json) -and (Get-Date) -lt $deadline)
    Start-Sleep -Seconds 1
    Get-Content C:\NexTone\bin\test-output.txt -ErrorAction SilentlyContinue
    '--- engine log (tail)'
    Get-Content "$env:LOCALAPPDATA\..\..\..\$user\AppData\Local\NexTone\engine.log" -Tail 25 -ErrorAction SilentlyContinue
  }
} finally {
  Remove-PSSession $s
}
