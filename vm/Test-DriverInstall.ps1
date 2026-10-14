# Checks the app's "Install drivers" path in a pristine test VM:
#  1. start the engine first (no drivers yet),
#  2. run `nextone-setup install-all` (what Settings › Devices › Install drivers runs, elevated),
#  3. the running engine must pick up the ASIO driver and the virtual devices by itself.
#   .\vm\Test-DriverInstall.ps1
param([string]$VmName = 'NexTone-DriverTest')
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'vmcommon.ps1')
$root = Resolve-Path (Join-Path $PSScriptRoot '..')
$bin = Join-Path $root 'build\x64-release\bin'
Restore-TestVM -VmName $VmName
$s = New-PSSession -VMName $VmName -Credential (Get-TestVMCredential)
try {
  Invoke-Command -Session $s { New-Item -ItemType Directory -Force C:\NexTone\bin, C:\NexTone\drivers\vad | Out-Null }
  foreach ($f in @("$bin\nextone-engine.exe", "$bin\nextone-setup.exe", (Get-Command node).Source, "$root\tools\nextone-ctl.mjs")) {
    Copy-Item -ToSession $s -Path $f -Destination 'C:\NexTone\bin\' -Force
  }
  Copy-Item -ToSession $s -Path "$bin\NexToneASIO64.dll", "$root\build\x86-release\bin\NexToneASIO32.dll" -Destination 'C:\NexTone\drivers\' -Force
  Copy-Item -ToSession $s -Path "$root\drivers\win\vad\build\package\*" -Destination 'C:\NexTone\drivers\vad\' -Force
  Invoke-Command -Session $s {
    $ErrorActionPreference = 'Continue'
    $ctl = { & C:\NexTone\bin\node.exe C:\NexTone\bin\nextone-ctl.mjs json 2>$null | ConvertFrom-Json }
    $env:NEXTONE_DISCOVERY = 'C:\Users\nextone\AppData\Local\NexTone\engine.json'
    schtasks /create /f /tn NexToneEngine /sc once /st 00:00 /ru nextone /it /rl highest /tr 'C:\NexTone\bin\nextone-engine.exe' 2>$null | Out-Null
    schtasks /run /tn NexToneEngine 2>$null | Out-Null
    $deadline = (Get-Date).AddSeconds(60)
    do { Start-Sleep -Seconds 2 } while (-not (Test-Path $env:NEXTONE_DISCOVERY) -and (Get-Date) -lt $deadline)
    Start-Sleep -Seconds 3
    $before = & $ctl
    "before: driverPack=$($before.status.driverPack | ConvertTo-Json -Compress) virtual=$(@($before.nodes | Where-Object kind -eq 'vad').Count)"
    $t0 = Get-Date
    & C:\NexTone\bin\nextone-setup.exe install-all C:\NexTone\drivers --engine C:\NexTone\bin\nextone-engine.exe --ui C:\NexTone\bin\nextone-engine.exe
    "install-all exit code: $LASTEXITCODE ($([int]((Get-Date) - $t0).TotalSeconds) s)"
    $deadline = (Get-Date).AddSeconds(30)
    do {
      Start-Sleep -Seconds 2
      $after = & $ctl
      $vad = @($after.nodes | Where-Object { $_.kind -eq 'vad' -and $_.state -ne 'unavailable' })
    } while (($vad.Count -lt 8 -or -not $after.status.driverPack.asio) -and (Get-Date) -lt $deadline)
    "after:  driverPack=$($after.status.driverPack | ConvertTo-Json -Compress) virtual=$($vad.Count) ($($vad.name -join ', '))"
    "engine noticed the drivers by itself: $(($vad.Count -ge 8) -and $after.status.driverPack.vad -and $after.status.driverPack.asio)"
  }
} finally {
  Remove-PSSession $s
}
